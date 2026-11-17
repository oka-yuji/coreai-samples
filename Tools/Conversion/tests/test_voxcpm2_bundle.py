import hashlib
import importlib
import json
import os
from pathlib import Path
from string import Template
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from Tools.Conversion.adapters.voxcpm2 import bundle, bundle_check
from Tools.Conversion.adapters.voxcpm2.bundle import (absolute, canaries, clone_tree, inventory, manifest, path_hits, patterns,
                                                      scan, verify_inventory)
from Tools.Conversion.adapters.voxcpm2.runtime import entries
from Tools.Conversion.adapters.voxcpm2.source import load_contract
from Tools.Conversion.storage import file_hash

RECIPE = Path(__file__).resolve().parents[1] / "recipes" / "voxcpm2"
CONTRACT = load_contract(RECIPE / "model.json")
CONFIGURATIONS = json.loads((RECIPE / "synthesis.json").read_text(encoding="utf-8"))["compare"]["configurations"]
TEMPLATE = Template((RECIPE / "bundle-notice.txt").read_text(encoding="utf-8"))


def fake_bundle(root):
    files = {"manifest.json": b"{}", "tokenizer.json": b"{}", "LICENSE": b"Apache License Version 2.0",
             "artifacts/proj-fp32.aimodel/main.mlirb": b"\x00\x01" * 64, "artifacts/proj-fp32.aimodel/metadata.json": b"{}"}
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    return files


def stripped_assets(directory):
    records = {}
    for asset in CONTRACT["assets"]:
        for precision in ("fp32", "fp16"):
            folder = directory / f"{asset}-{precision}.aimodel"
            folder.mkdir(parents=True)
            for name in ("main.mlirb", "main.hash", "metadata.json"):
                (folder / name).write_bytes(f"{asset} {precision} {name}".encode())
            records[(asset, precision)] = {name: file_hash(folder / name) for name in ("main.hash", "main.mlirb", "metadata.json")}
    return records


class Sources:
    def __init__(self, directory):
        directory = Path(directory)
        self.stripped = directory / "stripped"
        self.records = stripped_assets(self.stripped)
        self.tokenizer, self.license = directory / "tokenizer.json", directory / "LICENSE"
        self.tokenizer.write_bytes(b"{\"model\": {}}")
        self.license.write_bytes(b"Apache License Version 2.0")
        self.contract = {**CONTRACT, "files": {**CONTRACT["files"], "tokenizer.json": file_hash(self.tokenizer)}}
        self.pin = file_hash(self.license)
        self.root = directory / "bundle"

    def build(self, name="fp16_proj32"):
        return bundle.build(self.root, name, CONFIGURATIONS[name], self.contract, self.stripped, self.records, self.tokenizer,
                            self.license, TEMPLATE, self.pin)


class VoxCPM2ManifestTests(unittest.TestCase):
    def test_every_configuration_copies_the_contract(self):
        architecture = CONTRACT["architecture"]
        for name, precisions in CONFIGURATIONS.items():
            with self.subTest(name):
                content = manifest(CONTRACT, name, precisions)
                self.assertEqual(content["model"], {key: CONTRACT[key] for key in ("model_id", "revision", "upstream_revision", "license")})
                self.assertEqual(content["configuration"], {"name": name, "precisions": precisions})
                self.assertEqual((content["execution"], content["generation"]), (CONTRACT["execution"], CONTRACT["generation"]))
                self.assertEqual(content["structure"], {"patch": [1, architecture["feat_dim"], architecture["patch_size"]],
                                                        "sample_rate": architecture["sample_rate"]})
                self.assertEqual(content["tokenizer"], "tokenizer.json")
                for asset, entry in content["assets"].items():
                    precision = precisions[asset]
                    self.assertEqual((entry["path"], entry["precision"]), (f"artifacts/{asset}-{precision}.aimodel", precision))
                    for function, kinds in entry["functions"].items():
                        for kind, rows in kinds.items():
                            self.assertEqual(rows, entries(CONTRACT, asset, function, kind, precision))
                text = json.dumps(content)
                self.assertEqual([token for token in patterns(user="tester") if token.decode("utf-8", "ignore") in text], [])

    def test_half_precision_entries_keep_integer_inputs(self):
        content = manifest(CONTRACT, "fp16", CONFIGURATIONS["fp16"])
        inputs = dict((name, dtype) for name, dtype, _ in content["assets"]["tslm"]["functions"]["step"]["inputs"])
        self.assertEqual(inputs, {"embed": "float16", "position": "int32"})

    def test_incomplete_or_unknown_precisions_are_rejected(self):
        for precisions in ({"tslm": "fp32"}, {**CONFIGURATIONS["fp32"], "dit": "int8"}):
            with self.subTest(precisions=precisions), self.assertRaises(ValueError):
                manifest(CONTRACT, "broken", precisions)


class VoxCPM2NoticeTests(unittest.TestCase):
    def test_half_precision_assets_are_listed_only_when_present(self):
        full = bundle.notice(TEMPLATE, CONTRACT, "fp32", CONFIGURATIONS["fp32"])
        self.assertNotIn("16-bit", full)
        self.assertIn("All assets keep 32-bit floating point.", full)
        mixed = bundle.notice(TEMPLATE, CONTRACT, "fp16_proj32", CONFIGURATIONS["fp16_proj32"])
        self.assertIn("16-bit floating point: dit, locenc, ralm, tslm, vae. In those assets,", mixed)

    def test_the_usage_note_cites_the_model_card_without_adding_license_terms(self):
        text = bundle.notice(TEMPLATE, CONTRACT, "fp16", CONFIGURATIONS["fp16"])
        self.assertIn("The original model card asks", text)
        self.assertIn("This notice does not change the license terms.", text)
        self.assertNotIn("Do not use", text)
        self.assertNotIn("$", text)


class VoxCPM2InventoryTests(unittest.TestCase):
    def test_inventory_lists_every_file_with_its_size_and_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = fake_bundle(root)
            listing = inventory(root)
            self.assertEqual([entry["path"] for entry in listing["files"]], sorted(files))
            for entry in listing["files"]:
                data = files[entry["path"]]
                self.assertEqual((entry["bytes"], entry["sha256"]), (len(data), hashlib.sha256(data).hexdigest()))
            self.assertEqual((listing["count"], listing["bytes"]), (len(files), sum(map(len, files.values()))))
            verify_inventory(root, listing)

    def test_extra_missing_changed_and_linked_files_are_rejected(self):
        def extra(root):
            (root / "README.md").write_bytes(b"x")

        def missing(root):
            (root / "LICENSE").unlink()

        def changed(root):
            (root / "tokenizer.json").write_bytes(b"[]")

        def linked(root):
            (root / "LICENSE").unlink()
            os.symlink(root / "tokenizer.json", root / "LICENSE")

        for change in (extra, missing, changed, linked):
            with self.subTest(change.__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fake_bundle(root)
                listing = inventory(root)
                change(root)
                with self.assertRaises(ValueError):
                    verify_inventory(root, listing)

    def test_clones_copy_files_record_the_method_and_refuse_links(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            source.mkdir()
            files = fake_bundle(source)
            counts = clone_tree(source, target)
            self.assertEqual(sum(counts.values()), len(files))
            self.assertEqual(inventory(source)["files"], inventory(target)["files"])
            self.assertFalse(any(path.is_symlink() for path in target.rglob("*")))
            with mock.patch.object(bundle.ctypes, "CDLL", return_value=SimpleNamespace(clonefile=lambda *arguments: -1)):
                self.assertEqual(clone_tree(source, Path(directory) / "copied"), {"cloned": 0, "copied": len(files)})
            self.assertEqual(inventory(source)["files"], inventory(Path(directory) / "copied")["files"])
            os.symlink(source / "LICENSE", source / "linked")
            with self.assertRaises(ValueError):
                clone_tree(source, Path(directory) / "other")


class VoxCPM2BundleSourceTests(unittest.TestCase):
    def test_a_bundle_matching_its_sources_is_listed(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = Sources(directory)
            listing, copies = sources.build()
            self.assertEqual(listing, inventory(sources.root))
            self.assertEqual(listing["count"], 22)
            self.assertEqual(sum(copies.values()), 18)
            for asset, precision in CONFIGURATIONS["fp16_proj32"].items():
                recorded = sources.records[(asset, precision)]
                listed = {entry["path"].rsplit("/", 1)[-1]: {"bytes": entry["bytes"], "sha256": entry["sha256"]}
                          for entry in listing["files"] if entry["path"].startswith(f"artifacts/{asset}-{precision}.aimodel/")}
                self.assertEqual(listed, recorded)

    def test_one_byte_changed_while_copying_fails(self):
        def corrupt(source, target):
            target.parent.mkdir(parents=True, exist_ok=True)
            data = bytearray(source.read_bytes())
            if source.name == "main.mlirb" and source.parent.name == "dit-fp16.aimodel":
                data[0] ^= 1
            target.write_bytes(bytes(data))
            return False

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(bundle, "clone", side_effect=corrupt):
            with self.assertRaises(ValueError):
                Sources(directory).build()

    def test_stripped_files_outside_the_record_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = Sources(directory)
            (sources.stripped / "vae-fp16.aimodel" / "extra.bin").write_bytes(b"x")
            with self.assertRaises(ValueError):
                sources.build()

    def test_tokenizer_and_license_must_match_their_pins(self):
        for name in ("tokenizer", "license"):
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                sources = Sources(directory)
                getattr(sources, name).write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    sources.build()

    def test_the_license_pin_is_the_file_of_the_pinned_checkout(self):
        self.assertEqual(bundle.UPSTREAM_LICENSE, {"bytes": 11298,
                                                   "sha256": "4f10acc209addacfad28293315c74c4cd648f771ee1263a748f1781d1e0265e4"})


class VoxCPM2PathScanTests(unittest.TestCase):
    def samples(self):
        home = absolute("Users", "someone", "work")
        return {"home": ("x" + home + "y").encode(), "volume": absolute("Volumes", "disk").encode(),
                "temporary": absolute("private", "var", "folders", "aa", "T").encode(),
                "utf16le": home.encode("utf-16-le"), "utf16be": home.encode("utf-16-be"),
                "utf16be_end": absolute("Users", "").encode("utf-16-be"),
                "json": home.replace("/", "\\/").encode(), "unicode": home.replace("/", "\\u002F").encode(),
                "unicode_lower": home.replace("/", "\\u002f").encode(), "percent": home.replace("/", "%2F").encode(),
                "percent_lower": home.replace("/", "%2f").encode(),
                "json_utf16be": home.replace("/", "\\/").encode("utf-16-be"), "user": b"owner tester",
                "boundary": b"a" * 13 + absolute("Volumes", "").encode() + b"b" * 20}

    def test_local_paths_are_found_in_every_encoding_escape_and_across_chunks(self):
        tokens = patterns(user="tester")
        with tempfile.TemporaryDirectory() as directory:
            for name, data in self.samples().items():
                path = Path(directory) / name
                path.write_bytes(data)
                with self.subTest(name):
                    self.assertGreater(path_hits(path, tokens, chunk=16), 0)
            for name, data in {"clean": b"artifacts/tslm-fp32.aimodel/main.mlirb " * 10,
                               "escaped_relative": b"artifacts\\/tslm-fp32.aimodel%2Fmain.mlirb"}.items():
                path = Path(directory) / name
                path.write_bytes(data)
                with self.subTest(name):
                    self.assertEqual(path_hits(path, tokens, chunk=16), 0)

    def test_a_match_split_by_chunks_is_counted_once(self):
        tokens = patterns(user="tester")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data"
            volume = absolute("Volumes", "").encode()
            path.write_bytes(b"a" * 13 + volume + b"b" * 40 + volume + b"c" * 7)
            self.assertEqual([path_hits(path, tokens, chunk=size) for size in (16, 17, 1024)], [2, 2, 2])

    def test_scan_reports_counts_and_file_names_only(self):
        tokens = patterns(user="tester")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_bundle(root)
            (root / "notes.txt").write_bytes(absolute("Users", "someone", "work").encode())
            result = scan(root, tokens)
            self.assertEqual((result["files"], result["files_with_hits"]), (6, ["notes.txt"]))
            self.assertGreater(result["hits"], 0)
            self.assertNotIn("someone", json.dumps(result))

    def test_canaries_cover_escapes_and_both_utf16_byte_orders(self):
        found = canaries(patterns(user="tester"), user="tester")
        self.assertTrue({"utf16le", "utf16be", "json", "unicode", "percent", "boundary", "user"} <= set(found["detected"]))
        self.assertTrue(all(found["detected"].values()))
        self.assertFalse(found["clean_detected"])


class VoxCPM2VerdictTests(unittest.TestCase):
    def report(self, **changes):
        report = {"bundles": {name: {"path_scan": {"hits": 0}} for name in CONFIGURATIONS},
                  "canaries": {"detected": {"utf8": True, "json": True}, "clean_detected": False}}
        for key, value in changes.items():
            report[key] = value
        return report

    def test_bundles_pass_only_without_paths_with_working_canaries_and_every_configuration(self):
        self.assertTrue(bundle.verdict(self.report(), list(CONFIGURATIONS)))
        failing = {
            "path": self.report(bundles={**self.report()["bundles"], "fp16": {"path_scan": {"hits": 1}}}),
            "canary": self.report(canaries={"detected": {"utf8": True, "json": False}, "clean_detected": False}),
            "clean": self.report(canaries={"detected": {"utf8": True, "json": True}, "clean_detected": True}),
            "missing": self.report(bundles={"fp32": {"path_scan": {"hits": 0}}}),
        }
        for name, report in failing.items():
            with self.subTest(name):
                self.assertFalse(bundle.verdict(report, list(CONFIGURATIONS)))

    def test_strip_requires_changed_bytecode_and_equal_programs(self):
        strip = importlib.import_module("Tools.Conversion.adapters.voxcpm2.strip")
        before = {"main.hash": {"bytes": 32, "sha256": "1" * 64}, "main.mlirb": {"bytes": 100, "sha256": "2" * 64}}
        after = {"main.hash": {"bytes": 32, "sha256": "3" * 64}, "main.mlirb": {"bytes": 90, "sha256": "4" * 64}}
        digest, other = {"sha256": "5" * 64, "bytes": 9}, {"sha256": "6" * 64, "bytes": 9}
        good = strip.record("proj", "fp32", digest, digest, digest, before, after)
        self.assertTrue(strip.passed([good] * 12, 12))
        failing = {
            "unchanged bytecode": strip.record("proj", "fp32", digest, digest, digest, before, before),
            "changed program": strip.record("proj", "fp32", digest, other, other, before, after),
            "changed after reload": strip.record("proj", "fp32", digest, digest, other, before, after),
        }
        for name, row in failing.items():
            with self.subTest(name):
                self.assertFalse(strip.passed([good] * 11 + [row], 12))
        self.assertFalse(strip.passed([good] * 11, 12))

    def test_bundle_assets_are_loaded_from_inside_the_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root, scratch = Path(directory) / "fp16_proj32", Path(directory) / "scratch"
            root.mkdir()
            content = manifest(CONTRACT, "fp16_proj32", CONFIGURATIONS["fp16_proj32"])
            (root / "manifest.json").write_text(json.dumps(content))
            groups = bundle_check.shim(root, scratch, CONTRACT)
            self.assertEqual({precision: len(directories) for precision, directories in groups.items()}, {"fp16": 5, "fp32": 1})
            for precision, directories in groups.items():
                for folder in directories:
                    export = json.loads((folder / "export.json").read_text())
                    path = Path(export["precisions"][precision]["file"])
                    self.assertEqual(path, (root / content["assets"][export["asset"]]["path"]).resolve())
                    self.assertTrue(path.is_relative_to(root.resolve()))

    def test_the_recheck_passes_only_with_the_classification_before_removal(self):
        summary = {"dit.main.velocity": {"count": 2151, "passed": 2151, "gate": True, "normalized_max": 4.6e-6},
                   "vae.main.chunk_audio": {"count": 239, "passed": 229, "gate": False, "normalized_max": 1.3e-3}}
        fsq = {"calls": 234, "match": 233, "boundary": 1, "fail": 0}
        baseline = {"summary": summary, "fsq": fsq}
        result = bundle_check.judge(summary, fsq, True, baseline)
        self.assertTrue(result["passed"])
        self.assertTrue(result["errors_equal_before_removal"])
        changed_count = {**summary, "dit.main.velocity": {**summary["dit.main.velocity"], "passed": 2150}}
        failing = {
            "criterion": bundle_check.judge(summary, fsq, False, baseline),
            "counts": bundle_check.judge(changed_count, fsq, True, baseline),
            "fsq": bundle_check.judge(summary, {**fsq, "match": 232, "boundary": 2}, True, baseline),
        }
        for name, outcome in failing.items():
            with self.subTest(name):
                self.assertFalse(outcome["passed"])
        moved = {**summary, "dit.main.velocity": {**summary["dit.main.velocity"], "normalized_max": 4.7e-6}}
        outcome = bundle_check.judge(moved, fsq, True, baseline)
        self.assertEqual((outcome["passed"], outcome["errors_equal_before_removal"]), (True, False))


if __name__ == "__main__":
    unittest.main()
