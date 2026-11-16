import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from Tools.Conversion.adapters.voxcpm2.bundle import (absolute, canaries, clone_tree, inventory, manifest, path_hits, patterns,
                                                      scan, verify_inventory)
from Tools.Conversion.adapters.voxcpm2.runtime import entries
from Tools.Conversion.adapters.voxcpm2.source import load_contract

RECIPE = Path(__file__).resolve().parents[1] / "recipes" / "voxcpm2"
CONTRACT = load_contract(RECIPE / "model.json")
CONFIGURATIONS = json.loads((RECIPE / "synthesis.json").read_text(encoding="utf-8"))["compare"]["configurations"]


def fake_bundle(root):
    files = {"manifest.json": b"{}", "tokenizer.json": b"{}", "LICENSE": b"Apache License Version 2.0",
             "artifacts/proj-fp32.aimodel/main.mlirb": b"\x00\x01" * 64, "artifacts/proj-fp32.aimodel/metadata.json": b"{}"}
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    return files


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
                                                        "sample_rate": architecture["sample_rate"],
                                                        "samples_per_patch": CONTRACT["generation"]["samples_per_patch"]})
                self.assertEqual(content["tokenizer"], "tokenizer.json")
                for asset, entry in content["assets"].items():
                    precision = precisions[asset]
                    self.assertEqual((entry["path"], entry["precision"]), (f"artifacts/{asset}-{precision}.aimodel", precision))
                    for function, kinds in entry["functions"].items():
                        for kind, rows in kinds.items():
                            self.assertEqual(rows, entries(CONTRACT, asset, function, kind, precision))
                text = json.dumps(content)
                self.assertEqual([token.decode() for token in patterns() if token.decode("utf-8", "ignore") in text], [])

    def test_half_precision_entries_keep_integer_inputs(self):
        content = manifest(CONTRACT, "fp16", CONFIGURATIONS["fp16"])
        inputs = dict((name, dtype) for name, dtype, _ in content["assets"]["tslm"]["functions"]["step"]["inputs"])
        self.assertEqual(inputs, {"embed": "float16", "position": "int32"})

    def test_incomplete_or_unknown_precisions_are_rejected(self):
        for precisions in ({"tslm": "fp32"}, {**CONFIGURATIONS["fp32"], "dit": "int8"}):
            with self.subTest(precisions=precisions), self.assertRaises(ValueError):
                manifest(CONTRACT, "broken", precisions)


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

    def test_clones_copy_files_and_refuse_links(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            source.mkdir()
            files = fake_bundle(source)
            clone_tree(source, target)
            self.assertEqual(inventory(source)["files"], inventory(target)["files"])
            self.assertFalse(any(path.is_symlink() for path in target.rglob("*")))
            os.symlink(source / "LICENSE", source / "linked")
            with self.assertRaises(ValueError):
                clone_tree(source, Path(directory) / "other")
            self.assertEqual(len(files), 5)


class VoxCPM2PathScanTests(unittest.TestCase):
    def test_local_paths_are_found_in_every_encoding_and_across_chunks(self):
        tokens = patterns(user="someone")
        samples = {"home": ("x" + absolute("Users", "someone", "y")).encode(), "user": b"owner someone",
                   "volume": absolute("Volumes", "disk").encode(),
                   "temporary": absolute("private", "var", "folders", "aa", "T").encode(),
                   "utf16": absolute("Users", "someone").encode("utf-16-le"),
                   "boundary": b"a" * 13 + absolute("Volumes", "").encode() + b"b" * 20}
        with tempfile.TemporaryDirectory() as directory:
            for name, data in samples.items():
                path = Path(directory) / name
                path.write_bytes(data)
                with self.subTest(name):
                    self.assertGreater(path_hits(path, tokens, chunk=16), 0)
            clean = Path(directory) / "clean"
            clean.write_bytes(b"artifacts/tslm-fp32.aimodel/main.mlirb " * 10)
            self.assertEqual(path_hits(clean, tokens, chunk=16), 0)

    def test_a_match_split_by_chunks_is_counted_once(self):
        tokens = patterns(user="someone")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data"
            volume = absolute("Volumes", "").encode()
            path.write_bytes(b"a" * 13 + volume + b"b" * 40 + volume + b"c" * 7)
            self.assertEqual([path_hits(path, tokens, chunk=size) for size in (16, 17, 1024)], [2, 2, 2])

    def test_scan_reports_counts_and_file_names_only(self):
        tokens = patterns(user="someone")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_bundle(root)
            (root / "notes.txt").write_bytes(absolute("Users", "someone", "work").encode())
            result = scan(root, tokens)
            self.assertEqual((result["files"], result["files_with_hits"]), (6, ["notes.txt"]))
            self.assertGreater(result["hits"], 0)
            self.assertNotIn("someone", json.dumps(result))
        found = canaries(tokens, user="someone")
        self.assertTrue(all(found["detected"].values()))
        self.assertFalse(found["clean_detected"])


if __name__ == "__main__":
    unittest.main()
