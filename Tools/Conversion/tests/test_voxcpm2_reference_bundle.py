import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from Tools.Conversion.adapters.voxcpm2 import bundle, reference_bundle, strip
from Tools.Conversion.adapters.voxcpm2.source import load_contract
from Tools.Conversion.recipe import load_recipe
from Tools.Conversion.storage import PipelineError, check_results, file_hash, read_json, write_json

ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "Tools/Conversion/recipes/voxcpm2"
CONTRACT = load_contract(RECIPE / "model.json")
FP32 = {asset: "fp32" for asset in CONTRACT["assets"]}
FILES = ("main.hash", "main.mlirb", "metadata.json")
DESCRIPTOR = {"inputs": [["pcm", "float32", [1, 1, 161280]]], "outputs": [["mu", "float32", [1, 64, 252]]], "states": []}
REFERENCE = {"sample_rate": 16000, "minimum_samples": 80000, "maximum_samples": 160000, "hop_samples": 640,
             "start_token": 103, "end_token": 104, "padding": "right_zero"}
AUTHORING = """import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace


def log(event, path):
    with open(os.environ["VOXCPM2_FAKE_LOG"], "a", encoding="utf-8") as handle:
        handle.write(json.dumps([event, os.getpid(), Path(path).name]) + "\\n")


class Operation:
    def __init__(self, program):
        self.program = program

    def verify(self):
        if not isinstance(self.program.body["ops"], list):
            raise ValueError("invalid program")

    def print(self, *, file, binary, enable_debug_info, print_generic_op_form, large_elements_limit, large_resource_limit):
        if (binary, enable_debug_info, print_generic_op_form, large_elements_limit, large_resource_limit) != (True, False, True, None, None):
            raise ValueError("unexpected printing options")
        file.write(json.dumps(self.program.body["ops"]).encode())


class Program:
    def __init__(self, body, source):
        self.body, self.source = body, source
        self._module = SimpleNamespace(_mlir_module=SimpleNamespace(operation=Operation(self)))

    def save_asset(self, target, metadata):
        target = Path(target)
        target.mkdir()
        if os.environ.get("VOXCPM2_FAKE_FAULT") == "source":
            (self.source / "metadata.json").write_text("{}")
        data = json.dumps(self.body, sort_keys=True).encode()
        (target / "main.mlirb").write_bytes(data)
        (target / "main.hash").write_bytes(hashlib.sha256(data).digest())
        (target / "metadata.json").write_text(json.dumps(metadata, sort_keys=True))
        log("save", target)


class AIModelAsset:
    @staticmethod
    def load(path):
        path = Path(path)
        body = json.loads((path / "main.mlirb").read_bytes())
        if os.environ.get("VOXCPM2_FAKE_FAULT") == "reload" and "debug" not in body:
            body["ops"].append("reloaded")
        log("load", path)
        return SimpleNamespace(program=Program(body, path), metadata=json.loads((path / "metadata.json").read_bytes()))
"""
DEBUG_INFO = """import os


def strip_debug_info(program):
    fault = os.environ.get("VOXCPM2_FAKE_FAULT")
    if fault != "noop":
        program.body.pop("debug", None)
    if fault == "change":
        program.body["ops"].reverse()
    if fault == "leak":
        program.body["note"] = "/".join(("", "Volumes", "build", "cache"))
"""


def write_asset(folder, body, metadata):
    folder.mkdir(parents=True)
    data = json.dumps(body, sort_keys=True).encode()
    (folder / "main.mlirb").write_bytes(data)
    (folder / "main.hash").write_bytes(hashlib.sha256(data).digest())
    (folder / "metadata.json").write_text(json.dumps(metadata, sort_keys=True))


def flip(path):
    data = path.read_bytes()
    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])


def passing_report(files):
    sizes = {name: files[name]["bytes"] for name in FILES}
    measured = {"count": 4, "passed": 4, "nonfinite": 0, "normalized_max": 0.0005}
    return copy.deepcopy({"version": 1, "stage": "complete", "complete": True, "passed": True, "precision": "fp32",
                          "contract": {**{key: CONTRACT[key] for key in ("model_id", "revision", "upstream_revision", "files")},
                                       "assets": {"refenc": {"functions": {"main": DESCRIPTOR}}}},
                          "descriptor": DESCRIPTOR, "source_encoder": {"unchanged": True},
                          "summary": {"cases": 4, "required_cases": 4, "passed_cases": 4, "original_inputs_matched": 4,
                                      "fixed": measured, "coreai": measured},
                          "export": {"file": "refenc-fp32.aimodel", "files": sizes, "bytes": sum(sizes.values()),
                                     "main_mlirb_sha256": files["main.mlirb"]["sha256"]}})


def snapshot(*roots):
    return [{path.relative_to(root).as_posix(): file_hash(path) for path in sorted(root.rglob("*")) if path.is_file()}
            if root.is_dir() else file_hash(root) for root in roots]


class Fixture:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.base, self.run, self.output = self.directory / "base", self.directory / "run", self.directory / "out"
        self.raw, self.selection = self.run / "refenc-fp32.aimodel", self.directory / "selection.json"
        self.inventory, self.events = self.directory / "trusted.inventory.json", self.directory / "events.jsonl"
        self.base.mkdir()
        manifest = json.dumps(bundle.manifest(CONTRACT, "fp32", FP32), indent=2) + "\n"
        for name, text in (("LICENSE", "license"), ("NOTICE", "notice"), ("tokenizer.json", "{}"), ("manifest.json", manifest)):
            (self.base / name).write_text(text)
        for asset in CONTRACT["assets"]:
            write_asset(self.base / bundle.asset_path(asset, "fp32"), {"ops": [asset]}, {"asset": asset})
        self.trust()
        write_asset(self.raw, {"ops": ["encode", "pool"], "debug": bundle.absolute("Users", "builder", "encoder.py")},
                    {"asset": "refenc"})
        self.save_report(passing_report({name: file_hash(self.raw / name) for name in FILES}))

    def trust(self):
        write_json(self.inventory, bundle.inventory(self.base))

    def save_report(self, report):
        write_json(self.run / "report.json", report)
        self.select()

    def select(self, change=None):
        value = {"format": 1, "profile": "reference_fp32", "report": file_hash(self.run / "report.json"),
                 "files": {name: file_hash(self.raw / name) for name in FILES}}
        if change:
            change(value)
        write_json(self.selection, value)

    @contextlib.contextmanager
    def runtime(self, fault=""):
        shims = self.directory / "shims"
        sources = {"coreai/__init__.py": "", "coreai/authoring.py": AUTHORING, "coreai_torch/__init__.py": "",
                   "coreai_torch/debugging/__init__.py": "", "coreai_torch/debugging/debug_info.py": DEBUG_INFO}
        for name, text in sources.items():
            (shims / name).parent.mkdir(parents=True, exist_ok=True)
            (shims / name).write_text(text)
        environment = {"PYTHONPATH": os.pathsep.join((str(shims), str(ROOT))), "PYTHONDONTWRITEBYTECODE": "1",
                       "VOXCPM2_FAKE_LOG": str(self.events), "VOXCPM2_FAKE_FAULT": fault}
        with mock.patch.dict(os.environ, environment), mock.patch.dict(sys.modules), \
                mock.patch.object(sys, "path", [str(shims), *sys.path]), \
                mock.patch.object(reference_bundle, "INVENTORY", self.inventory):
            for name in [name for name in sys.modules if name.partition(".")[0] in ("coreai", "coreai_torch")]:
                del sys.modules[name]
            yield

    def build(self, fault="", output=None):
        printed = io.StringIO()
        with self.runtime(fault), contextlib.redirect_stdout(printed):
            reference_bundle.main(["--base-bundle", str(self.base), "--encoder-run", str(self.run),
                                   "--selection", str(self.selection), "--output", str(output or self.output)])
        return json.loads(printed.getvalue())

    def history(self):
        return [json.loads(line) for line in self.events.read_text().splitlines()] if self.events.exists() else []


class ReferenceBundleTests(unittest.TestCase):
    def fixture(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Fixture(directory.name)

    def test_the_selected_encoder_and_the_published_base_become_one_checked_bundle(self):
        f = self.fixture()
        before = snapshot(f.base, f.run, f.selection)
        printed = f.build()
        root = f.output / "reference_fp32"
        listing = bundle.inventory(root)
        files = {row["path"]: {"bytes": row["bytes"], "sha256": row["sha256"]} for row in listing["files"]}
        base = snapshot(f.base)[0]
        self.assertEqual(set(files), set(base) | {"artifacts/refenc-fp32.aimodel/" + name for name in FILES})
        self.assertEqual(listing["count"], 25)
        for name in set(base) - {"manifest.json", "NOTICE"}:
            self.assertEqual(files[name], base[name], name)
        expected = copy.deepcopy(read_json(f.base / "manifest.json"))
        expected["format"] = 2
        expected["configuration"] = {"name": "reference_fp32", "precisions": {**FP32, "refenc": "fp32"}}
        expected["assets"]["refenc"] = {"path": "artifacts/refenc-fp32.aimodel", "precision": "fp32",
                                        "functions": {"main": DESCRIPTOR}}
        expected["reference"] = REFERENCE
        self.assertEqual(read_json(root / "manifest.json"), expected)
        notice = (root / "NOTICE").read_text()
        self.assertIn("VoxCPM2 Core AI bundle (reference_fp32)", notice)
        self.assertIn("All assets keep 32-bit floating point.", notice)
        self.assertEqual(read_json(f.output / "reference_fp32.inventory.json"), listing)
        check = read_json(f.output / "check.json")
        self.assertEqual((check["complete"], check["passed"], check["files"], check["bytes"]), (True, True, 25, listing["bytes"]))
        self.assertEqual(check["selection_sha256"], file_hash(f.selection)["sha256"])
        self.assertEqual(check["base_inventory_sha256"], file_hash(f.inventory)["sha256"])
        self.assertEqual(check["manifest_sha256"], files["manifest.json"]["sha256"])
        self.assertEqual(check["inventory_sha256"], file_hash(f.output / "reference_fp32.inventory.json")["sha256"])
        self.assertEqual({key: printed[key] for key in ("passed", "selection_sha256", "manifest_sha256", "inventory_sha256")},
                         {key: check[key] for key in ("passed", "selection_sha256", "manifest_sha256", "inventory_sha256")})
        row = check["strip"]
        self.assertTrue(row["semantic_equal"] and row["bytecode_changed"])
        self.assertEqual(row["files_before"], snapshot(f.raw)[0])
        self.assertEqual(row["files_after"], snapshot(root / "artifacts/refenc-fp32.aimodel")[0])
        self.assertEqual((check["path_scan"]["files"], check["path_scan"]["hits"]), (25, 0))
        self.assertGreater(check["before_removal"]["hits"], 0)
        self.assertTrue(all(check["canaries"]["detected"].values()) and not check["canaries"]["clean_detected"])
        self.assertEqual(snapshot(f.base, f.run, f.selection), before)
        events = f.history()
        self.assertEqual({pid for event, pid, _ in events if event == "save"}, {os.getpid()})
        self.assertTrue({pid for event, pid, _ in events if event == "load"} - {os.getpid()})

    def test_bad_inputs_are_rejected_before_any_asset_is_loaded(self):
        def linked(path):
            outside = path.parent.parent / f"{path.parent.name}-{path.name}"
            path.rename(outside)
            path.symlink_to(outside)

        def untrusted_manifest(f):
            content = read_json(f.base / "manifest.json")
            content["configuration"]["name"] = "fp16"
            (f.base / "manifest.json").write_text(json.dumps(content, indent=2) + "\n")
            f.trust()

        def unlisted_notice(f):
            (f.base / "NOTICE").unlink()
            f.trust()

        def boolean_one_byte(f):
            (f.raw / "metadata.json").write_bytes(b"0")
            f.save_report(passing_report({name: file_hash(f.raw / name) for name in FILES}))
            f.select(lambda v: v["files"]["metadata.json"].update(bytes=True))

        cases = {
            "duplicate key": lambda f: f.selection.write_text(f.selection.read_text().replace('"format": 1', '"format": 1, "format": 1')),
            "unknown field": lambda f: f.select(lambda v: v.update(other=1)),
            "missing pin": lambda f: f.select(lambda v: v["files"].pop("main.hash")),
            "extra pin": lambda f: f.select(lambda v: v["files"].update(aux=v["report"])),
            "path as name": lambda f: f.select(lambda v: v["files"].update({"../metadata.json": v["files"].pop("metadata.json")})),
            "boolean size": lambda f: f.select(lambda v: v["report"].update(bytes=True)),
            "boolean size of a one-byte file": boolean_one_byte,
            "zero size": lambda f: f.select(lambda v: v["files"]["main.hash"].update(bytes=0)),
            "uppercase digest": lambda f: f.select(lambda v: v["report"].update(sha256=v["report"]["sha256"].upper())),
            "boolean format": lambda f: f.select(lambda v: v.update(format=True)),
            "other profile": lambda f: f.select(lambda v: v.update(profile="fp16")),
            "other digest": lambda f: f.select(lambda v: v["files"]["main.mlirb"].update(sha256="0" * 64)),
            "other size": lambda f: f.select(lambda v: v["files"]["metadata.json"].update(bytes=v["files"]["metadata.json"]["bytes"] + 1)),
            "failed report": lambda f: f.save_report({**read_json(f.run / "report.json"), "passed": False}),
            "changed encoder": lambda f: flip(f.raw / "metadata.json"),
            "extra encoder file": lambda f: (f.raw / "extra.bin").write_bytes(b"x"),
            "missing encoder file": lambda f: (f.raw / "main.hash").unlink(),
            "linked encoder file": lambda f: linked(f.raw / "main.hash"),
            "changed base file": lambda f: flip(f.base / "tokenizer.json"),
            "extra base file": lambda f: (f.base / "README.md").write_bytes(b"x"),
            "linked base file": lambda f: linked(f.base / "LICENSE"),
            "base manifest of another configuration": untrusted_manifest,
            "trusted inventory without a file": unlisted_notice,
        }
        for name, change in cases.items():
            with self.subTest(name):
                f = self.fixture()
                change(f)
                with self.assertRaises((ValueError, OSError)):
                    f.build()
                self.assertEqual(f.history(), [])
                self.assertFalse(os.path.lexists(f.output))

    def test_reports_must_meet_the_fixed_fp32_criteria(self):
        files = {name: {"bytes": 10 + index, "sha256": str(index) * 64} for index, name in enumerate(FILES)}
        reference_bundle.eligible(passing_report(files), files, CONTRACT)
        changes = [(["version"], True), (["stage"], "export"), (["complete"], False), (["passed"], False),
                   (["precision"], "fp16"), (["contract", "revision"], "f" * 40), (["contract", "upstream_revision"], "f" * 40),
                   (["contract", "assets", "refenc", "functions", "main", "states"], [["s", "float32", [1]]]),
                   (["descriptor", "outputs"], [["mu", "float32", [1, 252, 64]]]), (["source_encoder", "unchanged"], False),
                   (["summary", "cases"], 3), (["summary", "passed_cases"], 3), (["summary", "original_inputs_matched"], 3),
                   (["summary", "fixed", "passed"], 3), (["summary", "coreai", "count"], 4.0), (["summary", "coreai", "nonfinite"], 1),
                   (["summary", "fixed", "normalized_max"], None), (["summary", "coreai", "normalized_max"], True),
                   (["summary", "coreai", "normalized_max"], math.nextafter(0.001, math.inf)),
                   (["export", "file"], "../refenc-fp32.aimodel"), (["export", "files", "main.hash"], 11),
                   (["export", "bytes"], 1), (["export", "main_mlirb_sha256"], "f" * 64)]
        for keys, value in changes:
            with self.subTest(keys=keys, value=value):
                report = passing_report(files)
                target = report
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                with self.assertRaisesRegex(ValueError, "not an eligible FP32 encoder run"):
                    reference_bundle.eligible(report, files, CONTRACT)
        boundary = passing_report(files)
        boundary["summary"]["coreai"]["normalized_max"] = 0.001
        reference_bundle.eligible(boundary, files, CONTRACT)

    def test_a_failed_removal_or_a_remaining_path_leaves_no_passing_record(self):
        broken = {"detected": {"utf8": False}, "clean_detected": False}
        noisy = {"detected": {"utf8": True}, "clean_detected": True}
        for fault, canaries, field in (("noop", None, "bytecode_changed"), ("change", None, "semantic_equal"),
                                       ("reload", None, "semantic_equal"), ("leak", None, "path_scan"),
                                       ("source", None, None), ("", broken, "canaries"), ("", noisy, "canaries")):
            with self.subTest(fault=fault, canaries=canaries):
                f = self.fixture()
                scanner = mock.patch.object(bundle, "canaries", return_value=canaries) if canaries else contextlib.nullcontext()
                with scanner, self.assertRaises((RuntimeError, ValueError)):
                    f.build(fault)
                check = read_json(f.output / "check.json")
                self.assertIs(check["passed"], False)
                self.assertFalse((f.output / "reference_fp32.inventory.json").exists())
                if field in ("bytecode_changed", "semantic_equal"):
                    self.assertIs(check["strip"][field], False)
                elif field == "path_scan":
                    self.assertGreater(check["path_scan"]["hits"], 0)
                elif field == "canaries":
                    self.assertEqual((check["complete"], check["path_scan"]["hits"]), (True, 0))
                else:
                    self.assertNotIn("strip", check)
        f = self.fixture()
        original = strip.strip_one

        def damaged(asset, precision, source, target):
            row = original(asset, precision, source, target)
            (target / "metadata.json").write_text("{}")
            return row

        with mock.patch.object(strip, "strip_one", damaged), self.assertRaisesRegex(ValueError, "does not match its inventory"):
            f.build()
        self.assertIs(read_json(f.output / "check.json")["passed"], False)

    def test_an_existing_or_overlapping_output_is_refused(self):
        f = self.fixture()
        f.output.mkdir()
        (f.output / "keep").write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            f.build()
        self.assertEqual([path.name for path in f.output.iterdir()], ["keep"])
        before = snapshot(f.base, f.run)
        for output in (f.base / "artifacts" / "new", f.run / "new"):
            with self.subTest(output.parent.name), self.assertRaisesRegex(ValueError, "outside the base bundle"):
                f.build(output=output)
        self.assertEqual(snapshot(f.base, f.run), before)
        self.assertEqual(f.history(), [])

    def test_strip_one_compares_three_programs_and_needs_a_new_target(self):
        f = self.fixture()
        with f.runtime():
            row = strip.strip_one("refenc", "fp32", f.raw, f.directory / "stripped.aimodel")
            self.assertEqual(sorted(row), ["asset", "bytecode_changed", "files_after", "files_before", "precision", "semantic",
                                           "semantic_equal"])
            self.assertTrue(row["semantic_equal"] and row["bytecode_changed"])
            self.assertEqual((row["files_before"], row["files_after"]), tuple(snapshot(f.raw, f.directory / "stripped.aimodel")))
            self.assertEqual(row["semantic"]["bytes"], len(json.dumps(["encode", "pool"])))
            loads = [pid for event, pid, _ in f.history() if event == "load"]
            self.assertEqual(len(loads), 2)
            self.assertNotEqual(loads[0], loads[1])
            for target, error in ((f.directory / "stripped.aimodel", FileExistsError), (f.raw / "nested.aimodel", ValueError)):
                with self.subTest(target.name), self.assertRaises(error):
                    strip.strip_one("refenc", "fp32", f.raw, target)
            self.assertEqual(len(f.history()), 3)
            self.assertEqual(sorted(path.name for path in f.raw.iterdir()), sorted(FILES))

    def test_the_component_strip_shares_strip_one_and_still_requires_twelve_assets(self):
        f = self.fixture()
        export = f.directory / "proj-export"
        for precision in ("fp32", "fp16"):
            write_asset(export / f"proj-{precision}.aimodel", {"ops": [precision], "debug": "loc"}, {"precision": precision})
        write_json(export / "export.json", {"complete": True, "asset": "proj", "contract": CONTRACT,
                                            "precisions": {p: {"file": f"proj-{p}.aimodel"} for p in ("fp32", "fp16")}})
        calls, original = [], strip.strip_one

        def counted(*arguments):
            calls.append(arguments[:2])
            return original(*arguments)

        arguments = SimpleNamespace(contract=RECIPE / "model.json", export=[export], output=f.directory / "stripped")
        with f.runtime(), mock.patch.object(strip, "strip_one", counted), self.assertRaises(RuntimeError):
            strip.strip(arguments)
        report = read_json(f.directory / "stripped" / "strip.json")
        self.assertEqual((report["complete"], report["passed"], len(report["assets"])), (True, False, 2))
        self.assertEqual(sorted(calls), [("proj", "fp16"), ("proj", "fp32")])
        self.assertTrue(all(row["semantic_equal"] and row["bytecode_changed"] for row in report["assets"]))


class ReferenceBundleRecipeTests(unittest.TestCase):
    def stage(self):
        bindings = {"python": sys.executable, "base_bundle": "/base", "encoder_run": "/run", "selection": "/selection.json"}
        recipe = load_recipe(RECIPE / "reference-bundle-recipe.json", bindings, ROOT)
        self.assertEqual(len(recipe["steps"]), 1)
        return recipe["steps"][0]

    def test_the_recipe_passes_the_selected_run_to_the_bundle_command(self):
        stage = self.stage()
        self.assertEqual((stage["role"], stage["heavy"], stage["cwd"]), ("verify", True, "$repo"))
        self.assertEqual(stage["command"], ["$python", "-m", "Tools.Conversion.adapters.voxcpm2.reference_bundle",
                                            "--base-bundle", "$base_bundle", "--encoder-run", "$encoder_run",
                                            "--selection", "$selection", "--output", "$out/bundle"])
        self.assertEqual(stage["outputs"], ["bundle/reference_fp32", "bundle/reference_fp32.inventory.json", "bundle/check.json"])
        required = {"$adapters/" + name + ".py" for name in ("reference_bundle", "strip", "bundle", "runtime", "source")}
        required |= {"$repo/Tools/Conversion/storage.py", "$recipe_dir/model.json", "$recipe_dir/bundle-notice.txt",
                     "$repo/CoreAISamplesKit/Sources/SpeechAssets/Resources/voxcpm2-fp32.inventory.json", "$base_bundle",
                     "$selection", "$encoder_run/report.json", "$encoder_run/refenc-fp32.aimodel"}
        self.assertEqual(set(stage["inputs"]), required)
        with self.assertRaises(PipelineError):
            load_recipe(RECIPE / "reference-bundle-recipe.json", {"python": sys.executable}, ROOT)

    def test_only_a_complete_passing_check_is_accepted(self):
        stage = self.stage()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle" / "check.json"
            path.parent.mkdir()
            path.write_text(json.dumps({"complete": True, "passed": True}))
            check_results(Path(directory), stage["checks"])
            for change in ({"complete": False}, {"passed": False}, {"passed": 1}):
                path.write_text(json.dumps({"complete": True, "passed": True, **change}))
                with self.subTest(change), self.assertRaises(PipelineError):
                    check_results(Path(directory), stage["checks"])

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "The exported encoder descriptor needs PyTorch")
    def test_the_manifest_describes_the_exported_encoder(self):
        from Tools.Conversion.adapters.voxcpm2 import reference_encoder

        crops = {seconds: count * reference_encoder.SAMPLE_RATE // reference_encoder.SOURCE_RATE
                 for seconds, count in reference_encoder.CROPS.items()}
        self.assertEqual(reference_bundle.DESCRIPTOR, reference_encoder.descriptor()["functions"]["main"])
        self.assertEqual(reference_bundle.REFERENCE, {**REFERENCE, "sample_rate": reference_encoder.SAMPLE_RATE,
                                                      "hop_samples": reference_encoder.HOP_SAMPLES,
                                                      "minimum_samples": crops["5"], "maximum_samples": crops["10"]})
        self.assertLess(reference_bundle.REFERENCE["maximum_samples"], reference_encoder.MAXIMUM_SAMPLES)


if __name__ == "__main__":
    unittest.main()
