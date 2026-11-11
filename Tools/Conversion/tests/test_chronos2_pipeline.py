import asyncio
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from Tools.Conversion.adapters.chronos2 import bundle, verify
from Tools.Conversion.adapters.chronos2.artifacts import inventory, require_equal
from Tools.Conversion.storage import read_json, write_json
from Tools.Conversion.tests.test_chronos2_support import ChronosFiles, inspected_asset, predictions


class Array:
    def __init__(self, values):
        self.values = values

    def numpy(self):
        return self.values


class NativeTests(ChronosFiles, unittest.TestCase):
    def test_native_runner_preserves_missing_values_and_all_output_vectors(self):
        calls = []
        async def function(*, inputs, state):
            self.assertEqual(state, {})
            calls.append(inputs["context"].numpy().copy())
            return {"quantiles": Array(np.asarray(predictions(len(calls) - 1), dtype=np.float32))}
        actuals = asyncio.run(verify.execute_cases(function, self.fixtures, Array))
        self.assertEqual(actuals, self.values)
        self.assertEqual(len(calls), 18)
        self.assertEqual(calls[0].shape, (1, 16))
        self.assertEqual(calls[0].dtype, np.float32)
        self.assertTrue(np.isnan(calls[0][0, :15]).all())
        self.assertEqual(np.isnan(calls[12]).sum(), 6)
        self.assertEqual(calls[7].shape, (1, 8192))

    def test_native_runner_rejects_wrong_dtype_shape_nonfinite_and_output_names(self):
        arrays = [np.zeros((1, 21, 16), dtype=np.float16), np.zeros((1, 21, 16), dtype=np.float64),
                  np.zeros((1, 16, 21), dtype=np.float32), np.zeros((1, 21, 15), dtype=np.float32),
                  np.full((1, 21, 16), np.nan, dtype=np.float32), np.full((1, 21, 16), np.inf, dtype=np.float32)]
        for array in arrays:
            async def function(*, inputs, state):
                return {"quantiles": Array(array)}
            with self.subTest(dtype=array.dtype, shape=array.shape), self.assertRaises(ValueError):
                asyncio.run(verify.execute_cases(function, self.fixtures, Array))
        async def extra(*, inputs, state):
            return {"quantiles": Array(np.zeros((1, 21, 16), dtype=np.float32)), "state": Array(np.zeros(1))}
        with self.assertRaises(ValueError):
            asyncio.run(verify.execute_cases(extra, self.fixtures, Array))

    def test_inspection_requires_exact_stateless_fp32_contract(self):
        mutations = [
            lambda r: r["summary"]["functions"].append(r["summary"]["functions"][0]),
            lambda r: r["summary"]["functions"][0].update(name="other"),
            lambda r: r["summary"]["functions"][0].update(states=["cache"]),
            lambda r: r["summary"]["functions"][0]["inputs"][0].update(type="NDArray (Float32, 1 × 1008)"),
            lambda r: r["summary"]["functions"][0]["outputs"][0].update(type="NDArray (Float32, 1 × 16 × 21)"),
            lambda r: r["summary"]["storageTypes"][0].update(name="Float16"),
            lambda r: r["summary"]["storageTypes"][0].update(count=True),
            lambda r: r["summary"]["storageTypes"].clear(),
            lambda r: r["summary"]["computeTypes"].append("Float16"),
            lambda r: r["summary"].update(stateNames=["cache"]),
            lambda r: r["metadata"]["creatorDefinedMetadata"].update(source_revision="0" * 40),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                report = inspected_asset()
                mutate(report)
                with self.assertRaises(ValueError):
                    verify.validate_inspection(report)
        sanitized = verify.validate_inspection(inspected_asset())
        self.assertNotIn("path", sanitized)
        self.assertNotIn("creationDate", sanitized["metadata"])

    def test_verify_uses_argv_and_preserves_failing_comparison_with_vectors(self):
        self.make_export()
        bad = copy.deepcopy(self.values)
        bad[-1][0][-1][-1] += 1
        import json
        with patch.object(verify.subprocess, "check_output", return_value=json.dumps(inspected_asset())) as command, \
                patch.object(verify, "run_native", return_value=bad), self.assertRaises(ValueError):
            verify.verify(self.model_path, self.fixture_path, self.ready, self.reference_path,
                          self.export, self.root / "failed")
        argv = command.call_args.args[0]
        self.assertEqual(argv[:3], ["xcrun", "coreai-build", "inspect"])
        self.assertEqual(argv[4:], ["--io", "--metadata", "--storage", "--compute", "--json"])
        result = read_json(self.root / "failed/comparison.json")
        self.assertFalse(result["passed"])
        self.assertEqual((result["count"], result["elements"]), (18, 6048))
        self.assertEqual(read_json(self.root / "failed/actuals.json")["cases"][-1]["quantiles"], bad[-1])

    def test_verify_detects_asset_changes_before_native_load(self):
        self.make_export()
        (self.export / "Model.aimodel/functions/main.mlirb").write_bytes(b"tampered")
        with patch.object(verify, "run_native") as native, self.assertRaises(ValueError):
            verify.verify(self.model_path, self.fixture_path, self.ready, self.reference_path,
                          self.export, self.root / "out")
        native.assert_not_called()


class BundleTests(ChronosFiles, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.make_verified()

    def package(self, name="out"):
        return bundle.bundle(self.model_path, self.fixture_path, self.snapshot, self.ready,
                             self.reference_path, self.export, self.verified, self.root / name)

    def test_bundle_inventories_every_file_and_sanitizes_native_paths(self):
        manifest = self.package()
        self.assertEqual(manifest["format"], "coreai-samples-chronos2/1")
        target = self.root / "out/bundle"
        bundle.validate_bundle(target, manifest)
        complete = read_json(self.root / "out/inventory.json")["files"]
        self.assertIn("chronos2.json", complete)
        self.assertNotIn("chronos2.json", manifest["files"])
        require_equal(complete, inventory(target), "external inventory")
        self.assertEqual((target / "SOURCE_README.md").read_bytes(), (self.snapshot / "README.md").read_bytes())
        self.assertEqual((target / "LICENSE").read_bytes(), (self.recipe / "LICENSE").read_bytes())
        self.assertEqual((target / "NOTICE").read_bytes(), (self.recipe / "NOTICE").read_bytes())
        self.assertNotIn("inspection-raw.json", complete)
        for name in ("chronos2.json", "reference.json", "verification.json"):
            self.assertNotIn(str(self.root), (target / name).read_text())
            self.assertNotIn(inspected_asset()["path"], (target / name).read_text())
        with self.assertRaises(ValueError):
            self.package()
        bundle.validate_bundle(target, manifest)

    def test_actual_vectors_override_forged_pass_flags(self):
        actuals = read_json(self.verified / "actuals.json")
        actuals["cases"][-1]["quantiles"][0][-1][-1] += 1
        write_json(self.verified / "actuals.json", actuals)
        self.assertTrue(read_json(self.verified / "comparison.json")["passed"])
        with self.assertRaises(ValueError):
            self.package()
        self.assertFalse((self.root / "out/bundle").exists())

    def test_incomplete_extra_duplicate_and_reordered_cases_cannot_package(self):
        original = read_json(self.verified / "actuals.json")
        for index, mutate in enumerate((lambda c: c.pop(), lambda c: c.append(c[0]),
                                        lambda c: c[1].update(id=c[0]["id"]), lambda c: c.reverse(),
                                        lambda c: c[0].pop("quantiles"))):
            actuals = copy.deepcopy(original)
            mutate(actuals["cases"])
            write_json(self.verified / "actuals.json", actuals)
            with self.assertRaises(ValueError):
                self.package(f"out-{index}")

    def test_changed_reference_asset_and_wrapper_results_cannot_package(self):
        reference = read_json(self.reference_path)
        reference["cases"][0]["quantiles"][0][0][0] += 1
        write_json(self.reference_path, reference)
        with self.assertRaises(ValueError):
            self.package("reference-changed")
        write_json(self.reference_path, self.reference)
        wrapper = read_json(self.export / "wrapper-actuals.json")
        wrapper["cases"][-1]["quantiles"][0][-1][-1] += 1
        write_json(self.export / "wrapper-actuals.json", wrapper)
        with self.assertRaises(ValueError):
            self.package("wrapper-changed")

    def test_unknown_files_and_symlink_assets_cannot_package(self):
        asset = self.export / "Model.aimodel/functions/main.mlirb"
        original = asset.read_bytes()
        (self.export / "Model.aimodel/extra").write_text("extra")
        with self.assertRaises(ValueError):
            self.package("extra")
        (self.export / "Model.aimodel/extra").unlink()
        asset.unlink()
        other = self.root / "asset"
        other.write_bytes(original)
        asset.symlink_to(other)
        with self.assertRaises(ValueError):
            self.package("symlink")

    def test_copy_failures_and_changed_copies_leave_no_partial_bundle(self):
        with patch.object(bundle.shutil, "copytree", side_effect=OSError("copy failed")), self.assertRaises(OSError):
            self.package("copy-failed")
        self.assertEqual(list((self.root / "copy-failed").iterdir()), [])
        copyfile = bundle.shutil.copyfile
        def corrupt(source, target, *args, **kwargs):
            result = copyfile(source, target, *args, **kwargs)
            if Path(target).name == "NOTICE":
                Path(target).write_text("altered notice")
            return result
        with patch.object(bundle.shutil, "copyfile", side_effect=corrupt), self.assertRaises(ValueError):
            self.package("changed-copy")
        self.assertEqual(list((self.root / "changed-copy").iterdir()), [])

    def test_completed_bundle_rejects_extra_files_and_unsafe_manifest_paths(self):
        manifest = self.package()
        target = self.root / "out/bundle"
        extra = target / "unexpected"
        extra.write_text("extra")
        with self.assertRaises(ValueError):
            bundle.validate_bundle(target, manifest)
        extra.unlink()
        value = manifest["files"].pop("NOTICE")
        manifest["files"]["../NOTICE"] = value
        write_json(target / "chronos2.json", manifest)
        with self.assertRaises(ValueError):
            bundle.validate_bundle(target, manifest)


if __name__ == "__main__":
    unittest.main()
