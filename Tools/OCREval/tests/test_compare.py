import json
import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

from Tools.OCREval.compare import metrics, output_comparison, prepared_inputs
from Tools.OCREval import compare
from Tools.OCREval.page_reference import page_plan, read_page, restore_weights
from Tools.OCREval.reference import sha256


class PageTests(unittest.TestCase):
    def test_page_uses_original_crops_table_prompt_and_preserves_skips(self):
        calls = []
        layout = "<box:500 0 500 100><label:text><up><box:0 100 1000 900><label:table><up>"
        def recognize(image, prompt):
            calls.append((image.size, prompt))
            return {"text": layout if prompt == "layout" else "<fcel>A<nl>", "finished_by": "eos"}
        result = read_page(Image.new("RGB", (100, 200)), recognize)
        self.assertEqual(calls, [((1036, 1036), "layout"), ((100, 160), "table")])
        self.assertEqual(result["blocks"][0]["box_px"], [0, 20, 100, 180])
        self.assertEqual(result["blocks"][0]["otsl"], "<fcel>A<nl>")
        self.assertEqual(len(result["layout"]["skipped_boxes"]), 1)
        self.assertTrue(output_comparison(result, result)["exact_prediction"])
        result["layout"]["finished_by"] = "max_new_tokens"
        self.assertFalse(output_comparison(result, result)["both_eos"])

    def test_invalid_aspect_falls_back_but_outside_pixels_remain_valid(self):
        found, skipped = page_plan("<box:0 0 1000 1><label:text>", 1000, 1000)
        self.assertIsNone(found[0]["label"])
        self.assertEqual(len(skipped), 1)
        found, skipped = page_plan("<box:900 900 1100 1100><label:text>", 1000, 1000)
        self.assertEqual(found[0]["box_px"], [900, 900, 1100, 1100])
        self.assertEqual(skipped, [])
        found, skipped = page_plan("<box:0 0 999999999999999999999999 1><label:text>", 1000, 1000)
        self.assertIsNone(found[0]["label"])
        self.assertEqual(skipped, [])

    def test_comparison_detects_layout_and_text_differences_independently(self):
        def recognize(image, prompt):
            return {"text": "", "finished_by": "eos"}
        left = read_page(Image.new("RGB", (20, 20)), recognize)
        right = json.loads(json.dumps(left))
        right["blocks"][0]["box_px"][2] = 19
        result = output_comparison(left, right)
        self.assertFalse(result["exact_prediction"])
        self.assertTrue(result["text_equal"])


class InputsTests(unittest.TestCase):
    def test_inputs_verify_hashes_and_apply_annotation_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "set"
            source.mkdir()
            Image.new("RGB", (100, 100)).save(source / "page.png")
            (source / "A.json").write_text(json.dumps({"id": "A", "image": "page.png", "crop_px": [10, 20, 40, 60]}))
            manifest = {"samples": [{"id": "A", "image_sha256": sha256(source / "page.png"), "gt_sha256": sha256(source / "A.json")}]}
            (source / "manifest.json").write_text(json.dumps(manifest))
            result = prepared_inputs({"public": str(source)}, root / "out")
            with Image.open(result["public"]["A"]["image"]) as image:
                self.assertEqual(image.size, (30, 40))
            (source / "A.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                prepared_inputs({"public": str(source)}, root / "rejected")

    def test_metrics_keep_missing_memory_and_require_identified_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bundle").mkdir()
            (root / "bundle/weight").write_bytes(b"abc")
            (root / "cache/hash").mkdir(parents=True)
            (root / "cache/hash/data").write_bytes(b"abcde")
            record = {"generate_s": 2, "load_s": 1, "assets": [{"main_hash": "hash"}], "layout": {"tokens": {"output": 2}},
                      "blocks": [{"tokens": {"output": 5}}], "peak_rss_bytes": 100}
            result = metrics(record, {"bundle": str(root / "bundle")}, str(root / "cache"), True)
            self.assertNotIn("peak_footprint_bytes", result)
            self.assertEqual(result["prepared_bytes"]["value"], 5)
            self.assertEqual(result["bundle_bytes"]["value"], 3)
            self.assertEqual(result["generated_tokens"]["value"], 7)
            with self.assertRaisesRegex(ValueError, "could not be located"):
                metrics(record, {"bundle": str(root / "bundle")}, str(root / "missing"), False)


class WeightsTests(unittest.TestCase):
    def test_restoration_verifies_manifest_file_tensors_and_target_coverage(self):
        import torch
        from safetensors.torch import save_file
        from Tools.Conversion.adapters.teleocr.quantization import tensor_record

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            module = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
            model = SimpleNamespace(visual=module, language_model=None)
            values = {"0.weight": torch.ones(3, 4), "1.weight": torch.ones(2, 3)}
            save_file(values, folder / "reference.safetensors")
            record = {"kind": "vision", "bits": 8, "reference_sha256": sha256(folder / "reference.safetensors"),
                      "layers": [{"name": str(i), "effective": tensor_record(values[f"{i}.weight"])} for i in range(2)]}
            def manifest():
                (folder / "compression.json").write_text(json.dumps(record))
                return {"compression_sha256": sha256(folder / "compression.json"), "reference_sha256": record["reference_sha256"]}
            expected = manifest()
            self.assertEqual(restore_weights(model, folder, expected)["targets"], 2)
            self.assertTrue(torch.equal(module[0].weight, values["0.weight"]))
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                restore_weights(model, folder, expected | {"compression_sha256": "bad"})
            record["layers"][0]["effective"]["sha256"] = "bad"
            with self.assertRaisesRegex(ValueError, "Invalid reference tensor"):
                restore_weights(model, folder, manifest())
            record["layers"].pop()
            with self.assertRaisesRegex(ValueError, "target keys differ"):
                restore_weights(model, folder, manifest())
            expected = manifest()
            with (folder / "reference.safetensors").open("ab") as output:
                output.write(b"bad")
            with self.assertRaisesRegex(ValueError, "weights differ"):
                restore_weights(model, folder, expected)


class RunnerTests(unittest.TestCase):
    def test_failures_remain_in_denominator_and_lock_is_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cli").write_text("test")
            plan = {"cli": str(root / "cli"), "cli_sha256": sha256(root / "cli"), "sets": {"test": str(root)}, "comparisons": [],
                    "variants": [{"id": "A", "engine": "reference"}], "cache_root": "unused"}
            (root / "plan.json").write_text(json.dumps(plan))
            lock = root / "lock"
            samples = {"test": {sid: {"image": sid} for sid in ("first", "second")}}
            record = {"layout": {"finished_by": "eos"}, "blocks": []}
            with mock.patch.dict(os.environ, {"COREAI_HEAVY_JOB_LOCK": str(lock)}), \
                    mock.patch.object(compare.reference, "model_processes", return_value=[]), \
                    mock.patch.object(compare.reference, "environment", return_value={"machine": {"cpu": "test"}, "os": {"build": "test"}}), \
                    mock.patch.object(compare.reference, "command", return_value="test"), \
                    mock.patch.object(compare.signal, "signal"), \
                    mock.patch.object(compare, "prepared_inputs", return_value=samples), \
                    mock.patch.object(compare, "run_child", side_effect=[RuntimeError("failed run"), record]), \
                    mock.patch.object(compare, "score_set", return_value={}):
                rows = compare.run_plan(root / "plan.json", root / "out", performance=False)
            self.assertEqual([row["status"] for row in rows], ["failed", "passed"])
            self.assertEqual(rows[0]["error"], "RuntimeError: failed run")
            self.assertFalse(lock.exists())
            self.assertEqual(len(list(root.glob("heavy-job-done-333-*"))), 1)
            self.assertIsNone(json.loads((root / "out/invocation.json").read_text())["aborted"])

    def test_existing_lock_prevents_model_environment_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plan.json").write_text("{}")
            lock = root / "lock"
            lock.mkdir()
            (lock / "owner.txt").write_text("other session")
            with mock.patch.dict(os.environ, {"COREAI_HEAVY_JOB_LOCK": str(lock)}), \
                    mock.patch.object(compare.reference, "model_processes", return_value=[]), \
                    mock.patch.object(compare.reference, "environment") as environment, \
                    mock.patch.object(compare.signal, "signal"):
                with self.assertRaises(TimeoutError):
                    compare.run_plan(root / "plan.json", root / "out", performance=False)
            environment.assert_not_called()
            self.assertEqual((lock / "owner.txt").read_text(), "other session")
            self.assertIn("TimeoutError", json.loads((root / "out/invocation.json").read_text())["aborted"])


if __name__ == "__main__":
    unittest.main()
