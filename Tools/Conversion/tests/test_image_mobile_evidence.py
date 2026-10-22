import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Tools.Conversion.adapters.qwen_image21.fixtures import file_record, save_json
from Tools.Conversion.adapters.qwen_image21.evaluation import REVISION, SIGMAS
from Tools.Conversion.adapters.qwen_image21.mobile_evidence import POLICY_SHA256, trace_files, validate_run_result, validate_trace
from Tools.Conversion.adapters.qwen_image21.mobile_profile import make_profile
from Tools.Conversion.adapters.qwen_image21.mobile_verify import compare_example, completed_run


class ImageMobileEvidenceTests(unittest.TestCase):
    def test_repeated_results_remain_bound_to_the_run_seed_payload_and_completion(self):
        expected = {"run_id": "g01-20261018", "fixture_id": "g01", "seed": 20261018, "noise": "noise/20261018.npy",
                    "operation": "generate", "source_revision": REVISION, "sigmas": SIGMAS, "reference_sha256": None,
                    **{name: "a" * 64 for name in ("prompt_sha256", "noise_sha256", "processed_tokens_sha256", "layout_sha256")}}
        result = {"run_id": expected["run_id"], "status": "completed", "input": expected}
        validate_run_result(result, expected)
        for defect in ("id", "status", "seed", "noise"):
            altered = copy.deepcopy(result)
            if defect == "id":
                altered["run_id"] = "g02-20261018"
            elif defect == "status":
                altered["status"] = "failed"
            elif defect == "seed":
                altered["input"]["seed"] += 1
            else:
                altered["input"]["noise"] = "another.npy"
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                validate_run_result(altered, expected)

    def capture(self, root):
        names = trace_files("generate")
        layouts = {f"transformer-{index}-layout.json" for index in range(8)}
        for name in names | layouts | {"output.png"}:
            (root / name).write_bytes(b"captured fixture")
        counts = {"vision": 0, "text": 1, "denoiser": 8, "vae_encoder": 0, "vae_decoder": 1,
                  "scheduler": 8, "callback": 8, "encoder_inputs": 1, "language_inputs": 1, "vision_inputs": 0}
        trace = {"complete": True, "counts": counts, "steps": [{"index": index} for index in range(8)],
                 "tensors": {name: {**file_record(root / name), "nonfinite_elements": 0} for name in names},
                 "layouts": {name: file_record(root / name) for name in layouts},
                 "not_applicable": [{"component": kind, "reason": "generation has no reference image"}
                                    for kind in ("vision", "vae_encoder")]}
        save_json(root / "trace.json", trace)
        save_json(root / "result.json", {"status": "completed", "files": {
            path.name: file_record(path) for path in root.iterdir()}})

    def test_missing_changed_extra_or_incomplete_source_capture_is_rejected(self):
        for defect in (None, "missing", "changed", "extra", "step", "region"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                self.capture(root)
                if defect == "missing":
                    (root / "latent-7.npy").unlink()
                elif defect == "changed":
                    (root / "rgba.npy").write_bytes(b"different source")
                elif defect == "extra":
                    (root / "extra.npy").write_bytes(b"extra")
                elif defect in ("step", "region"):
                    trace = json.loads((root / "trace.json").read_text())
                    if defect == "step":
                        trace["steps"].pop()
                    else:
                        trace["tensors"].pop("joint-noise-7.npy")
                    save_json(root / "trace.json", trace)
                if defect is None:
                    self.assertEqual(validate_trace(root, "generate")[0]["status"], "completed")
                else:
                    with self.assertRaises(ValueError):
                        validate_trace(root, "generate")

    def test_joint_and_target_errors_are_checked_separately_without_padded_rows(self):
        profile = make_profile("w4", "a" * 64, {"test.py": "b" * 64})
        expected = np.ones((1, 1050, 64), np.float32)
        actual = np.ones((1, 2304, 64), np.float32)
        actual[:, 1050:] = 100
        example = {"id": "denoiser-0", "expected": [expected], "target_indices": np.arange(26, 1050)}
        self.assertTrue(compare_example("denoiser", example, [actual], profile)["passed"])
        actual[:, 0] += 1
        result = compare_example("denoiser", example, [actual], profile)
        self.assertFalse(result["comparisons"][0]["metrics"]["passed"])
        self.assertTrue(result["comparisons"][1]["metrics"]["passed"])
        self.assertFalse(result["passed"])
        with self.assertRaises(ValueError):
            compare_example("denoiser", example, [actual[:, :1050]], profile)
        with self.assertRaises(ValueError):
            compare_example("vision", {"id": "vision", "expected": [expected] * 4}, [expected] * 3, profile)

    def test_shrunken_or_failed_comparison_sets_never_complete_a_run(self):
        def metric(boundary, output=0):
            return {"boundary": boundary, "output": output, "policy_sha256": POLICY_SHA256, "metrics": {"passed": True}}
        rows = [{"id": name, "passed": True, "comparisons": [metric("component")]} for name in ("text", "vae_decoder")]
        rows += [{"id": f"denoiser-{index}", "passed": True,
                  "comparisons": [metric("component"), metric("fixed-step-target")]} for index in range(8)]
        record = {"fixed": rows, "rollout": {"latent": metric("final-latent"), "rgba": metric("final-rgba")}}
        self.assertTrue(completed_run(record, "generate"))
        for defect in ("step", "target", "rollout", "failure", "policy"):
            altered = copy.deepcopy(record)
            if defect == "step":
                altered["fixed"].pop()
            elif defect == "target":
                altered["fixed"][-1]["comparisons"].pop()
            elif defect == "rollout":
                altered["rollout"].pop("rgba")
            elif defect == "failure":
                altered["fixed"][0]["comparisons"][0]["metrics"]["passed"] = False
            else:
                altered["rollout"]["rgba"]["policy_sha256"] = "0" * 64
            self.assertFalse(completed_run(altered, "generate"))
        self.assertFalse(completed_run(record, "edit"))


if __name__ == "__main__":
    unittest.main()
