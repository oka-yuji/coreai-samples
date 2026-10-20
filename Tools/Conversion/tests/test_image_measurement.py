import copy
import json
from pathlib import Path
import unittest

from Tools.Conversion.adapters.qwen_image21.evaluation import quality_summary
from Tools.Conversion.adapters.qwen_image21.fixtures import validate_cases
from Tools.Conversion.adapters.qwen_image21.measurement import PhaseRecorder


class ImageMeasurementTests(unittest.TestCase):
    def test_failed_phase_preserves_its_interval_and_missing_memory_without_succeeding(self):
        times = iter([100, 200, 500, 700])
        recorder = PhaseRecorder(capture=lambda: {"memory": None}, clock=lambda: next(times))
        with self.assertRaises(ValueError), recorder.phase("decoder"):
            raise ValueError("fixture failure")
        report = recorder.report()
        self.assertEqual(report["phases"][0]["status"], "failed")
        self.assertEqual(report["phases"][0]["error_type"], "ValueError")
        self.assertEqual(report["phases"][0]["seconds"], 300 / 1e9)
        self.assertEqual(len(report["observations"]), 2)
        self.assertIsNone(report["specialization_seconds"]["value"])

    def test_quality_does_not_turn_missing_ratings_into_success_or_drop_original_failures(self):
        axes = {str(index): ["instruction", "preservation"] for index in range(60)}
        reference = {key: {axis: 2 for axis in values} for key, values in axes.items()}
        candidate = copy.deepcopy(reference)
        for key in ("0", "1"):
            candidate[key]["instruction"] = 0
        self.assertTrue(quality_summary(reference, candidate, axes)["candidate_meets_provisional_target"])
        candidate["2"]["instruction"] = 0
        self.assertFalse(quality_summary(reference, candidate, axes)["candidate_meets_provisional_target"])
        reference["2"]["instruction"] = 0
        result = quality_summary(reference, candidate, axes)
        self.assertEqual(result["new_failures"], ["0", "1"])
        self.assertEqual(result["axes"]["instruction"]["reference_successes"], 59)
        candidate["59"]["preservation"] = None
        result = quality_summary(reference, candidate, axes)
        self.assertFalse(result["complete"])
        self.assertFalse(result["candidate_meets_provisional_target"])
        del candidate["59"]
        with self.assertRaises(ValueError):
            quality_summary(reference, candidate, axes)

    def test_published_suite_keeps_all_inputs_seeds_and_applicable_quality_axes(self):
        path = Path(__file__).resolve().parents[1] / "recipes/qwen_image21/evaluation-cases.json"
        definition = json.loads(path.read_text())
        validate_cases(definition)
        for mutation in ("remove", "duplicate", "seed"):
            changed = copy.deepcopy(definition)
            if mutation == "remove":
                changed["cases"].pop()
            elif mutation == "duplicate":
                changed["cases"][1]["id"] = changed["cases"][0]["id"]
            else:
                changed["seeds"][0] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_cases(changed)


if __name__ == "__main__":
    unittest.main()
