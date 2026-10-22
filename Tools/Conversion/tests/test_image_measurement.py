import copy
import errno
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

from Tools.Conversion.adapters.qwen_image21.evaluation import quality_summary
from Tools.Conversion.adapters.qwen_image21.fixtures import quality_axes, validate_cases, validate_suite_runs
from Tools.Conversion.adapters.qwen_image21.measurement import PhaseRecorder, write_metrics


class ImageMeasurementTests(unittest.TestCase):
    def test_failed_report_replacement_preserves_the_last_complete_json_and_original_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text('{"complete":false,"phase":"previous"}\n')
            original = path.read_bytes()
            error = KeyboardInterrupt()
            with patch.object(Path, "replace", side_effect=OSError(errno.ENOSPC, "no space")):
                write_metrics(path, {"complete": False, "status": "cancelled"}, preserving=error)
                with self.assertRaises(OSError):
                    write_metrics(path, {"complete": True})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            self.assertIn("could not be saved", error.__notes__[0])

    def test_report_io_failure_cannot_replace_a_generation_cancellation(self):
        from Tools.Conversion.adapters.qwen_image21 import generate
        from Tools.Conversion.adapters.qwen_image21.contracts import KINDS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(generate, "prompt_inputs", side_effect=KeyboardInterrupt), \
                    patch.object(Path, "replace", side_effect=OSError(errno.ENOSPC, "no space")):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    generate.generate(dict.fromkeys(KINDS, root), root, root / "output", "fixture")
            self.assertIn("could not be saved", caught.exception.__notes__[0])

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
        axes = quality_axes()
        keys = list(axes)
        reference = {key: {axis: 2 for axis in values} for key, values in axes.items()}
        candidate = copy.deepcopy(reference)
        for key in keys[:2]:
            candidate[key]["instruction"] = 0
        self.assertTrue(quality_summary(reference, candidate, axes)["candidate_meets_provisional_target"])
        candidate[keys[2]]["instruction"] = 0
        self.assertFalse(quality_summary(reference, candidate, axes)["candidate_meets_provisional_target"])
        reference[keys[2]]["instruction"] = 0
        result = quality_summary(reference, candidate, axes)
        self.assertEqual(result["new_failures"], keys[:2])
        self.assertEqual(result["axes"]["instruction"]["reference_successes"], 59)
        candidate[keys[-1]]["preservation"] = None
        result = quality_summary(reference, candidate, axes)
        self.assertFalse(result["complete"])
        self.assertFalse(result["candidate_meets_provisional_target"])
        del candidate[keys[-1]]
        with self.assertRaises(ValueError):
            quality_summary(reference, candidate, axes)

    def test_empty_axes_and_duplicate_suite_ids_cannot_make_an_incomplete_suite_pass(self):
        axes = {key: [] for key in quality_axes()}
        rows = {key: {} for key in axes}
        with self.assertRaises(ValueError):
            quality_summary(rows, rows, axes)
        runs = [{"run_id": key, "fixture_id": key.rsplit("-", 1)[0], "seed": int(key.rsplit("-", 1)[1]),
                 "noise": f"noise/{key.rsplit('-', 1)[1]}.npy"}
                for key in quality_axes()]
        validate_suite_runs(runs)
        changed = copy.deepcopy(runs)
        changed[0]["noise"] = "noise/20261021.npy"
        with self.assertRaises(ValueError):
            validate_suite_runs(changed)
        runs[-1] = runs[0]
        with self.assertRaises(ValueError):
            validate_suite_runs(runs)

    def test_an_existing_failure_does_not_hide_new_failures_on_other_axes(self):
        axes = quality_axes()
        reference = {key: {axis: 2 for axis in values} for key, values in axes.items()}
        selected = [key for key, values in axes.items() if {"instruction", "detail"} <= set(values)][:3]
        self.assertEqual(len(selected), 3)
        for key in selected:
            reference[key]["instruction"] = 0
        candidate = copy.deepcopy(reference)
        for key in selected:
            candidate[key]["detail"] = 0
        result = quality_summary(reference, candidate, axes)
        self.assertTrue(all(row["candidate_rate"] >= 0.9 for row in result["axes"].values()))
        self.assertEqual(result["new_failures"], selected)
        self.assertEqual(result["new_failure_axes"], {key: ["detail"] for key in selected})
        self.assertFalse(result["candidate_meets_provisional_target"])

    def test_generation_persists_failures_before_any_model_can_load(self):
        from Tools.Conversion.adapters.qwen_image21 import generate
        from Tools.Conversion.adapters.qwen_image21.contracts import KINDS
        with tempfile.TemporaryDirectory() as name:
            output = Path(name) / "failed"
            with patch.object(generate, "prompt_inputs", side_effect=ValueError("invalid prompt")), \
                    patch.object(generate, "Component", side_effect=AssertionError("Model loading ran")):
                with self.assertRaises(ValueError):
                    generate.generate(dict.fromkeys(KINDS, Path(name)), Path(name), output, "fixture")
            report = json.loads((output / "metrics.json").read_text())
            self.assertFalse(report["complete"])
            self.assertEqual(report["error_type"], "ValueError")
            self.assertEqual(report["measurement"]["phases"][-1]["status"], "failed")

    def test_published_suite_keeps_all_inputs_seeds_and_applicable_quality_axes(self):
        path = Path(__file__).resolve().parents[1] / "recipes/qwen_image21/evaluation-cases.json"
        definition = json.loads(path.read_text())
        validate_cases(definition)
        for mutation in ("remove", "duplicate", "seed", "null-reference", "remove-text-axis"):
            changed = copy.deepcopy(definition)
            if mutation == "remove":
                changed["cases"].pop()
            elif mutation == "duplicate":
                changed["cases"][1]["id"] = changed["cases"][0]["id"]
            elif mutation == "seed":
                changed["seeds"][0] += 1
            elif mutation == "null-reference":
                for case in changed["cases"]:
                    if "reference" in case:
                        case["reference"] = None
            else:
                for case in changed["cases"]:
                    case["axes"] = [axis for axis in case["axes"] if axis != "text"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_cases(changed)


if __name__ == "__main__":
    unittest.main()
