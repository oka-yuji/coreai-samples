import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from Tools.Conversion.recipe import load_recipe
from Tools.Conversion.storage import PipelineError, check_results

ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "Tools/Conversion/recipes/voxcpm2"


def bindings():
    return {"python": sys.executable, "snapshot": str(ROOT / "snapshot"), "upstream": str(ROOT / "upstream"), "voices": str(ROOT / "voices")}


def passing_report():
    summary = {"count": 4, "passed": 4, "nonfinite": 0, "normalized_max": 0.000001}
    return {"complete": True, "passed": True, "precision": "fp32", "source_encoder": {"unchanged": True}, "summary": {"cases": 4, "required_cases": 4, "passed_cases": 4, "original_inputs_matched": 4, "fixed": copy.deepcopy(summary), "coreai": copy.deepcopy(summary)}}


class ReferenceRecipeTests(unittest.TestCase):
    def test_the_reference_recipe_requires_voices_without_changing_the_old_recipe(self):
        values = bindings()
        original = load_recipe(RECIPE / "recipe.json", {key: value for key, value in values.items() if key != "voices"}, ROOT)
        self.assertEqual(original["name"], "voxcpm2_components")
        with self.assertRaises(PipelineError):
            load_recipe(RECIPE / "reference-recipe.json", {key: value for key, value in values.items() if key != "voices"}, ROOT)
        actual = load_recipe(RECIPE / "reference-recipe.json", values, ROOT)
        self.assertEqual(actual["name"], "voxcpm2_reference")
        self.assertEqual(len(actual["steps"]), 1)
        stage = actual["steps"][0]
        self.assertEqual((stage["id"], stage["role"], stage["heavy"]), ("reference_encoder", "verify", True))
        command = stage["command"]
        self.assertEqual(command[command.index("--precision") + 1], "fp32")
        self.assertEqual(command[command.index("--output") + 1], "$out/encoder")

    def test_every_code_data_and_comparison_input_is_declared_for_reuse(self):
        stage = load_recipe(RECIPE / "reference-recipe.json", bindings(), ROOT)["steps"][0]
        required = {"$adapters/" + name + ".py" for name in ["reference_encoder", "source", "convert", "runtime", "wrappers"]}
        required |= {"$repo/Tools/Conversion/storage.py", "$recipe_dir/model.json", "$recipe_dir/requirements.lock", "$recipe_dir/reference-voices.json", "$snapshot", "$upstream/src/voxcpm", "$upstream/pyproject.toml", "$voices/README.md", "$voices/voice-donations/0a67.wav", "$voices/voice-donations/1410.wav"}
        self.assertEqual(set(stage["inputs"]), required)
        expected = {"encoder/report.json", "encoder/export.json"}
        expected |= {"encoder/refenc-fp32.aimodel/" + name for name in ["main.hash", "main.mlirb", "metadata.json"]}
        for case in ["0a67-5s", "0a67-10s", "1410-5s", "1410-10s"]:
            expected.add("encoder/voices/" + case + ".wav")
            expected |= {"encoder/cases/" + case + "/" + name + ".npy" for name in ["pcm", "original", "fixed-mu", "fixed", "coreai-mu", "coreai"]}
        self.assertEqual(set(stage["outputs"]), expected)

    def test_incomplete_failed_wrong_precision_count_nonfinite_and_error_reports_are_rejected(self):
        stage = load_recipe(RECIPE / "reference-recipe.json", bindings(), ROOT)["steps"][0]
        changes = [(["complete"], False), (["passed"], False), (["precision"], "fp16"), (["summary", "cases"], 3), (["summary", "passed_cases"], 3), (["summary", "original_inputs_matched"], 3), (["summary", "coreai", "nonfinite"], 1), (["summary", "fixed", "normalized_max"], 0.002), (["summary", "coreai", "normalized_max"], 0.002), (["source_encoder", "unchanged"], False)]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "encoder/report.json"
            path.parent.mkdir()
            path.write_text(json.dumps(passing_report()))
            check_results(directory, stage["checks"])
            for keys, value in changes:
                report = passing_report()
                target = report
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                path.write_text(json.dumps(report))
                with self.subTest(keys=keys), self.assertRaises(PipelineError):
                    check_results(directory, stage["checks"])


if __name__ == "__main__":
    unittest.main()
