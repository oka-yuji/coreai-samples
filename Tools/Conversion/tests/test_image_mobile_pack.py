import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from Tools.Conversion.adapters.qwen_image21.fixtures import file_record
from Tools.Conversion.adapters.qwen_image21.mobile_compare import quality_template, score_quality
from Tools.Conversion.adapters.qwen_image21.mobile_pack import compression_summary, copy_file, pack
from Tools.Conversion.adapters.qwen_image21.mobile_profile import CANDIDATES, make_profile
from Tools.Conversion.adapters.qwen_image21.mobile_recipe import recipe
from Tools.Conversion.recipe import load_recipe


class ImageMobilePackTests(unittest.TestCase):
    def test_compression_summary_accounts_for_retained_dense_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            asset = root / "denoiser/denoiser.aimodel"
            asset.mkdir(parents=True)
            (asset / "model").write_bytes(b"x" * 100)
            compression = {"layers": [{"shape": [2, 2], "payload_bits_scenario": 20}],
                           "dense_layers": [{"weight": {"shape": [2, 2]}, "payload_bits": 64}]}
            result = compression_summary({"denoiser": ({"asset_bytes": 80}, compression)}, root)["denoiser"]
            self.assertEqual(result["quantized_linear_count"], 1)
            self.assertEqual(result["retained_dense_linear_count"], 1)
            self.assertEqual(result["selected_linear_bits_per_weight_scenario"], 5)
            self.assertEqual(result["declared_linear_weight_bits_per_weight_scenario"], 10.5)
            self.assertEqual(result["asset_size_ratio"], 0.8)

    def test_every_generated_candidate_is_accepted_by_the_actual_recipe_parser(self):
        repo = Path(__file__).parents[3]
        root = repo / "Tools/Conversion/recipes/qwen_image21_mobile"
        bindings = {name: "/fixture/" + name for name in
                    ("python", "snapshot", "heavy_lock", "baseline_reference", "baseline_bundle", "quality")}
        for candidate in CANDIDATES:
            for mode in ("", "-export", "-reviewed"):
                filename = "recipe" + mode + ("" if candidate == "w4" else "-" + candidate) + ".json"
                with self.subTest(candidate=candidate, mode=mode):
                    parsed = load_recipe(root / filename, bindings, repo)
                    self.assertEqual(len(parsed["steps"]), 7 if mode == "-export" else 13)

    def test_pack_cannot_write_inside_the_frozen_fixture_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixtures = root / "fixtures"
            fixtures.mkdir()
            (fixtures / "original").write_bytes(b"frozen input")
            profile = make_profile("w4", "a" * 64, {"test.py": "b" * 64})
            args = SimpleNamespace(output=fixtures / "candidate", fixtures=fixtures, profile=root / "profile.json",
                                   dependency_lock=root / "lock", snapshot=root / "snapshot", reference=root / "reference",
                                   mixed_reference=root / "mixed", baseline=root / "baseline", baseline_bundle=root / "baseline-bundle",
                                   verification=root / "verification", component=[kind + "=" + str(root / kind) for kind in profile["components"]])
            with patch("Tools.Conversion.adapters.qwen_image21.mobile_pack.require_heavy_job"), \
                 patch("Tools.Conversion.adapters.qwen_image21.mobile_pack.read_profile", return_value=profile), \
                 patch("Tools.Conversion.adapters.qwen_image21.mobile_pack.compare_cohorts") as comparison:
                with self.assertRaises(ValueError):
                    pack(args)
                comparison.assert_not_called()
            self.assertEqual(list(fixtures.iterdir()), [fixtures / "original"])

    def cohorts(self):
        return {name: {"summary_sha256": name.lower() * 32} for name in ("A0", "A1", "B", "C")}

    def test_unreviewed_changed_or_shrunken_quality_records_cannot_pass(self):
        cohorts = self.cohorts()
        template = quality_template(cohorts)
        result = score_quality(template, cohorts)
        self.assertFalse(result["complete"])
        self.assertFalse(result["passed"])
        for defect in ("identity", "cohort", "row", "duplicate", "axis"):
            changed = copy.deepcopy(template)
            if defect == "identity":
                changed["cohorts"]["C"]["identity"] = {}
            elif defect == "cohort":
                changed["cohorts"].pop("A0")
            elif defect == "row":
                changed["cohorts"]["C"]["ratings"].pop()
            elif defect == "duplicate":
                changed["cohorts"]["C"]["ratings"][1] = changed["cohorts"]["C"]["ratings"][0]
            else:
                changed["cohorts"]["C"]["ratings"][0]["axes"] = {}
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                score_quality(changed, cohorts)

    def test_new_quality_failures_are_not_hidden_by_a_high_average_success_rate(self):
        cohorts = self.cohorts()
        value = quality_template(cohorts)
        for cohort in value["cohorts"].values():
            for row in cohort["ratings"]:
                row["axes"] = {key: 2 for key in row["axes"]}
        self.assertTrue(score_quality(value, cohorts)["passed"])
        for row in value["cohorts"]["C"]["ratings"][:3]:
            row["axes"]["instruction"] = 0
        result = score_quality(value, cohorts)
        self.assertTrue(result["complete"])
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["comparisons"]["A0-to-C"]["new_failures"]), 3)

    def test_bundle_copy_preserves_existing_files_and_requires_the_verified_source_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, target = root / "source", root / "copied"
            source.write_bytes(b"verified asset bytes")
            expected = file_record(source)
            copy_file(source, target, expected)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            with self.assertRaises(FileExistsError):
                copy_file(source, target, expected)
            link = root / "link"
            link.symlink_to(root / "missing")
            with self.assertRaises(FileExistsError):
                copy_file(source, link, expected)
            self.assertFalse((root / "missing").exists())
            source.write_bytes(b"changed after validation")
            with self.assertRaises(ValueError):
                copy_file(source, root / "changed", expected)

    def test_generated_recipes_keep_export_numeric_and_human_quality_gates_separate(self):
        root = Path(__file__).parents[1] / "recipes/qwen_image21_mobile"
        for candidate in CANDIDATES:
            for mode in ("", "-export", "-reviewed"):
                expected = recipe(candidate, export_only=mode == "-export", reviewed=mode == "-reviewed")
                filename = "recipe" + mode + ("" if candidate == "w4" else "-" + candidate) + ".json"
                self.assertEqual(json.loads((root / filename).read_text()), expected)
                steps = {row["id"]: row for row in expected["steps"]}
                if mode == "-export":
                    self.assertFalse(any(row["role"] == "verify" for row in steps.values()))
                    self.assertNotIn("reference", steps)
                    continue
                self.assertEqual(steps["numeric"]["role"], "verify")
                self.assertIn("mixed_reference", steps["bundle"]["depends"])
                self.assertNotIn("--component", steps["mixed_reference"]["command"])
                command = steps["mixed_profile"]["command"]
                self.assertEqual(command[command.index("--candidate") + 1], "mixed-f32" if candidate.endswith("-f32") else "mixed")
                reviewed = mode == "-reviewed"
                self.assertEqual("--quality" in steps["bundle"]["command"], reviewed)
                self.assertEqual(any(row["key"] == ["verified"] for row in steps["bundle"]["checks"]), reviewed)


if __name__ == "__main__":
    unittest.main()
