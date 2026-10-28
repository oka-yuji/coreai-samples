import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Tools.Conversion.adapters.qwen_image21.evaluation import canonical_hash
from Tools.Conversion.adapters.qwen_image21.fixtures import file_record
from Tools.Conversion.adapters.qwen_image21.mobile_acceptance import OWNER_REVIEW, quality_acceptance
from Tools.Conversion.adapters.qwen_image21.mobile_recipe import recipe
from Tools.Conversion.recipe import load_recipe


class ImageMobileAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.profile = {"id": "test-candidate", "source": {"revision": "test"}}
        self.numeric = self.root / "numeric.json"
        self.numeric.write_text(json.dumps({"numeric_complete": True, "numeric_passed": True}))
        self.comparison = self.root / "comparison.json"
        self.result = {"cohorts": {"C": {"verification_file": file_record(self.numeric),
                        "profile_sha256": canonical_hash(self.profile)}}, "quality_passed": False,
                       "quality": {"passed": False, "complete": False, "comparisons": {}}}
        self.comparison.write_text(json.dumps(self.result))
        self.review = self.root / "owner.json"
        self.decision = {"format": 1, "id": OWNER_REVIEW,
                         "decision": "accepted_for_current_local_research_candidate_visual_quality",
                         "acceptance_amendment": {"criterion_id": "QMI-OWNER-VISUAL-01", "status": "met"},
                         "candidate_profile_sha256": canonical_hash(self.profile),
                         "numeric_verification_sha256": file_record(self.numeric)["sha256"],
                         "comparison_sha256": file_record(self.comparison)["sha256"]}
        self.review.write_text(json.dumps(self.decision))
        patcher = patch("Tools.Conversion.adapters.qwen_image21.mobile_acceptance.OWNER_REVIEW_PATH", self.review)
        patcher.start()
        self.addCleanup(patcher.stop)

    def accept(self, identifier=OWNER_REVIEW):
        return quality_acceptance(self.comparison, self.profile, self.numeric, identifier)

    def test_owner_review_accepts_only_visual_quality_and_preserves_incomplete_benchmark(self):
        original = self.comparison.read_bytes()
        value = self.accept()
        self.assertTrue(value["accepted"])
        self.assertEqual(value["method"], OWNER_REVIEW)
        self.assertFalse(value["benchmark_passed"])
        self.assertEqual(value["device_verification"], "pending")
        self.assertEqual(value["review_file"], file_record(self.review))
        self.assertEqual(self.comparison.read_bytes(), original)
        self.assertFalse(self.accept(None)["accepted"])

    def test_benchmark_method_remains_separate(self):
        self.result["quality_passed"] = True
        self.result["quality"].update(passed=True, complete=True)
        self.comparison.write_text(json.dumps(self.result))
        value = self.accept(None)
        self.assertTrue(value["accepted"])
        self.assertTrue(value["benchmark_passed"])
        self.assertEqual(value["method"], "case-axis-benchmark")
        self.assertNotIn("review_file", value)

    def test_unknown_or_changed_approval_cannot_enable_quality(self):
        with self.assertRaises(ValueError):
            self.accept("../../arbitrary-approval.json")
        for key, value in (("id", "other"), ("decision", "pending"), ("format", 2),
                           ("candidate_profile_sha256", "a" * 64), ("numeric_verification_sha256", "b" * 64),
                           ("comparison_sha256", "c" * 64),
                           ("acceptance_amendment", {"criterion_id": "QMI-OWNER-VISUAL-01", "status": "pending"})):
            with self.subTest(key=key):
                changed = {**self.decision, key: value}
                self.review.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    self.accept()

    def test_changed_candidate_and_gallery_are_rejected(self):
        self.profile["id"] = "different-candidate"
        with self.assertRaises(ValueError):
            self.accept()
        self.profile["id"] = "test-candidate"
        self.comparison.write_text(self.comparison.read_text() + "\n")
        with self.assertRaises(ValueError):
            self.accept()

    def test_changed_or_failed_numeric_evidence_is_rejected_even_with_rebound_hash(self):
        for field in ("numeric_complete", "numeric_passed"):
            with self.subTest(field=field):
                self.numeric.write_text(json.dumps({"numeric_complete": True, "numeric_passed": True, field: False}))
                result = copy.deepcopy(self.result)
                result["cohorts"]["C"]["verification_file"] = file_record(self.numeric)
                self.comparison.write_text(json.dumps(result))
                decision = {**self.decision, "numeric_verification_sha256": file_record(self.numeric)["sha256"],
                            "comparison_sha256": file_record(self.comparison)["sha256"]}
                self.review.write_text(json.dumps(decision))
                with self.assertRaises(ValueError):
                    self.accept()

    def test_inconsistent_benchmark_result_is_rejected(self):
        self.result["quality_passed"] = True
        self.comparison.write_text(json.dumps(self.result))
        with self.assertRaises(ValueError):
            self.accept(None)

    def test_owner_recipe_is_bound_and_tracks_the_acceptance_code_and_record(self):
        candidate = "w4-dit-g16-blocks16-23-f32"
        generated = recipe(candidate, owner_review=True)
        bundle = next(step for step in generated["steps"] if step["id"] == "bundle")
        self.assertIn(OWNER_REVIEW, bundle["command"])
        self.assertIn("$repo/Tools/Conversion/adapters/qwen_image21/mobile_acceptance.py", bundle["inputs"])
        self.assertIn("$repo/docs/research/qwen-image21-owner-quality-20261028.json", bundle["inputs"])
        self.assertIn({"file": "receipt.json", "key": ["verified"], "equals": True}, bundle["checks"])
        repo = Path(__file__).parents[3]
        path = repo / "Tools/Conversion/recipes/qwen_image21_mobile/recipe-owner-reviewed-w4-dit-g16-blocks16-23-f32.json"
        self.assertEqual(json.loads(path.read_text()), generated)
        bindings = {key: "/fixture/" + key for key in ("python", "snapshot", "heavy_lock", "baseline_reference", "baseline_bundle")}
        self.assertEqual(len(load_recipe(path, bindings, repo)["steps"]), 13)
        for kwargs in ({"candidate": "w4"}, {"candidate": candidate, "reviewed": True},
                       {"candidate": candidate, "export_only": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                recipe(**kwargs, owner_review=True)


if __name__ == "__main__":
    unittest.main()
