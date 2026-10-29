import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Tools.Conversion.adapters.qwen_image21.evaluation import canonical_hash
from Tools.Conversion.adapters.qwen_image21.fixtures import file_record, quality_axes
from Tools.Conversion.adapters.qwen_image21.mobile_acceptance import OWNER_IDENTICAL_REVIEW, OWNER_REVIEW, quality_acceptance
from Tools.Conversion.adapters.qwen_image21.mobile_recipe import recipe
from Tools.Conversion.adapters.qwen_image21.mobile_profile import NATIVE_COMPLEX_CANDIDATE
from Tools.Conversion.adapters.qwen_image21.mobile_visual_equivalence import EQUIVALENCE_FILES, REVIEWED_GALLERY_RECEIPT
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

    def equivalent_proof(self):
        root = Path(__file__).parents[1] / "adapters/qwen_image21"
        return {"format": 1, "same_reviewed_images": True, "same_reviewed_presentation": True,
                "original_owner_review": {"id": OWNER_REVIEW, "file": file_record(self.review)},
                "previous": {"comparison_file": {"sha256": self.decision["comparison_sha256"]},
                             "verification_file": {"sha256": self.decision["numeric_verification_sha256"]},
                             "profile_sha256": self.decision["candidate_profile_sha256"]},
                "candidate": {"comparison_file": file_record(self.comparison), "verification_file": file_record(self.numeric),
                              "profile_sha256": canonical_hash(self.profile)},
                "presentation": {"previous": {"receipt_file": REVIEWED_GALLERY_RECEIPT}},
                "equivalence_code": {name: file_record(root / name) for name in EQUIVALENCE_FILES},
                "run_count": 60, "image_count": 240,
                "runs": [{"run_id": key, "images": {name: {} for name in ("A0", "A1", "B", "C")}} for key in quality_axes()]}

    def test_identical_gallery_method_preserves_the_original_review_and_benchmark(self):
        proof = self.root / "equivalence.json"
        proof.write_text(json.dumps(self.equivalent_proof()))
        original = self.review.read_bytes()
        with patch("Tools.Conversion.adapters.qwen_image21.mobile_acceptance.OWNER_IDENTICAL_PATH", proof), \
             patch("Tools.Conversion.adapters.qwen_image21.mobile_acceptance.OWNER_IDENTICAL_RECORD", file_record(proof)):
            value = self.accept(OWNER_IDENTICAL_REVIEW)
        self.assertTrue(value["accepted"])
        self.assertFalse(value["benchmark_passed"])
        self.assertEqual(value["method"], OWNER_IDENTICAL_REVIEW)
        self.assertEqual(value["device_verification"], "pending")
        self.assertEqual(value["equivalence_file"], file_record(proof))
        self.assertEqual(value["equivalence_evidence"], "evidence/visual-equivalence.json")
        self.assertEqual(self.review.read_bytes(), original)

    def test_partial_unbound_or_old_code_equivalence_is_rejected(self):
        proof = self.root / "equivalence.json"
        changes = [lambda d: d.__setitem__("same_reviewed_presentation", False),
                   lambda d: d.__setitem__("same_reviewed_images", False),
                   lambda d: d["original_owner_review"].__setitem__("file", {}),
                   lambda d: d["previous"].__setitem__("profile_sha256", "a" * 64),
                   lambda d: d["candidate"].__setitem__("profile_sha256", "b" * 64),
                   lambda d: d["candidate"].__setitem__("verification_file", {}),
                   lambda d: d["candidate"].__setitem__("comparison_file", {}),
                   lambda d: d["presentation"]["previous"].__setitem__("receipt_file", {}),
                   lambda d: d.__setitem__("equivalence_code", {}), lambda d: d["runs"].pop(),
                   lambda d: d["runs"][0]["images"].pop("A0"), lambda d: d.__setitem__("image_count", 239)]
        with patch("Tools.Conversion.adapters.qwen_image21.mobile_acceptance.OWNER_IDENTICAL_PATH", proof):
            for index, change in enumerate(changes):
                with self.subTest(index=index):
                    value = self.equivalent_proof()
                    change(value)
                    proof.write_text(json.dumps(value))
                    with patch("Tools.Conversion.adapters.qwen_image21.mobile_acceptance.OWNER_IDENTICAL_RECORD", file_record(proof)):
                        with self.assertRaises(ValueError):
                            self.accept(OWNER_IDENTICAL_REVIEW)

    def test_equivalence_cannot_accept_missing_proof_or_failed_current_numerics(self):
        proof = self.root / "equivalence.json"
        with patch("Tools.Conversion.adapters.qwen_image21.mobile_acceptance.OWNER_IDENTICAL_PATH", proof):
            with patch("Tools.Conversion.adapters.qwen_image21.mobile_acceptance.OWNER_IDENTICAL_RECORD", None):
                with self.assertRaisesRegex(ValueError, "not pinned"):
                    self.accept(OWNER_IDENTICAL_REVIEW)
            with patch("Tools.Conversion.adapters.qwen_image21.mobile_acceptance.OWNER_IDENTICAL_RECORD", {"bytes": 1, "sha256": "a" * 64}):
                with self.assertRaises(FileNotFoundError):
                    self.accept(OWNER_IDENTICAL_REVIEW)
            for field in ("numeric_complete", "numeric_passed"):
                with self.subTest(field=field):
                    self.numeric.write_text(json.dumps({"numeric_complete": True, "numeric_passed": True, field: False}))
                    result = copy.deepcopy(self.result)
                    result["cohorts"]["C"]["verification_file"] = file_record(self.numeric)
                    self.comparison.write_text(json.dumps(result))
                    proof.write_text(json.dumps(self.equivalent_proof()))
                    with self.assertRaisesRegex(ValueError, "complete matching numerical verification"):
                        self.accept(OWNER_IDENTICAL_REVIEW)

    def test_a_modified_proof_is_rejected_even_if_its_completion_flags_remain_true(self):
        proof = self.root / "equivalence.json"
        original = self.equivalent_proof()
        proof.write_text(json.dumps(original))
        expected = file_record(proof)
        original["runs"][0]["input_sha256"] = "not-the-bound-input"
        original["runs"][0]["images"]["C"] = None
        proof.write_text(json.dumps(original))
        with patch("Tools.Conversion.adapters.qwen_image21.mobile_acceptance.OWNER_IDENTICAL_PATH", proof), \
             patch("Tools.Conversion.adapters.qwen_image21.mobile_acceptance.OWNER_IDENTICAL_RECORD", expected):
            with self.assertRaisesRegex(ValueError, "differs from its recorded identity"):
                self.accept(OWNER_IDENTICAL_REVIEW)

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

    def test_identical_gallery_recipe_tracks_both_records_and_the_proof_implementation(self):
        generated = recipe(NATIVE_COMPLEX_CANDIDATE, owner_review=True)
        bundle = next(step for step in generated["steps"] if step["id"] == "bundle")
        self.assertIn(OWNER_IDENTICAL_REVIEW, bundle["command"])
        for path in ("docs/research/qwen-image21-owner-quality-20261028.json",
                     "docs/research/qwen-image21-native-complex-visual-equivalence-20261029.json",
                     "Tools/Conversion/adapters/qwen_image21/mobile_visual_equivalence.py",
                     "Tools/Conversion/adapters/qwen_image21/mobile_gallery.py"):
            self.assertIn("$repo/" + path, bundle["inputs"])
        path = Path(__file__).parents[1] / "recipes/qwen_image21_mobile" / ("recipe-owner-reviewed-" + NATIVE_COMPLEX_CANDIDATE + ".json")
        self.assertEqual(json.loads(path.read_text()), generated)
        bindings = {key: "/fixture/" + key for key in ("python", "snapshot", "heavy_lock", "baseline_reference", "baseline_bundle")}
        self.assertEqual(len(load_recipe(path, bindings, Path(__file__).parents[3])["steps"]), 13)


if __name__ == "__main__":
    unittest.main()
