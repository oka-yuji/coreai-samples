import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Tools.Conversion.adapters.qwen_image21.evaluation import canonical_hash
from Tools.Conversion.adapters.qwen_image21.fixtures import file_record, quality_axes
from Tools.Conversion.adapters.qwen_image21.mobile_gallery import bytes_record
from Tools.Conversion.adapters.qwen_image21.mobile_visual_equivalence import cohort_roots, prove_equivalence, prove_presentation


class ImageVisualEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.keys = list(quality_axes())
        self.old = self.make_cohorts("old")
        self.new = self.make_cohorts("new")
        comparison = self.read(self.old / "comparison/comparison.json")
        self.review = self.root / "owner.json"
        self.save(self.review, {"format": 1, "id": "qmi-owner-visual-v1",
                  "decision": "accepted_for_current_local_research_candidate_visual_quality",
                  "acceptance_amendment": {"criterion_id": "QMI-OWNER-VISUAL-01", "status": "met"},
                  "comparison_sha256": file_record(self.old / "comparison/comparison.json")["sha256"],
                  "numeric_verification_sha256": comparison["cohorts"]["C"]["verification_file"]["sha256"],
                  "candidate_profile_sha256": comparison["cohorts"]["C"]["profile_sha256"]})
        patcher = patch("Tools.Conversion.adapters.qwen_image21.mobile_visual_equivalence.OWNER_REVIEW_PATH", self.review)
        patcher.start()
        self.addCleanup(patcher.stop)

    def save(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def read(self, path):
        return json.loads(path.read_text())

    def roots(self, root):
        return {name: root / name for name in ("A0", "A1", "B", "C")}

    def make_cohorts(self, name):
        root = self.root / name
        inputs = [{"run_id": key, "input": {"run_id": key, "seed": 7, "prompt_sha256": "a" * 64}} for key in self.keys]
        candidate = "w4-dit-g16-blocks16-23-f32" if name == "old" else "w4-dit-g16-native-complex-blocks16-23-f32"
        code = {key: name for key in ("metal_rotary.py", "mobile_convert.py", "mobile_models.py", "mobile_profile.py")}
        code["unchanged.py"] = "same"
        profile = {"id": "qmi-mobile-" + candidate + "-v1", "source": {"revision": "pinned", "converter_sha256": code},
                   "workload": {"steps": 8}, "components": {"denoiser": {"bits": 4}}}
        identity = {"profile": profile, "profile_sha256": canonical_hash(profile), "fixtures_file": {"sha256": "b" * 64},
                    "policy_sha256": "c" * 64, "versions": {"coreai": "fixed"},
                    "source_identity": {"source_files": {"weights": "fixed"}, "effective_compression": {"bits": 4},
                                        "sigmas": [1, 0], "prefix_kv_cache": False}}
        cohorts = {}
        for cohort in self.roots(root):
            report = {"runs": [], "complete": True, "numeric_complete": True, "numeric_passed": True,
                      "identity": identity, "evidence_files": {}}
            for entry in inputs:
                key = entry["run_id"]
                path = ("rollout/" if cohort == "C" else "") + key + "/output.png"
                target = root / cohort / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((cohort + key).encode())
                record = file_record(target)
                report["evidence_files"][path] = record
                report["runs"].append({**entry, "status": "completed", "files": {"output.png": record}})
            filename = "verification.json" if cohort == "C" else "summary.json"
            self.save(root / cohort / filename, report)
            binding = "verification_file" if cohort == "C" else "summary_file"
            cohorts[cohort] = {binding: file_record(root / cohort / filename), "profile_sha256": canonical_hash(profile)}
        self.save(root / "comparison/comparison.json", {"format": 2, "cohorts": cohorts, "runs": inputs})
        return root

    def prove(self):
        return prove_equivalence(self.old / "comparison", self.roots(self.old), self.new / "comparison", self.roots(self.new))

    def mutate_report(self, cohort, mutate):
        filename = "verification.json" if cohort == "C" else "summary.json"
        path = self.new / cohort / filename
        report = self.read(path)
        mutate(report)
        self.save(path, report)
        comparison = self.read(self.new / "comparison/comparison.json")
        binding = "verification_file" if cohort == "C" else "summary_file"
        comparison["cohorts"][cohort][binding] = file_record(path)
        self.save(self.new / "comparison/comparison.json", comparison)

    def test_all_four_cohorts_match_without_inventing_approval_or_benchmark_scores(self):
        before = self.review.read_bytes()
        result = self.prove()
        self.assertTrue(result["same_reviewed_images"])
        self.assertEqual(result["image_count"], 240)
        self.assertEqual(result["run_count"], 60)
        self.assertNotIn("accepted", result)
        self.assertNotIn("quality_passed", result)
        self.assertEqual(self.review.read_bytes(), before)
        self.assertNotEqual(result["previous"]["profile_sha256"], result["candidate"]["profile_sha256"])

    def test_a_changed_image_is_rejected_even_when_the_candidate_reports_are_rebound(self):
        for cohort in ("A0", "A1", "B", "C"):
            with self.subTest(cohort=cohort):
                self.new = self.make_cohorts("new")
                key = self.keys[-1]
                name = ("rollout/" if cohort == "C" else "") + key + "/output.png"
                path = self.new / cohort / name
                path.write_bytes(b"changed visual content")
                record = file_record(path)
                def mutate(report):
                    report["evidence_files"][name] = record
                    report["runs"][-1]["files"]["output.png"] = record
                self.mutate_report(cohort, mutate)
                with self.assertRaisesRegex(ValueError, "changes a reviewed image"):
                    self.prove()

    def test_missing_duplicate_unfinished_changed_input_or_weight_contract_cannot_pass(self):
        mutations = [lambda d: d["runs"].pop(), lambda d: d["runs"].__setitem__(0, copy.deepcopy(d["runs"][1])),
                     lambda d: d.__setitem__("numeric_passed", False), lambda d: d["runs"][0].__setitem__("status", "not_run"),
                     lambda d: d["runs"][0]["input"].__setitem__("seed", 8),
                     lambda d: d["identity"]["source_identity"]["effective_compression"].__setitem__("bits", 3),
                     lambda d: d["identity"].__setitem__("versions", {"coreai": "different"})]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                self.new = self.make_cohorts("new")
                self.mutate_report("C", mutate)
                with self.assertRaises(ValueError):
                    self.prove()

    def test_changed_previous_approval_stale_inventory_and_links_are_rejected(self):
        image = self.new / "B" / self.keys[0] / "output.png"
        original = image.read_bytes()
        image.write_bytes(b"unrecorded bytes")
        with self.assertRaises(ValueError):
            self.prove()
        image.unlink()
        other = self.root / "elsewhere.png"
        other.write_bytes(original)
        image.symlink_to(other)
        with self.assertRaises(ValueError):
            self.prove()
        image.unlink()
        image.write_bytes(original)
        review = self.read(self.review)
        review["comparison_sha256"] = "f" * 64
        self.save(self.review, review)
        with self.assertRaises(ValueError):
            self.prove()

    def test_root_bindings_require_each_cohort_once(self):
        for values in (["C=/tmp"], ["A0=/tmp", "A0=/tmp"], ["unexpected=/tmp"], ["C="]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                cohort_roots(values)


    def make_gallery(self, root, result, side, mutate=None):
        gallery = root / "gallery"
        images, cases = {}, []
        for row in result["runs"]:
            key = row["run_id"]
            case = {"run_id": key, "prompt": "unchanged", "seed": 7, "images": {}, "guidance": {"checks": ["instruction"]}}
            for cohort, record in row["images"].items():
                name = "images/" + cohort + "/" + key + ".png"
                case["images"][cohort] = name
                images[name] = record
                path = gallery / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((cohort + key).encode())
            cases.append(case)
        payload = {"format": 1, "comparison_sha256": result[side]["comparison_file"]["sha256"],
                   "cases": cases, "review": {"language": "ja", "cohorts": ["A0", "A1", "B", "C"]}}
        payload["ratings"] = {"format": 2, "cohorts": {name: {"identity": identity, "ratings": [
            {"run_id": key, "axes": {axis: None for axis in axes}} for key, axes in quality_axes().items()]}
            for name, identity in result[side]["cohorts"].items()}}
        if mutate:
            mutate(payload)
        template = (Path(__file__).parents[1] / "recipes/qwen_image21_mobile/quality-review.html").read_bytes()
        html = template.replace(b"__REVIEW_PAYLOAD__", json.dumps(payload).encode())
        (gallery / "index.html").write_bytes(html)
        self.save(gallery / "review-receipt.json", {"format": 1, "complete": True, "html": bytes_record(html),
                  "comparison_file": result[side]["comparison_file"], "images": images,
                  "template_file": bytes_record(template), "display_guide_file": {"sha256": "d" * 64}})
        return gallery

    def test_presentation_reuses_the_same_images_prompts_guidance_and_template(self):
        result = self.prove()
        old = self.make_gallery(self.old, result, "previous")
        new = self.make_gallery(self.new, result, "candidate")
        with patch("Tools.Conversion.adapters.qwen_image21.mobile_visual_equivalence.REVIEWED_GALLERY_RECEIPT",
                   file_record(old / "review-receipt.json")):
            proof = prove_presentation(result, old, new)
        self.assertTrue(proof["same_reviewed_presentation"])
        self.assertNotIn("accepted", proof)

    def test_presentation_rejects_reordered_cases_changed_prompts_labels_and_mapping(self):
        result = self.prove()
        old = self.make_gallery(self.old, result, "previous")
        mutations = [lambda d: d["cases"].reverse(), lambda d: d["cases"][0].__setitem__("prompt", "changed"),
                     lambda d: d["review"]["cohorts"].reverse(),
                     lambda d: d["cases"][0]["images"].__setitem__("C", d["cases"][0]["images"]["B"]),
                     lambda d: d.pop("ratings"),
                     lambda d: d["ratings"]["cohorts"]["C"]["ratings"][0]["axes"].__setitem__("instruction", 2),
                     lambda d: d["ratings"]["cohorts"]["C"].__setitem__("identity", {})]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                new = self.make_gallery(self.new, result, "candidate", mutate)
                with patch("Tools.Conversion.adapters.qwen_image21.mobile_visual_equivalence.REVIEWED_GALLERY_RECEIPT",
                           file_record(old / "review-receipt.json")):
                    with self.assertRaises(ValueError):
                        prove_presentation(result, old, new)

    def test_matching_mutations_to_both_galleries_cannot_replace_the_original_presentation(self):
        result = self.prove()
        old = self.make_gallery(self.old, result, "previous")
        expected = file_record(old / "review-receipt.json")
        def mutate(payload):
            payload["cases"][0].update(prompt="changed in both galleries", seed=8)
        self.make_gallery(self.old, result, "previous", mutate)
        new = self.make_gallery(self.new, result, "candidate", mutate)
        with patch("Tools.Conversion.adapters.qwen_image21.mobile_visual_equivalence.REVIEWED_GALLERY_RECEIPT", expected):
            with self.assertRaisesRegex(ValueError, "not the preserved owner-reviewed gallery"):
                prove_presentation(result, old, new)


if __name__ == "__main__":
    unittest.main()
