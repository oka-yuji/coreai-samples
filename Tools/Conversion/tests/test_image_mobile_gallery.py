import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from Tools.Conversion.adapters.qwen_image21.fixtures import quality_axes
from Tools.Conversion.adapters.qwen_image21.mobile_gallery import build, bytes_record, validate_ratings


class ImageMobileGalleryTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node is required to exercise asynchronous gallery restoration")
    def test_a_stale_file_read_cannot_replace_newer_ratings_or_manual_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = self.fixture(Path(directory))
            with patch("Tools.Conversion.adapters.qwen_image21.mobile_gallery.mobile_fixtures"):
                build(args)
            template = Path(__file__).parents[1] / "recipes/qwen_image21_mobile/quality-review.html"
            result = subprocess.run([shutil.which("node"), str(Path(__file__).with_name("image_gallery_ui.cjs")),
                                     str(template), str(args.output / "index.html")], capture_output=True, text=True, check=True)
            cases = json.loads(result.stdout)["results"]
            self.assertEqual(len(cases), 2)
            self.assertTrue(all(case["passed"] for case in cases))

    def fixture(self, root):
        from PIL import Image

        stream = io.BytesIO()
        Image.new("RGBA", (512, 512), (210, 50, 40, 120)).save(stream, format="PNG")
        png = stream.getvalue()
        args = SimpleNamespace(**{key: root / key for key in (
            "comparison", "fixtures", "baseline", "mixed_reference", "reference", "verification", "output")})
        for name in vars(args):
            if name != "output":
                getattr(args, name).mkdir()
        runs, fixture_files = [], {}
        for key in quality_axes():
            identifier, seed = key.split("-")
            runs.append({"run_id": key, "fixture_id": identifier, "seed": int(seed)})
            path = identifier + "/input.json"
            if path not in fixture_files:
                reference = "images/" + identifier + ".png" if identifier.startswith("e") else None
                metadata = {"prompt": "Literal </script><script>alert(1)</script> & text", "reference": reference}
                fixture_files[path] = self.write(args.fixtures / path, metadata)
                if reference:
                    target = args.fixtures / reference
                    target.parent.mkdir(exist_ok=True)
                    target.write_bytes(png)
                    fixture_files[reference] = bytes_record(png)
        self.write(args.fixtures / "fixtures.json", {"runs": runs, "inventory": fixture_files})
        cohorts = {}
        for name, folder in zip(("A0", "A1", "B", "C"), (
                args.baseline, args.mixed_reference, args.reference, args.verification), strict=True):
            recorded, files = [], {}
            for run in runs:
                path = ("rollout/" if name == "C" else "") + run["run_id"] + "/output.png"
                target = folder / path
                target.parent.mkdir(parents=True)
                target.write_bytes(png)
                files[path] = bytes_record(png)
                recorded.append({"run_id": run["run_id"], "input": run, "status": "completed",
                                 "files": {"output.png": bytes_record(png)}})
            report = {"runs": recorded, "complete": True, "numeric_passed": True, "evidence_files": files}
            filename, field = ("verification.json", "verification_file") if name == "C" else ("summary.json", "summary_file")
            cohorts[name] = {field: self.write(folder / filename, report)}
        ratings = {"format": 2, "cohorts": {name: {"identity": identity,
                   "ratings": [{"run_id": key, "axes": {axis: None for axis in axes}}
                               for key, axes in quality_axes().items()]} for name, identity in cohorts.items()}}
        quality_file = self.write(args.comparison / "quality-ratings.json", ratings)
        self.write(args.comparison / "comparison.json", {"format": 2, "cohorts": cohorts, "quality_file": quality_file,
                                                        "runs": [{"run_id": row["run_id"], "input": row} for row in runs]})
        return args, cohorts, ratings

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(value, ensure_ascii=False) + "\n").encode()
        path.write_bytes(data)
        return bytes_record(data)

    def test_complete_review_copies_bound_images_and_keeps_all_axes_unreviewed(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, ratings = self.fixture(Path(directory))
            with patch("Tools.Conversion.adapters.qwen_image21.mobile_gallery.mobile_fixtures") as validation:
                receipt = build(args)
                validation.assert_called_once()
            html = (args.output / "index.html").read_text()
            embedded = html.split('<script id="payload" type="application/json">', 1)[1].split("</script>", 1)[0]
            payload = json.loads(embedded)
            self.assertEqual(payload["ratings"], ratings)
            self.assertEqual(len(payload["cases"]), 60)
            self.assertEqual(len(receipt["images"]), 255)
            self.assertEqual(html.count("<script"), 2)
            self.assertIn("</script><script>alert(1)</script>", payload["cases"][0]["prompt"])
            for case in payload["cases"]:
                self.assertEqual(set(case["images"]), {"A0", "A1", "B", "C"})
            self.assertTrue(all(score is None for cohort in ratings["cohorts"].values()
                                for row in cohort["ratings"] for score in row["axes"].values()))

    def test_changed_inputs_are_rejected_without_creating_a_reviewable_page(self):
        for defect in ("image", "summary", "ratings", "input", "overlap", "link"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                args, _, _ = self.fixture(Path(directory))
                if defect == "image":
                    (args.baseline / "g01-20261018/output.png").write_bytes(b"different")
                elif defect == "summary":
                    (args.baseline / "summary.json").write_text("{}")
                elif defect == "ratings":
                    (args.comparison / "quality-ratings.json").write_text("{}")
                elif defect == "input":
                    value = json.loads((args.comparison / "comparison.json").read_text())
                    value["runs"][0]["input"]["seed"] = 0
                    self.write(args.comparison / "comparison.json", value)
                elif defect == "overlap":
                    args.output = args.fixtures / "review"
                else:
                    original = args.baseline / "g01-20261018/output.png"
                    original.unlink()
                    original.symlink_to(args.reference / "g01-20261018/output.png")
                with patch("Tools.Conversion.adapters.qwen_image21.mobile_gallery.mobile_fixtures"):
                    with self.assertRaises((ValueError, FileExistsError)):
                        build(args)
                self.assertFalse((args.output / "index.html").exists())

    def test_scores_cannot_drop_rows_axes_cohorts_or_output_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            _, cohorts, initial = self.fixture(Path(directory))
            for defect in ("cohort", "identity", "row", "duplicate", "axis", "bool", "range"):
                changed = copy.deepcopy(initial)
                row = changed["cohorts"]["C"]["ratings"][0]
                if defect == "cohort":
                    changed["cohorts"].pop("A0")
                elif defect == "identity":
                    changed["cohorts"]["C"]["identity"] = {"changed": True}
                elif defect == "row":
                    changed["cohorts"]["C"]["ratings"].pop()
                elif defect == "duplicate":
                    changed["cohorts"]["C"]["ratings"][-1] = row
                elif defect == "axis":
                    row["axes"].pop("instruction")
                else:
                    row["axes"]["instruction"] = True if defect == "bool" else 3
                with self.subTest(defect=defect), self.assertRaises(ValueError):
                    validate_ratings(changed, cohorts)
            accepted = copy.deepcopy(initial)
            accepted["cohorts"]["C"]["ratings"][0]["axes"]["instruction"] = 1
            validate_ratings(accepted, cohorts)


if __name__ == "__main__":
    unittest.main()
