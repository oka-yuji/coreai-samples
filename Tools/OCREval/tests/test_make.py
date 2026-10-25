import json
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from Tools.OCREval.fixtures import DATA, F_PIPELINES, make

BASELINE = Path(__file__).resolve().parents[3] / "docs/results/teleocr-eval/2026-10-25"
F178_IMAGES = {
    "F1.png": "5e7bd11137589429c1c19e73f1c944ecaa514fe8441cc6b5cc51d01b2f9befdc",
    "F2.png": "1e3f442bc924bc1b919dea0cfa3782b60fd83c051b095e0243353d6283c6a72f",
    "F3.png": "459cb5959af1adbb6be74c50808674d879c35a1b8241ffee314b596bb20c376e",
    "F4.png": "50e6741afea1be14839e67c01e6a8355cdf0e634379f6838efc359758ab8c747",
    "F5.jpg": "b354c5139f3416c3cdb243f5051016b6f1e5241a5954c86c7ab32575ae3648e0",
}


class MakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name) / "a"
        cls.first = make(cls.dir)
        cls.second = make(Path(cls.tmp.name) / "b")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_regeneration_is_byte_identical(self):
        self.assertEqual(len(self.first["samples"]), 35)
        self.assertEqual(self.first, self.second)

    def test_baseline_used_the_same_inputs(self):
        self.assertEqual(json.loads((BASELINE / "set.json").read_text()), self.first)

    def test_earlier_fixtures_are_reproduced(self):
        images = {row["image"]: row["image_sha256"] for row in self.first["samples"]}
        self.assertEqual({name: images[name] for name in F178_IMAGES}, F178_IMAGES)
        for fid in F_PIPELINES:
            original = json.loads((DATA / f"{fid}.json").read_text())
            generated = json.loads((self.dir / f"{fid}.json").read_text())
            self.assertEqual({k: generated[k] for k in original}, original)

    def test_each_stratum_uses_every_condition_twice(self):
        gts = [json.loads((self.dir / f"S{i:02d}.json").read_text()) for i in range(1, 31)]
        for stratum in ("body", "numbers", "table", "columns", "camera"):
            group = [g["conditions"] for g in gts if g["stratum"] == stratum]
            self.assertEqual(len(group), 6)
            self.assertEqual(Counter(c["font"] for c in group), Counter({"mincho": 2, "gothic_w3": 2, "gothic_w6": 2}))
            self.assertEqual(Counter(c["size_px"] for c in group), Counter({28: 2, 36: 2, 44: 2}))
            expected = {"camera": 6} if stratum == "camera" else {"none": 2, "blur": 2, "perspective": 2}
            self.assertEqual(Counter(c["degradation"] for c in group), Counter(expected))

    def test_crop_and_line_boxes_stay_on_the_page(self):
        for path in self.dir.glob("*.json"):
            gt = json.loads(path.read_text())
            for box in [gt["table_bbox"]] if "table_bbox" in gt else gt.get("line_boxes", []):
                self.assertTrue(0 <= box[0] < box[2] <= 1240 and 0 <= box[1] < box[3] <= 1754, (path.name, box))


class RepositoryTests(unittest.TestCase):
    def test_no_evaluation_images_are_tracked(self):
        files = subprocess.run(["git", "ls-files", "--", "docs/results/teleocr-eval", "Tools/OCREval"], cwd=BASELINE.parents[3],
                               capture_output=True, text=True, check=True).stdout.split("\n")
        self.assertIn("docs/results/teleocr-eval/2026-10-25/scores.json", files)
        self.assertEqual([f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))], [])


if __name__ == "__main__":
    unittest.main()
