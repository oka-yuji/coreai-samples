import json
import tempfile
import unittest
from pathlib import Path

from Tools.OCREval.fixtures import make
from Tools.OCREval.score import L, load_set, reference_prediction, score_set, selftest

BASELINE = Path(__file__).resolve().parents[3] / "docs/results/teleocr-eval/2026-10-25"


def result_blocks(gt, pred):
    pipeline = gt["pipelines"][0]
    if pipeline == "two_stage":
        texts = [p.read_text() for p in sorted((BASELINE / "outputs").glob(f"{gt['id']}-block[0-9][0-9]-text-bf16.txt"))]
        return [{"label": b["label"], "box_px": b["crop_px"], "prompt": "text", "text": t} for b, t in zip(pred["blocks"], texts)]
    if pipeline == "table":
        return [{"label": "table", "box_px": gt["table_bbox"], "prompt": "table", "text": "", "otsl": pred["table"]}]
    return [{"label": "text", "box_px": [0, 0, *gt["size_px"]], "prompt": "text", "text": pred["text"]}]


class ScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.set = cls.root / "set"
        make(cls.set)
        cls.baseline = json.loads((BASELINE / "scores.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_selftest(self):
        selftest()

    def test_baseline_scores_are_reproduced(self):
        self.assertEqual(score_set(self.set, outputs=BASELINE / "outputs"), self.baseline)

    def test_result_json_gives_the_reference_metrics(self):
        gts = load_set(self.set)
        results = self.root / "results"
        results.mkdir()
        ids = ("F1", "F2", "F4", "S07", "S13", "S19", "S25")
        for sid in ids:
            blocks = result_blocks(gts[sid], reference_prediction(BASELINE / "outputs", gts[sid]))
            doc = {"schema": "teleocr-ocr-result/0", "layout": {"status": "parsed", "resize": 1036}, "blocks": blocks}
            (results / f"{sid}.json").write_text(json.dumps(doc, ensure_ascii=False))
        scored = score_set(self.set, results=results)["samples"]
        for sid in ids:
            self.assertEqual(scored[sid], self.baseline["samples"][sid], sid)
        self.assertEqual(scored["S01"], {"missing": True, "judgment": L["not_scored"]})

    def test_layout_without_boxes_scores_an_empty_prediction(self):
        outputs = self.root / "no-boxes"
        outputs.mkdir()
        (outputs / "S19-layout-bf16.txt").write_text("no boxes\n")
        scored = score_set(self.set, outputs=outputs)["samples"]["S19"]
        self.assertEqual((scored["cer"], scored["layout_missing_rate"], scored["judgment"]), (1.0, 1.0, L["insufficient"]))

    def test_unparsed_layout_is_marked(self):
        results = self.root / "unparsed"
        results.mkdir()
        doc = {"layout": {"status": "unparsed"}, "blocks": [{"label": "text", "box_px": [0, 0, 1240, 1754], "text": "x"}]}
        (results / "F4.json").write_text(json.dumps(doc))
        scored = score_set(self.set, results=results)["samples"]["F4"]
        self.assertTrue(scored["layout_unparsed"])
        self.assertEqual(scored["layout_missing_rate"], 0)
        self.assertEqual(scored["judgment"], L["insufficient"])


if __name__ == "__main__":
    unittest.main()
