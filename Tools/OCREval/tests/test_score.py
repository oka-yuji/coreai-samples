import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from Tools.OCREval.fixtures import make
from Tools.OCREval.score import TEXT_KEYS, L, judge, load_set, reference_prediction, score_set, selftest

BASELINE = Path(__file__).resolve().parents[3] / "docs/results/teleocr-eval/2026-10-25"


def result_blocks(gt, pred):
    pipeline = gt["pipelines"][0]
    if pipeline == "two_stage":
        texts = [p.read_text() for p in sorted((BASELINE / "outputs").glob(f"{gt['id']}-block[0-9][0-9]-text-bf16.txt"))]
        found = [{"label": b["label"], "box_px": b["crop_px"], "prompt": "text", "text": t} for b, t in zip(pred["blocks"], texts)]
    elif pipeline == "table":
        found = [{"label": "table", "box_px": gt["table_bbox"], "prompt": "table", "text": pred["table"], "otsl": pred["table"]}]
    else:
        found = [{"label": "text", "box_px": [0, 0, *gt["size_px"]], "prompt": "text", "text": pred["text"]}]
    for index, block in enumerate(found):
        block.update(index=index, box_norm=[round(v * 1000 / size) for v, size in zip(block["box_px"], gt["size_px"] * 2)],
                     seconds={"vision": 0.1, "first_token": 0.2, "decode": 0.3, "total": 0.5},
                     tokens={"prompt": 100, "image": 64, "output": 20}, finished_by="eos")
    return found


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
            doc = {"record": "ocr", "schema": "teleocr-ocr-result/1", "image": f"{sid}.png", "image_px": gts[sid]["size_px"],
                   "layout": {"status": "parsed", "resize": 1036, "skipped_boxes": []}, "blocks": blocks}
            (results / f"{sid}.json").write_text(json.dumps(doc, ensure_ascii=False))
        scored = score_set(self.set, results=results)["samples"]
        for sid in ids:
            self.assertEqual(scored[sid], self.baseline["samples"][sid], sid)
        self.assertEqual(scored["S01"], {"missing": True, "judgment": L["not_scored"]})

    def test_cli_record_fixture_is_scored_by_the_command(self):
        fixtures = Path(__file__).parent / "fixtures"
        record = json.loads((fixtures / "F4.json").read_text())
        self.assertEqual(record["schema"], "teleocr-ocr-result/1")
        self.assertEqual(record["layout"]["skipped_boxes"], [])
        output = self.root / "cli-fixture-scores.json"
        subprocess.run([sys.executable, "-m", "Tools.OCREval", "score", "--set", str(self.set),
                        "--results", str(fixtures), "--out", str(output)],
                       cwd=BASELINE.parents[3], check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(output.read_text())["samples"]["F4"], self.baseline["samples"]["F4"])

    def test_layout_without_boxes_scores_an_empty_prediction(self):
        outputs = self.root / "no-boxes"
        outputs.mkdir()
        (outputs / "S19-layout-bf16.txt").write_text("no boxes\n")
        scored = score_set(self.set, outputs=outputs)["samples"]["S19"]
        self.assertEqual((scored["cer"], scored["layout_missing_rate"], scored["judgment"]), (1.0, 1.0, L["insufficient"]))

    def test_missing_block_outputs_are_not_scored(self):
        outputs = self.root / "partial"
        outputs.mkdir()
        (outputs / "S19-layout-bf16.txt").write_text("<box:088 088 468 121><label:title><up>\n<box:087 153 462 499><label:text><up>\n")
        (outputs / "S19-block01-text-bf16.txt").write_text("x\n")
        scored = score_set(self.set, outputs=outputs)["samples"]["S19"]
        self.assertEqual(scored, {"missing": True, "incomplete_blocks": [1, 2], "block_runs": ["S19-block01-text-bf16"],
                                  "judgment": L["not_scored"]})

    def test_samples_that_must_not_be_redistributed_keep_only_numbers(self):
        text = {"id": "P1", "kind": "text", "stratum": "public", "rule": "F1", "pipelines": ["text"],
                "lines": ["The quick brown fox jumps over the lazy dog 12 times."], "crop_px": [0, 0, 10, 10],
                "image": "page.png", "size_px": [10, 10]}
        table = {"id": "P2", "kind": "table", "stratum": "public", "rule": "F2", "pipelines": ["table"], "title": "",
                 "cells": [["Item", "Count"], ["Bolt", "12"]], "crop_px": [0, 0, 10, 10], "image": "page.png",
                 "size_px": [10, 10]}
        outputs = self.root / "public-outputs"
        outputs.mkdir()
        (outputs / "P1-text-bf16.txt").write_text("The quick brown fox jumps over the lazy cat 12 times.\n")
        (outputs / "P2-table-bf16.txt").write_text("<fcel>Item<fcel>Count<nl><fcel>Bolt<fcel>13<nl>\n")
        for redistribute in (False, None):
            folder = self.root / f"public-{redistribute}"
            folder.mkdir()
            for gt in (text, table):
                extra = {} if redistribute is None else {"redistribute": redistribute}
                (folder / f"{gt['id']}.json").write_text(json.dumps(gt | extra))
            samples = score_set(folder, outputs=outputs)["samples"]
            kept = {k for m in samples.values() for k in m}
            if redistribute is False:
                self.assertFalse(kept & set(TEXT_KEYS), kept)
                self.assertTrue({"cer", "cer_raw", "numbers_match", "cell_match", "shape_strict_match"} <= kept, kept)
            else:
                self.assertTrue(set(TEXT_KEYS) <= kept, kept)

    def test_judgment_thresholds_are_inclusive(self):
        above = lambda x: math.nextafter(x, 1)
        good, cond, bad = L["good"], L["conditional"], L["insufficient"]
        cases = [
            ("F1", {"cer": 0.02}, good), ("F1", {"cer": above(0.02)}, cond),
            ("F1", {"cer": 0.10}, cond), ("F1", {"cer": above(0.10)}, bad),
            ("F3", {"cer": 0.03, "numbers_match": 0.90}, good), ("F3", {"cer": above(0.03), "numbers_match": 0.90}, cond),
            ("F3", {"cer": 0.03, "numbers_match": math.nextafter(0.90, 0)}, cond),
            ("F3", {"cer": 0.10, "numbers_match": 0.70}, cond), ("F3", {"cer": above(0.10), "numbers_match": 1.0}, bad),
            ("F3", {"cer": 0.10, "numbers_match": math.nextafter(0.70, 0)}, bad),
            ("F3", {"cer": 0.03, "numbers_match": None}, good), ("F3", {"cer": 0.10, "numbers_match": None}, cond),
            ("F4", {"cer": 0.05, "read_order_edit": 0.10}, good), ("F4", {"cer": 0.05, "read_order_edit": above(0.10)}, cond),
            ("F4", {"cer": above(0.05), "read_order_edit": 0.0}, cond), ("F4", {"cer": 0.05, "read_order_edit": None}, cond),
            ("F4", {"cer": 0.15}, cond), ("F4", {"cer": above(0.15)}, bad),
            ("F5", {"cer": 0.10, "numbers_match": 0.80}, good), ("F5", {"cer": above(0.10), "numbers_match": 1.0}, cond),
            ("F5", {"cer": 0.10, "numbers_match": math.nextafter(0.80, 0)}, cond),
            ("F5", {"cer": 0.25, "numbers_match": 0.0}, cond), ("F5", {"cer": above(0.25), "numbers_match": 1.0}, bad),
            ("F5", {"cer": 0.10, "numbers_match": None}, good),
        ]
        for rule, metrics, expected in cases:
            with self.subTest(rule=rule, metrics=metrics):
                self.assertEqual(judge(rule, metrics), expected)

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
