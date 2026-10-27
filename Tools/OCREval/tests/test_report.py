import json
import tempfile
import unittest
from pathlib import Path

from Tools.OCREval.compare import write
from Tools.OCREval.report import build, judgment_changes
from Tools.OCREval.score import L


class ReportTests(unittest.TestCase):
    def prepare(self, root):
        context = {"model": "teleocr", "variant": "coreai-fp16", "device": "test", "os": "test", "workload": "test:A", "cache": "prepared", "plan_sha256": "a" * 64}
        plan = {"source": "test", "cli_sha256": "b" * 64, "conditions": {"backend": "test"}, "fixture_revision": {"test": "c" * 64}, "timing_boundary": "test",
                "variants": [{"id": "coreai-fp16", "engine": "coreai", "bundle": "/PRIVATE_MODEL_PATH", "asset_hashes": {"metadata.json": "d" * 64}}],
                "sets": {"test": "/PRIVATE_SET_PATH"},
                "comparisons": [{"id": "self", "axis": "conversion", "left": "coreai-fp16", "right": "coreai-fp16"}],
                "performance": {"set": "test", "sample": "A"}}
        write(root / "plan.json", plan)
        write(root / "invocation.json", {"aborted": None, "only": None, "performance": True, "plan_sha256": "a" * 64, "finished_at": "test"})
        write(root / "inputs/inputs.json", {"test": {"A": {"image": "/PRIVATE_INPUT_PATH"}}})
        write(root / "environment.json", {key: "test" for key in ("os", "machine", "python", "packages", "model", "xcode", "swift")})
        write(root / "scores/test/coreai-fp16.json", {"samples": {"A": {"judgment": L["good"], "cer": 0.1, "diff_ops_gt_vs_pred": ["PRIVATE_TEXT"]}},
                                                    "strata": {"test": {"n": 1, "n_scored": 1, "mean_cer": 0.1}}})
        write(root / "comparisons.json", {"test": {"self": {"samples": {"A": {"exact_prediction": True, "both_eos": True, "text_equal": True}}}}})
        rows = []
        for phase, count in [("quality", 1), ("cold", 1), ("warmup", 1), ("measured", 20)]:
            for i in range(count):
                row = {"run_id": f"{phase}-{i}", "set": "test", "sample": "A", "variant": "coreai-fp16", "phase": phase, "iteration": i,
                       "context": context | {"cache": "cold" if phase == "cold" else "prepared"},
                       "status": "warmup" if phase == "warmup" else "passed", "metrics": {"latency_s": {"value": i + 1, "unit": "s"}},
                       "conditions_before": {"power": "AC Power PRIVATE_BATTERY_ID", "thermal": "no warning"},
                       "conditions_after": {"power": "AC Power PRIVATE_BATTERY_ID", "thermal": "no warning"}}
                rows.append(row)
        write(root / "runs.json", rows)
        return rows

    def export(self, root):
        spec = Path(__file__).resolve().parents[3] / "docs/model-reports/metrics.json"
        return build(root, root / "public", spec)

    def test_complete_report_preserves_counts_and_uses_common_p95_without_private_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.prepare(root)
            report = self.export(root)
            measured = report["performance"]["coreai-fp16"]["prepared"]
            self.assertEqual(measured["counts"], {"passed": 20, "failed": 0, "warmup": 1})
            self.assertEqual(measured["metrics"]["latency_s"]["p95"], 19)
            self.assertEqual(measured["power_sources"], ["AC Power"])
            self.assertEqual(report["quality"]["test"]["coreai-fp16"]["metrics"]["cer"]["n"], 1)
            for path in (root / "public").rglob("*"):
                if path.is_file():
                    self.assertNotIn("PRIVATE_", path.read_text())

    def test_failed_measurements_keep_the_denominator_and_suppress_under_twenty_p95(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.prepare(root)
            rows[-1] |= {"status": "failed", "metrics": {}, "error": "PRIVATE_ERROR_PATH"}
            write(root / "runs.json", rows)
            measured = self.export(root)["performance"]["coreai-fp16"]["prepared"]
            self.assertEqual(measured["counts"]["failed"], 1)
            self.assertEqual(measured["metrics"]["latency_s"]["n"], 19)
            self.assertIsNone(measured["metrics"]["latency_s"]["p95"])

    def test_incomplete_or_duplicate_invocations_cannot_be_exported_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.prepare(root)
            for bad in (rows[:-1], rows + [rows[0]]):
                write(root / "runs.json", bad)
                with self.assertRaisesRegex(ValueError, "Missing, duplicate"):
                    self.export(root)
            write(root / "runs.json", rows)
            invocation = json.loads((root / "invocation.json").read_text())
            write(root / "invocation.json", invocation | {"aborted": "interrupted"})
            with self.assertRaisesRegex(ValueError, "complete full invocation"):
                self.export(root)

    def test_quality_judgment_drops_and_missing_samples_remain_separate(self):
        left = {"A": {"judgment": L["good"]}, "B": {"judgment": L["conditional"]}, "C": {"judgment": L["insufficient"]}}
        right = {"A": {"judgment": L["insufficient"]}, "B": {"judgment": L["good"]}}
        self.assertEqual(judgment_changes(left, right), {"same": 0, "improved": 1, "degraded": 1, "missing": 1, "degraded_samples": ["A"]})

    def test_equal_text_does_not_hide_other_prediction_differences_or_incomplete_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.prepare(root)
            path = root / "comparisons.json"
            pairs = json.loads(path.read_text())
            pairs["test"]["self"]["samples"]["A"]["exact_prediction"] = False
            write(path, pairs)
            pair = self.export(root)["comparisons"]["test"]["self"]
            self.assertEqual(pair["text_equal"], 1)
            self.assertEqual(pair["prediction_difference_with_equal_text"], 1)
            self.assertEqual(pair["non_eos_pairs"], 0)
            self.assertEqual(pair["mismatch_fraction"], 1)
            pairs["test"]["self"]["samples"]["A"] |= {"exact_prediction": True, "both_eos": False}
            write(path, pairs)
            pair = self.export(root)["comparisons"]["test"]["self"]
            self.assertEqual(pair["prediction_difference_with_equal_text"], 0)
            self.assertEqual(pair["non_eos_pairs"], 1)
            self.assertEqual(pair["mismatch_fraction"], 1)

    def test_a_good_score_from_an_incomplete_generation_is_not_a_completed_good_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.prepare(root)
            rows[0]["status"] = "failed"
            write(root / "runs.json", rows)
            quality = self.export(root)["quality"]["test"]["coreai-fp16"]
            self.assertEqual(quality["good_fraction"], 1)
            self.assertEqual(quality["completed_good_fraction"], 0)

    def test_warmup_cannot_be_relabelled_as_a_successful_measured_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.prepare(root)
            next(row for row in rows if row["phase"] == "warmup")["status"] = "passed"
            write(root / "runs.json", rows)
            with self.assertRaisesRegex(ValueError, "phase"):
                self.export(root)

    def test_derived_score_and_pair_sets_must_cover_exactly_the_fixed_inputs(self):
        for mutation in ("score-missing", "score-extra", "pair-missing", "pair-extra", "set-missing", "pair-id-missing", "input-set-extra"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.prepare(root)
                score_path = root / "scores/test/coreai-fp16.json"
                pair_path = root / "comparisons.json"
                score, pairs = json.loads(score_path.read_text()), json.loads(pair_path.read_text())
                if mutation == "score-missing":
                    score["samples"].pop("A")
                elif mutation == "score-extra":
                    score["samples"]["extra"] = score["samples"]["A"]
                elif mutation == "pair-missing":
                    pairs["test"]["self"]["samples"].pop("A")
                elif mutation == "pair-extra":
                    pairs["test"]["self"]["samples"]["extra"] = pairs["test"]["self"]["samples"]["A"]
                elif mutation == "set-missing":
                    pairs.pop("test")
                elif mutation == "pair-id-missing":
                    pairs["test"].pop("self")
                else:
                    write(root / "inputs/inputs.json", {"test": {"A": {}}, "extra": {}})
                write(score_path, score)
                write(pair_path, pairs)
                with self.assertRaisesRegex(ValueError, "set|sample|pair"):
                    self.export(root)


if __name__ == "__main__":
    unittest.main()
