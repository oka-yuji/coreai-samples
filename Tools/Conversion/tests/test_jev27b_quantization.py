import copy
import importlib.util
import json
import unittest

from Tools.Conversion.adapters.jev27b.decision import HASH_SCHEMA, compare_hashes
from Tools.Conversion.adapters.jev27b import summary
from Tools.Conversion.adapters.jev27b.probe import estimate, speed_rows
from Tools.Conversion.adapters.jev27b.verify_quantization import compare, complete, differences, finite, joined
from Tools.Conversion.storage import PipelineError

NUMPY = importlib.util.find_spec("numpy") is not None

SETTINGS = {"schema": HASH_SCHEMA, "model_id": "owner/merged", "revision": "1" * 40, "model_type": "qwen3_5", "device": "cpu",
            "versions": {"torch": "2.9.0"}, "recipe_sha256": "a" * 64, "compute_precision": "bfloat16",
            "compression": "qwen3_5_cfg", "context_length": 262144}


def record(*digests):
    return {**SETTINGS, "layout": [{"name": "w.parametrizations.weight.0", "block": [1, 32]}],
            "tensors": [{"name": f"t{index}", "kind": "buffer", "shape": [2], "dtype": "torch.uint8", "sha256": value}
                        for index, value in enumerate(digests)]}


def row(identifier, kind="noul", rejected=False, **stages):
    entry = {"set": "F", "id": identifier, "kind": kind, "band": None if rejected else "<=512", "row_sha256": "r",
             "text_sha256": "t",
             "token_count": 9000 if rejected else 3, "tokens": [1] * (9000 if rejected else 3),
             "options": ["false", "true"], "rejected": rejected}
    for stage, probabilities in stages.items():
        entry.update({f"logits{stage}": None if rejected else [0.0] * 24, f"p{stage}": probabilities,
                      f"p{stage}_T": probabilities, f"elapsed{stage}_s": None if rejected else 0.1})
    return entry


def expected(rows):
    return [{key: item[key] for key in ("set", "id", "kind", "band", "row_sha256", "token_count", "rejected")}
            | {"n_options": len(item["options"])} for item in rows]


class HashComparisonTests(unittest.TestCase):
    def test_identical_records_match_and_any_difference_does_not(self):
        first = record("a" * 64, "b" * 64)
        self.assertTrue(compare_hashes(first, copy.deepcopy(first))["hashes_match"])
        changed = compare_hashes(first, record("a" * 64, "c" * 64))
        self.assertFalse(changed["hashes_match"])
        self.assertEqual((changed["differing"], changed["matching"]), (["t1"], 1))
        self.assertFalse(compare_hashes(first, record("a" * 64))["hashes_match"])
        self.assertFalse(compare_hashes(first, {**record("a" * 64, "b" * 64), "device": "mps"})["hashes_match"])
        self.assertFalse(compare_hashes(first, {**record("a" * 64, "b" * 64), "layout": []})["hashes_match"])
        self.assertFalse(compare_hashes(record(None, "b" * 64), record(None, "b" * 64))["hashes_match"])


class ReferenceRowTests(unittest.TestCase):
    def test_complete_rows_pass_and_missing_results_or_reordered_rows_fail(self):
        rows = [row("a", **{"2": [0.4, 0.6]}), row("b", rejected=True, **{"2": None})]
        self.assertEqual(complete(rows, expected(rows), "2"), [])
        missing = copy.deepcopy(rows)
        missing[0]["logits2"] = None
        self.assertEqual(complete(missing, expected(rows), "2"), ["a: forward results do not match the rejection"])
        self.assertEqual(complete(rows[::-1], expected(rows), "2"), ["rows differ from the fixtures in identity or order"])

    def test_probabilities_must_be_finite_bounded_and_normalized(self):
        self.assertEqual(finite([row("a", **{"2": [0.4, 0.6]})], "2"), [])
        self.assertEqual(finite([row("a", **{"2": [0.4, 0.7]})], "2"), ["a"])
        self.assertEqual(finite([row("a", **{"2": [-0.1, 1.1]})], "2"), ["a"])
        self.assertEqual(finite([row("a", rejected=True, **{"2": None})], "2"), [])
        unrecorded = row("a", **{"2": [0.4, 0.6]})
        unrecorded.update(logits2=None, p2=None, p2_T=None)
        self.assertEqual(finite([unrecorded], "2"), ["a"])

    def test_the_comparison_counts_top_changes_and_rows_within_twice_the_error(self):
        rows = [row("near", **{"2": [0.52, 0.48], "1m": [0.48, 0.52]}),
                row("far", **{"2": [0.10, 0.90], "1m": [0.12, 0.88]})]
        result = compare(rows, "2", "1m")
        plain = result["temperature_1"]
        self.assertEqual(plain["top1_changed"], {"rows": 1, "ids": ["near"]})
        self.assertAlmostEqual(plain["max_probability_error"], 0.04)
        self.assertAlmostEqual(plain["mean_probability_error"], 0.03)
        self.assertEqual(plain["margin_within_2e"], {"rows": 1, "ids": ["near"]})
        self.assertEqual(result["shipping_temperature"]["margin_within_2e"], {"rows": 1, "ids": ["near"]})
        self.assertEqual(set(plain["by_kind"]), {"noul"})
        shifted = copy.deepcopy(rows)
        shifted[1]["p2_T"] = [0.60, 0.40]
        moved = compare(shifted, "2", "1m")
        shipped = moved["shipping_temperature"]
        self.assertEqual((shipped["margin_within_2e"]["rows"], shipped["top1_changed"]["rows"]), (2, 2))
        self.assertEqual(moved["temperature_1"]["margin_within_2e"]["rows"], 1)


class JoinTests(unittest.TestCase):
    def test_rows_join_by_identity_and_q_rows_keep_only_the_quantized_stages(self):
        first = row("a", **{"2": [0.4, 0.6]})
        second = row("b", **{"2": [0.7, 0.3]})
        q_row = {**row("q", **{"2": [0.5, 0.5]}), "set": "Q"}
        merged = {"rows": [first, second, q_row]}
        base = {"rows": [{**item, **{key.replace("2", "0q"): value for key, value in item.items() if "2" in key}}
                         for item in copy.deepcopy(merged["rows"])]}
        base["rows"][1]["p0q"] = [0.6, 0.4]
        stage0 = {"rows": [row("b", **{"0": [0.9, 0.1], "1": [0.8, 0.2]}), row("a", **{"0": [0.3, 0.7], "1": [0.4, 0.6]})]}
        stage1m = {"rows": [row("a", **{"1m": [0.45, 0.55]}), row("b", **{"1m": [0.75, 0.25]})]}
        rows = joined((merged, base), (stage0, stage1m))
        self.assertEqual([item["id"] for item in rows], ["a", "b", "q"])
        self.assertEqual((rows[0]["p0"], rows[0]["p1m"], rows[1]["p1"]), ([0.3, 0.7], [0.45, 0.55], [0.8, 0.2]))
        self.assertNotIn("p0", rows[2])
        self.assertEqual(rows[2]["p0q"], [0.5, 0.5])
        result = differences(rows[:2], "2", "0q")
        self.assertEqual(result["rows"], 2)
        self.assertAlmostEqual(result["mean"], 0.05)
        self.assertAlmostEqual(result["max"], 0.1)


class SpeedEstimateTests(unittest.TestCase):
    def test_speed_rows_take_the_closest_distinct_rows(self):
        rows = [{"id": name, "token_count": count} for name, count in (("s", 40), ("m", 500), ("l", 1500), ("x", 3900))]
        self.assertEqual([item["id"] for item in speed_rows(rows)], ["s", "m", "l", "x"])
        self.assertEqual([item["id"] for item in speed_rows(rows[:2])], ["s", "m"])

    @unittest.skipUnless(NUMPY, "numpy is required for the time estimate")
    def test_the_estimate_fits_a_quadratic_and_falls_back_to_a_line(self):
        rows = [{"token_count": count} for count in (100, 1000)]
        quadratic = estimate([(100, 2.0 + 0.01 * 100 + 1e-6 * 100 ** 2), (1000, 2.0 + 10.0 + 1.0),
                              (3000, 2.0 + 30.0 + 9.0)], rows)
        self.assertEqual(quadratic["model"], "quadratic")
        self.assertAlmostEqual(quadratic["estimate_s"], 3.01 + 13.0, places=6)
        concave = estimate([(100, 1.0), (1000, 9.0), (3000, 20.0)], rows)
        self.assertEqual(concave["model"], "linear")
        single = estimate([(100, 5.0)], rows)
        self.assertEqual((single["model"], single["estimate_s"]), ("per-token", 5.0 / 100 * 1100))


class QuantizationSummaryTests(unittest.TestCase):
    report = {"version": 1, "status": "failed", "verification": "failed", "stages": []}

    def comparison(self):
        temperature = {"max_probability_error": 0.2, "mean_probability_error": 0.02, "by_kind": {}, "by_band": {},
                       "top1_changed": {"rows": 3, "ids": ["private-example-sentinel"]},
                       "margin_within_2e": {"rows": 5, "ids": ["private-example-sentinel"]}}
        side = {"rows": 301, "temperature_1": temperature, "shipping_temperature": copy.deepcopy(temperature)}
        return {"schema": "typed-decision-quantization/1", "passed": True,
                "checks": {"hashes_match": True, "rows_complete": True, "finite": True},
                "hashes": {"items": 1622, "matching": 1622},
                "rows": {"F": 302, "Q": 900, "F_forwarded": 301, "Q_forwarded": 900, "rejected": 1},
                "stage2_vs_1m": side, "stage0q_vs_0": copy.deepcopy(side), "devices": {"2": "private-example-sentinel"},
                "adaptation": {"F": {"stage 2 against 0q": {"rows": 301, "mean": 0.25, "max": 0.9}},
                               "Q": {"stage 2 against 0q": {"rows": 0}}}}

    def test_the_quantization_check_projects_only_its_numbers(self):
        result = summary.project(self.report, self.comparison())
        self.assertEqual(result["status"], "failed")
        comparison = result["comparison"]
        self.assertEqual(comparison["stage2_vs_1m"]["temperature_1"],
                         {"max_probability_error": 0.2, "mean_probability_error": 0.02, "top1_changed": 3,
                          "margin_within_2e": 5})
        self.assertEqual((comparison["passed"], comparison["hashes_match"], comparison["hashed_tensors"],
                          comparison["rows"]["F"]), (True, True, 1622, 302))
        self.assertEqual(comparison["adaptation_mean"], {"F: stage 2 against 0q": 0.25})
        self.assertNotIn("private-example-sentinel", json.dumps(result))

    def test_the_quantization_check_rejects_invalid_values_and_other_schemas(self):
        for mutate in (lambda c: c["checks"].update(finite="yes"), lambda c: c["hashes"].update(items=-1),
                       lambda c: c["stage2_vs_1m"]["temperature_1"].update(max_probability_error=float("nan")),
                       lambda c: c["rows"].update(Q=1.5), lambda c: c.update(schema="typed-decision-merge/1")):
            comparison = self.comparison()
            mutate(comparison)
            with self.subTest(comparison=comparison), self.assertRaises(PipelineError):
                summary.project(self.report, comparison)

    def test_an_incomplete_check_projects_its_results_without_statistics(self):
        comparison = self.comparison()
        for key in ("stage2_vs_1m", "stage0q_vs_0", "adaptation"):
            del comparison[key]
        comparison.update(passed=False, checks={"hashes_match": False, "rows_complete": True, "finite": True})
        result = summary.project(self.report, comparison)["comparison"]
        self.assertEqual((result["passed"], result["hashes_match"]), (False, False))
        self.assertNotIn("stage2_vs_1m", result)


if __name__ == "__main__":
    unittest.main()
