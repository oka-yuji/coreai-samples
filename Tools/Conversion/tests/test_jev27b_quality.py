from fractions import Fraction
import math
import unittest

from Tools.Conversion.adapters.jev27b.compare import tokens_sha256
from Tools.Conversion.adapters.jev27b.score import (
    auroc, bounds, ece, gain_condition, gains, judge, log_softmax, percentile, relative, resampled, resamples, score,
    summary, teacher, units,
)
from Tools.Conversion.adapters.jev27b.typed_head import decide, softmax

CARD = {"noul": 1.25, "choice": 0.75, "score": 1.5}
OPTIONS = {"noul": ["false", "true"], "score": ["0", "1", "2", "3", "4", "5"], "choice": ["a", "b", "c"]}


def logits(seed, offset=0.0):
    return [((7 * row + 3 * seed) % 11 - 5) / 4 + offset * (row % 3) for row in range(24)]


def stage(name, values, kind):
    shipped, plain = decide(values, kind, len(OPTIONS[kind]), CARD[kind])
    return {f"logits{name}": values, f"p{name}": plain, f"p{name}_T": shipped, f"elapsed{name}_s": 0.1}


def documents(specs):
    expected, reference, peft, merged, swift, restored = [], [], [], [], [], []
    for index, spec in enumerate(specs):
        kind, tokens = spec["kind"], [index, index + 1, index + 2]
        options = OPTIONS[kind]
        base = {"set": "Q", "id": f"row{index}", "kind": kind, "options": options, "band": "<=512",
                "row_sha256": f"{index:064x}", "text_sha256": "0" * 64, "token_count": 3, "tokens": tokens,
                "rejected": False}
        expected.append({key: base[key] for key in ("set", "id", "kind", "band", "row_sha256", "token_count", "rejected")}
                        | {"n_options": len(options)})
        values = spec["logits"]
        reference.append({**base, **stage("2", values["2"], kind), **stage("0q", values["0q"], kind)})
        peft.append({**base, **stage("1", values["1m"], kind), **stage("0", values["0"], kind)})
        merged.append({**base, **stage("1m", values["1m"], kind)})
        shipped, plain = decide(values["3"], kind, len(options), CARD[kind])
        swift.append({"id": base["id"], "kind": kind, "token_count": 3, "tokens_sha256": tokens_sha256(tokens),
                      "logits": values["3"], "probabilities": shipped, "probabilities_t1": plain, "capacity": 256,
                      "top": 0, "elapsed_seconds": 0.5})
        restored.append({"id": base["id"], "kind": kind, "state": spec.get("state", f"state {index}"), "question": "q",
                         "options": options, "target": spec["target"]})
    return ({"temperatures": CARD, "rows": reference}, {"passed": True, "rows": peft},
            {"passed": True, "rows": merged}, {"schema": "typed-decide/1", "rows": swift}, expected, restored)


def specs(adapted=0.5):
    targets = {"noul": [[0.3, 0.7], [0.8, 0.2], [0.5, 0.5]], "choice": [[0.1, 0.2, 0.7], [0.6, 0.3, 0.1], [0.2, 0.5, 0.3]],
               "score": [[0, 0, 0.1, 0.6, 0.3, 0], [0.5, 0.5, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]}
    rows = []
    for kind, values in targets.items():
        for index, target in enumerate(values):
            seed = len(rows)
            rows.append({"kind": kind, "target": target, "logits": {
                "0": logits(seed), "0q": logits(seed, 0.1), "1m": logits(seed, adapted), "2": logits(seed, adapted + 0.05),
                "3": logits(seed, adapted + 0.1)}})
    return rows


class MetricTests(unittest.TestCase):
    def test_targets_are_checked_and_normalized(self):
        self.assertEqual(teacher([0.3, 0.7], 2), ([0.3, 0.7], 0.0))
        normalized, deviation = teacher([0.3, 0.6995], 2)
        self.assertAlmostEqual(math.fsum(normalized), 1.0, places=15)
        self.assertAlmostEqual(deviation, -5e-4, places=12)
        self.assertEqual(teacher([0, 1], 2), ([0.0, 1.0], 0))
        for broken, count in (([0.3, 0.698], 2), ([0.3, 0.7], 3), ([-0.1, 1.1], 2), ([True, False], 2),
                              ([math.nan, 1.0], 2), ("0.3,0.7", 2), (None, 2)):
            self.assertIsNone(teacher(broken, count), broken)

    def test_noul_brier_soft_hard_and_multiclass(self):
        values = [0.0] * 24
        values[1] = 1.0
        row = judge(values, "noul", [0.3, 0.7], 1.0)
        p = softmax([0.0, 1.0])
        self.assertAlmostEqual(row["brier_soft"], (p[1] - 0.7) ** 2, places=15)
        self.assertAlmostEqual(row["brier_hard"], (p[1] - 1) ** 2, places=15)
        self.assertAlmostEqual(row["brier_multiclass"], 2 * row["brier_soft"], places=15)
        tied = judge(values, "noul", [0.5, 0.5], 1.0)
        self.assertTrue(tied["tie"])
        self.assertFalse(tied["positive"])
        self.assertAlmostEqual(tied["brier_hard"], p[1] ** 2, places=15)

    def test_score_mae_uses_the_expected_score_and_its_normalized_form(self):
        row = judge([0.0] * 24, "score", [0, 0, 0, 0, 0, 1], 1.0)
        self.assertAlmostEqual(row["mae"], 2.5, places=12)
        self.assertAlmostEqual(row["mae_normalized"], 0.5, places=12)

    def test_kl_takes_log_probabilities_from_logsumexp_and_skips_zero_targets(self):
        values = [0.0] * 24
        values[0] = 1000.0
        self.assertAlmostEqual(judge(values, "noul", [1.0, 0.0], 1.0)["kl"], 0.0, places=12)
        self.assertAlmostEqual(judge(values, "noul", [0.0, 1.0], 1.0)["kl"], 1000.0, places=9)
        self.assertEqual(log_softmax([1000.0, 0.0])[0], 0.0)
        mixed = [0.2, 1.5, -0.7]
        target = [0.25, 0.75, 0.0]
        values = [0.0] * 8 + mixed + [0.0] * 13
        direct = math.fsum(t * math.log(t / p) for t, p in zip(target, softmax(mixed)) if t > 0)
        self.assertAlmostEqual(judge(values, "choice", target, 1.0)["kl"], direct, places=14)
        same = softmax(mixed)
        self.assertAlmostEqual(judge(values, "choice", same, 1.0)["kl"], 0.0, places=14)

    def test_auroc_gives_ties_one_half_and_is_undefined_with_one_class(self):
        self.assertEqual(auroc([0.9, 0.8, 0.8, 0.1], [True, True, False, False]), 0.875)
        self.assertIsNone(auroc([0.9, 0.8], [True, True]))
        self.assertIsNone(auroc([], []))

    def test_auroc_leaves_out_rows_with_a_half_target(self):
        rows = [judge(logits(seed), "noul", target, 1.0)
                for seed, target in enumerate(([0.2, 0.8], [0.9, 0.1], [0.5, 0.5], [0.4, 0.6]))]
        result = summary(rows)["noul"]
        self.assertEqual((result["auroc_rows"], result["auroc_excluded_ties"]), (3, 1))
        ranked = [row for row in rows if not row["tie"]]
        self.assertEqual(result["auroc"], auroc([row["p_true"] for row in ranked], [row["positive"] for row in ranked]))

    def test_ece_bins_the_top_confidence_into_fifteen_equal_bins(self):
        rows = [{"confidence": 1.0, "teacher_probability": 0.9, "hit": 1},
                {"confidence": 0.95, "teacher_probability": 0.7, "hit": 0},
                {"confidence": 0.5, "teacher_probability": 0.5, "hit": 1},
                {"confidence": 0.52, "teacher_probability": 0.2, "hit": 0}]
        result = ece(rows)
        self.assertAlmostEqual(result["soft"], 0.5 * 0.175 + 0.5 * 0.16, places=12)
        self.assertAlmostEqual(result["hard"], 0.5 * 0.475 + 0.5 * 0.01, places=12)
        self.assertEqual(ece([{"confidence": 1.0, "teacher_probability": 1.0, "hit": 1}]), {"soft": 0.0, "hard": 0.0})


class BootstrapTests(unittest.TestCase):
    ROWS = [{"kind": "noul", "state": "a"}, {"kind": "noul", "state": "a"}, {"kind": "choice", "state": "a"},
            {"kind": "noul", "state": "b"}]

    def test_units_join_rows_that_share_a_state_within_a_kind(self):
        groups = units(self.ROWS)
        self.assertEqual(groups, {"noul": [[0, 1], [3]], "choice": [[2]], "score": []})
        self.assertEqual(resampled(groups["noul"], [0, 0, 1]), [0, 1, 0, 1, 3])

    def test_draws_repeat_with_the_seed(self):
        groups = units(self.ROWS)
        first = resamples(groups, replicates=50, seed=7)
        self.assertEqual(first, resamples(groups, replicates=50, seed=7))
        self.assertNotEqual(first, resamples(groups, replicates=50, seed=8))
        self.assertEqual([len(draw["noul"]) for draw in first], [2] * 50)
        self.assertEqual([draw["score"] for draw in first], [[]] * 50)

    def test_percentiles_interpolate_linearly(self):
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(percentile(values, 0.5), 2.5)
        self.assertAlmostEqual(percentile(values, 0.025), 1.075, places=12)
        self.assertAlmostEqual(percentile(values, 0.975), 3.925, places=12)
        self.assertEqual(percentile([5.0], 0.975), 5.0)
        self.assertIsNone(percentile([], 0.5))
        self.assertEqual(bounds([None, 3.0, 1.0, 2.0])["invalid"], 1)
        self.assertEqual(bounds([None, 3.0, 1.0, 2.0])["valid"], 3)

    def paired_values(self, offsets):
        rows, values = [], {"3": [], "1m": []}
        for index in range(40):
            kind = ("noul", "choice", "score")[index % 3]
            rows.append({"kind": kind, "state": f"state {index // 2}"})
            field = {"noul": "brier_soft", "choice": "accuracy", "score": "mae_normalized"}[kind]
            first = {"kind": kind, field: (index % 7) / 10}
            values["1m"].append(first)
            values["3"].append({"kind": kind, field: first[field] + offsets[kind](index)})
        return rows, values

    def test_the_interval_of_a_difference_uses_the_same_rows_for_both_stages(self):
        rows, values = self.paired_values({"noul": lambda index: 0.003, "choice": lambda index: -1.0,
                                           "score": lambda index: 0.0})
        groups = units(rows)
        result = relative(values, groups, resamples(groups, replicates=200))
        self.assertAlmostEqual(result["noul_brier"]["interval"]["low"], 0.003, places=12)
        self.assertAlmostEqual(result["noul_brier"]["interval"]["high"], 0.003, places=12)
        self.assertAlmostEqual(result["choice_accuracy"]["difference"], -1.0, places=12)
        self.assertTrue(all(item["passed"] for item in result.values()))

    def test_the_accuracy_limit_is_inclusive_and_exact(self):
        others = [{"kind": "noul", "brier_soft": 0.1}, {"kind": "score", "mae_normalized": 0.1}]
        for misses, passed in ((1, True), (2, False)):
            rows = [{"kind": "choice", "state": f"s{index}"} for index in range(50)] + [
                {"kind": "noul", "state": "n"}, {"kind": "score", "state": "s"}]
            values = {"1m": [{"kind": "choice", "accuracy": 100} for _ in range(50)] + others,
                      "3": [{"kind": "choice", "accuracy": 0 if index < misses else 100} for index in range(50)] + others}
            groups = units(rows)
            item = relative(values, groups, resamples(groups, replicates=20))["choice_accuracy"]
            self.assertEqual(item["difference"], -2.0 * misses)
            self.assertEqual(item["passed"], passed)


class GainTests(unittest.TestCase):
    def summaries(self, hits, brier, mae, kl):
        return {stage: {"choice": {"hits": hits[stage], "rows": 300}, "noul": {"brier_soft": brier[stage]},
                        "score": {"mae_normalized": mae[stage]}, "all": {"kl": kl[stage]}}
                for stage in ("0", "0q", "1m", "2", "3")}

    def test_gains_follow_the_direction_of_each_metric(self):
        result = gains(self.summaries({"0": 200, "1m": 230, "0q": 195, "2": 222, "3": 220},
                                      {"0": 0.05, "1m": 0.02, "0q": 0.06, "2": 0.04, "3": 0.045},
                                      {"0": 0.10, "1m": 0.05, "0q": 0.10, "2": 0.09, "3": 0.095},
                                      {"0": 0.30, "1m": 0.10, "0q": 0.32, "2": 0.15, "3": 0.16}))
        accuracy = result["choice_accuracy"]
        self.assertAlmostEqual(accuracy["gain_bf16"], 10.0, places=12)
        self.assertAlmostEqual(accuracy["gain_4bit"], 25 / 3, places=12)
        self.assertEqual(accuracy["status"], "pass")
        self.assertAlmostEqual(result["noul_brier"]["gain_bf16"], 0.03, places=12)
        self.assertAlmostEqual(result["noul_brier"]["gain_4bit"], 0.015, places=12)
        self.assertEqual(result["noul_brier"]["status"], "fail")
        self.assertEqual(result["score_mae_normalized"]["status"], "fail")
        self.assertAlmostEqual(result["kl"]["gain_4bit"], 0.16, places=12)
        self.assertEqual(result["kl"]["status"], "pass")
        self.assertAlmostEqual(result["kl"]["gain_pytorch_4bit"], 0.17, places=12)
        self.assertAlmostEqual(result["kl"]["stage3_against_stage2"], -0.01, places=12)

    def test_the_share_boundary_is_exact_for_accuracy(self):
        hits = {"0": 150, "1m": 155, "0q": 151, "2": 153, "3": 155}
        result = gain_condition({stage: Fraction(100 * count, 300) for stage, count in hits.items()}, True)
        self.assertEqual((result["judged"], result["status"]), (0.0, "pass"))
        floats = {stage: 100 * count / 300 for stage, count in hits.items()}
        self.assertLess(floats["3"] - floats["0q"] - 0.8 * (floats["1m"] - floats["0"]), 0)
        summaries = {stage: {"choice": {"hits": count, "rows": 300}, "noul": {"brier_soft": 0.1},
                             "score": {"mae_normalized": 0.1}, "all": {"kl": 0.1}} for stage, count in hits.items()}
        self.assertEqual(gains(summaries)["choice_accuracy"]["status"], "pass")

    def test_an_undefined_gain_is_not_a_pass(self):
        for bf16 in (0, -1):
            result = gain_condition({"0": 10.0, "1m": 10.0 + bf16, "0q": 5.0, "2": 8.0, "3": 9.0}, True)
            self.assertEqual(result["status"], "undefined")
            self.assertIsNone(result["share"])
            self.assertEqual(result["gain_4bit"], 4.0)


class ScoreTests(unittest.TestCase):
    def test_consistent_inputs_are_recorded_with_the_conditions(self):
        result = score(*documents(specs()))
        self.assertTrue(result["recorded"])
        self.assertEqual(result["targets"]["rows"], 9)
        self.assertEqual(set(result["stages"]["1"]), {"0", "0q", "1m", "2", "3"})
        self.assertEqual(result["bootstrap"]["units"], {"noul": 3, "choice": 3, "score": 3})
        rows = specs()
        expected = math.fsum(
            (softmax(row["logits"]["3"][0:2])[1] - row["target"][1]) ** 2
            - (softmax(row["logits"]["1m"][0:2])[1] - row["target"][1]) ** 2 for row in rows if row["kind"] == "noul") / 3
        self.assertAlmostEqual(result["conditions"]["D1-1"]["noul_brier"]["difference"], expected, places=14)
        self.assertEqual(result["passed"], all(result["conditions_passed"].values()))
        card = result["stages"]["card"]["3"]["noul"]
        self.assertNotEqual(card["brier_soft"], result["stages"]["1"]["3"]["noul"]["brier_soft"])

    def test_no_adaptation_leaves_every_gain_undefined_and_d1_2_failed(self):
        result = score(*documents(specs(adapted=0.0)))
        self.assertTrue(result["recorded"])
        self.assertEqual({item["status"] for item in result["conditions"]["D1-2"].values()}, {"undefined"})
        self.assertFalse(result["conditions_passed"]["D1-2"])
        self.assertFalse(result["passed"])

    def test_inconsistent_inputs_are_not_recorded(self):
        def target(documents):
            documents[5][0]["target"] = [0.3, 0.69]

        def token(documents):
            documents[3]["rows"][1]["tokens_sha256"] = "0" * 64

        def nonfinite(documents):
            documents[3]["rows"][2]["logits"][3] = math.nan

        def reference(documents):
            documents[1]["passed"] = False

        def order(documents):
            documents[2]["rows"].reverse()

        for change, name in ((target, "targets"), (token, "3"), (nonfinite, "nonfinite"), (reference, "references"),
                             (order, "1m")):
            inputs = list(documents(specs()))
            change(inputs)
            result = score(*inputs)
            self.assertFalse(result["recorded"], name)
            self.assertTrue(result["problems"][name], name)


if __name__ == "__main__":
    unittest.main()
