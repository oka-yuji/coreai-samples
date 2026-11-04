import copy
import json
from pathlib import Path
import tempfile
import unittest

from Tools.Conversion.adapters.jev27b.compare import compare, report, temperatures, tokens_sha256
from Tools.Conversion.adapters.jev27b.typed_head import decide

TEMPERATURES = {"noul": 1.25, "choice": 0.75, "score": 1.5}
OPTIONS = {"noul": ["false", "true"], "score": ["0", "1", "2", "3", "4", "5"]}


def logits(seed):
    return [((7 * row + 3 * seed) % 11 - 5) / 4 for row in range(24)]


def reference_row(identifier, kind, band, count=None, tokens=None, rejected=False, seed=0):
    options = OPTIONS.get(kind) or [f"option {index}" for index in range(count)]
    tokens = tokens or [seed, seed + 1, seed + 2]
    row = {"set": "F", "id": identifier, "kind": kind, "options": options, "band": band, "token_count": len(tokens),
           "tokens": tokens, "rejected": rejected}
    if not rejected:
        values = logits(seed)
        shipped, plain = decide(values, kind, len(options), TEMPERATURES[kind])
        row.update(logits2=values, p2=plain, p2_T=shipped, p1m=plain)
    return row


def swift_row(reference):
    if reference["rejected"]:
        return {"id": reference["id"], "kind": reference["kind"], "error": "tooManyTokens",
                "token_count": reference["token_count"]}
    shipped, plain = decide(reference["logits2"], reference["kind"], len(reference["options"]),
                            TEMPERATURES[reference["kind"]])
    return {"id": reference["id"], "kind": reference["kind"], "token_count": reference["token_count"],
            "tokens_sha256": tokens_sha256(reference["tokens"]), "capacity": 256, "logits": reference["logits2"],
            "probabilities": shipped, "probabilities_t1": plain, "top": 0, "elapsed_seconds": 0.5}


def documents():
    rows = [reference_row("a", "noul", "<=512", seed=1), reference_row("b", "choice", "<=512", 3, seed=2),
            reference_row("c", "score", "<=512", seed=3), reference_row("d", "choice", "513-2048", 16, seed=4),
            reference_row("e", "noul", None, tokens=list(range(8193)), rejected=True)]
    reference = {"temperatures": TEMPERATURES,
                 "rows": rows + [{**reference_row("q", "noul", "<=512", seed=5), "set": "Q"}]}
    swift = {"schema": "typed-decide/1", "rows": [swift_row(row) for row in rows]}
    return reference, swift


def shift(row, delta, index=0):
    values = list(row["probabilities_t1"])
    values[index] += delta
    values[index + 1] -= delta
    row["probabilities_t1"] = values


class CompareTests(unittest.TestCase):
    def test_identical_outputs_pass_and_bands_without_rows_are_unevaluated(self):
        reference, swift = documents()
        result = compare(reference, swift, TEMPERATURES)
        self.assertTrue(result["passed"])
        self.assertEqual(result["rows"], {"total": 5, "forwarded": 4, "rejected": 1})
        self.assertEqual(sorted(result["temperature_1"]["bands"]), ["513-2048", "<=512"])
        self.assertEqual(result["temperature_1"]["unevaluated"], ["2049-4096", "4097-8192"])
        self.assertEqual(result["temperature_1"]["max_probability_error"], 0.0)
        self.assertTrue(result["cli_shipping_consistent"])
        self.assertEqual(result["capacity"], {"256": 4})

    def test_a_band_fails_on_its_largest_error_and_on_its_mean(self):
        reference, swift = documents()
        large = copy.deepcopy(swift)
        shift(large["rows"][3], 0.031)
        result = compare(reference, large, TEMPERATURES)
        self.assertFalse(result["passed"])
        self.assertFalse(result["temperature_1"]["bands"]["513-2048"]["passed"])
        self.assertTrue(result["temperature_1"]["bands"]["<=512"]["passed"])
        mean = copy.deepcopy(swift)
        for row in mean["rows"][:3]:
            shift(row, 0.02)
        result = compare(reference, mean, TEMPERATURES)
        group = result["temperature_1"]["bands"]["<=512"]
        self.assertAlmostEqual(group["max"], 0.02)
        self.assertGreater(group["mean"], 0.01)
        self.assertFalse(group["passed"])
        self.assertFalse(result["passed"])

    def test_top_changes_are_reported_with_the_rule_and_do_not_decide_the_result(self):
        reference, swift = documents()
        row = reference["rows"][0]
        row["p2"] = [0.515, 0.485]
        swift["rows"][0]["probabilities_t1"] = [0.49, 0.51]
        row2 = reference["rows"][3]
        row2["p2"] = [0.5, 0.42] + [0.08 / 14] * 14
        swift["rows"][3]["probabilities_t1"] = [0.45, 0.47] + [0.08 / 14] * 14
        result = compare(reference, swift, TEMPERATURES)
        changes = {change["id"]: change for change in result["temperature_1"]["top_changes"]}
        self.assertEqual(sorted(changes), ["a", "d"])
        self.assertTrue(changes["a"]["within_rule"])
        self.assertAlmostEqual(changes["a"]["gap"], 0.03)
        self.assertFalse(changes["d"]["within_rule"])
        self.assertAlmostEqual(changes["d"]["reference_margin"], 0.08)
        self.assertAlmostEqual(changes["d"]["gap"], 0.08)
        self.assertEqual(result["temperature_1"]["top_changes_within_rule"], 1)
        self.assertTrue(result["temperature_1"]["bands"]["<=512"]["passed"])
        self.assertEqual(result["temperature_1"]["top1_changed"]["rows"], 2)

    def test_token_ids_lengths_ids_and_rejections_are_checked(self):
        reference, swift = documents()
        cases = {
            "token IDs differ": lambda rows: rows[1].update(tokens_sha256=tokens_sha256([9])),
            "wrong length": lambda rows: rows[2].update(probabilities=rows[2]["probabilities"][:-1]),
            "repeats ids": lambda rows: rows.insert(1, copy.deepcopy(rows[0])),
            "identity or order": lambda rows: rows.reverse(),
            "expected a tooManyTokens": lambda rows: rows[4].update(error="invalidOptions"),
            "unexpected error": lambda rows: rows[0].update(error="nonFinite"),
            "capacity": lambda rows: rows[0].pop("capacity"),
            "kind differs": lambda rows: rows[1].update(kind="score"),
        }
        for fragment, change in cases.items():
            broken = copy.deepcopy(swift)
            change(broken["rows"])
            result = compare(reference, broken, TEMPERATURES)
            self.assertFalse(result["passed"], fragment)
            self.assertFalse(result["checks"]["rows_complete"], fragment)
            self.assertTrue(any(fragment in problem for problem in result["problems"]), (fragment, result["problems"]))
        missing = copy.deepcopy(swift)
        missing["rows"][4]["token_count"] = 8192
        self.assertFalse(compare(reference, missing, TEMPERATURES)["checks"]["rows_complete"])
        schema = {**swift, "schema": "typed-decide/0"}
        self.assertFalse(compare(reference, schema, TEMPERATURES)["checks"]["rows_complete"])

    def test_values_outside_a_distribution_fail_the_finite_check(self):
        reference, swift = documents()
        broken = copy.deepcopy(swift)
        broken["rows"][0]["probabilities"] = [0.6, 0.6]
        result = compare(reference, broken, TEMPERATURES)
        self.assertEqual(result["nonfinite"], ["a"])
        self.assertFalse(result["passed"])

    def test_the_shipping_temperature_is_recomputed_from_logits_with_the_given_temperatures(self):
        reference, swift = documents()
        swift["rows"][0]["logits"] = [value + (0.5 if index == 0 else 0.0)
                                      for index, value in enumerate(swift["rows"][0]["logits"])]
        result = compare(reference, swift, TEMPERATURES)
        self.assertGreater(result["shipping_temperature"]["max_probability_error"], 0.0)
        self.assertFalse(result["cli_shipping_consistent"])
        other = compare(reference, swift, {"noul": 2.0, "choice": 2.0, "score": 2.0})
        self.assertEqual(other["shipping_temperature"]["temperatures"]["noul"], 2.0)
        self.assertNotEqual(other["shipping_temperature"]["max_probability_error"],
                            result["shipping_temperature"]["max_probability_error"])
        self.assertTrue(result["passed"])

    def test_the_report_marks_unevaluated_checks_and_keeps_the_band_order(self):
        reference, swift = documents()
        broken = copy.deepcopy(swift)
        broken["rows"].reverse()
        result = compare(reference, broken, TEMPERATURES)
        self.assertIsNone(result["checks"]["finite"])
        self.assertIn("finite unevaluated", report(result))
        swift["rows"][3]["capacity"] = 1024
        result = compare(reference, swift, TEMPERATURES)
        text = report(result)
        self.assertIn("| 2049-4096 | 0 | - | - | unevaluated |", text)
        shipping = text.split("## Shipping temperature")[1].split("## Stage 3")[0]
        self.assertNotIn("Passed", shipping)
        self.assertNotIn("passed", result["shipping_temperature"]["bands"]["<=512"])
        self.assertEqual(list(result["stage_1m_reference"]["by_band"]), ["<=512", "513-2048"])
        self.assertEqual(list(result["capacity"]), ["256", "1024"])

    def test_temperatures_come_from_the_reference_or_a_per_kind_file(self):
        reference, _ = documents()
        self.assertEqual(temperatures(None, reference), TEMPERATURES)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "temperatures.json"
            path.write_text(json.dumps({"per_kind": {"noul": 2.0, "choice": 1.0, "score": 0.5}}))
            self.assertEqual(temperatures(path, reference)["score"], 0.5)
            for values in ({"noul": 0, "choice": 1, "score": 1}, {"noul": 1, "choice": 1},
                           {"noul": True, "choice": 1, "score": 1}):
                path.write_text(json.dumps({"per_kind": values}))
                with self.assertRaises(ValueError):
                    temperatures(path, reference)


if __name__ == "__main__":
    unittest.main()
