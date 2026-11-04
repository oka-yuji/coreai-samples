import math
import unittest

import numpy as np

from Tools.Conversion.adapters.jev27b.typed_head import band, decide, render, slot_range, tokenize

TEMPERATURES = {"noul": 1.0143134751376188, "choice": 1.0161280671450397, "score": 1.0036213515883572}
SPANS = {"noul": 2, "score": 6}


def reference(logits, start, end, temperature):
    selected = np.asarray(logits, dtype=np.float64)[start:end] / temperature
    weights = np.exp(selected - selected.max())
    return weights / weights.sum()


class TypedHeadRenderTests(unittest.TestCase):
    def test_choice_renders_the_card_template(self):
        text = render("choice", "SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.",
                      "Supplier response for this scenario.", ["issue_warning", "renegotiate", "dual_source", "maintain"])
        self.assertEqual(text, "[kind] choice\n[state] SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.\n"
                               "[question] Supplier response for this scenario.\n[options]\nA) issue_warning\nB) renegotiate\n"
                               "C) dual_source\nD) maintain\n[decision]:")

    def test_noul_and_score_render_their_fixed_option_lines(self):
        self.assertEqual(render("noul", "s", "q", ["false", "true"]),
                         "[kind] noul\n[state] s\n[question] q\n[options]\nfalse\ntrue\n[decision]:")
        self.assertEqual(render("score", "", "q", ("0", "1", "2", "3", "4", "5")),
                         "[kind] score\n[state] \n[question] q\n[options]\n0\n1\n2\n3\n4\n5\n[decision]:")

    def test_choice_accepts_two_and_sixteen_options_with_labels_a_to_p(self):
        self.assertTrue(render("choice", "s", "q", ["x", "y"]).endswith("\n[options]\nA) x\nB) y\n[decision]:"))
        lines = render("choice", "s\nmore", "q\nmore", [f"o{i}" for i in range(16)]).split("\n")
        self.assertEqual(lines[6:22], [f"{label}) o{i}" for i, label in enumerate("ABCDEFGHIJKLMNOP")])
        self.assertEqual(lines[-1], "[decision]:")

    def test_render_rejects_invalid_options_and_inputs(self):
        cases = [("choice", ["only"]), ("choice", [f"o{i}" for i in range(17)]), ("choice", ["a", "a"]),
                 ("choice", ["a\nb", "c"]), ("choice", ["a\rb", "c"]), ("choice", ["a b", "c"]), ("choice", ["a", 1]),
                 ("choice", "ab"), ("noul", ["true", "false"]), ("noul", None), ("score", ["0", "1", "2", "3", "4"]),
                 ("unknown", ["a", "b"])]
        for kind, options in cases:
            with self.subTest(kind=kind, options=options), self.assertRaises(ValueError):
                render(kind, "s", "q", options)
        with self.assertRaises(ValueError):
            render("noul", {"state": 1}, "q", ["false", "true"])


class TypedHeadDecisionTests(unittest.TestCase):
    def test_slot_ranges_follow_the_head_layout(self):
        self.assertEqual(slot_range("noul", 2), (0, 2))
        self.assertEqual(slot_range("score", 6), (2, 8))
        self.assertEqual(slot_range("choice", 2), (8, 10))
        self.assertEqual(slot_range("choice", 16), (8, 24))
        for kind, n in (("noul", 3), ("score", 5), ("choice", 1), ("choice", 17), ("choice", 4.0), ("other", 2)):
            with self.subTest(kind=kind, n=n), self.assertRaises(ValueError):
                slot_range(kind, n)

    def test_probabilities_match_a_numpy_reference_with_and_without_temperature(self):
        rng = np.random.default_rng(20261102)
        for kind, n in (("noul", 2), ("score", 6), ("choice", 2), ("choice", 4), ("choice", 16)):
            logits = rng.normal(scale=6.0, size=24)
            start, end = slot_range(kind, n)
            for temperature in (TEMPERATURES[kind], 1.0, 0.25):
                with self.subTest(kind=kind, n=n, temperature=temperature):
                    probs, probs_t1 = decide(logits.astype(np.float32), kind, n, temperature)
                    np.testing.assert_allclose(probs, reference(logits.astype(np.float32), start, end, temperature),
                                               rtol=0, atol=1e-6)
                    np.testing.assert_allclose(probs_t1, reference(logits.astype(np.float32), start, end, 1.0), rtol=0, atol=1e-6)
                    self.assertEqual(len(probs), n)
                    self.assertAlmostEqual(math.fsum(probs), 1.0, places=12)

    def test_slots_outside_the_kind_do_not_change_the_result(self):
        logits = np.zeros(24)
        logits[8:12] = [1.0, 2.0, 3.0, 4.0]
        changed = logits.copy()
        changed[[0, 1, 5, 12, 23]] = [50.0, -50.0, 9.0, 80.0, -3.0]
        self.assertEqual(decide(logits, "choice", 4, 1.0), decide(changed, "choice", 4, 1.0))

    def test_decide_rejects_nonfinite_logits_bad_temperatures_and_lengths(self):
        good = [0.0] * 24
        for values in ([math.nan] + [0.0] * 23, [0.0] * 23 + [math.inf], [0.0] * 23):
            with self.subTest(values=values[:2]), self.assertRaises(ValueError):
                decide(values, "noul", 2, 1.0)
        for temperature in (0.0, -1.0, math.nan, math.inf, True, "1"):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                decide(good, "noul", 2, temperature)
        for values, temperature in (([1.0] * 24, 5e-324), ([1e308, -1e308] + [0.0] * 22, 0.5)):
            with self.subTest(scaled=values[:2], temperature=temperature), self.assertRaises(ValueError):
                decide(values, "noul", 2, temperature)


class TypedHeadBandTests(unittest.TestCase):
    def test_bands_split_at_the_declared_token_limits(self):
        expected = {0: "<=512", 1: "<=512", 512: "<=512", 513: "513-2048", 2048: "513-2048", 2049: "2049-4096",
                    4096: "2049-4096", 4097: "4097-8192", 8192: "4097-8192", 8193: None, 262144: None}
        for count, label in expected.items():
            with self.subTest(count=count):
                self.assertEqual(band(count), label)
        for value in (-1, 1.5, True, "12"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                band(value)

    def test_tokenize_asks_for_ids_without_special_tokens(self):
        calls = []

        def tokenizer(text, **options):
            calls.append((text, options))
            return {"input_ids": (7, 8)}

        self.assertEqual(tokenize(tokenizer, "x"), [7, 8])
        self.assertEqual(calls, [("x", {"add_special_tokens": False})])


if __name__ == "__main__":
    unittest.main()
