import copy
import unittest

from Tools.Conversion.adapters.jev27b.decision import HASH_SCHEMA, compare_hashes
from Tools.Conversion.adapters.jev27b.verify_quantization import compare, complete, finite

SETTINGS = {"schema": HASH_SCHEMA, "model_id": "owner/merged", "revision": "1" * 40, "model_type": "qwen3_5", "device": "cpu",
            "versions": {"torch": "2.9.0"}, "recipe_sha256": "a" * 64, "compute_precision": "bfloat16",
            "compression": "qwen3_5_cfg", "context_length": 262144}


def record(*digests):
    return {**SETTINGS, "layout": [{"name": "w.parametrizations.weight.0", "block": [1, 32]}],
            "tensors": [{"name": f"t{index}", "kind": "buffer", "shape": [2], "dtype": "torch.uint8", "sha256": value}
                        for index, value in enumerate(digests)]}


def row(identifier, kind="noul", rejected=False, **stages):
    entry = {"set": "F", "id": identifier, "kind": kind, "band": None if rejected else "<=512", "row_sha256": "r",
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
        self.assertEqual(result["margin_within_2e"], {"rows": 1, "ids": ["near"]})
        self.assertEqual(set(plain["by_kind"]), {"noul"})


if __name__ == "__main__":
    unittest.main()
