from fractions import Fraction
import json
from pathlib import Path
import unittest

from Tools.LLMEvaluation import items, scoring

TABLE = items.read_json(Path(__file__).resolve().parent / "scoring_cases.json")
CASES = {case["id"]: case for case in TABLE["cases"]}
BUNDLE = items.load()
CONTRACT = BUNDLE["contract"]
FIELDS = {"id", "kind", "output", "answer", "rules", "parsed", "correct", "f1", "tp", "fp", "fn"}


def table_item(case):
    return {"scorer": {"kind": case["kind"], **case.get("rules", {})}, "answer": case.get("answer")}


class ScoringTableTests(unittest.TestCase):
    def test_the_table_belongs_to_this_contract(self):
        self.assertEqual(TABLE["contract"], CONTRACT["version"])
        self.assertEqual(TABLE["entity_keys"], CONTRACT["entity_keys"])
        self.assertEqual(len(TABLE["cases"]), 101)
        self.assertEqual(len(CASES), 101)
        self.assertEqual({case["kind"] for case in TABLE["cases"]}, set(CONTRACT["scorers"]) - {"python_tests"})

    def test_every_case_gives_the_result_in_the_table(self):
        for case in TABLE["cases"]:
            with self.subTest(case["id"]):
                self.assertLessEqual(set(case), FIELDS)
                found = scoring.score(table_item(case), case["output"])
                self.assertEqual(found["kind"], case["kind"])
                self.assertIs(found["correct"], case["correct"])
                if "parsed" in case:
                    self.assertEqual(found["parsed"], case["parsed"])
                if "f1" in case:
                    self.assertEqual(found["f1"], Fraction(*case["f1"]))
                for name in ("tp", "fp", "fn"):
                    if name in case:
                        self.assertEqual(found[name], case[name])

    def test_unreadable_outputs_record_the_reason(self):
        expected = {"CD-01": "scored", "CD-06": "no_digit", "CD-08": "no_digit", "RE-13": "scored",
                    "FI-01": "scored", "FI-07": "no_integer", "FI-08": "no_marker", "FI-14": "no_integer",
                    "JE-01": "scored", "JE-11": "duplicate_key", "JE-12": "no_json", "JE-13": "no_json",
                    "ES-08": "duplicate_key", "ES-09": "no_json", "FR-15": "scored", "UN-01": "unscored"}
        for name, state in expected.items():
            with self.subTest(name):
                self.assertEqual(scoring.score(table_item(CASES[name]), CASES[name]["output"])["state"], state)


class FirstObjectTests(unittest.TestCase):
    def test_a_duplicate_key_inside_an_object_that_does_not_decode_is_skipped(self):
        answer = CASES["JE-17"]["answer"]
        text = json.dumps({"wrap": answer}, ensure_ascii=False)[:-1] + ', "x": {"a": 1, "a": 2} ...'
        found = scoring.score({"scorer": {"kind": "json_exact"}, "answer": answer}, text)
        self.assertEqual((found["state"], found["correct"]), ("scored", True))
        nested = '{"rows": [{"a": 2, "a": 2}]}'
        found = scoring.score({"scorer": {"kind": "json_exact"}, "answer": {"rows": [{"a": 2}]}}, nested)
        self.assertEqual((found["state"], found["correct"]), ("duplicate_key", False))

    def test_nesting_deeper_than_the_decoder_limit_does_not_stop_the_search(self):
        answer = CASES["JE-17"]["answer"]
        text = '{"a": ' + "[" * 5000 + "\n" + json.dumps(answer, ensure_ascii=False)
        self.assertIs(scoring.score({"scorer": {"kind": "json_exact"}, "answer": answer}, text)["correct"], True)


if __name__ == "__main__":
    unittest.main()
