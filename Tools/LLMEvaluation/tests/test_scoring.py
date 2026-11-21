from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

from Tools.LLMEvaluation import items, scoring

TABLE = items.read_json(Path(__file__).resolve().parent / "scoring_cases.json")
CASES = {case["id"]: case for case in TABLE["cases"]}
BUNDLE = items.load()
CONTRACT = BUNDLE["contract"]
CATEGORIES = {category["id"]: category for category in CONTRACT["categories"]}
FIELDS = {"id", "kind", "output", "answer", "rules", "parsed", "correct", "f1", "tp", "fp", "fn"}
STOP = 106


def table_item(case):
    return {"scorer": {"kind": case["kind"], **case.get("rules", {})}, "answer": case.get("answer")}


def committed_items(category, count):
    entries = [entry for entry in BUNDLE["sets"]["primary"]["items"] if entry["category"] == category][:count]
    return [items.item(CATEGORIES[category], entry, None, BUNDLE) for entry in entries]


def record(item_id, ids, stop=True):
    generated = ids + ([STOP] if stop else [])
    value = {"id": item_id, "input_sha256": "0" * 64, "input_tokens": 1, "output_ids": ids,
             "stop": "stop_token" if stop else "limit", "top2": [[token, 0.75, 0, 0.25] for token in generated],
             "non_finite": 0}
    if stop:
        value["stop_token"] = STOP
    return value


class Pieces:
    def __init__(self, pieces, special=()):
        self.pieces, self.special = pieces, set(special)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces[token] for token in ids if not (skip_special_tokens and token in self.special))


def write_lines(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


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


class RecordCheckTests(unittest.TestCase):
    def setUp(self):
        self.items = (committed_items("free-ja", 2) + committed_items("synth-format", 2)
                      + committed_items("synth-extract", 1) + committed_items("synth-long-4k", 1))
        self.records = [record(item["id"], [7, 8]) for item in self.items]
        self.hashes = scoring.committed(BUNDLE)

    def checked(self, items_list=None, records=None):
        return scoring.record_checks(self.items if items_list is None else items_list,
                                     self.records if records is None else records, self.hashes)

    def only(self, name, ids):
        return {**{key: [] for key in ("missing", "not_in_set", "duplicate", "top2_differs", "item_hash_differs")},
                name: ids}

    def test_clean_records_report_nothing(self):
        self.assertEqual(self.checked(), self.only("missing", []))
        limited = [record(item["id"], [7], stop=False) for item in self.items]
        self.assertEqual(self.checked(records=limited), self.only("missing", []))

    def test_each_problem_is_reported_in_its_own_field(self):
        first, second, third = (item["id"] for item in self.items[:3])
        self.assertEqual(self.checked(records=self.records[1:]), self.only("missing", [first]))
        self.assertEqual(self.checked(records=self.records + [record("ja-choice-9999", [7])]),
                         self.only("not_in_set", ["ja-choice-9999"]))
        self.assertEqual(self.checked(records=self.records + [record(second, [9])]), self.only("duplicate", [second]))
        wrong = record(third, [7, 8])
        wrong["top2"][1][0] = 9
        self.assertEqual(self.checked(records=[wrong if row["id"] == third else row for row in self.records]),
                         self.only("top2_differs", [third]))
        changed = json.loads(json.dumps(self.items))
        changed[1]["max_new_tokens"] += 1
        self.assertEqual(self.checked(items_list=changed), self.only("item_hash_differs", [second]))

    def test_top2_covers_every_generated_token_including_the_stop_token(self):
        first = self.items[0]["id"]
        cases = {"no stop entry": {**record(first, [7, 8]), "top2": record(first, [7, 8])["top2"][:-1]},
                 "stop entry after a limit": {**record(first, [7, 8], stop=False),
                                              "top2": record(first, [7, 8])["top2"]},
                 "another stop token": {**record(first, [7, 8]), "stop_token": 1}}
        for name, wrong in cases.items():
            with self.subTest(name):
                self.assertEqual(self.checked(records=[wrong] + self.records[1:]), self.only("top2_differs", [first]))


class ScoreRunTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        keys = CONTRACT["entity_keys"]
        empty = {key: [] for key in keys}
        self.synthetic = committed_items("free-ja", 1)
        self.items = self.synthetic + [
            {"id": "ja-choice-0001", "category": "ja-choice", "scorer": {"kind": "choice_digit"}, "answer": 2},
            {"id": "code-fix-0001", "category": "code-fix",
             "scorer": {"kind": "python_tests", "setup": "", "test": "assert False", "call": ""}, "answer": None},
            {"id": "ja-reading-0001", "category": "ja-reading", "scorer": {"kind": "reading_exact"}, "answer": ["AB"]},
            {"id": "ja-reading-0002", "category": "ja-reading", "scorer": {"kind": "reading_exact"}, "answer": ["AB"]},
            {"id": "extract-ja-0001", "category": "extract-ja", "scorer": {"kind": "entity_sets"},
             "answer": {**empty, keys[0]: ["A", "B"]}},
            {"id": "extract-ja-0002", "category": "extract-ja", "scorer": {"kind": "entity_sets"},
             "answer": {**empty, keys[1]: ["D"]}},
            {"id": "synth-long-4k-0001", "category": "synth-long-4k", "scorer": {"kind": "final_integer"},
             "answer": 5},
        ]
        self.tokenizer = Pieces(["4", "2", "AB", "C", json.dumps({**empty, keys[0]: ["A", "C"]}, ensure_ascii=False),
                                 json.dumps({**empty, keys[1]: ["D"]}, ensure_ascii=False), "x", "3"], special=[0])
        self.outputs = {"free-ja-0001": [6], "ja-choice-0001": [0, 1], "code-fix-0001": [6],
                        "ja-reading-0001": [2], "ja-reading-0002": [2, 3], "extract-ja-0001": [4],
                        "extract-ja-0002": [5]}
        self.items_path = self.root / "primary.jsonl"
        write_lines(self.items_path, self.items)

    def tearDown(self):
        self.folder.cleanup()

    def run_folder(self, name, rows, run=None):
        folder = self.root / name
        folder.mkdir()
        write_lines(folder / "outputs.jsonl", rows)
        (folder / "run.json").write_text(json.dumps(run or {"role": name}), encoding="utf-8")
        return folder

    def scored(self, runs, output="scores"):
        result = scoring.score_runs(BUNDLE, self.items_path, runs, self.root / output, self.tokenizer, "f" * 64)
        return result, self.root / output

    def test_the_scored_text_is_the_decoded_output_ids_without_special_tokens(self):
        rows = [{**record(key, ids), "text": "3"} for key, ids in self.outputs.items()]
        run = self.run_folder("candidate", rows)
        _, output = self.scored([run])
        scores = {row["id"]: row for row in read_lines(output / "candidate" / "scores.jsonl")}
        self.assertEqual([row["id"] for row in read_lines(output / "candidate" / "scores.jsonl")],
                         [item["id"] for item in self.items])
        self.assertEqual(scores["ja-choice-0001"]["parsed"], 2)
        self.assertIs(scores["ja-choice-0001"]["correct"], True)
        self.assertEqual(scores["code-fix-0001"],
                         {"id": "code-fix-0001", "category": "code-fix", "kind": "python_tests", "correct": None,
                          "state": "pending_code", "parsed": None})
        self.assertEqual((scores["free-ja-0001"]["state"], scores["free-ja-0001"]["correct"]), ("unscored", None))
        self.assertEqual((scores["synth-long-4k-0001"]["state"], scores["synth-long-4k-0001"]["correct"]),
                         ("missing", None))

    def test_the_summary_counts_categories_and_keeps_the_checks_and_the_run_record(self):
        rows = [record(key, ids) for key, ids in self.outputs.items()]
        rows += [record("ja-choice-9999", [1]), record("ja-reading-0001", [3])]
        run = self.run_folder("baseline", rows, {"role": "reference", "set": "primary"})
        result, output = self.scored([run])
        summary = items.read_json(output / "baseline" / "summary.json")
        self.assertEqual(summary["contract"], {"version": CONTRACT["version"],
                                               "sha256": BUNDLE["hashes"]["contract.json"]})
        self.assertEqual(summary["items"], {"file": "primary.jsonl", "sha256": items.sha256(self.items_path)})
        self.assertEqual(summary["tokenizer"], {"sha256": "f" * 64})
        self.assertEqual(summary["run"], {"role": "reference", "set": "primary"})
        self.assertEqual(summary["checks"], {"missing": ["synth-long-4k-0001"], "not_in_set": ["ja-choice-9999"],
                                             "duplicate": ["ja-reading-0001"], "top2_differs": [],
                                             "item_hash_differs": sorted(item["id"] for item in self.items[1:])})
        categories = summary["categories"]
        self.assertEqual(categories["ja-choice"], {"kind": "choice_digit", "items": 1, "correct": 1, "unscored": 0,
                                                   "states": {"scored": 1}})
        self.assertEqual(categories["code-fix"], {"kind": "python_tests", "items": 1, "correct": 0, "unscored": 1,
                                                  "states": {"pending_code": 1}})
        self.assertEqual(categories["ja-reading"]["mean_f1"], 0.9)
        self.assertEqual(categories["ja-reading"]["correct"], 1)
        self.assertEqual({key: categories["extract-ja"][key] for key in ("correct", "tp", "fp", "fn", "micro_f1")},
                         {"correct": 1, "tp": 2, "fp": 1, "fn": 1, "micro_f1": 2 / 3})
        self.assertEqual(categories["synth-long-4k"], {"kind": "final_integer", "items": 1, "correct": 0,
                                                       "unscored": 1, "states": {"missing": 1}})
        self.assertEqual(result["baseline"]["checks"]["not_in_set"], 1)

    def test_runs_are_scored_together_into_a_new_folder(self):
        first = self.run_folder("one", [record(key, ids) for key, ids in self.outputs.items()])
        second = self.run_folder("two", [record(key, []) for key in self.outputs])
        result, output = self.scored([first, second])
        self.assertEqual(sorted(path.name for path in output.iterdir()), ["one", "two"])
        self.assertEqual(result["one"]["categories"]["ja-choice"]["correct"], 1)
        self.assertEqual(result["two"]["categories"]["ja-choice"]["correct"], 0)
        with self.assertRaises(FileExistsError):
            self.scored([first])
        nested = self.root / "nested"
        nested.mkdir()
        twin = self.run_folder("nested/one", [])
        with self.assertRaises(ValueError):
            self.scored([first, twin], output="twins")
        self.assertFalse((self.root / "twins").exists())

    def test_a_tokenizer_that_differs_from_the_pin_is_rejected(self):
        path = self.root / "tokenizer.json"
        path.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            scoring.pinned(path, CONTRACT)


if __name__ == "__main__":
    unittest.main()
