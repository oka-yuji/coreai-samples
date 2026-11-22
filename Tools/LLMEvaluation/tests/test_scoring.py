from collections import Counter
from fractions import Fraction
import json
import os
from pathlib import Path
import platform
import tempfile
import unittest
from unittest import mock

from Tools.LLMEvaluation import execution, items, scoring

TABLE = items.read_json(Path(__file__).resolve().parent / "scoring_cases.json")
CASES = {case["id"]: case for case in TABLE["cases"]}
BUNDLE = items.load()
CONTRACT = BUNDLE["contract"]
CATEGORIES = {category["id"]: category for category in CONTRACT["categories"]}
FIELDS = {"id", "kind", "output", "answer", "rules", "parsed", "correct", "f1", "tp", "fp", "fn"}
STOP = 106
VOCABULARY = 200


def table_item(case):
    return {"scorer": {"kind": case["kind"], **case.get("rules", {})}, "answer": case.get("answer")}


def committed_items(category, count):
    entries = [entry for entry in BUNDLE["sets"]["primary"]["items"] if entry["category"] == category][:count]
    return [items.item(CATEGORIES[category], entry, None, BUNDLE) for entry in entries]


def bundle_with(primary, reserve=()):
    entries = {entry["id"]: entry for name in ("primary", "reserve") for entry in BUNDLE["sets"][name]["items"]}
    return {**BUNDLE, "sets": {"primary": {"items": [entries[key] for key in primary]},
                               "reserve": {"items": [entries[key] for key in reserve]}}}


def frozen_as(listed):
    bundle = bundle_with([item["id"] for item in listed])
    entries = [{**entry, "item_sha256": items.digest(item)} for entry, item in zip(bundle["sets"]["primary"]["items"], listed)]
    return {**bundle, "sets": {"primary": {"items": entries}, "reserve": {"items": []}}}


def record(item_id, ids, stop=True):
    generated = ids + ([STOP] if stop else [])
    value = {"id": item_id, "input_sha256": "0" * 64, "input_tokens": 1, "output_ids": ids,
             "stop": "stop_token" if stop else "limit",
             "top2": [[token, 0.75, 0 if token else 1, 0.25] for token in generated],
             "non_finite": 0}
    if stop:
        value["stop_token"] = STOP
    return value


class Pieces:
    def __init__(self, pieces, special=()):
        self.pieces, self.special = pieces, set(special)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces[token] for token in ids if not (skip_special_tokens and token in self.special))

    def get_vocab_size(self, with_added_tokens=True):
        return VOCABULARY


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

    def test_an_object_with_a_constant_that_is_not_json_is_skipped(self):
        item = {"scorer": {"kind": "json_exact"}, "answer": {"x": 1}}
        for name in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(name):
                found = scoring.score(item, '{"x": ' + name + '} {"x": 1}')
                self.assertEqual((found["state"], found["correct"]), ("scored", True))
        found = scoring.score(item, '{"x": NaN} {"x": [Infinity, -Infinity]}')
        self.assertEqual((found["state"], found["correct"], found["parsed"]), ("no_json", False, None))

    def test_a_number_that_overflows_is_read_and_compared_as_a_number(self):
        for answer in ({"x": 1}, {"x": "Infinity"}):
            with self.subTest(answer):
                found = scoring.score({"scorer": {"kind": "json_exact"}, "answer": answer}, '{"x": 1e400}')
                self.assertEqual((found["state"], found["correct"], found["parsed"]),
                                 ("scored", False, {"x": float("inf")}))

    def test_nesting_deeper_than_the_decoder_limit_does_not_stop_the_search(self):
        answer = CASES["JE-17"]["answer"]
        text = '{"a": ' + "[" * 5000 + "\n" + json.dumps(answer, ensure_ascii=False)
        self.assertIs(scoring.score({"scorer": {"kind": "json_exact"}, "answer": answer}, text)["correct"], True)


class RulesOutsideTheTableTests(unittest.TestCase):
    def test_reading_strips_before_and_after_the_final_mark(self):
        rule = CONTRACT["scorers"]["reading_exact"]
        opening, closing = rule["enclosing"][0]
        word = CASES["RE-01"]["output"]
        for output in (word + rule["trailing"] + "\n", opening + word + closing + " " + rule["trailing"]):
            with self.subTest(output):
                self.assertIs(scoring.score({"scorer": {"kind": "reading_exact"}, "answer": [word]}, output)["correct"],
                              True)

    def test_an_entity_value_of_another_type_is_incorrect_even_without_names(self):
        keys = CONTRACT["entity_keys"]
        answer = {key: [] for key in keys}
        for value in (None, ""):
            with self.subTest(value):
                found = scoring.score({"scorer": {"kind": "entity_sets"}, "answer": answer},
                                      json.dumps({**answer, keys[5]: value}))
                self.assertEqual((found["correct"], found["tp"], found["fp"], found["fn"]), (False, 0, 0, 0))

    def test_the_text_after_the_bullet_is_stripped_before_it_is_counted(self):
        case = CASES["FR-09"]
        bullet = case["rules"]["bullet"]
        output = case["output"].replace(bullet, bullet + " ")
        self.assertIs(scoring.score(table_item(case), output)["correct"], True)


class RecordCheckTests(unittest.TestCase):
    def setUp(self):
        self.items = (committed_items("free-ja", 2) + committed_items("synth-format", 2)
                      + committed_items("synth-extract", 1) + committed_items("synth-long-4k", 1))
        self.records = [record(item["id"], [7, 8]) for item in self.items]
        self.hashes = scoring.committed(BUNDLE)

    def checked(self, items_list=None, records=None):
        numbered = list(enumerate(self.records if records is None else records, 1))
        listed = self.items if items_list is None else items_list
        return scoring.record_checks(listed, *scoring.partition(numbered, VOCABULARY),
                                     scoring.differing(listed, self.hashes))

    def only(self, name, ids):
        names = ("missing", "not_in_set", "duplicate", "top2_differs", "item_hash_differs", "malformed")
        return {**{key: [] for key in names}, name: ids}

    def test_clean_records_report_nothing(self):
        self.assertEqual(self.checked(), self.only("missing", []))
        limited = [record(item["id"], [7], stop=False) for item in self.items]
        self.assertEqual(self.checked(records=limited), self.only("missing", []))
        failed = [{**row, "error": "stopped"} for row in self.records]
        self.assertEqual(self.checked(records=failed), self.only("missing", []))
        for name, (first, second) in {"integers": (1, 0), "floats": (1.0, 0.0), "a tie": (0.5, 0.5)}.items():
            with self.subTest(name):
                edges = [{**row, "top2": [[entry[0], first, entry[2], second] for entry in row["top2"]]}
                         for row in self.records]
                self.assertEqual(self.checked(records=edges), self.only("missing", []))

    def test_each_malformed_record_is_reported_only_as_malformed(self):
        first = self.items[0]["id"]
        good = record(first, [7, 8])
        cases = {f"no {name}": {key: value for key, value in good.items() if key != name}
                 for name in ("input_sha256", "input_tokens", "output_ids", "stop", "stop_token", "top2", "non_finite")}
        cases.update({
            "input_sha256 of another type": {**good, "input_sha256": 0},
            "input_tokens of another type": {**good, "input_tokens": "1"},
            "input_tokens a boolean": {**good, "input_tokens": True},
            "non_finite of another type": {**good, "non_finite": None},
            "output_ids of another type": {**good, "output_ids": "7"},
            "output_ids with a string": {**good, "output_ids": [7, "8"]},
            "output_ids with a float": {**good, "output_ids": [7, 8.0]},
            "output_ids with a negative id": {**good, "output_ids": [7, -1]},
            "output_ids past the vocabulary": {**good, "output_ids": [7, VOCABULARY]},
            "stop_token of another type": {**good, "stop_token": "106"},
            "stop_token past the vocabulary": {**good, "stop_token": VOCABULARY},
            "stop_token negative": {**good, "stop_token": -1},
            "another stop value": {**good, "stop": "eos"},
            "top2 of another type": {**good, "top2": {}},
            "top2 entry of another type": {**good, "top2": [7, 8, STOP]},
            "top2 entry too short": {**good, "top2": [entry[:3] for entry in good["top2"]]},
            "top2 entry with a string id": {**good, "top2": [["7", 0.75, 0, 0.25]] + good["top2"][1:]},
            "top2 entry with a string probability": {**good, "top2": [[7, "0.75", 0, 0.25]] + good["top2"][1:]},
            "top2 probability infinite": {**good, "top2": [[7, float("inf"), 0, 0.25]] + good["top2"][1:]},
            "top2 probability not a number": {**good, "top2": [[7, 0.75, 0, float("nan")]] + good["top2"][1:]},
            "top2 first id past the vocabulary": {**good, "top2": [[VOCABULARY, 0.75, 0, 0.25]] + good["top2"][1:]},
            "top2 second id past the vocabulary": {**good, "top2": [[7, 0.75, VOCABULARY, 0.25]] + good["top2"][1:]},
            "top2 second id negative": {**good, "top2": [[7, 0.75, -1, 0.25]] + good["top2"][1:]},
            "top2 ids equal": {**good, "top2": [[7, 0.75, 7, 0.25]] + good["top2"][1:]},
            "top2 probability above one": {**good, "top2": [[7, 1.5, 0, 0.25]] + good["top2"][1:]},
            "top2 probability negative": {**good, "top2": [[7, 0.75, 0, -0.25]] + good["top2"][1:]},
            "top2 probability too large for a float": {**good, "top2": [[7, 10 ** 400, 0, 0.25]] + good["top2"][1:]},
            "top2 probability a boolean": {**good, "top2": [[7, True, 0, 0.25]] + good["top2"][1:]},
            "top2 first probability below the second": {**good, "top2": [[7, 0.25, 0, 0.75]] + good["top2"][1:]},
            "input_tokens zero": {**good, "input_tokens": 0},
            "input_tokens negative": {**good, "input_tokens": -1},
            "non_finite negative": {**good, "non_finite": -1},
            "error of another type": {**good, "error": None},
        })
        for name, broken in cases.items():
            with self.subTest(name):
                self.assertEqual(self.checked(records=[broken] + self.records[1:]), self.only("malformed", [first]))

    def test_malformed_records_with_a_string_id_count_for_duplicates_and_the_set(self):
        first = self.items[0]["id"]
        good = self.records[0]
        broken = {key: value for key, value in good.items() if key != "top2"}
        for name, rows in {"well-formed first": [good, broken], "malformed first": [broken, good],
                           "two malformed": [broken, broken]}.items():
            with self.subTest(name):
                self.assertEqual(self.checked(records=rows + self.records[1:]),
                                 {**self.only("duplicate", [first]), "malformed": [first] * rows.count(broken)})
        outside = {**broken, "id": "ja-choice-9999"}
        self.assertEqual(self.checked(records=self.records + [outside]),
                         {**self.only("not_in_set", ["ja-choice-9999"]), "malformed": ["ja-choice-9999"]})

    def test_a_record_without_a_readable_id_is_listed_by_line_and_its_item_is_missing(self):
        first = self.items[0]["id"]
        good = record(first, [7, 8])
        cases = {"no id": {key: value for key, value in good.items() if key != "id"},
                 "id of another type": {**good, "id": 1}, "not an object": [good], "not JSON": None}
        for name, broken in cases.items():
            with self.subTest(name):
                self.assertEqual(self.checked(records=self.records[1:] + [broken]),
                                 {**self.only("malformed", [len(self.records)]), "missing": [first]})

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
        self.bundle = bundle_with([item["id"] for item in self.items])

    def tearDown(self):
        self.folder.cleanup()

    def run_folder(self, name, rows, run=None):
        folder = self.root / name
        folder.mkdir()
        write_lines(folder / "outputs.jsonl", rows)
        (folder / "run.json").write_text(json.dumps(run or {"role": name}), encoding="utf-8")
        return folder

    def scored(self, runs, output="scores"):
        result = scoring.score_runs(self.bundle, self.items_path, runs, self.root / output, self.tokenizer, "f" * 64)
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
                          "state": "item_hash_differs", "parsed": None})
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
        self.assertEqual(summary["items"], {"file": "primary.jsonl", "sha256": items.sha256(self.items_path),
                                            "set": "primary"})
        self.assertEqual(summary["tokenizer"], {"sha256": "f" * 64, "vocabulary": VOCABULARY})
        self.assertEqual(summary["run"], {"role": "reference", "set": "primary"})
        self.assertEqual(summary["outputs"], {"sha256": items.sha256(run / "outputs.jsonl")})
        self.assertEqual(summary["checks"], {"missing": ["synth-long-4k-0001"], "not_in_set": ["ja-choice-9999"],
                                             "duplicate": ["ja-reading-0001"], "top2_differs": [],
                                             "item_hash_differs": sorted(item["id"] for item in self.items[1:]),
                                             "malformed": []})
        categories = summary["categories"]
        self.assertEqual(categories["ja-choice"], {"kind": "choice_digit", "items": 1, "correct": 1, "unscored": 0,
                                                   "states": {"scored": 1}})
        self.assertEqual(categories["code-fix"], {"kind": "python_tests", "items": 1, "correct": 0, "unscored": 1,
                                                  "states": {"item_hash_differs": 1}})
        self.assertEqual(categories["ja-reading"]["mean_f1"], 0.9)
        self.assertEqual(categories["ja-reading"]["correct"], 1)
        self.assertEqual({key: categories["extract-ja"][key] for key in ("correct", "tp", "fp", "fn", "micro_f1")},
                         {"correct": 1, "tp": 2, "fp": 1, "fn": 1, "micro_f1": 2 / 3})
        self.assertEqual(categories["synth-long-4k"], {"kind": "final_integer", "items": 1, "correct": 0,
                                                       "unscored": 1, "states": {"missing": 1}})
        self.assertEqual(result["baseline"]["checks"]["not_in_set"], 1)

    def test_broken_records_are_reported_and_the_other_items_are_scored(self):
        rows = [record(key, ids) for key, ids in self.outputs.items()]
        rows[1] = {**rows[1], "output_ids": [0, VOCABULARY]}
        rows[3] = {key: value for key, value in rows[3].items() if key != "id"}
        rows.insert(4, {**record("ja-reading-0002", [2]), "top2": None})
        folder = self.root / "broken"
        folder.mkdir()
        lines = [json.dumps(row).encode("utf-8") for row in rows] + [b'{"id": ', b"\xff\xfe", b"[]"]
        (folder / "outputs.jsonl").write_bytes(b"\n".join(lines) + b"\n")
        (folder / "run.json").write_text("{}", encoding="utf-8")
        _, output = self.scored([folder])
        scores = {row["id"]: row for row in read_lines(output / "broken" / "scores.jsonl")}
        checks = items.read_json(output / "broken" / "summary.json")["checks"]
        self.assertEqual(checks["malformed"], ["ja-choice-0001", 4, "ja-reading-0002", 9, 10, 11])
        self.assertEqual(checks["missing"], ["ja-reading-0001", "synth-long-4k-0001"])
        self.assertEqual((checks["duplicate"], checks["not_in_set"], checks["top2_differs"]),
                         (["ja-reading-0002"], [], []))
        self.assertEqual(scores["ja-choice-0001"], {"id": "ja-choice-0001", "category": "ja-choice",
                                                    "kind": "choice_digit", "correct": None, "state": "malformed",
                                                    "parsed": None})
        self.assertEqual(scores["ja-reading-0001"]["state"], "missing")
        self.assertEqual((scores["ja-reading-0002"]["state"], scores["ja-reading-0002"]["f1"]), ("scored", 0.8))
        self.assertIs(scores["extract-ja-0002"]["correct"], True)

    def test_scores_are_standard_json_when_a_number_overflows(self):
        item = {"id": "synth-extract-0001", "category": "synth-extract", "scorer": {"kind": "json_exact"},
                "answer": {"x": 1}}
        write_lines(self.items_path, [item])
        self.bundle = bundle_with([item["id"]])
        self.tokenizer = Pieces(['{"x": [1e400, -1e400]}'])
        _, output = self.scored([self.run_folder("overflow", [record(item["id"], [0])])])

        def strict(name):
            raise ValueError(name)

        text = (output / "overflow" / "scores.jsonl").read_text(encoding="utf-8")
        rows = [json.loads(line, parse_constant=strict) for line in text.splitlines()]
        self.assertEqual([(row["state"], row["correct"], row["parsed"]) for row in rows],
                         [("scored", False, {"x": ["Infinity", "-Infinity"]})])

    def test_a_run_folder_given_as_a_relative_dot_path_is_named_after_the_folder(self):
        run = self.run_folder("current", [record(key, ids) for key, ids in self.outputs.items()])
        (run / "inner").mkdir()
        start = os.getcwd()
        try:
            for name, folder, relative in (("dot", run, "."), ("dot-dot", run / "inner", "..")):
                with self.subTest(name):
                    os.chdir(folder)
                    _, output = self.scored([Path(relative)], output=f"out-{name}")
                    self.assertEqual([path.name for path in output.iterdir()], ["current"])
        finally:
            os.chdir(start)
        with self.assertRaises(ValueError):
            self.scored([Path("/")], output="root")
        self.assertFalse((self.root / "root").exists())

    def test_items_that_are_not_exactly_one_frozen_set_are_rejected(self):
        run = self.run_folder("run", [record(key, values) for key, values in self.outputs.items()])
        ids = [item["id"] for item in self.items]
        cases = {"one missing": (bundle_with(ids + ["ja-choice-0002"]), "1 missing, 0 extra"),
                 "one extra": (bundle_with(ids[1:]), "0 missing, 1 extra"),
                 "two sets": (bundle_with(ids[1:], ids[:1]), "0 missing, 1 extra")}
        for name, (bundle, counts) in cases.items():
            with self.subTest(name):
                self.bundle = bundle
                with self.assertRaisesRegex(ValueError, f"not the primary set: {counts}"):
                    self.scored([run], output=f"out-{name}")
                self.assertFalse((self.root / f"out-{name}").exists())

    def test_the_summary_names_the_frozen_set_of_the_items(self):
        ids = [item["id"] for item in self.items]
        rows = [record(key, values) for key, values in self.outputs.items()]
        for name, bundle in (("primary", bundle_with(ids)), ("reserve", bundle_with([], ids))):
            with self.subTest(name):
                self.bundle = bundle
                _, output = self.scored([self.run_folder(f"run-{name}", rows)], output=f"out-{name}")
                self.assertEqual(items.read_json(output / f"run-{name}" / "summary.json")["items"]["set"], name)

    def test_run_folder_names_that_match_after_folding_are_rejected(self):
        rows = [record(key, values) for key, values in self.outputs.items()]
        for name, (left, right) in {"case": ("Baseline", "baseline"),
                                    "normalization": ("caf\u00e9", "cafe\u0301")}.items():
            with self.subTest(name):
                (self.root / f"left-{name}").mkdir()
                (self.root / f"right-{name}").mkdir()
                runs = [self.run_folder(f"left-{name}/{left}", rows), self.run_folder(f"right-{name}/{right}", rows)]
                with self.assertRaises(ValueError):
                    self.scored(runs, output=f"out-{name}")
                self.assertFalse((self.root / f"out-{name}").exists())

    def test_a_failure_while_making_the_run_folders_removes_the_output(self):
        rows = [record(key, values) for key, values in self.outputs.items()]
        runs = [self.run_folder("one", rows), self.run_folder("two", rows)]
        make = Path.mkdir

        def failing(path, *arguments, **options):
            if path.name == "two":
                raise OSError("cannot make the folder")
            return make(path, *arguments, **options)

        with mock.patch.object(Path, "mkdir", failing), self.assertRaises(OSError):
            self.scored(runs)
        self.assertFalse((self.root / "scores").exists())

    def test_records_with_probabilities_that_are_not_finite_are_malformed(self):
        text = json.dumps(record("ja-choice-0001", [0, 1]))

        def probability(value):
            return text.replace("0.75", value, 1)

        lines = [probability("NaN"), probability("Infinity"), probability("1e400"), text]
        lines += [json.dumps(record(key, values)) for key, values in self.outputs.items() if key != "ja-choice-0001"]
        folder = self.root / "finite"
        folder.mkdir()
        (folder / "outputs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (folder / "run.json").write_text("{}", encoding="utf-8")
        _, output = self.scored([folder])
        checks = items.read_json(output / "finite" / "summary.json")["checks"]
        scores = {row["id"]: row for row in read_lines(output / "finite" / "scores.jsonl")}
        self.assertEqual((checks["malformed"], checks["duplicate"]), ([1, 2, "ja-choice-0001"], ["ja-choice-0001"]))
        self.assertEqual((scores["ja-choice-0001"]["state"], scores["ja-choice-0001"]["correct"]), ("scored", True))

    def test_summaries_are_standard_ascii_json_with_unencodable_ids_and_constants(self):
        surrogate, other = chr(0xD800), chr(0xDFFF)
        rows = [record(key, values) for key, values in self.outputs.items()]
        rows += [{key: value for key, value in record(surrogate, [7]).items() if key != "top2"}, record(other, [7])]
        folder = self.root / "surrogate"
        folder.mkdir()
        (folder / "outputs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        (folder / "run.json").write_text('{"role": "candidate", "note": NaN}', encoding="utf-8")
        _, output = self.scored([folder])

        def strict(name):
            raise ValueError(name)

        summary = json.loads((output / "surrogate" / "summary.json").read_text(encoding="ascii"), parse_constant=strict)
        for line in (output / "surrogate" / "scores.jsonl").read_text(encoding="ascii").splitlines():
            json.loads(line, parse_constant=strict)
        self.assertEqual((summary["checks"]["malformed"], summary["checks"]["not_in_set"]),
                         ([surrogate], [surrogate, other]))
        self.assertEqual(summary["run"], {"role": "candidate", "note": "NaN"})

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


class CodeScoringTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        self.trace = self.root / "trace"
        self.items = [{"id": f"code-gen-mbpp-000{number}", "category": "code-gen-mbpp",
                       "scorer": {"kind": "python_tests", "setup": "", "test": "assert value == 1", "call": ""},
                       "answer": None} for number in (1, 2, 3)]
        self.items_path = self.root / "primary.jsonl"
        write_lines(self.items_path, self.items)
        self.bundle = frozen_as(self.items)
        self.tokenizer = Pieces(["```python\nvalue = 1\n```", "value = 2", "slow", "flaky",
                                 f"open({str(self.trace)!r}, 'w').close()\nvalue = 1"])

    def run_folder(self, name, outputs):
        folder = self.root / name
        folder.mkdir()
        write_lines(folder / "outputs.jsonl", [record(key, ids) for key, ids in outputs.items()])
        (folder / "run.json").write_text(json.dumps({"role": name}), encoding="utf-8")
        return folder

    def scored(self, runs, output="scores", **options):
        scoring.score_runs(self.bundle, self.items_path, runs, self.root / output, self.tokenizer, "f" * 64, **options)
        return {run.name: ({row["id"]: row for row in read_lines(self.root / output / run.name / "scores.jsonl")},
                           items.read_json(self.root / output / run.name / "summary.json")) for run in runs}

    def test_the_same_output_ids_of_an_item_run_once_and_share_the_result(self):
        names = [item["id"] for item in self.items]
        baseline = self.run_folder("baseline", dict(zip(names, ([0], [0], [1]))))
        candidate = self.run_folder("candidate", dict(zip(names, ([0], [1], [1]))))
        with mock.patch.object(execution, "run", wraps=execution.run) as run:
            found = self.scored([baseline, candidate])
        self.assertEqual(run.call_count, 4)
        before, after = found["baseline"][0], found["candidate"][0]
        self.assertEqual([before[name]["correct"] for name in names], [True, True, False])
        self.assertEqual([after[name]["correct"] for name in names], [True, False, False])
        self.assertEqual(before[names[0]], after[names[0]])
        self.assertEqual(before[names[2]], after[names[2]])

    def test_timeouts_and_passes_on_the_rerun_are_recorded(self):
        calls = Counter()

        def attempt(units):
            code = dict(units)["code"]
            calls[code] += 1
            return {"slow": "timeout", "flaky": "timeout" if calls[code] == 1 else "passed"}.get(code, "passed")

        names = [item["id"] for item in self.items]
        run = self.run_folder("run", dict(zip(names, ([2], [3], [0]))))
        with mock.patch.object(execution, "attempt", attempt):
            rows, summary = self.scored([run])["run"]
        self.assertEqual([{key: rows[name][key] for key in ("correct", "failure", "rerun")} for name in names],
                         [{"correct": False, "failure": "timeout", "rerun": True},
                          {"correct": True, "failure": None, "rerun": True},
                          {"correct": True, "failure": None, "rerun": False}])
        self.assertEqual(summary["code"], {"timeouts": 2, "rerun_passed": 1})
        self.assertEqual(summary["categories"]["code-gen-mbpp"]["correct"], 2)

    def test_code_items_that_differ_from_the_frozen_set_are_not_run(self):
        names = [item["id"] for item in self.items]
        rewritten = {**self.items[1], "scorer": {**self.items[1]["scorer"], "test": "assert value == 2"}}
        without_units = {**self.items[2], "scorer": {"kind": "python_tests"}}
        write_lines(self.items_path, [self.items[0], rewritten, without_units])
        run = self.run_folder("run", dict(zip(names, ([0], [1], [0]))))
        with mock.patch.object(execution, "run", wraps=execution.run) as executed:
            rows, summary = self.scored([run])["run"]
        self.assertEqual([dict(call.args[0])["test"] for call in executed.call_args_list], ["assert value == 1"])
        self.assertIs(rows[names[0]]["correct"], True)
        for name in names[1:]:
            self.assertEqual(rows[name], {"id": name, "category": "code-gen-mbpp", "kind": "python_tests", "correct": None,
                                          "state": "item_hash_differs", "parsed": None})
        self.assertEqual(summary["checks"]["item_hash_differs"], names[1:])
        self.assertEqual(summary["categories"]["code-gen-mbpp"]["states"], {"scored": 1, "item_hash_differs": 2})

    def test_a_differing_code_item_is_item_hash_differs_whatever_its_records(self):
        names = [item["id"] for item in self.items]
        rewritten = {**self.items[1], "scorer": {**self.items[1]["scorer"], "test": "assert value == 2"}}
        write_lines(self.items_path, [self.items[0], rewritten, self.items[2]])
        good = record(names[1], [0])
        broken = {key: value for key, value in good.items() if key != "top2"}
        damaged = {key: value for key, value in record(names[2], [0]).items() if key != "top2"}
        cases = {"a well-formed record": [good], "no record": [], "only a malformed record": [broken],
                 "a malformed and a well-formed record": [broken, good]}
        for number, (name, records) in enumerate(cases.items()):
            with self.subTest(name):
                folder = self.root / f"case-{number}"
                folder.mkdir()
                write_lines(folder / "outputs.jsonl", records + [damaged])
                (folder / "run.json").write_text("{}", encoding="utf-8")
                with mock.patch.object(execution, "run", wraps=execution.run) as executed:
                    rows, summary = self.scored([folder], output=f"scores-{number}")[folder.name]
                self.assertEqual(executed.call_count, 0)
                self.assertEqual(rows[names[1]], {"id": names[1], "category": "code-gen-mbpp", "kind": "python_tests",
                                                  "correct": None, "state": "item_hash_differs", "parsed": None})
                self.assertEqual((rows[names[0]]["state"], rows[names[2]]["state"]), ("missing", "malformed"))
                self.assertEqual(summary["categories"]["code-gen-mbpp"],
                                 {"kind": "python_tests", "items": 3, "correct": 0, "unscored": 3,
                                  "states": {"missing": 1, "item_hash_differs": 1, "malformed": 1}})
                self.assertEqual(summary["code"], {"timeouts": 0, "rerun_passed": 0})
                self.assertEqual(summary["checks"]["item_hash_differs"], [names[1]])
                self.assertEqual(summary["checks"]["missing"], [names[0]] + ([names[1]] if not records else []))
                self.assertEqual(summary["checks"]["malformed"], [names[1]] * records.count(broken) + [names[2]])

    def test_code_is_not_run_when_the_sandbox_cannot_be_used(self):
        run = self.run_folder("run", {self.items[0]["id"]: [4]})
        unusable = {"no sandbox-exec": mock.patch.object(execution, "SANDBOX", str(self.root / "missing")),
                    "a profile that does not compile": mock.patch.object(execution, "PROFILE", "(version 1) (no-rule)")}
        for number, (name, patch) in enumerate(unusable.items()):
            with self.subTest(name):
                with patch, self.assertRaises((OSError, RuntimeError)):
                    self.scored([run], output=f"scores-{number}")
                self.assertFalse(self.trace.exists())
                self.assertFalse((self.root / f"scores-{number}").exists())
        self.assertIs(self.scored([run])["run"][0][self.items[0]["id"]]["correct"], False)
        self.assertFalse(self.trace.exists())

    def test_the_summary_records_the_scorer_and_the_code_counts(self):
        run = self.run_folder("run", {self.items[0]["id"]: [0]})
        _, summary = self.scored([run], commit="c" * 40)["run"]
        self.assertEqual(summary["scorer"], {"commit": "c" * 40, "python": platform.python_version()})
        self.assertEqual(summary["code"], {"timeouts": 0, "rerun_passed": 0})
        text = {"id": "ja-choice-0001", "category": "ja-choice", "scorer": {"kind": "choice_digit"}, "answer": 1}
        write_lines(self.items_path, [text])
        self.bundle = bundle_with([text["id"]])
        _, summary = self.scored([self.run_folder("text", {text["id"]: [1]})], output="text-scores")["text"]
        self.assertEqual(summary["scorer"], {"commit": None, "python": platform.python_version()})
        self.assertEqual(summary["code"], {"timeouts": 0, "rerun_passed": 0})


if __name__ == "__main__":
    unittest.main()
