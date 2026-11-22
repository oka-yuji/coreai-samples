import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from Tools.LLMEvaluation import comparison, items, scoring

TABLE = items.read_json(Path(__file__).resolve().parent / "comparison_cases.json")
BUNDLE = items.load()
CONTRACT = BUNDLE["contract"]
LAYER2 = CONTRACT["layer2"]
CATEGORIES = {category["id"]: category for category in CONTRACT["categories"]}
VOCABULARY = 300000
RUN = {"asset": {"main_hash": "a"}, "inputs": {"sha256": "i"}, "implementation": {"commit": "c"},
       "call_shape": {"prefill_chunk": 1}}


def step(node, part):
    if not isinstance(part, dict):
        return part
    found = [index for index, entry in enumerate(node) if entry.get("id") == part["id"]]
    if len(found) != 1:
        raise ValueError(f"{len(found)} elements have the id {part['id']}")
    return found[0]


def applied(document, operations):
    document = copy.deepcopy(document)
    for name, path, *value in operations:
        node = document
        for part in path[:-1]:
            node = node[step(node, part)]
        key = step(node, path[-1])
        if name == "set":
            node[key] = value[0]
        elif name == "delete":
            del node[key]
        elif name == "append":
            node[key].append(value[0])
        else:
            raise ValueError(f"Unknown operation: {name}")
    return document


def same(found, expected):
    if isinstance(expected, float):
        return type(found) is float and math.isclose(found, expected, rel_tol=0, abs_tol=1e-12)
    if isinstance(expected, dict):
        return (isinstance(found, dict) and found.keys() == expected.keys()
                and all(same(found[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return isinstance(found, list) and len(found) == len(expected) and all(map(same, found, expected))
    return type(found) is type(expected) and found == expected


def case_input(name):
    return applied(TABLE["base"]["input"], next(case for case in TABLE["cases"] if case["name"] == name)["input"])


def output(document, side, item_id):
    return next(entry for entry in document[side]["outputs"] if entry["id"] == item_id)


def score_row(document, side, item_id):
    return next(row for row in document[side]["scores"] if row["id"] == item_id)


class ComparisonTableTests(unittest.TestCase):
    def test_the_table_belongs_to_this_contract(self):
        self.assertEqual(TABLE["contract"], CONTRACT["version"])
        self.assertEqual(len(TABLE["cases"]), 57)
        self.assertEqual(len({case["name"] for case in TABLE["cases"]}), 57)
        self.assertEqual(list(comparison.REASONS), TABLE["reasons"])

    def test_every_case_gives_the_record_in_the_table(self):
        for case in TABLE["cases"]:
            with self.subTest(case["name"]):
                found = comparison.compare(applied(TABLE["base"]["input"], case["input"]), LAYER2)
                for key, value in applied(TABLE["base"]["expected"], case["expected"]).items():
                    self.assertIn(key, found)
                    self.assertTrue(same(found[key], value), f"{key}: {found[key]!r} != {value!r}")


class RulesOutsideTheTableTests(unittest.TestCase):
    def test_every_control_token_of_the_contract_is_a_critical_failure(self):
        for token in LAYER2["critical_failures"]["control_token_ids"]:
            with self.subTest(token):
                document = case_input("clean_pair")
                record = output(document, "candidate", "free-ja-0002")
                record["output_ids"][1] = token
                record["top2"][1][0] = token
                found = comparison.compare(document, LAYER2)
                self.assertEqual(found["critical_failures"], [{"id": "free-ja-0002", "kind": "control_token"}])

    def test_equal_output_ids_that_stopped_differently_and_scored_differently_are_inconsistent(self):
        document = case_input("clean_pair")
        record = output(document, "candidate", "ja-reading-0001")
        record.update({"stop": "limit", "top2": record["top2"][:-1]})
        del record["stop_token"]
        score_row(document, "candidate", "ja-reading-0001")["correct"] = False
        found = comparison.compare(document, LAYER2)
        self.assertEqual(found["verdict"], "incomplete")
        self.assertEqual(found["incomplete"], [{"reason": "score_inconsistent", "ids": ["ja-reading-0001"]}])
        self.assertEqual(found["regressions"], ["ja-reading-0001"])
        self.assertTrue(same(found["changed"], [{"id": "ja-reading-0001", "position": 3, "baseline_gap": 0.625}]))

    def test_the_record_keeps_the_contract_and_both_run_records(self):
        document = case_input("clean_pair")
        found = comparison.compare(document, LAYER2)
        self.assertEqual(found["contract"], document["contract"])
        self.assertEqual(found["runs"], {"baseline": document["baseline"]["run"],
                                         "candidate": document["candidate"]["run"]})


def built(category, count):
    entries = [entry for entry in BUNDLE["sets"]["primary"]["items"] if entry["category"] == category][:count]
    return [items.item(CATEGORIES[category], entry, None, BUNDLE) for entry in entries]


def write_lines(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class Pieces:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces[token] for token in ids)

    def get_vocab_size(self, with_added_tokens=True):
        return VOCABULARY


class CompareRunTests(unittest.TestCase):
    def setUp(self):
        self.items = built("free-ja", 1) + built("synth-long-4k", 1)
        self.bundle = {**BUNDLE, "sets": {"primary": {"items": [entry for entry in BUNDLE["sets"]["primary"]["items"]
                                                                if entry["id"] in {item["id"] for item in self.items}]},
                                          "reserve": {"items": []}}}
        marker = CONTRACT["scorers"]["final_integer"]["marker"]
        self.tokenizer = Pieces({7: "x", 8: f"{marker} {self.items[1]['answer']}", 9: f"{marker} -1"})
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        self.items_path = self.folder / "primary.jsonl"
        write_lines(self.items_path, self.items)

    def run_folder(self, name, role, answer=8, **fields):
        folder = self.folder / name
        folder.mkdir(parents=True)
        contract = {"version": CONTRACT["version"], "sha256": self.bundle["hashes"]["contract.json"]}
        (folder / "run.json").write_text(json.dumps({"contract": contract, "set": "primary", "role": role, **fields}),
                                         encoding="utf-8")
        records = [{"id": item["id"], "input_sha256": "0" * 64, "input_tokens": item.get("token_count", 1),
                    "output_ids": [7] if item["category"] == "free-ja" else [answer], "stop": "stop_token",
                    "stop_token": 106, "non_finite": 0} for item in self.items]
        for record in records:
            record["top2"] = [[token, 1.0, 0, 0.0] for token in record["output_ids"] + [106]]
        write_lines(folder / "outputs.jsonl", records)
        return folder

    def scored(self, *runs, name="scores"):
        scores = self.folder / name
        scoring.score_runs(self.bundle, self.items_path, runs, scores, self.tokenizer, "t" * 64)
        return scores

    def compared(self, baseline, candidate, scores, name="verdict", **options):
        return comparison.compare_runs(self.bundle, self.items_path, baseline, candidate, scores, self.folder / name,
                                       **options)

    def test_a_pair_scored_together_gives_the_verdict_and_the_summary(self):
        baseline, candidate = self.run_folder("base", "reference"), self.run_folder("cand", "fp16", answer=9)
        layer1 = self.folder / "layer1.json"
        layer1.write_text("{}", encoding="utf-8")
        shown = self.compared(baseline, candidate, self.scored(baseline, candidate), layer1=layer1)
        record = items.read_json(self.folder / "verdict" / "verdict.json")
        self.assertEqual(shown["verdict"], "regression_found")
        self.assertEqual(record["verdict"], "regression_found")
        self.assertEqual(record["regressions"], [self.items[1]["id"]])
        self.assertEqual(record["changed"], [{"id": self.items[1]["id"], "position": 0, "baseline_gap": 1.0}])
        self.assertEqual(record["layer1"], {"file": "layer1.json", "sha256": items.sha256(layer1)})
        self.assertEqual(record["runs"]["candidate"]["role"], "fp16")
        self.assertIsNone(record["substitute_reason"])
        self.assertIsNone(record["claim"])
        text = (self.folder / "verdict" / "summary.md").read_text(encoding="utf-8")
        self.assertIn("`regression_found`", text)
        self.assertIn(f"Regressions: `{self.items[1]['id']}`", text)

    def test_an_id_with_a_lone_surrogate_is_written_to_both_files(self):
        baseline, candidate = self.run_folder("base", "reference"), self.run_folder("cand", "fp16")
        path = candidate / "outputs.jsonl"
        stray = {**json.loads(path.read_text(encoding="utf-8").splitlines()[0]), "id": "free-ja-9999\ud800"}
        path.write_text(path.read_text(encoding="utf-8") + json.dumps(stray) + "\n", encoding="utf-8")
        self.compared(baseline, candidate, self.scored(baseline, candidate))
        record = items.read_json(self.folder / "verdict" / "verdict.json")
        self.assertEqual(record["incomplete"], [{"reason": "not_in_set", "ids": [stray["id"]]}])
        text = (self.folder / "verdict" / "summary.md").read_text(encoding="utf-8")
        self.assertIn("`not_in_set`: `free-ja-9999\\ud800`", text)

    def test_the_claim_names_the_categories_that_could_not_be_evaluated(self):
        baseline, candidate = self.run_folder("base", "fp16"), self.run_folder("cand", "w8")
        self.compared(baseline, candidate, self.scored(baseline, candidate))
        record = items.read_json(self.folder / "verdict" / "verdict.json")
        claim = f"{LAYER2['claim']}; not evaluated: synth-long-4k"
        self.assertEqual((record["verdict"], record["claim"]), ("no_regression_found", claim))
        self.assertIn("\n" + claim + "\n", (self.folder / "verdict" / "summary.md").read_text(encoding="utf-8"))

    def test_only_the_pairs_of_the_contract_are_compared(self):
        roles = {"base": "reference", "fp16": "fp16", "w8": "w8", "w4": "w4", "anchor": "anchor"}
        runs = {name: self.run_folder(name, role) for name, role in roles.items()}
        scores = self.scored(*runs.values())
        refused = [("fp16", "base", None), ("w8", "w4", None), ("anchor", "fp16", None), ("base", "w8", None),
                   ("base", "w4", None), ("base", "fp16", "fp16 failed"), ("fp16", "w8", "fp16 failed"),
                   ("base", "w8", " ")]
        for number, (before, after, reason) in enumerate(refused):
            with self.subTest((before, after, reason)):
                with self.assertRaises(ValueError):
                    self.compared(runs[before], runs[after], scores, name=f"refused-{number}", substitute_reason=reason)
                self.assertFalse((self.folder / f"refused-{number}").exists())
        for number, (before, after, reason) in enumerate([("base", "fp16", None), ("fp16", "w8", None),
                                                          ("fp16", "w4", None), ("base", "w8", "fp16 failed"),
                                                          ("base", "w4", "fp16 failed")]):
            with self.subTest((before, after, reason)):
                self.compared(runs[before], runs[after], scores, name=f"accepted-{number}", substitute_reason=reason)
                record = items.read_json(self.folder / f"accepted-{number}" / "verdict.json")
                self.assertEqual(record["substitute_reason"], reason)
                self.assertEqual(record["verdict"], "no_regression_found")

    def test_an_existing_output_folder_is_rejected(self):
        baseline, candidate = self.run_folder("base", "reference"), self.run_folder("cand", "fp16")
        scores = self.scored(baseline, candidate)
        (self.folder / "verdict").mkdir()
        with self.assertRaises(FileExistsError):
            self.compared(baseline, candidate, scores)
        self.assertEqual(list((self.folder / "verdict").iterdir()), [])

    def test_run_folders_with_the_same_name_are_rejected(self):
        baseline, candidate = self.run_folder("a/run", "reference"), self.run_folder("b/Run", "fp16")
        with self.assertRaises(ValueError):
            self.compared(baseline, candidate, self.scored(baseline))
        self.assertFalse((self.folder / "verdict").exists())

    def test_a_run_folder_that_changed_after_scoring_is_rejected(self):
        for case in ("outputs", "run", "summary"):
            with self.subTest(case):
                baseline = self.run_folder(f"{case}/base", "reference")
                candidate = self.run_folder(f"{case}/cand", "fp16")
                scores = self.scored(baseline, candidate, name=f"{case}/scores")
                if case == "outputs":
                    path = candidate / "outputs.jsonl"
                    path.write_bytes(path.read_bytes().replace(b'"0', b'"1', 1))
                elif case == "run":
                    path = candidate / "run.json"
                    path.write_text(json.dumps({**items.read_json(path), "implementation": "again"}), encoding="utf-8")
                else:
                    path = scores / "cand" / "summary.json"
                    path.write_text(json.dumps({key: value for key, value in items.read_json(path).items()
                                                if key != "outputs"}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.compared(baseline, candidate, scores, name=f"{case}/verdict")
                self.assertFalse((self.folder / case / "verdict").exists())

    def test_scores_that_are_not_of_the_items_are_rejected(self):
        baseline, candidate = self.run_folder("base", "reference"), self.run_folder("cand", "fp16")
        scores = self.scored(baseline, candidate)
        path = scores / "cand" / "scores.jsonl"
        write_lines(path, [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()][:1])
        with self.assertRaises(ValueError):
            self.compared(baseline, candidate, scores)
        self.assertFalse((self.folder / "verdict").exists())

    def test_an_items_file_other_than_the_scored_one_is_rejected(self):
        baseline, candidate = self.run_folder("base", "reference"), self.run_folder("cand", "fp16")
        scores = self.scored(baseline, candidate)
        write_lines(self.items_path, [self.items[0], {**self.items[1], "answer": self.items[1]["answer"] + 1}])
        with self.assertRaises(ValueError):
            self.compared(baseline, candidate, scores)
        self.assertFalse((self.folder / "verdict").exists())

    def test_items_that_differ_from_the_frozen_set_are_rejected(self):
        baseline, candidate = self.run_folder("base", "reference"), self.run_folder("cand", "fp16")
        scores = self.scored(baseline, candidate)
        longer = {**self.items[1], "token_count": self.items[1]["token_count"] + 1}
        write_lines(self.items_path, [self.items[0], longer])
        with self.assertRaises(ValueError):
            self.compared(baseline, candidate, scores)
        write_lines(self.items_path, self.items[:1])
        with self.assertRaises(ValueError):
            self.compared(baseline, candidate, scores)
        self.assertFalse((self.folder / "verdict").exists())

    def test_a_failure_while_writing_removes_the_output(self):
        baseline, candidate = self.run_folder("base", "reference"), self.run_folder("cand", "fp16")
        scores = self.scored(baseline, candidate)
        with mock.patch.object(comparison, "summary", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.compared(baseline, candidate, scores)
        self.assertFalse((self.folder / "verdict").exists())

    def test_the_rerun_counts_swaps_against_the_candidate_scores(self):
        baseline, candidate = self.run_folder("base", "reference"), self.run_folder("cand", "fp16", **RUN)
        rerun = self.run_folder("again", "fp16", answer=9, **{**RUN, "call_shape": {"prefill_chunk": 8}})
        scores = self.scored(baseline, candidate, rerun)
        self.compared(baseline, candidate, scores, rerun=scores / "again")
        record = items.read_json(self.folder / "verdict" / "verdict.json")
        self.assertEqual(record["rerun"], {"correct_to_incorrect": 1, "incorrect_to_correct": 0})
        self.assertEqual(record["runs"]["rerun"], items.read_json(rerun / "run.json"))
        self.assertEqual(record["verdict"], "no_regression_found")

    def test_a_rerun_scoring_of_other_conditions_is_rejected(self):
        changes = {"duplicate": {}, "items": {}, "asset": {"asset": {"main_hash": "b"}},
                   "inputs": {"inputs": {"sha256": "j"}}, "implementation": {"implementation": {"commit": "d"}},
                   "set": {"set": "reserve"}, "role": {"role": "w8"}, "call_shape": {"call_shape": RUN["call_shape"]}}
        for case in (*changes, "no_asset"):
            with self.subTest(case):
                fields = {key: value for key, value in RUN.items() if case != "no_asset" or key != "asset"}
                baseline = self.run_folder(f"{case}/base", "reference")
                candidate = self.run_folder(f"{case}/cand", "fp16", **fields)
                change = dict(changes.get(case, {}))
                rerun = self.run_folder(f"{case}/again", change.pop("role", "fp16"), answer=9,
                                        **{**fields, "call_shape": {"prefill_chunk": 8}, **change})
                scores = self.scored(baseline, candidate, rerun, name=f"{case}/scores")
                if case == "duplicate":
                    path = scores / "again" / "scores.jsonl"
                    text = path.read_text(encoding="utf-8")
                    path.write_text(text + text.splitlines(keepends=True)[1], encoding="utf-8")
                elif case == "items":
                    path = scores / "again" / "summary.json"
                    summary = items.read_json(path)
                    path.write_text(json.dumps({**summary, "items": {**summary["items"], "sha256": "0" * 64}}),
                                    encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.compared(baseline, candidate, scores, name=f"{case}/verdict", rerun=scores / "again")
                self.assertFalse((self.folder / case / "verdict").exists())


if __name__ == "__main__":
    unittest.main()
