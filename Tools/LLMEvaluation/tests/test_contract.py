import copy
import gzip
import hashlib
import json
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import unittest

from Tools.LLMEvaluation import checks, items, selection
from Tools.LLMEvaluation.synthetic.generate import generate

BUNDLE = items.load()


def entries(bundle, category, name="primary"):
    return [entry for entry in bundle["sets"][name]["items"] if entry["category"] == category]


def category(bundle, name):
    return next(value for value in bundle["contract"]["categories"] if value["id"] == name)


def scored(scorer, code, root):
    marker = secrets.token_hex(16)
    parts = {**scorer, "code": code, "marker": f"print({marker!r})"}
    program = "\n".join(parts[name] for name in BUNDLE["contract"]["scorers"]["python_tests"]["program"])
    result = subprocess.run([sys.executable, "-I", "-c", program], cwd=root, capture_output=True, text=True,
                            stdin=subprocess.DEVNULL, check=False)
    return result.returncode == 0 and result.stdout.splitlines()[-1:] == [marker]

class ContractTests(unittest.TestCase):
    def rejected(self, change, text):
        bundle = copy.deepcopy(BUNDLE)
        change(bundle)
        found = checks.problems(bundle)
        self.assertTrue(any(text in problem for problem in found), found)

    def test_committed_contract_sets_and_document_agree(self):
        self.assertEqual(checks.problems(copy.deepcopy(BUNDLE)), [])

    def test_unpinned_revisions_are_rejected(self):
        self.rejected(lambda bundle: bundle["contract"]["sources"]["jsquad"].update(revision="main"),
                      "source jsquad revision is not a 40-digit commit")
        self.rejected(lambda bundle: bundle["contract"]["sources"]["ner"].update(revision="d6c4d48"),
                      "source ner revision is not a 40-digit commit")
        self.rejected(lambda bundle: bundle["contract"]["model"].update(revision="main"),
                      "model revision is not a 40-digit commit")

    def test_duplicate_ids_are_rejected(self):
        def change(bundle):
            entries(bundle, "ja-choice", "reserve")[0]["id"] = entries(bundle, "ja-choice")[0]["id"]

        self.rejected(change, "duplicate item id ja-choice-0001")

    def test_the_same_original_in_two_places_is_rejected(self):
        def primary_and_reserve(bundle):
            entries(bundle, "extract-ja", "reserve")[0]["source"] = dict(entries(bundle, "extract-ja")[0]["source"])

        def reading_and_long(bundle):
            long = entries(bundle, "long-ja-4k")[0]["source"]
            entries(bundle, "ja-reading")[0]["source"] = {"article": long["article"], "question": long["question"]}

        def filler_from_question_article(bundle):
            long = entries(bundle, "long-ja-16k")[0]["source"]
            question = entries(bundle, "ja-reading")[0]["source"]["question"]
            index = next(index for index, key in enumerate(long["paragraphs"]) if key != checks.placement(long))
            long["paragraphs"][index] = question[:question.rindex("q")]

        def even_fix(bundle):
            entries(bundle, "code-fix")[0]["source"]["task"] = entries(bundle, "code-gen-he")[0]["source"]["task"]

        self.rejected(primary_and_reserve, "same original as extract-ja-")
        self.rejected(reading_and_long, "same original as")
        self.rejected(filler_from_question_article, "filler paragraph comes from a question article")
        self.rejected(even_fix, "problem number parity differs")

    def test_count_mismatches_are_rejected(self):
        self.rejected(lambda bundle: bundle["sets"]["primary"]["items"].pop(), "primary: item counts differ")
        self.rejected(lambda bundle: category(bundle, "ja-choice").update(primary=199), "item counts differ")
        self.rejected(lambda bundle: category(bundle, "synth-long-16k").update(reserve=0), "reserve: item counts differ")

    def test_unknown_scorer_kinds_are_rejected(self):
        self.rejected(lambda bundle: category(bundle, "ja-reading").update(scorer="fuzzy_match"),
                      "ja-reading: unknown scorer kind fuzzy_match")
        self.rejected(lambda bundle: bundle["contract"]["scorers"].update(fuzzy_match={}),
                      "unknown scorer kind in the contract: fuzzy_match")

    def test_missing_thresholds_and_limits_are_rejected(self):
        for path in (("max_probability_error",), ("band_mean_max_probability_error",), ("near_tie_probability_margin",),
                     ("same_assets", "float16_ticks"), ("positions", "max_per_item")):
            def change(bundle, path=path):
                parent = bundle["contract"]["layer1"]
                for key in path[:-1]:
                    parent = parent[key]
                del parent[path[-1]]

            self.rejected(change, f"contract key layer1/{'/'.join(path)} is missing")
        self.rejected(lambda bundle: category(bundle, "code-fix").pop("max_new_tokens"), "code-fix: max_new_tokens is missing")
        self.rejected(lambda bundle: category(bundle, "synth-long-4k").pop("token_range"), "synth-long-4k: token_range is missing")
        self.rejected(lambda bundle: bundle["contract"]["layer2"]["critical_failures"]["repetition"].pop("min_length"),
                      "contract key layer2/critical_failures/repetition/min_length is missing")
        self.rejected(lambda bundle: bundle["contract"]["layer1"]["bands"].pop(1), "bands are missing or not contiguous")

    def test_code_execution_limit_and_reruns_are_required(self):
        tests = ("contract", "scorers", "python_tests")
        self.assertEqual(checks.lookup(BUNDLE, tests + ("wall_clock_seconds",)), 10)
        self.assertEqual(checks.lookup(BUNDLE, tests + ("timeout_reruns",)), 1)
        self.rejected(lambda bundle: bundle["contract"]["scorers"]["python_tests"].pop("wall_clock_seconds"),
                      "contract key scorers/python_tests/wall_clock_seconds is missing")
        self.rejected(lambda bundle: bundle["contract"]["scorers"]["python_tests"].pop("timeout_reruns"),
                      "contract key scorers/python_tests/timeout_reruns is missing")
        self.rejected(lambda bundle: bundle["contract"]["scorers"]["python_tests"].update(timeout_reruns=0),
                      "python_tests timeout_reruns is missing")

    def test_critical_failure_lists_and_the_evaluable_minimum_are_required(self):
        self.assertEqual(checks.lookup(BUNDLE, ("contract", "layer2", "min_baseline_correct")), 10)

        def rename(bundle):
            failures = bundle["contract"]["layer2"]["critical_failures"]
            failures["relative_to_reference"] = failures.pop("relative_to_baseline")

        self.rejected(rename, "contract key layer2/critical_failures/relative_to_baseline is missing")
        self.rejected(lambda bundle: bundle["contract"]["layer2"].pop("min_baseline_correct"),
                      "contract key layer2/min_baseline_correct is missing")
        self.rejected(lambda bundle: bundle["contract"]["layer2"].update(min_baseline_correct=0),
                      "layer2 min_baseline_correct is missing")

    def humaneval_scorer(self, root):
        prompt = 'def helper(value):\n    return value * 2\n\n\ndef target(value):\n    """Return helper(value) plus one."""\n'
        test = "\n\ndef check(candidate):\n    assert candidate(3) == 7\n    assert helper(2) == 4\n"
        row = {"task_id": "HumanEval/0", "prompt": prompt, "test": test, "entry_point": "target"}
        data = gzip.compress((json.dumps(row) + "\n").encode("utf-8"), mtime=0)
        (Path(root) / "fake").mkdir()
        (Path(root) / "fake" / "rows.jsonl.gz").write_bytes(data)
        bundle = copy.deepcopy(BUNDLE)
        bundle["contract"]["sources"]["humaneval"].update(folder="fake", path="rows.jsonl.gz", bytes=len(data),
                                                          sha256=hashlib.sha256(data).hexdigest())
        entry = {"id": "code-gen-he-0001", "category": "code-gen-he", "unit": "problem", "source": {"task": 0}}
        scorer = items.item(category(bundle, "code-gen-he"), entry, items.Sources(bundle["contract"], root),
                            bundle)["scorer"]
        self.assertEqual(scorer["setup"], prompt)
        return scorer

    def test_humaneval_program_starts_with_the_problem_prompt(self):
        with tempfile.TemporaryDirectory() as root:
            scorer = self.humaneval_scorer(root)
            solution = "def target(value):\n    return helper(value) + 1\n"
            self.assertTrue(scored(scorer, solution, root))
            self.assertFalse(scored({**scorer, "setup": ""}, solution, root))

    def test_code_is_correct_only_when_the_tests_ran_to_the_end(self):
        with tempfile.TemporaryDirectory() as root:
            scorer = self.humaneval_scorer(root)
            self.assertTrue(scored(scorer, "def target(value):\n    return helper(value) + 1\n", root))
            self.assertFalse(scored(scorer, "import sys\n\n\ndef target(value):\n    return helper(value) + 1\n\n\nsys.exit(0)\n", root))
            self.assertFalse(scored(scorer, "def target(value):\n    return value\n", root))

    def test_every_required_contract_key_is_required(self):
        paths = list(checks.required_paths())
        self.assertGreater(len(paths), 100)
        for path in paths:
            contract = copy.deepcopy(BUNDLE["contract"])
            parent = checks.lookup(contract, path[:-1]) if len(path) > 1 else contract
            del parent[path[-1]]
            with self.subTest("/".join(path)):
                self.assertIn(f"contract key {'/'.join(path)} is missing", checks.problems({**BUNDLE, "contract": contract}))

    def test_verdict_precedence_order_is_fixed(self):
        self.rejected(lambda bundle: bundle["contract"]["layer2"]["precedence"].reverse(),
                      "layer2 precedence differs from the fixed order")

        def swap(bundle):
            order = bundle["contract"]["layer2"]["precedence"]
            order[1], order[2] = order[2], order[1]

        self.rejected(swap, "layer2 precedence differs from the fixed order")

    def test_long_context_items_keep_distinct_fillers_and_the_strata_order(self):
        def duplicate(bundle):
            source = entries(bundle, "long-ja-4k")[0]["source"]
            own = checks.placement(source)
            fillers = [index for index, key in enumerate(source["paragraphs"]) if key != own]
            source["paragraphs"][fillers[1]] = source["paragraphs"][fillers[0]]

        def all_front(bundle):
            for entry in entries(bundle, "long-ja-4k"):
                entry["source"]["position"] = "front"

        self.rejected(duplicate, "long-ja-4k-0001: a paragraph appears twice")
        self.rejected(all_front, "long-ja-4k-0002: stratum differs from its place in the set")

    def test_hash_mismatches_are_rejected(self):
        self.rejected(lambda bundle: bundle["hashes"].update({"sets/primary.json": "0" * 64}),
                      "document hash for sets/primary.json differs")
        self.rejected(lambda bundle: entries(bundle, "synth-format", "reserve")[0].update(item_sha256="0" * 64),
                      "item hash differs")
        self.rejected(lambda bundle: bundle["resources"].update({"synthetic/vocabulary.json": "0" * 64}),
                      "resource synthetic/vocabulary.json differs from its pin")
        self.rejected(lambda bundle: bundle.update(document=bundle["document"].replace("G31-QC-v1", "G31-QC-v0")),
                      "document header version differs from the contract")

    def test_primary_and_reserve_numbering_is_required(self):
        def swap(bundle):
            primary, reserve = bundle["sets"]["primary"]["items"], bundle["sets"]["reserve"]["items"]
            first = next(index for index, entry in enumerate(primary) if entry["category"] == "ja-choice")
            second = next(index for index, entry in enumerate(reserve) if entry["category"] == "ja-choice")
            primary[first], reserve[second] = reserve[second], primary[first]

        self.rejected(swap, "primary: ja-choice ids are not numbered 1 to 200")

    def test_the_same_humaneval_problem_in_both_code_categories_is_rejected(self):
        def both_even(bundle):
            category(bundle, "code-fix").update(parity=0)
            fix = entries(bundle, "code-fix")
            fix[0]["source"]["task"] = entries(bundle, "code-gen-he")[0]["source"]["task"]
            for entry in fix[1:]:
                entry["source"]["task"] -= 1

        self.rejected(both_even, "code-fix-0001: same original as code-gen-he-0001")

    def test_the_same_synthetic_combination_in_two_categories_is_rejected(self):
        bundle = copy.deepcopy(BUNDLE)
        seed, prompts, vocabulary = bundle["contract"]["seed"], bundle["prompts"], bundle["vocabulary"]

        def combination(name, index, records):
            return json.dumps(generate(category(bundle, name), index, seed, prompts, vocabulary, records)["combination"],
                              ensure_ascii=False)

        seen = {combination("synth-long-4k", entry["source"]["index"], entry["source"]["records"])
                for name in items.SETS for entry in entries(bundle, "synth-long-4k", name)}
        entry = entries(bundle, "synth-long-16k")[0]
        records = entry["source"]["records"]
        index = next(index for index in range(1000) if combination("synth-long-16k", index, records) in seen)
        template = generate(category(bundle, "synth-long-16k"), index, seed, prompts, vocabulary, records)["template"]
        entry["source"] = {"index": index, "template": template, "records": records}
        found = checks.problems(bundle)
        self.assertTrue(any("same template combination" in problem and "synth-long-16k-0001" in problem
                            and "synth-long-4k-" in problem for problem in found), found)

    def test_document_header_and_freeze_date_are_compared_exactly(self):
        self.rejected(lambda bundle: bundle.update(document=bundle["document"].replace(
            "| Version | `G31-QC-v1` |", "| Version | `G31-QC-v0` |")), "document header version differs from the contract")
        self.rejected(lambda bundle: bundle.update(document=bundle["document"].replace(
            "| Frozen | 2026-11-22 |", "| Frozen | 2026-11-21 |")), "document header frozen differs from the contract")
        for value in ("", "2026", "2026-11-31", "20261122"):
            self.rejected(lambda bundle, value=value: bundle["contract"].update(frozen=value),
                          "frozen is not a YYYY-MM-DD date")

    def test_item_scorer_kinds_must_match_their_category(self):
        self.rejected(lambda bundle: category(bundle, "synth-format").update(scorer="json_exact"),
                      "synth-format-0001: scorer kind differs from its category")

    def test_select_check_ignores_only_the_recorded_versions(self):
        committed = (items.ROOT / items.FILES["reserve"]).read_text(encoding="utf-8")
        document = json.loads(committed)
        self.assertEqual(selection.serialize(document), committed)
        self.assertEqual(selection.compare(document, committed), (True, None))
        newer = {**document, "selected_with": {**document["selected_with"], "tokenizers": "0.0.0"}}
        same, note = selection.compare(newer, committed)
        self.assertTrue(same)
        self.assertEqual(note["current"]["tokenizers"], "0.0.0")
        changed = {**document, "items": document["items"][1:] + document["items"][:1]}
        self.assertFalse(selection.compare(changed, committed)[0])
        recorded = document["selected_with"]["tokenizer_sha256"]
        self.assertFalse(selection.compare(document, committed.replace(recorded, "0" * 64))[0])
        recorded = document["selected_with"]["chat_template_sha256"]
        self.assertFalse(selection.compare(document, committed.replace(recorded, "0" * 64))[0])

    def test_build_rejects_source_and_item_hash_mismatches(self):
        with tempfile.TemporaryDirectory() as root:
            row = {"q_id": 7, "question": "q", "label": 2, **{f"choice{index}": f"c{index}" for index in range(5)}}
            data = (json.dumps(row) + "\n").encode("utf-8")
            (Path(root) / "fake").mkdir()
            (Path(root) / "fake" / "rows.jsonl").write_bytes(data)
            bundle = copy.deepcopy(BUNDLE)
            bundle["contract"]["sources"]["jcqa"].update(folder="fake", path="rows.jsonl", bytes=len(data),
                                                         sha256=hashlib.sha256(data).hexdigest())
            entry = {"id": "ja-choice-0001", "category": "ja-choice", "unit": "question", "source": {"q_id": 7}}
            value = items.item(category(bundle, "ja-choice"), entry, items.Sources(bundle["contract"], root), bundle)
            self.assertEqual(value["answer"], 2)
            entry["item_sha256"] = items.digest(value)
            bundle["sets"] = {"primary": {"items": [entry]}, "reserve": {"items": []}}
            manifest = items.build(bundle, root, Path(root) / "first")
            self.assertEqual(manifest["items"], {"primary": 1, "reserve": 0})
            entry["item_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "item hashes differ"):
                items.build(bundle, root, Path(root) / "second")
            entry["item_sha256"] = items.digest(value)
            (Path(root) / "fake" / "rows.jsonl").write_bytes(data + b"\n")
            with self.assertRaisesRegex(ValueError, "differs from its pin"):
                items.build(bundle, root, Path(root) / "third")
            with self.assertRaises(FileExistsError):
                items.build(bundle, root, Path(root) / "first")
            (Path(root) / "fake" / "rows.jsonl").write_bytes(data)
            category(bundle, "ja-choice").update(scorer="reading_exact")
            with self.assertRaisesRegex(ValueError, "scorer kinds differ"):
                items.build(bundle, root, Path(root) / "fourth")

    def test_validate_runs_with_only_the_standard_library(self):
        script = items.ROOT / "__main__.py"
        with tempfile.TemporaryDirectory() as root:
            result = subprocess.run([sys.executable, "-I", "-S", str(script), "validate"], cwd=root,
                                    capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS: G31-QC-v1", result.stdout)


if __name__ == "__main__":
    unittest.main()
