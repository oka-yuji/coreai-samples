import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

from Tools.LLMEvaluation import items, layer1

BUNDLE = items.load()
CONTRACT = BUNDLE["contract"]
ENTRIES = BUNDLE["sets"]["primary"]["items"]
IDENTITY = {"version": CONTRACT["version"], "sha256": BUNDLE["hashes"]["contract.json"]}
LIMITS = {category["id"]: category["max_new_tokens"] for category in CONTRACT["categories"]}
FAMILIES = {"long-ja-4k": 6, "long-ja-16k": 2, "synth-long-4k": 6, "synth-long-16k": 2}
CHOSEN = {entry["id"] for entry in ENTRIES if int(entry["id"][-4:]) <= FAMILIES.get(entry["category"], 8)}
ORDER = [f"{category}-{number:04d}" for category in dict.fromkeys(entry["category"] for entry in ENTRIES)
         for number in range(1, FAMILIES.get(category, 8) + 1)]
STOP = 106
ROOT = Path(__file__).resolve().parents[3]


def digest(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode("utf-8")).hexdigest()


def identity(run):
    value = {key: item for key, item in run.items() if key not in ("items", "started", "finished")}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def generation(output_ids, stop=STOP):
    output_ids = list(output_ids)
    record = {"output_ids": output_ids, "non_finite": 0, "top2": [[token, 0.75, token + 1, 0.125] for token in output_ids]}
    if stop is None:
        return {**record, "stop": "limit"}
    return {**record, "stop": "stop_token", "stop_token": stop, "top2": record["top2"] + [[stop, 0.75, stop + 1, 0.125]]}


def fixture():
    inputs, outputs = [], []
    for number, entry in enumerate(ENTRIES):
        length = entry["token_count"] if entry["id"] in CHOSEN and "token_count" in entry else 4 + number % 5
        ids = [2, 105] + [1000 + (number * 7 + index) % 3000 for index in range(length - 2)]
        inputs.append({"id": entry["id"], "input_ids": ids, "input_sha256": digest(ids),
                       "max_new_tokens": LIMITS[entry["category"]]})
        outputs.append({"id": entry["id"], "input_sha256": digest(ids), "input_tokens": len(ids),
                        **generation(range(5000 + number % 11, 5001 + number % 11 + number % 3))})
    run = {"contract": dict(IDENTITY), "set": "primary", "role": "reference", "items": len(ENTRIES), "id": "run-1",
           "asset": {"model": CONTRACT["model"]["repository"], "revision": CONTRACT["model"]["revision"]},
           "started": "2026-11-22T00:00:00Z", "finished": "2026-11-22T01:00:00Z"}
    return {"inputs": inputs, "outputs": outputs, "run": run, "summary": {}, "progress": {}}


def by_id(records, name):
    return next(record for record in records if record["id"] == name)


def generate(state, name, output_ids, stop=STOP):
    records = state["outputs"]
    index = next(number for number, record in enumerate(records) if record["id"] == name)
    old = records[index]
    records[index] = {"id": name, "input_sha256": old["input_sha256"], "input_tokens": old["input_tokens"],
                      **generation(output_ids, stop)}


def lines(records):
    return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records).encode("utf-8")


def save(state, folder):
    inputs, run = Path(folder) / "inputs", Path(folder) / "run"
    inputs.mkdir(parents=True)
    run.mkdir()
    data = lines(state["inputs"])
    (inputs / "inputs.jsonl").write_bytes(data)
    files = {"count": len(state["inputs"]), "sha256": hashlib.sha256(data).hexdigest()}
    (inputs / "inputs.json").write_text(json.dumps({"contract": IDENTITY, **files, **state["summary"]}))
    stored, outputs = {"inputs": files, **state["run"]}, lines(state["outputs"])
    (run / "run.json").write_text(json.dumps(stored))
    (run / "outputs.jsonl").write_bytes(outputs)
    if state["progress"] is not None:
        (run / "progress.json").write_text(json.dumps({"lines": len(state["inputs"]), "run": identity(stored),
                                                       "sha256": hashlib.sha256(outputs).hexdigest(), **state["progress"]}))
    return inputs, run


def bounded(call, seconds=3):
    outcome = {}

    def run():
        try:
            outcome["value"] = call()
        except BaseException as error:
            outcome["error"] = error

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    return thread, outcome


def release(path, thread):
    waited = thread.is_alive()
    if waited:
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_NONBLOCK))
        except OSError:
            pass
        thread.join(5)
    return waited


def replaced(path, kind):
    path.unlink()
    if kind == "fifo":
        os.mkfifo(path)
    elif kind == "folder":
        path.mkdir()
    elif kind == "broken link":
        path.symlink_to(path.parent / "nowhere")


def grouped(entries):
    found = {}
    for entry in entries:
        found.setdefault(entry["category"], []).append(entry["id"])
    return found


class Folder(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def forced(self, state, bundle=BUNDLE, name="forced"):
        inputs, run = save(state, self.root / f"{name}-sources")
        output = self.root / name
        summary = layer1.write(bundle, inputs, run, output)
        records = [json.loads(line) for line in (output / "forced.jsonl").read_text().splitlines()]
        return summary, {record["id"]: record for record in records}, output


class SelectionTests(unittest.TestCase):
    def test_the_first_items_of_each_category_are_chosen_in_ascending_id_order_with_the_long_families_split(self):
        self.assertEqual([entry["id"] for entry in layer1.chosen(BUNDLE)], ORDER)
        self.assertEqual(len(ORDER), 88)

    def test_the_choice_follows_the_item_ids_and_not_the_order_of_the_set_file(self):
        shuffled = copy.deepcopy(BUNDLE)
        random.Random(697).shuffle(shuffled["sets"]["primary"]["items"])
        self.assertEqual(grouped(layer1.chosen(shuffled)), grouped(layer1.chosen(BUNDLE)))

    def test_the_counts_of_the_choice_are_read_from_the_contract(self):
        changed = copy.deepcopy(BUNDLE)
        rule = changed["contract"]["layer1"]["items"]
        rule["per_category"] = 3
        rule["families"]["long-ja"] = {"long-ja-4k": 1, "long-ja-16k": 4}
        counts = {name: len(ids) for name, ids in grouped(layer1.chosen(changed)).items()}
        self.assertEqual((counts["ja-choice"], counts["free-ja"], counts["long-ja-4k"], counts["long-ja-16k"],
                          counts["synth-long-4k"], counts["synth-long-16k"]), (3, 3, 1, 4, 6, 2))


class ForcedInputTests(Folder):
    def test_a_generation_that_stopped_on_a_token_ends_with_that_token_as_its_last_target(self):
        state = fixture()
        generate(state, "ja-reading-0001", [11, 12, 13])
        _, records, _ = self.forced(state)
        record, ids = records["ja-reading-0001"], by_id(state["inputs"], "ja-reading-0001")["input_ids"]
        self.assertEqual(record["targets"], [11, 12, 13, STOP])
        self.assertEqual(record["tokens"], ids + [11, 12, 13])
        self.assertEqual((record["rows"], record["start"], record["input_tokens"]), (4, len(ids) - 1, len(ids)))

    def test_a_generation_that_reached_the_limit_gets_no_stop_token(self):
        state = fixture()
        generate(state, "ja-choice-0001", range(20, 28), stop=None)
        _, records, _ = self.forced(state)
        record, ids = records["ja-choice-0001"], by_id(state["inputs"], "ja-choice-0001")["input_ids"]
        self.assertEqual(record["targets"], list(range(20, 28)))
        self.assertEqual(record["tokens"], ids + list(range(20, 27)))
        self.assertEqual(record["rows"], 8)

    def test_at_most_the_first_128_generated_tokens_are_scored(self):
        state = fixture()
        generate(state, "code-gen-he-0001", range(4000, 4200))
        generate(state, "code-gen-he-0002", range(4000, 4127))
        generate(state, "code-gen-he-0003", range(4000, 4128))
        generate(state, "code-fix-0001", range(4000, 4512), stop=None)
        _, records, _ = self.forced(state)
        for name, targets in (("code-gen-he-0001", list(range(4000, 4128))),
                              ("code-gen-he-0002", list(range(4000, 4127)) + [STOP]),
                              ("code-gen-he-0003", list(range(4000, 4128))),
                              ("code-fix-0001", list(range(4000, 4128)))):
            with self.subTest(name):
                ids = by_id(state["inputs"], name)["input_ids"]
                self.assertEqual(records[name]["targets"], targets)
                self.assertEqual(records[name]["rows"], 128)
                self.assertEqual(records[name]["tokens"], ids + targets[:127])

    def test_every_record_scores_from_the_last_input_position_and_names_its_hashes(self):
        state = fixture()
        _, records, _ = self.forced(state)
        self.assertEqual(sorted(records), sorted(CHOSEN))
        for name, record in records.items():
            with self.subTest(name):
                source, output = by_id(state["inputs"], name), by_id(state["outputs"], name)
                ids, generated = source["input_ids"], output["output_ids"] + [output["stop_token"]]
                self.assertEqual(set(record), {"id", "input_sha256", "tokens", "tokens_sha256", "start", "rows",
                                               "input_tokens", "targets"})
                self.assertEqual(record["input_sha256"], source["input_sha256"])
                self.assertEqual(record["tokens"], ids + generated[:-1])
                self.assertEqual(record["tokens_sha256"], digest(record["tokens"]))
                self.assertEqual((record["start"], record["rows"], record["input_tokens"], record["targets"]),
                                 (len(ids) - 1, len(generated), len(ids), generated))

    def test_the_row_limit_is_read_from_the_contract(self):
        changed = copy.deepcopy(BUNDLE)
        changed["contract"]["layer1"]["positions"]["max_per_item"] = 2
        _, records, _ = self.forced(fixture(), changed)
        self.assertEqual({record["rows"] for record in records.values()}, {2})
        self.assertTrue(all(len(record["tokens"]) == record["input_tokens"] + 1 for record in records.values()))

    def test_the_summary_names_the_contract_the_inputs_the_run_and_the_counts(self):
        summary, _, output = self.forced(fixture())
        sources = self.root / "forced-sources"
        self.assertEqual(json.loads((output / "forced.json").read_text()), summary)
        self.assertEqual(sorted(path.name for path in output.iterdir()), ["forced.json", "forced.jsonl"])
        self.assertEqual([json.loads(line)["id"] for line in (output / "forced.jsonl").read_text().splitlines()], ORDER)
        self.assertEqual(summary, {
            "contract": IDENTITY, "set": "primary",
            "inputs": {"count": len(ENTRIES), "sha256": items.sha256(sources / "inputs/inputs.jsonl")},
            "run": {"id": "run-1", "outputs_sha256": items.sha256(sources / "run/outputs.jsonl")},
            "forced": {"count": 88, "sha256": items.sha256(output / "forced.jsonl")},
            "bands": {"1-512": 72, "513-2048": 0, "2049-4096": 12, "4097-16896": 4},
            "categories": {category: len(ids) for category, ids in grouped(
                [{"id": name, "category": name[:-5]} for name in ORDER]).items()}})

    def test_writing_the_same_records_again_gives_the_same_bytes(self):
        state = fixture()
        _, _, first = self.forced(state, name="first")
        _, _, second = self.forced(state, name="second")
        for name in ("forced.json", "forced.jsonl"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_the_command_writes_the_folder_and_prints_its_summary(self):
        inputs, run = save(fixture(), self.root / "sources")
        output = self.root / "forced"
        process = subprocess.run([sys.executable, "-m", "Tools.LLMEvaluation", "forced", "--inputs", str(inputs),
                                  "--run", str(run), "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout), json.loads((output / "forced.json").read_text()))


class ForcedRecordTests(Folder):
    def written(self):
        state = fixture()
        generate(state, "code-gen-he-0001", range(4000, 4200))
        return self.forced(state)[1]

    def test_every_written_record_is_consistent(self):
        rule = CONTRACT["layer1"]
        self.assertEqual({name: layer1.inconsistencies(record, rule) for name, record in self.written().items()},
                         {name: [] for name in ORDER})

    def test_a_constructed_record_that_is_inconsistent_stops_the_forced_input(self):
        original = layer1.forced_record

        def shifted(*arguments):
            made, wrong = original(*arguments)
            return {**made, "start": made["start"] + 1}, wrong

        inputs, run = save(fixture(), self.root / "sources")
        with mock.patch.object(layer1, "forced_record", side_effect=shifted):
            with self.assertRaises(ValueError) as caught:
                layer1.write(BUNDLE, inputs, run, self.root / "forced")
        self.assertIn("ja-choice-0001: the written record is inconsistent: its start is not input_tokens - 1",
                      str(caught.exception))
        self.assertFalse(os.path.lexists(self.root / "forced"))

    def test_each_broken_relation_of_a_forced_record_is_reported(self):
        records, rule = self.written(), CONTRACT["layer1"]

        def rehashed(record):
            record["tokens_sha256"] = digest(record["tokens"])

        def longer(record):
            record["tokens"].append(7)
            rehashed(record)

        def tail(record):
            record["tokens"][-1] += 1
            rehashed(record)

        def above(record):
            record["rows"] += 1
            record["targets"].append(9)
            record["tokens"].append(record["targets"][-2])
            rehashed(record)

        cases = (("ja-choice-0001", lambda record: record.update(rows=True), ["it does not have the fields of a forced record"]),
                 ("ja-choice-0001", lambda record: record["tokens"].__setitem__(0, "2"),
                  ["it does not have the fields of a forced record"]),
                 ("ja-choice-0001", lambda record: record.update(tokens_sha256="0" * 64),
                  ["its tokens_sha256 is not that of its tokens"]),
                 ("ja-choice-0001", lambda record: record.update(input_sha256="0" * 64),
                  ["its input_sha256 is not that of its first input_tokens tokens"]),
                 ("ja-choice-0001", lambda record: record.update(start=record["start"] + 1), ["its start is not input_tokens - 1"]),
                 ("ja-choice-0001", longer, ["its tokens are not input_tokens + rows - 1",
                                             "its tokens after the input are not the first rows - 1 targets"]),
                 ("ja-choice-0001", lambda record: record["targets"].append(9), ["its targets are not one per row"]),
                 ("ja-choice-0001", tail, ["its tokens after the input are not the first rows - 1 targets"]),
                 ("code-gen-he-0001", above, ["its rows are outside 1 to max_per_item"]))
        for name, change, expected in cases:
            with self.subTest(expected[0]):
                record = copy.deepcopy(records[name])
                change(record)
                self.assertEqual(layer1.inconsistencies(record, rule), expected)
        self.assertEqual(layer1.inconsistencies([], rule), ["it is not a JSON object"])
        empty = copy.deepcopy(records["ja-choice-0001"])
        empty.update(rows=0, targets=[], tokens=empty["tokens"][:empty["input_tokens"] - 1])
        self.assertIn("its rows are outside 1 to max_per_item", layer1.inconsistencies(empty, rule))
        narrow = {**rule, "bands": [[1, 4]]}
        self.assertEqual(layer1.inconsistencies(records["ja-choice-0002"], narrow), ["its input length is outside the bands"])


class RefusalTests(Folder):
    def refused(self, state, text, after=None, bundle=BUNDLE, error=ValueError):
        inputs, run = save(state, tempfile.mkdtemp(dir=self.root))
        if after is not None:
            after(inputs, run)
        output = self.root / "forced"
        with self.assertRaises(error) as caught:
            layer1.write(bundle, inputs, run, output)
        self.assertIn(text, str(caught.exception))
        self.assertFalse(os.path.lexists(output))

    def test_a_run_or_inputs_made_under_another_contract_version_or_hash_write_nothing(self):
        for key, value in (("version", "G31-QC-v0"), ("sha256", "0" * 64)):
            with self.subTest(run=key):
                state = fixture()
                state["run"]["contract"][key] = value
                self.refused(state, "the run was made under another contract")
            with self.subTest(inputs=key):
                state = fixture()
                state["summary"] = {"contract": {**IDENTITY, key: value}}
                self.refused(state, "inputs.json is not the summary of inputs.jsonl under this contract")

    def test_input_records_that_are_not_the_ones_the_run_read_write_nothing(self):
        cases = {"run inputs": (lambda state: state["run"].update(inputs={"count": len(ENTRIES), "sha256": "0" * 64}),
                                "the run did not read these input records"),
                 "summary hash": (lambda state: state["summary"].update(sha256="0" * 64), "inputs.json is not the summary"),
                 "summary count": (lambda state: state["summary"].update(count=len(ENTRIES) - 1),
                                   "inputs.json is not the summary"),
                 "a float count": (lambda state: state["run"].update(items=float(len(ENTRIES))), "the run did not finish"),
                 "a missing item": (lambda state: state["inputs"].pop(), "the input records are not one per item"),
                 "an extra item": (lambda state: state["inputs"].append({**state["inputs"][-1], "id": "free-ja-0099"}),
                                   "the input records are not one per item")}
        for name, (change, text) in cases.items():
            with self.subTest(name):
                state = fixture()
                change(state)
                self.refused(state, text)

    def test_an_input_sha256_or_input_length_that_differs_writes_nothing(self):
        def own(state):
            by_id(state["inputs"], "free-ja-0001")["input_ids"].append(7)

        def length(state):
            output = by_id(state["outputs"], "free-ja-0001")
            output["input_tokens"] += 1

        cases = {"an input hash that is not of its ids": (own, "the input record does not hold token ids with their"),
                 "an output of another input": (lambda state: by_id(state["outputs"], "free-ja-0001").update(
                     input_sha256="1" * 64), "free-ja-0001: the output was generated from another input"),
                 "an output of another length": (length, "free-ja-0001: the output was generated from another input")}
        for name, (change, text) in cases.items():
            with self.subTest(name):
                state = fixture()
                change(state)
                self.refused(state, text)

    def test_an_unfinished_run_writes_nothing(self):
        for name, change in (("no finished", lambda run: run.pop("finished")),
                             ("finished null", lambda run: run.update(finished=None)),
                             ("fewer items", lambda run: run.update(items=len(ENTRIES) - 1))):
            with self.subTest(name):
                state = fixture()
                change(state["run"])
                self.refused(state, "the run did not finish")

    def test_an_item_that_ended_with_an_error_or_saw_non_finite_logits_writes_nothing(self):
        for name, change, text in (("error", {"error": "RuntimeError: stopped"}, "the output ended with an error"),
                                   ("non-finite", {"non_finite": 1}, "the output saw non-finite logits")):
            with self.subTest(name):
                state = fixture()
                by_id(state["outputs"], "extract-ja-0002").update(change)
                self.refused(state, f"extract-ja-0002: {text}")

    def test_a_run_of_another_model_or_revision_writes_nothing(self):
        for name, change in (("another revision", lambda asset: asset.update(revision="0" * 40)),
                             ("another model", lambda asset: asset.update(model="google/gemma-4-26B-it")),
                             ("no model", lambda asset: asset.pop("model"))):
            with self.subTest(name):
                state = fixture()
                change(state["run"]["asset"])
                self.refused(state, "the run is not of the model and revision of the contract")

    def test_a_run_of_another_role_or_set_writes_nothing(self):
        for key, value, text in (("role", "anchor", "the run is not of the reference role"),
                                 ("role", "fp16", "the run is not of the reference role"),
                                 ("set", "reserve", "the run is not of the primary set")):
            with self.subTest(f"{key} {value}"):
                state = fixture()
                state["run"][key] = value
                self.refused(state, text)

    def test_output_records_that_fail_the_record_checks_write_nothing(self):
        def other_top2(state):
            by_id(state["outputs"], "ja-choice-0003")["top2"][0][0] += 2

        cases = {"top2 of other tokens": (other_top2, "top2_differs ['ja-choice-0003']"),
                 "a missing record": (lambda state: state["outputs"].pop(), "missing ['free-ja-0050']"),
                 "a repeated record": (lambda state: state["outputs"].append(dict(state["outputs"][0])),
                                       "duplicate ['ja-choice-0001']"),
                 "a malformed record": (lambda state: state["outputs"][5].update(input_tokens=True),
                                        "malformed ['ja-choice-0006']")}
        for name, (change, text) in cases.items():
            with self.subTest(name):
                state = fixture()
                change(state)
                self.refused(state, text)

    def test_output_records_replaced_after_the_run_finished_write_nothing(self):
        def replaced(inputs, run):
            state = fixture()
            generate(state, "ja-reading-0002", [21, 22, 23])
            (run / "outputs.jsonl").write_bytes(lines(state["outputs"]))

        self.refused(fixture(), "progress.json does not confirm these output records of this finished run: sha256",
                     replaced)

    def test_a_missing_or_unconfirmed_progress_record_writes_nothing(self):
        def other_run(state):
            state["progress"]["run"] = identity({"inputs": {}, **state["run"], "id": "run-2"})

        cases = (("a line in writing", lambda state: state["progress"].update(next_sha256="0" * 64), "next_sha256"),
                 ("fewer lines", lambda state: state["progress"].update(lines=len(ENTRIES) - 1), "lines"),
                 ("a float line count", lambda state: state["progress"].update(lines=float(len(ENTRIES))), "lines"),
                 ("a flag as the line count", lambda state: state["progress"].update(lines=True), "lines"),
                 ("a null line in writing", lambda state: state["progress"].update(next_sha256=None), "next_sha256"),
                 ("another run", other_run, ": run"),
                 ("other bytes", lambda state: state["progress"].update(sha256="0" * 64), "sha256"))
        for name, change, text in cases:
            with self.subTest(name):
                state = fixture()
                change(state)
                self.refused(state, f"progress.json does not confirm these output records of this finished run{text}"
                             if text == ": run" else text)
        state = fixture()
        state["progress"] = None
        self.refused(state, "progress.json", error=FileNotFoundError)

    def test_an_output_record_with_a_repeated_key_writes_nothing_even_when_its_last_value_is_right(self):
        inputs, run = save(fixture(), tempfile.mkdtemp(dir=self.root))
        rows = (run / "outputs.jsonl").read_bytes().split(b"\n")
        number = next(index for index, row in enumerate(rows) if b'"id": "free-ja-0001"' in row)
        rows[number] = b'{"input_sha256": "' + b"1" * 64 + b'", ' + rows[number][1:]
        data = b"\n".join(rows)
        (run / "outputs.jsonl").write_bytes(data)
        progress = json.loads((run / "progress.json").read_text())
        (run / "progress.json").write_text(json.dumps({**progress, "sha256": hashlib.sha256(data).hexdigest()}))
        with self.assertRaises(ValueError) as caught:
            layer1.write(BUNDLE, inputs, run, self.root / "forced")
        self.assertIn(f"malformed [{number + 1}]", str(caught.exception))
        self.assertFalse(os.path.lexists(self.root / "forced"))

    def test_an_output_whose_stop_or_length_does_not_follow_the_contract_writes_nothing(self):
        def other_limit(state):
            by_id(state["inputs"], "ja-choice-0005")["max_new_tokens"] = 9

        cases = (("another stop token", lambda state: generate(state, "ja-reading-0003", [11], stop=7),
                  "ja-reading-0003: the output stopped on a token that is not a stop token"),
                 ("a stop token inside", lambda state: generate(state, "ja-reading-0003", [11, STOP, 12]),
                  "ja-reading-0003: the output holds a stop token before its end"),
                 ("a limit below max_new_tokens", lambda state: generate(state, "ja-choice-0004", [301, 302, 303], stop=None),
                  "ja-choice-0004: the output length does not fit its stop and max_new_tokens"),
                 ("a stop at max_new_tokens", lambda state: generate(state, "ja-choice-0004", range(20, 28)),
                  "ja-choice-0004: the output length does not fit its stop and max_new_tokens"),
                 ("another max_new_tokens", other_limit,
                  "ja-choice-0005: the input record has another max_new_tokens than its category"))
        for name, change, text in cases:
            with self.subTest(name):
                state = fixture()
                change(state)
                self.refused(state, text)

    def test_a_long_context_input_whose_length_differs_from_the_set_writes_nothing(self):
        state = fixture()
        source = by_id(state["inputs"], "long-ja-4k-0001")
        source["input_ids"].append(7)
        source["input_sha256"] = digest(source["input_ids"])
        by_id(state["outputs"], "long-ja-4k-0001").update(input_sha256=source["input_sha256"],
                                                          input_tokens=len(source["input_ids"]))
        self.refused(state, "long-ja-4k-0001: the input length differs from the token_count of the set")

    def test_an_input_length_outside_the_bands_of_the_contract_writes_nothing(self):
        changed = copy.deepcopy(BUNDLE)
        changed["contract"]["layer1"]["bands"] = [[1, 4096]]
        self.refused(fixture(), "long-ja-16k-0001: the input length is outside the bands", bundle=changed)

    def test_an_item_without_a_generated_token_writes_nothing(self):
        state = fixture()
        generate(state, "synth-format-0002", [], stop=None)
        self.refused(state, "synth-format-0002: the output has no generated token")

    def test_records_with_repeated_keys_or_non_standard_numbers_write_nothing(self):
        def repeated(inputs, run):
            text = (run / "run.json").read_text()
            (run / "run.json").write_text(text[:-1] + ', "role": "fp16"}')

        def constant(inputs, run):
            text = (inputs / "inputs.json").read_text()
            (inputs / "inputs.json").write_text(text[:-1] + ', "extra": NaN}')

        for name, change, text in (("repeated key", repeated, "Duplicate JSON key"),
                                   ("NaN", constant, "Not a JSON value: NaN")):
            with self.subTest(name):
                self.refused(fixture(), text, change)

    def test_a_source_that_is_not_a_regular_file_stops_without_waiting(self):
        names = (("inputs", "inputs.jsonl"), ("inputs", "inputs.json"), ("run", "run.json"), ("run", "progress.json"),
                 ("run", "outputs.jsonl"))
        for folder, name in names:
            for kind, error, text in (("fifo", ValueError, f"{name} is not a regular file"),
                                      ("folder", ValueError, f"{name} is not a regular file"),
                                      ("missing", FileNotFoundError, name), ("broken link", FileNotFoundError, name)):
                with self.subTest(file=name, kind=kind):
                    inputs, run = save(fixture(), tempfile.mkdtemp(dir=self.root))
                    source = {"inputs": inputs, "run": run}[folder] / name
                    replaced(source, kind)
                    output = Path(tempfile.mkdtemp(dir=self.root)) / "forced"
                    thread, outcome = bounded(lambda: layer1.write(BUNDLE, inputs, run, output))
                    self.assertFalse(release(source, thread), "the forced input waited on the source")
                    self.assertIsInstance(outcome.get("error"), error)
                    self.assertIn(text, str(outcome["error"]))
                    if error is ValueError:
                        self.assertNotIn(str(source.parent), str(outcome["error"]))
                    self.assertFalse(os.path.lexists(output))

    def test_an_existing_output_or_link_is_refused_before_anything_is_read(self):
        missing = self.root / "missing"
        for name, make in (("folder", lambda path: path.mkdir()),
                           ("dangling link", lambda path: path.symlink_to(self.root / "nowhere"))):
            with self.subTest(name):
                output = self.root / name.replace(" ", "-")
                make(output)
                with self.assertRaises(FileExistsError):
                    layer1.write(BUNDLE, missing, missing, output)
                self.assertTrue(os.path.lexists(output))
                self.assertFalse(os.path.exists(missing))

if __name__ == "__main__":
    unittest.main()
