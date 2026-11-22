import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    import transformers
except ImportError:
    torch = transformers = None

from Tools.Conversion.adapters.gemma4_31b import baseline, reference
from Tools.Conversion.adapters.gemma4_31b.inputs import line, tokens_sha256
from Tools.Conversion.storage import read_json, write_json
from Tools.LLMEvaluation import scoring

STOPS = {1, 106, 50}
TEXT = {"vocab_size": 128, "vocab_size_per_layer_input": 128, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 6, "layer_types": ["sliding_attention"] * 5 + ["full_attention"], "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 8, "num_global_key_value_heads": 1, "global_head_dim": 16,
        "sliding_window": 8, "attention_k_eq_v": True, "final_logit_softcapping": 30.0, "hidden_size_per_layer_input": 0,
        "max_position_embeddings": 4096,
        "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0,
                                               "rope_type": "proportional"},
                            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}}
VISION = {"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
          "num_key_value_heads": 2, "head_dim": 8, "global_head_dim": 8}
MODEL = {"model_id": "google/gemma-4-31B-it", "revision": "a" * 40,
         "files": {"config.json": {"bytes": 1, "sha256": "b" * 64},
                   "model-00001-of-00002.safetensors": {"bytes": 2, "sha256": "e" * 64},
                   "model-00002-of-00002.safetensors": {"bytes": 3, "sha256": "f" * 64}}}
VERSIONS = {"python": "3.11.9", "packages": {"torch": "2.15.0"}, "os_build": "26A428", "platform": "macOS"}
MEMORY = {"max_rss_bytes": 10, "footprint_bytes": 5, "lifetime_peak_footprint_bytes": 12,
          "mps_driver_allocated_bytes": 8, "mps_current_allocated_bytes": 7}
OBSERVED = {"samples": 3, "interval_s": 5, "aborts": 0, "failure": None, "unreadable_samples": 0,
            "first": {"swap_used_bytes": 5}, "peak": {"swap_used_bytes": 5, "rss_bytes": 11},
            "lowest": {"free_percent": 40, "free_bytes": {}}}
RECIPE = Path(__file__).resolve().parents[1] / "recipes/gemma4_31b"
CONTRACT = Path(__file__).resolve().parents[2] / "LLMEvaluation/contract.json"


def item(identifier, ids, limit=8):
    return {"id": identifier, "input_ids": ids, "input_sha256": tokens_sha256(ids), "max_new_tokens": limit}


def conditions(**changes):
    run = baseline.conditions({"version": "G31-QC-v1"}, "c" * 64, "d" * 64, 4, MODEL, "c0ffee", VERSIONS)
    run.update(changes)
    return run


def inputs_of(count):
    return [item(f"ja-choice-{number:04d}", [2, 105, number] * number) for number in range(1, count + 1)]


def produce(calls):
    def made(entry):
        calls.append(entry["id"])
        steps = len(entry["input_ids"]) % 3 + 1
        return ({"id": entry["id"], "input_sha256": entry["input_sha256"], "input_tokens": len(entry["input_ids"]),
                 "output_ids": [7] * steps, "stop": "stop_token", "stop_token": 1,
                 "top2": [[7, 0.5, 8, 0.25]] * steps + [[1, 0.5, 7, 0.25]], "non_finite": 0}, {"seconds": 0.0})

    return made


def session(folder, entries, calls=None, run=None, deadline=None, made=None):
    stored, done, hasher = baseline.start(folder, run or conditions(), entries)
    with contextlib.redirect_stdout(io.StringIO()):
        hasher = baseline.run_items(folder, stored, entries, done, hasher, made or produce([] if calls is None else calls),
                                    deadline)
    baseline.finish(folder, stored, hasher)
    return stored


def contents(folder):
    return {path.name: path.read_bytes() for path in sorted(Path(folder).iterdir())}


def complete_lines(data):
    return [part + b"\n" for part in data.split(b"\n")[:-1]]


def digest(data):
    return hashlib.sha256(data).hexdigest()


class FakeModel:
    config = None

    def __init__(self, rows):
        self.rows, self.calls = list(rows), []

    def __call__(self, input_ids, past_key_values, use_cache, logits_to_keep):
        self.calls.append(input_ids.shape[1])
        row = self.rows.pop(0)
        if isinstance(row, Exception):
            raise row
        return SimpleNamespace(logits=torch.tensor([[row]], dtype=torch.bfloat16))


def softmax(row, index):
    top = max(row)
    return math.exp(row[index] - top) / sum(math.exp(value - top) for value in row)


@unittest.skipIf(np is None or torch is None, "NumPy and PyTorch are required for the generation")
class GenerationTests(unittest.TestCase):
    def generate(self, rows, ids=(2, 5, 6), limit=8, chunk=baseline.CHUNK):
        model = FakeModel(rows)
        record, _ = baseline.generate(model, item("ja-choice-0001", list(ids), limit), {1}, "cpu", chunk)
        return record, model.calls

    def test_the_record_has_the_contract_form_and_each_top2_leads_with_the_generated_token_with_ties_to_the_lower_id(self):
        rows = [[0, 0, 0, 2, 0, 2, 1, 0], [0, 0, 0, 0, 1.5, 0, 2.5, 0], [0, 3, 0, 1, 0, 1, 0, 0]]
        record, calls = self.generate(rows)
        self.assertEqual({key: record[key] for key in record if key != "top2"},
                         {"id": "ja-choice-0001", "input_sha256": tokens_sha256([2, 5, 6]), "input_tokens": 3,
                          "output_ids": [3, 6], "stop": "stop_token", "stop_token": 1, "non_finite": 0})
        self.assertEqual([[entry[0], entry[2]] for entry in record["top2"]], [[3, 5], [6, 4], [1, 3]])
        for entry, row in zip(record["top2"], rows):
            self.assertAlmostEqual(entry[1], softmax(row, entry[0]), places=12)
            self.assertAlmostEqual(entry[3], softmax(row, entry[2]), places=12)
        self.assertEqual(record["top2"][0][1], record["top2"][0][3])
        self.assertTrue(scoring.well_formed(record, 8))
        self.assertEqual([entry[0] for entry in record["top2"]], scoring.generated(record))
        self.assertEqual(calls, [3, 1, 1])
        self.assertEqual(json.loads(line(record)), record)

    def test_an_item_that_reaches_its_limit_ends_on_limit_with_one_top2_entry_per_output_token(self):
        record, calls = self.generate([[0, 0, 0, 0, 2, 0, 1, 0]] * 3, limit=3)
        self.assertEqual((record["output_ids"], record["stop"], "stop_token" in record, len(record["top2"])),
                         ([4, 4, 4], "limit", False, 3))
        self.assertEqual(calls, [3, 1, 1])
        self.assertTrue(scoring.well_formed(record, 8))
        self.assertEqual([entry[0] for entry in record["top2"]], scoring.generated(record))

    def test_a_stop_token_at_the_last_allowed_step_ends_on_the_stop_token(self):
        record, _ = self.generate([[0, 0, 0, 0, 2, 0, 1, 0], [0, 3, 0, 0, 2, 0, 1, 0]], limit=2)
        self.assertEqual((record["output_ids"], record["stop"], record["stop_token"], len(record["top2"])),
                         ([4], "stop_token", 1, 2))

    def test_non_finite_logits_or_a_runtime_error_end_the_item_with_an_error_and_the_tokens_so_far(self):
        first = [0, 0, 0, 0, 2, 0, 1, 0]
        cases = {"non-finite": ([first, [math.nan, 0, 0, 0, 1, 0, 0, math.inf]], 2),
                 "runtime error": ([first, RuntimeError("out of memory")], 0)}
        for label, (rows, count) in cases.items():
            with self.subTest(label=label):
                record, _ = self.generate(rows)
                self.assertEqual((record["output_ids"], record["stop"], len(record["top2"]), record["non_finite"]),
                                 ([4], "limit", 1, count))
                self.assertIsInstance(record["error"], str)
                self.assertTrue(scoring.well_formed(record, 8))
        self.assertEqual(self.generate(cases["runtime error"][0])[0]["error"], "RuntimeError: out of memory")

    def test_a_failure_while_generating_or_after_it_is_written_as_the_first_error_and_stops_the_run(self):
        first, stop = [0, 0, 0, 0, 2, 0, 1, 0], [0, 3, 0, 0, 2, 0, 1, 0]
        cases = {"the model, then the release": ([first, RuntimeError("out of memory")], "release",
                                                  [None, RuntimeError("synchronize failed")], "RuntimeError: out of memory"),
                 "the model, then the memory record": ([first, RuntimeError("out of memory")], "memory",
                                                        RuntimeError("memory failed"), "RuntimeError: out of memory"),
                 "only the memory record after a stop token": ([first, stop], "memory", RuntimeError("memory failed"),
                                                               "RuntimeError: memory failed")}
        for label, (rows, target, effect, message) in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as base, \
                    mock.patch.object(baseline, target, side_effect=effect):
                folder, calls = Path(base) / "run", []

                def made(entry):
                    calls.append(entry["id"])
                    return baseline.generate(FakeModel(rows), entry, {1}, "cpu")

                with self.assertRaisesRegex(RuntimeError, "ended with an error"):
                    session(folder, inputs_of(2), made=made, run=conditions(inputs={"sha256": "d" * 64, "count": 2}))
                lines = complete_lines((folder / "outputs.jsonl").read_bytes())
                record = json.loads(lines[0])
                self.assertEqual((calls, len(lines), record["error"], record["output_ids"]),
                                 (["ja-choice-0001"], 1, message, [4]))
                self.assertTrue(scoring.well_formed(record, 8))
                self.assertNotIn("finished", read_json(folder / "run.json"))

    def test_inputs_longer_than_the_chunk_are_read_in_chunks_with_at_least_one_token_in_the_last(self):
        row = [0, 3, 0, 0, 0, 0, 0, 0]
        for length, expected in ((3, [3]), (4, [4]), (8, [4, 4]), (9, [4, 4, 1])):
            with self.subTest(length=length):
                _, calls = self.generate([row] * len(expected), ids=[2] * length, chunk=4)
                self.assertEqual(calls, expected)


def tiny(seed=0):
    torch.manual_seed(seed)
    config = transformers.Gemma4Config(text_config=copy.deepcopy(TEXT), vision_config=copy.deepcopy(VISION))
    return transformers.Gemma4ForConditionalGeneration(config).eval()


def prompt(length, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return [2] + [int(token) for token in torch.randint(10, TEXT["vocab_size"], (length - 1,), generator=generator)]


@unittest.skipIf(np is None or torch is None or not hasattr(transformers, "Gemma4ForConditionalGeneration"),
                 "NumPy, PyTorch and a Transformers with Gemma 4 are required for the small model")
class SmallModelTests(unittest.TestCase):
    def run_model(self, model, ids, chunk, limit=12):
        with torch.no_grad():
            return baseline.generate(model, item("ja-choice-0001", ids, limit), STOPS, "cpu", chunk)[0]

    def test_reading_in_chunks_gives_the_tokens_and_probabilities_of_a_single_read(self):
        model = tiny()
        for length in (41, 40, 8, 5):
            with self.subTest(length=length):
                ids = prompt(length)
                chunked, single = self.run_model(model, ids, 8), self.run_model(model, ids, 4096)
                self.assertEqual(chunked["output_ids"], single["output_ids"])
                self.assertEqual(chunked["stop"], single["stop"])
                self.assertEqual([[e[0], e[2]] for e in chunked["top2"]], [[e[0], e[2]] for e in single["top2"]])
                for left, right in zip(chunked["top2"], single["top2"]):
                    self.assertAlmostEqual(left[1], right[1], delta=1e-6)
                    self.assertAlmostEqual(left[3], right[3], delta=1e-6)

    def test_the_greedy_tokens_equal_those_of_generate_on_inputs_without_ties(self):
        model = tiny()
        for length in (41, 9):
            with self.subTest(length=length):
                ids = prompt(length, seed=length)
                record = self.run_model(model, ids, 8, limit=16)
                self.assertTrue(all(entry[1] != entry[3] for entry in record["top2"]))
                with torch.no_grad():
                    sequence = model.generate(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids),
                                                                                                       dtype=torch.long),
                                              do_sample=False, num_beams=1, max_new_tokens=16, eos_token_id=sorted(STOPS))
                self.assertEqual(scoring.generated(record), sequence[0, len(ids):].tolist())
                self.assertTrue(scoring.well_formed(record, TEXT["vocab_size"]))
                self.assertEqual([entry[0] for entry in record["top2"]], scoring.generated(record))

    def test_exactly_tied_logits_on_the_small_model_pick_the_lower_id(self):
        model = tiny()
        ids = prompt(20)
        first = self.run_model(model, ids, 8)["output_ids"][0]
        lower = next(token for token in range(3, first) if token not in ids and token not in STOPS)
        with torch.no_grad():
            weights = model.get_input_embeddings().weight
            weights[lower] = weights[first]
        record = self.run_model(model, ids, 8)
        self.assertEqual(record["top2"][0][0], lower)
        self.assertEqual(record["top2"][0][2], first)
        self.assertEqual(record["top2"][0][1], record["top2"][0][3])
        self.assertEqual(record["output_ids"][0], lower)

    def test_each_output_record_carries_the_input_sha256_and_length_of_its_input_record(self):
        model = tiny()
        entries = [item("ja-choice-0001", prompt(9)), item("long-ja-4k-0001", prompt(30, seed=3), limit=4)]
        read = baseline.read_inputs(b"".join(line(entry) for entry in entries), TEXT["vocab_size"])
        self.assertEqual(read, entries)
        for entry in read:
            record = self.run_model(model, entry["input_ids"], 8, limit=entry["max_new_tokens"])
            self.assertEqual((record["input_sha256"], record["input_tokens"]),
                             (entry["input_sha256"], len(entry["input_ids"])))


class InputRecordTests(unittest.TestCase):
    def test_input_records_with_another_sha256_or_values_out_of_range_are_rejected(self):
        good = item("ja-choice-0001", [2, 105, 7])
        variants = {"another sha256": {**good, "input_sha256": tokens_sha256([2, 105, 8])},
                    "an id at the vocabulary size": item("ja-choice-0001", [2, 105, 128]),
                    "a negative id": item("ja-choice-0001", [2, -1]), "a boolean id": {**good, "input_ids": [2, True]},
                    "a float id": {**good, "input_ids": [2, 105.0]}, "no ids": item("ja-choice-0001", []),
                    "a limit of zero": {**good, "max_new_tokens": 0}, "a boolean limit": {**good, "max_new_tokens": True},
                    "a float limit": {**good, "max_new_tokens": 8.0}, "an id that is not a string": {**good, "id": 1}}
        for label, value in variants.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                baseline.read_inputs(line(value), 128)
        texts = {"a repeated id": line(good) + line(good), "a repeated key": b'{"id": "a", "id": "b"}\n',
                 "NaN": line(good).replace(b'"max_new_tokens":8', b'"max_new_tokens":NaN'),
                 "a blank line": line(good) + b"\n" + line(item("ja-choice-0002", [2])), "not an object": b"[1]\n"}
        for label, text in texts.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                baseline.read_inputs(text, 128)
        self.assertEqual(baseline.read_inputs(line(good), 128), [good])


class ReadyMonitor:
    def __init__(self, path, volumes, patterns, on_abort):
        self.ready = SimpleNamespace(wait=lambda timeout: True)

    def start(self):
        pass

    def finish(self):
        return copy.deepcopy(OBSERVED)


@unittest.skipIf(torch is None, "PyTorch is required for the command")
class SessionTests(unittest.TestCase):
    def run_main(self, folder, failing=None):
        folder = Path(folder)
        (folder / "prepare").mkdir()
        write_json(folder / "prepare/ready.json",
                   {"ready": True, "contract_sha256": digest((RECIPE / "model.json").read_bytes())})
        (folder / "inputs.jsonl").write_bytes(b"".join(map(line, inputs_of(2))))
        argv = ["baseline", "--model", str(RECIPE / "model.json"), "--snapshot", str(folder), "--prepare",
                str(folder / "prepare"), "--contract", str(CONTRACT), "--inputs", str(folder / "inputs.jsonl"), "--commit",
                "c0ffee", "--output", str(folder / "run"), "--log", str(folder / "log")]
        loading = {"info": {}, "parameter_dtypes": ["torch.bfloat16"], "parameter_devices": ["mps:0"], "training": False,
                   "after_load": dict(MEMORY)}

        def writing(path, value):
            if Path(path).name == failing:
                raise OSError("No space left on device")
            return write_json(path, value)

        def made(model, entry, stops, device):
            record, info = produce([])(entry)
            return record, {**info, **MEMORY}

        patches = [mock.patch.object(sys, "argv", argv), mock.patch.object(reference, "ReferenceMonitor", ReadyMonitor),
                   mock.patch.object(baseline, "environment", return_value=VERSIONS),
                   mock.patch.object(baseline, "verify_inputs"),
                   mock.patch.object(baseline, "load_model", return_value=(SimpleNamespace(eval=lambda: None), {})),
                   mock.patch.object(baseline, "loading_record", return_value=loading),
                   mock.patch.object(baseline, "memory", return_value=dict(MEMORY)),
                   mock.patch.object(baseline, "generate", side_effect=made),
                   mock.patch.object(baseline, "write_json", side_effect=writing)]
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            return baseline.main()

    def test_a_session_writes_its_record_and_then_the_finish_of_the_run(self):
        with tempfile.TemporaryDirectory() as folder:
            code = self.run_main(folder)
            run, session_record = read_json(Path(folder) / "run/run.json"), read_json(Path(folder) / "log/session.json")
        self.assertEqual((code, run["items"], "finished" in run, session_record["passed"]), (0, 2, True, True))

    def test_a_session_record_that_cannot_be_written_leaves_the_run_without_its_finish(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(OSError):
                self.run_main(folder, failing="session.json")
            run = read_json(Path(folder) / "run/run.json")
            lines = complete_lines((Path(folder) / "run/outputs.jsonl").read_bytes())
            self.assertFalse((Path(folder) / "log/session.json").exists())
        self.assertEqual((run["items"], "finished" in run, len(lines)), (0, False, 2))


class RunFolderTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.base, True)

    def full(self, name, entries, **options):
        session(self.base / name, entries, **options)
        return (self.base / name / "outputs.jsonl").read_bytes()

    def rejected(self, folder, entries, message, run=None):
        before, calls = contents(folder), []
        with self.assertRaises(ValueError) as caught:
            session(folder, entries, calls, run=run)
        self.assertEqual(calls, [])
        self.assertEqual(contents(folder), before)
        self.assertIn(message, str(caught.exception))

    def test_a_new_run_writes_the_three_files_and_ends_with_every_item_and_its_finish(self):
        entries, calls = inputs_of(4), []
        run = session(self.base / "run", entries, calls)
        stored = read_json(self.base / "run" / "run.json")
        outputs = (self.base / "run" / "outputs.jsonl").read_bytes()
        self.assertEqual(sorted(contents(self.base / "run")), ["outputs.jsonl", "progress.json", "run.json"])
        self.assertEqual(calls, [entry["id"] for entry in entries])
        self.assertEqual((stored, stored["items"], "finished" in stored), (run, 4, True))
        self.assertEqual(read_json(self.base / "run" / "progress.json"),
                         {"lines": 4, "sha256": digest(outputs), "run": baseline.identity(stored)})
        self.assertEqual({key: stored[key] for key in conditions()}, conditions())

    def test_a_resumed_run_skips_the_written_items_and_ends_with_the_outputs_of_an_uninterrupted_run(self):
        entries = inputs_of(4)
        expected = self.full("full", entries)
        folder, calls = self.base / "run", []
        first = session(folder, entries, calls, deadline=0)
        self.assertEqual((calls, first["items"], "finished" in read_json(folder / "run.json")), (["ja-choice-0001"], 1, True))
        calls.clear()
        resumed = session(folder, entries, calls)
        self.assertEqual(calls, [entry["id"] for entry in entries[1:]])
        self.assertEqual((folder / "outputs.jsonl").read_bytes(), expected)
        self.assertEqual((resumed["id"], resumed["started"], resumed["items"]), (first["id"], first["started"], 4))

    def test_an_item_with_an_error_is_written_and_then_stops_the_run(self):
        entries, folder, calls = inputs_of(4), self.base / "run", []
        made = produce(calls)

        def failing(entry):
            record, info = made(entry)
            return ({**record, "stop": "limit", "error": "RuntimeError: out of memory"}
                    if entry["id"] == "ja-choice-0002" else record), info

        with self.assertRaises(RuntimeError):
            session(folder, entries, made=failing)
        lines = complete_lines((folder / "outputs.jsonl").read_bytes())
        self.assertEqual(calls, ["ja-choice-0001", "ja-choice-0002"])
        self.assertEqual((len(lines), json.loads(lines[1])["error"]), (2, "RuntimeError: out of memory"))
        self.assertNotIn("finished", read_json(folder / "run.json"))

    def test_items_of_the_long_context_categories_run_last_in_input_order(self):
        names = ["long-ja-4k-0001", "ja-choice-0001", "synth-long-16k-0001", "code-fix-0001", "long-ja-16k-0001",
                 "synth-long-4k-0001", "free-ja-0001"]
        entries, calls = [item(name, [2, 105]) for name in names], []
        session(self.base / "run", entries, calls, run=conditions(inputs={"sha256": "d" * 64, "count": 7}))
        self.assertEqual(calls, ["ja-choice-0001", "code-fix-0001", "free-ja-0001", "long-ja-4k-0001",
                                 "synth-long-16k-0001", "long-ja-16k-0001", "synth-long-4k-0001"])
        self.assertEqual(read_json(self.base / "run" / "run.json")["order"], {"last": list(baseline.LAST)})

    def test_a_run_stopped_after_announcing_a_line_before_or_while_writing_it_writes_that_item_once(self):
        entries = inputs_of(3)
        expected = self.full("full", entries, run=conditions(inputs={"sha256": "d" * 64, "count": 3}))
        lines = complete_lines(expected)
        for cut in (False, True):
            with self.subTest(cut=cut):
                folder, calls = self.base / f"run-{cut}", []
                run = conditions(inputs={"sha256": "d" * 64, "count": 3})
                stored, _, _ = baseline.start(folder, run, entries)
                (folder / "outputs.jsonl").write_bytes(lines[0] + lines[1] + (lines[2][:20] if cut else b""))
                write_json(folder / "progress.json", {"lines": 2, "sha256": digest(lines[0] + lines[1]),
                                                      "next_sha256": digest(expected), "run": baseline.identity(stored)})
                session(folder, entries, calls, run=run)
                self.assertEqual(calls, ["ja-choice-0003"])
                self.assertEqual((folder / "outputs.jsonl").read_bytes(), expected)

    def test_a_run_that_cannot_replace_progress_json_before_a_line_stops_before_writing_it_and_resumes_once(self):
        entries = inputs_of(4)
        expected = self.full("full", entries)
        folder, calls = self.base / "run", []
        made = produce(calls)

        def failing(entry):
            if entry["id"] == "ja-choice-0003":
                folder.chmod(0o555)
            return made(entry)

        self.addCleanup(folder.chmod, 0o755)
        with self.assertRaises(PermissionError):
            session(folder, entries, made=failing)
        folder.chmod(0o755)
        self.assertEqual(len(complete_lines((folder / "outputs.jsonl").read_bytes())), 2)
        calls.clear()
        session(folder, entries, calls)
        self.assertEqual(calls, ["ja-choice-0003", "ja-choice-0004"])
        self.assertEqual((folder / "outputs.jsonl").read_bytes(), expected)

    def test_changed_lines_or_progress_and_files_of_another_run_stop_the_resume_without_removing_anything(self):
        entries = inputs_of(3)
        run = conditions(inputs={"sha256": "d" * 64, "count": 3})
        neither = "which match neither"

        def lines_of(folder):
            return complete_lines((folder / "outputs.jsonl").read_bytes())

        def another(folder, **fields):
            other = self.base / f"other-{len(list(self.base.iterdir()))}"
            session(other, entries, run={**run, **fields})
            return other

        def copy_from(folder, other, names):
            for name in names:
                shutil.copyfile(other / name, folder / name)

        changes = {
            "a changed byte": (lambda f: (f / "outputs.jsonl").write_bytes(
                (f / "outputs.jsonl").read_bytes().replace(b'"output_ids":[7', b'"output_ids":[8', 1)), neither),
            "a removed line": (lambda f: (f / "outputs.jsonl").write_bytes(b"".join(lines_of(f)[:2])), "fewer than the 3"),
            "swapped lines": (lambda f: (f / "outputs.jsonl").write_bytes(b"".join([lines_of(f)[i] for i in (1, 0, 2)])),
                              neither),
            "an older progress.json": (lambda f: write_json(f / "progress.json", {
                "lines": 1, "sha256": digest(lines_of(f)[0]), "next_sha256": digest(b"".join(lines_of(f)[:2])),
                "run": read_json(f / "progress.json")["run"]}), neither),
            "a line without an announcement": (lambda f: (f / "outputs.jsonl").write_bytes(
                b"".join(lines_of(f) + lines_of(f)[:1])), neither),
            "a line other than the announced one": (lambda f: write_json(f / "progress.json", {
                "lines": 2, "sha256": digest(b"".join(lines_of(f)[:2])),
                "next_sha256": digest(b"".join(lines_of(f)[:2] + lines_of(f)[:1])),
                "run": read_json(f / "progress.json")["run"]}), neither),
            "files of a run with other conditions": (lambda f: copy_from(f, another(f, role="anchor"),
                                                                         ["outputs.jsonl", "progress.json"]),
                                                     "progress.json belongs to another run"),
            "files of another run with the same conditions": (lambda f: copy_from(f, another(f),
                                                                                  ["outputs.jsonl", "progress.json"]),
                                                              "progress.json belongs to another run"),
            "run.json of another run with the same conditions": (lambda f: copy_from(f, another(f), ["run.json"]),
                                                                 "progress.json belongs to another run"),
            "a missing progress.json": (lambda f: (f / "progress.json").unlink(), "progress.json is missing"),
            "a run.json without an id": (lambda f: write_json(f / "run.json", {
                key: value for key, value in read_json(f / "run.json").items() if key != "id"}), "no run id"),
        }
        for label, (change, message) in changes.items():
            with self.subTest(label=label):
                folder = self.base / label.replace(" ", "-")
                session(folder, entries, run=run)
                change(folder)
                self.rejected(folder, entries, message, run=run)

    def test_outputs_and_progress_put_back_together_to_an_earlier_point_stop_the_resume(self):
        entries, folder = inputs_of(4), self.base / "run"
        session(folder, entries, deadline=0)
        session(folder, entries, deadline=0)
        earlier = {name: (folder / name).read_bytes() for name in ("outputs.jsonl", "progress.json")}
        session(folder, entries)
        for name, data in earlier.items():
            (folder / name).write_bytes(data)
        self.rejected(folder, entries, "outputs.jsonl has 2 complete lines, fewer than the 4 finished items")

    def test_counts_out_of_range_in_progress_json_or_run_json_stop_the_resume_without_removing_anything(self):
        entries = inputs_of(3)
        run = conditions(inputs={"sha256": "d" * 64, "count": 3})
        for field, value in (("lines", 2 ** 63), ("lines", -1), ("lines", True), ("items", -1), ("items", 4),
                             ("items", True)):
            with self.subTest(field=field, value=value):
                folder = self.base / f"{field}-{value}"
                stored, _, _ = baseline.start(folder, run, entries)
                if field == "lines":
                    write_json(folder / "progress.json", {"lines": value, "sha256": digest(b""),
                                                          "next_sha256": digest(b""), "run": baseline.identity(stored)})
                    message = "which match neither"
                else:
                    write_json(folder / "run.json", {**stored, "items": value})
                    message = "finished items, not a count from 0 to 3"
                self.rejected(folder, entries, message, run=run)

    def test_written_lines_with_an_id_outside_the_inputs_or_an_id_twice_stop_the_resume(self):
        entries = inputs_of(3)
        run = conditions(inputs={"sha256": "d" * 64, "count": 3})
        for repeated in (False, True):
            with self.subTest(repeated=repeated):
                folder = self.base / f"ids-{repeated}"
                session(folder, entries, run=run, deadline=0)
                written = (folder / "outputs.jsonl").read_bytes()
                changed = written + (written if repeated else written.replace(b"ja-choice-0001", b"ja-choice-0009"))
                (folder / "outputs.jsonl").write_bytes(changed)
                write_json(folder / "progress.json", {"lines": 2, "sha256": digest(changed),
                                                      "run": read_json(folder / "progress.json")["run"]})
                self.rejected(folder, entries, "repeats an id" if repeated else "has no id of the input records",
                              run=run)

    def test_a_run_folder_with_other_conditions_is_rejected_without_generating(self):
        entries = inputs_of(3)
        run = conditions(inputs={"sha256": "d" * 64, "count": 3})
        others = {"contract": {"version": "G31-QC-v1", "sha256": "0" * 64}, "set": "reserve", "role": "anchor",
                  "asset": {**run["asset"], "revision": "0" * 40},
                  "inputs": {"sha256": "0" * 64, "count": 3},
                  "implementation": {**run["implementation"], "commit": "0ther"},
                  "call_shape": {"prefill_chunk": 512}, "order": {"last": []}}
        for field, value in others.items():
            with self.subTest(field=field):
                folder = self.base / field
                session(folder, entries, run=run, deadline=0)
                self.rejected(folder, entries, f"differ from its run.json: {field}", run={**run, field: value})
        for field, value in (("weights", {"model.safetensors": "0" * 64}), ("environment", {"python": "3.12.0"})):
            with self.subTest(field=field):
                folder = self.base / field
                session(folder, entries, run=run, deadline=0)
                key = "asset" if field == "weights" else "implementation"
                self.rejected(folder, entries, f"differ from its run.json: {key}",
                              run={**run, key: {**run[key], field: value}})

    def test_a_folder_without_run_json_starts_only_when_empty_or_left_by_a_start_that_stopped_before_run_json(self):
        empty = json.dumps({"lines": 0, "sha256": digest(b""), "run": "x"})
        variants = [({}, True), ({"outputs.jsonl": ""}, True), ({"outputs.jsonl": "", "progress.json": empty}, True),
                    ({"outputs.jsonl": "{}\n"}, False), ({"notes.txt": ""}, False),
                    ({"outputs.jsonl": "", "notes.txt": ""}, False), ({"outputs.jsonl": "", "progress.json": "{}"}, False),
                    ({"outputs.jsonl": "", "progress.json": empty.replace('"lines": 0', '"lines": false')}, False),
                    ({"progress.json": empty}, False)]
        entries = inputs_of(2)
        run = conditions(inputs={"sha256": "d" * 64, "count": 2})
        for index, (files, starts) in enumerate(variants):
            with self.subTest(files=files):
                folder = self.base / f"folder-{index}"
                folder.mkdir()
                for name, text in files.items():
                    (folder / name).write_text(text)
                if starts:
                    calls = []
                    session(folder, entries, calls, run=run)
                    self.assertEqual(calls, [entry["id"] for entry in entries])
                else:
                    self.rejected(folder, entries, "not a run folder", run=run)


if __name__ == "__main__":
    unittest.main()
