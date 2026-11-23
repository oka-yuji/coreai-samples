import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

from Tools.Conversion.adapters.gemma4_31b import export, layers, native
from Tools.Conversion.adapters.gemma4_31b.prepare import TOKENIZER_FILES, load_contract
from Tools.Conversion.adapters.pinned import sha256
from Tools.Conversion.storage import digest, read_json, write_json
from Tools.Conversion.tests.test_gemma4_31b_layers import (BUNDLE_DEFECTS, MAIN_ASSET, TEXT, complete_export,
                                                            write_contract)

ENGINE = Path(__file__).resolve().parents[3] / "CoreAISamplesKit/Sources/CoreAIBackend/Engine.swift"
LOGITS = Path(__file__).resolve().parents[3] / "Tools/Verify/Sources/coreai-verify/PositionLogits.swift"
VOCABULARY = TEXT["vocab_size"]
MAIN_HASH = hashlib.sha256(MAIN_ASSET).hexdigest()
FAKE = """#!{python}
import hashlib, json, sys, time
from pathlib import Path
import numpy as np

plan = json.loads(Path(__file__).with_suffix(".json").read_text())
arguments = dict(zip(sys.argv[2::2], sys.argv[3::2]))
ids = json.loads(Path(arguments["--tokens"]).read_text())
start, chunk = int(arguments["--start"]), int(arguments["--chunk"])
name = next(key for key, value in plan["conditions"].items() if value == [len(ids), start, chunk])
change = plan["changes"].get(name, {{}})
with Path(plan["calls"]).open("a") as calls:
    calls.write(name + "\\n")
time.sleep(change.get("sleep", 0))
source = "short" if name.startswith("short") else "context"
rows = plan["rows"][source]
values = np.load(plan["arrays"][source])[[rows.index(position) for position in range(start, len(ids))]]
values = values.astype(np.float16)
for row, column, value in change.get("set", []):
    values[row, column] = value
for row in change.get("constant", []):
    values[row] = 1.0
values = values[:len(values) + change.get("rows", 0)] if change.get("rows", 0) < 0 else np.concatenate(
    [values, values[:change.get("rows", 0)]])
output = Path(arguments["--output"])
if not change.get("no_output"):
    output.write_bytes(values.astype("<f2").tobytes())
if change.get("unreadable"):
    output.chmod(0)
prefill = plan["prefill_chunk"]
record = {{"record": "logits", "token_count": len(ids), "start": start, "chunk": chunk,
          "tokens_sha256": hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest(),
          "shape": [len(ids) - start, values.shape[1]], "dtype": "float16", "byte_order": "little",
          "prefix_chunk": prefill, "prefix_calls": -(-start // prefill),
          "main_calls": -(-(len(ids) - start) // chunk), "main_hash": plan["main_hash"], "cached": True,
          "non_finite_count": 0, "load_s": 0.1, "prefix_s": 0.0, "scoring_s": 0.2, "specialize_s": 0.0}}
if start:
    record["prefix_function"] = "prefill"
record.update(change.get("record", {{}}))
for key in change.get("drop", []):
    record.pop(key)
text = json.dumps(record)
for old, new in change.get("replace", []):
    text = text.replace(old, new)
print(change.get("stdout", text))
sys.exit(change.get("exit", 0))
"""


def bounded(call, seconds=5):
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


OPEN = io.open


@contextlib.contextmanager
def watched():
    reads = []

    def opener(file, *arguments, **options):
        handle = OPEN(file, *arguments, **options)
        status = os.fstat(handle.fileno())
        original = handle.read

        def read(*sizes):
            reads.append((status.st_dev, status.st_ino))
            return original(*sizes)

        handle.read = read
        return handle

    with mock.patch("builtins.open", new=opener), mock.patch("io.open", new=opener):
        yield reads


def inode(path):
    status = os.stat(path)
    return status.st_dev, status.st_ino


def model_record(contract):
    pinned = load_contract(contract)
    return {"id": pinned["model_id"], "revision": pinned["revision"], "contract_sha256": sha256(contract),
            "verified_before_loading": {name: entry for name, entry in pinned["files"].items()
                                        if name in TOKENIZER_FILES}}


def edit(path, change):
    record = read_json(path)
    change(record)
    write_json(path, record)


@unittest.skipIf(np is None, "NumPy is required for the comparison")
class MeasureTests(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(0)
        self.reference = generator.normal(size=(40, VOCABULARY)).astype(np.float32)
        self.reference[:, 0] += 20
        for row, first, second in ((11, 10.0, 9.0), (13, 10.0, 9.9921875)):
            self.reference[row, :2] = first, second
        self.reference[9, 50] = 1.5

    def measure(self, change=None, known=()):
        swift = self.reference.astype(np.float16)
        if change:
            change(swift)
        mask = np.zeros(len(swift), dtype=bool)
        mask[list(known)] = True
        return native.measure(swift, self.reference, np.arange(len(swift)), mask)

    def failing(self, result):
        return {name for name, passed in result["checks"].items() if not passed}

    def test_matching_rows_pass_and_each_criterion_fails_on_its_own(self):
        result = self.measure()
        self.assertEqual(self.failing(result), set())
        self.assertLess(result["max_abs_logit"], 0.01)
        self.assertEqual((result["rows"], result["non_finite"], result["constant_rows"]), (40, 0, 0))

        def at(row, column, value):
            return lambda swift: swift.__setitem__((row, column), value)

        for label, change in {"a NaN": at(3, 5, np.nan), "an infinity": at(4, 6, np.inf),
                              "a negative infinity": at(5, 7, -np.inf)}.items():
            with self.subTest(case=label):
                result = self.measure(change)
                self.assertIn("finite", self.failing(result))
                self.assertIsNone(result["max_abs_logit"])
        cases = {"a logit difference of 2.0": (at(9, 50, 3.5), {"max_abs_logit"}),
                 "a logit difference just below 2.0": (at(9, 50, 3.490234375), set()),
                 "another top-1 outside the near tie": (lambda swift: swift.__setitem__((11, slice(0, 2)), (9.0, 10.0)),
                                                        {"top1"})}
        for label, (change, failing) in cases.items():
            with self.subTest(case=label):
                self.assertEqual(self.failing(self.measure(change)), failing)

    def test_a_constant_row_fails(self):
        self.reference[15] = 1.0
        self.reference[15, 0] = 1.0001
        result = self.measure(lambda swift: swift.__setitem__(15, 1.0))
        self.assertEqual(result["constant_rows"], 1)
        self.assertEqual(self.failing(result), {"no_constant_rows"})

    def test_a_top_one_change_within_the_near_tie_is_counted_but_does_not_fail(self):
        result = self.measure(lambda swift: swift.__setitem__((13, slice(0, 2)), (9.9921875, 10.0)))
        self.assertEqual(self.failing(result), set())
        self.assertEqual(result["top1_mismatches"], {"near_tie": 1, "outside": 0})

    def test_only_the_rows_of_the_known_call_shape_are_left_out_and_their_difference_is_recorded(self):
        shifted = lambda swift: swift.__setitem__((20, 50), self.reference[20, 50] + 6.0)
        result = self.measure(shifted, known=(20, 21))
        self.assertEqual(self.failing(result), set())
        self.assertEqual(result["known"]["rows"], 2)
        self.assertGreater(result["known"]["max_abs_logit"], 5.9)
        self.assertLess(result["max_abs_logit"], 0.01)
        self.assertEqual(self.failing(self.measure(shifted, known=(21,))), {"max_abs_logit"})
        self.assertEqual(self.failing(self.measure(shifted)), {"max_abs_logit"})
        swapped = lambda swift: swift.__setitem__((11, slice(0, 2)), (9.0, 10.0))
        self.assertEqual(self.failing(self.measure(swapped, known=(11,))), set())
        nan = lambda swift: swift.__setitem__((20, 7), np.nan)
        self.assertIn("finite", self.failing(self.measure(nan, known=(20,))))


class KnownCallTests(unittest.TestCase):
    def test_only_calls_of_eight_or_more_tokens_that_end_at_97_to_112_positions_are_known(self):
        rows = {name: native.known_rows(start, length, chunk) for name, _, length, start, chunk in layers.CONDITIONS}
        self.assertEqual(rows["window-c8"], list(range(96, 112)))
        self.assertEqual({name for name, value in rows.items() if value}, {"window-c8"})
        self.assertEqual(native.known_rows(0, 120, 7), [])
        self.assertEqual(native.known_rows(97, 120, 8), list(range(97, 105)))
        self.assertEqual(native.known_rows(90, 200, 22), list(range(90, 112)))
        self.assertEqual(native.known_rows(0, 100, 100), list(range(100)))
        self.assertEqual(native.known_rows(0, 113, 113), [])

    def test_the_rule_is_the_one_the_engine_applies_to_gemma_4_31b(self):
        self.assertIn('"google/gemma-4-31B-it": (avoidedEnds: 97...112, shortestAffectedLength: 8)', ENGINE.read_text())
        self.assertEqual((native.KNOWN_ENDS, native.KNOWN_SHORTEST), (range(97, 113), 8))

    def test_the_reads_before_the_start_follow_the_engine_and_the_verifier(self):
        self.assertEqual(native.call_lengths(0, 100, 4096), [96, 4])
        self.assertEqual(native.call_lengths(0, 112, 4096), [96, 7, 7, 2])
        self.assertEqual(native.call_lengths(0, 8184, 4096), [4096, 4088])
        self.assertEqual(native.call_lengths(0, 16832, 4096), [4096, 4096, 4096, 4096, 448])
        self.assertEqual(native.prefix_record(0, 4096, True), {"prefix_calls": 0, "prefix_chunk": 4096})
        self.assertEqual(native.prefix_record(16832, 4096, True),
                         {"prefix_function": "prefill", "prefix_calls": 5, "prefix_chunk": 4096})
        self.assertEqual(native.prefix_record(16832, 4096, False)["prefix_function"], "main")
        self.assertEqual(native.prefix_record(100, 4096, True),
                         {"prefix_function": "main", "prefix_calls": 2, "prefix_chunk": 4096})
        self.assertEqual(native.prefix_record(1000, 2048, True),
                         {"prefix_function": "prefill", "prefix_calls": 1, "prefix_chunk": 2048})
        self.assertIn("return physicalMemory >= 64 << 30 ? 4096 : 2048", ENGINE.read_text())
        self.assertIn("let unsplit = lengths == pieces(0..<start, size: chunkSize).map(\\.count)", LOGITS.read_text())
        self.assertIn('return hasPrefillFunction && unsplit ? "prefill" : "main"', LOGITS.read_text())
        self.assertIn(native.prefill_chunk(), (2048, 4096))

    def test_the_field_check_takes_any_json_value_and_keeps_counts_within_64_bits(self):
        largest = 2**63 - 1
        self.assertEqual((native.valid("count", largest), native.valid("seconds", largest), native.valid("seconds", 0.25)),
                         (True, True, True))
        for kind in ("count", "seconds", "shape"):
            for value in (2**63, 10**400, -1, None, "1", True, {"a": 1}, [2**63], [10**400]):
                with self.subTest(kind=kind, value=repr(value)[:24]):
                    self.assertFalse(native.valid(kind, value))
        for value in (float("inf"), float("nan"), -0.5, 10**400):
            with self.subTest(seconds=repr(value)[:24]):
                self.assertFalse(native.valid("seconds", value))

    def test_the_prefill_chunk_is_4096_from_64_gib_of_memory_like_the_engine(self):
        for pages, chunk in ((2**22 - 1, 2048), (2**22, 4096), (2**23, 4096)):
            sizes = {"SC_PAGE_SIZE": 2**14, "SC_PHYS_PAGES": pages}
            with self.subTest(pages=pages), mock.patch.object(native.os, "sysconf", side_effect=sizes.__getitem__):
                self.assertEqual(native.prefill_chunk(), chunk)


@unittest.skipIf(np is None, "NumPy is required for the comparison")
class CompareTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        (self.folder / "snapshot").mkdir()
        for index, item in enumerate(TOKENIZER_FILES):
            (self.folder / "snapshot" / item).write_text("abc"[index])
        files = {item: {"bytes": 1, "sha256": sha256(self.folder / "snapshot" / item)} for item in TOKENIZER_FILES}
        self.contract = write_contract(self.folder, files)
        self.config = load_contract(self.contract)["config"]
        self.kinds = layers.check_layers(self.config, 6)
        generator = np.random.default_rng(1)
        self.reference = self.folder / "reference"
        self.reference.mkdir()
        inputs, self.arrays, self.rows = {}, {}, {}
        for name in ("short", "context"):
            ids, rows = layers.tokens(name, self.config), layers.kept_rows(name)
            array = generator.normal(size=(len(rows), VOCABULARY)).astype(np.float32)
            array[:, 0] += 20
            np.save(self.reference / f"{name}.npy", array)
            self.arrays[name], self.rows[name] = str(self.reference / f"{name}.npy"), rows
            inputs[name] = {"positions": len(ids), "tokens_sha256": digest(ids), "rows": rows,
                            "logits": layers.file_record(self.reference / f"{name}.npy")}
        write_json(self.reference / "record.json", {"schema": layers.CONTEXT_SCHEMA, "implementation": "original",
                                                     "layers": 6, "layer_types": self.kinds, "chunk": 1024,
                                                     "weights": layers.WEIGHTS, "model": model_record(self.contract),
                                                     "inputs": inputs})
        self.export = complete_export(self.folder / "export", self.contract)
        self.verifier = self.folder / "verifier"
        self.calls = self.folder / "calls.txt"

    def run_compare(self, changes=None, label="out"):
        plan = {"conditions": {name: [length, start, chunk] for name, _, length, start, chunk in layers.CONDITIONS},
                "changes": changes or {}, "arrays": self.arrays, "rows": self.rows, "main_hash": MAIN_HASH,
                "calls": str(self.calls), "prefill_chunk": native.prefill_chunk()}
        self.verifier.write_text(FAKE.format(python=sys.executable))
        self.verifier.chmod(0o755)
        self.verifier.with_suffix(".json").write_text(json.dumps(plan))
        return native.compare(self.export, self.reference, self.verifier, self.folder / label, self.contract)

    def conditions(self, result):
        return {entry["name"]: entry for entry in result["conditions"]}

    def test_matching_logits_pass_all_twelve_conditions_and_the_comparison_is_saved(self):
        result = self.run_compare()
        self.assertTrue(result["passed"], result["problems"])
        self.assertEqual(read_json(self.folder / "out/comparison.json"), result)
        conditions = self.conditions(result)
        self.assertEqual(list(conditions), [name for name, *_ in layers.CONDITIONS])
        self.assertEqual(conditions["window-c8"]["known"]["rows"], 16)
        self.assertEqual({name: entry["rows"] for name, entry in conditions.items()},
                         {name: length - start for name, _, length, start, _ in layers.CONDITIONS})
        for entry in conditions.values():
            data = (self.folder / "out" / f"{entry['name']}.f16").read_bytes()
            self.assertEqual(entry["output"], {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        self.assertEqual(json.loads((self.folder / "out/tail.tokens.json").read_text()),
                         layers.tokens("context", self.config))

    def test_each_condition_fails_on_its_own_defect_and_the_known_rows_are_left_out(self):
        changes = {"short-c26": {"rows": -1}, "prefix-c1": {"rows": 1}, "window-c1": {"exit": 1},
                   "window-c2": {"stdout": "not a record"}, "window-c4": {"record": {"tokens_sha256": "0" * 64}},
                   "window-c8": {"set": [[100, 5, 30.0], [111, 6, 30.0]]},
                   "capacity-2048": {"set": [[2, 3, float("nan")]]}, "capacity-4096": {"constant": [4]},
                   "capacity-8192": {"set": [[1, 7, 10.0]]}, "capacity-16384": {"set": [[0, 9, 40.0]]},
                   "tail": {"record": {"main_hash": "cd" * 32}}}
        result = self.run_compare(changes)
        self.assertFalse(result["passed"])
        conditions = self.conditions(result)
        passed = {name for name, entry in conditions.items() if entry["passed"]}
        self.assertEqual(passed, {"short-c1", "window-c8"})
        self.assertGreater(conditions["window-c8"]["known"]["max_abs_logit"], 5)
        for name in ("short-c26", "prefix-c1", "window-c1", "window-c2", "window-c4", "tail"):
            with self.subTest(condition=name):
                self.assertTrue(conditions[name]["problems"])
        failing = {name: {key for key, value in conditions[name]["checks"].items() if not value}
                   for name in ("capacity-2048", "capacity-4096", "capacity-8192", "capacity-16384")}
        self.assertEqual(failing, {"capacity-2048": {"finite", "max_abs_logit", "top1"},
                                   "capacity-4096": {"no_constant_rows", "max_abs_logit"},
                                   "capacity-8192": {"max_abs_logit"}, "capacity-16384": {"max_abs_logit", "top1"}})

    def test_the_record_of_the_reads_before_the_start_must_match_the_engine(self):
        other = 2048 if native.prefill_chunk() == 4096 else 4096
        changes = {"prefix-c1": {"record": {"prefix_function": "main"}}, "capacity-2048": {"drop": ["prefix_function"]},
                   "capacity-4096": {"record": {"prefix_calls": 2}}, "capacity-8192": {"drop": ["prefix_calls"]},
                   "capacity-16384": {"record": {"prefix_chunk": other}}, "short-c1": {"record": {"prefix_function": "prefill"}},
                   "short-c26": {"drop": ["prefix_calls"]}, "tail": {"record": {"prefix_function": None}}}
        result = self.run_compare(changes)
        failing = {name for name, entry in self.conditions(result).items() if not entry["passed"]}
        self.assertEqual(failing, set(changes))

    def test_a_condition_that_cannot_be_compared_is_recorded_and_the_others_are_still_compared(self):
        result = self.run_compare({"window-c2": {"no_output": True}, "window-c4": {"unreadable": True}})
        self.assertFalse(result["passed"])
        self.assertEqual(read_json(self.folder / "out/comparison.json"), result)
        conditions = self.conditions(result)
        self.assertEqual(conditions["window-c2"]["problems"], ["the condition could not be compared (FileNotFoundError)"])
        self.assertEqual(conditions["window-c4"]["problems"], ["the condition could not be compared (PermissionError)"])
        self.assertEqual({name for name, entry in conditions.items() if not entry["passed"]}, {"window-c2", "window-c4"})

    def test_a_verifier_record_with_a_missing_wrong_or_non_finite_field_fails_only_its_condition(self):
        result = self.run_compare({"window-c2": {"replace": [['"load_s": 0.1', '"load_s": 1e999']]},
                                   "window-c4": {"drop": ["cached"]}, "prefix-c1": {"record": {"main_calls": "300"}},
                                   "tail": {"record": {"extra": "dropped"}}})
        self.assertEqual(read_json(self.folder / "out/comparison.json"), result)
        conditions = self.conditions(result)
        self.assertEqual({name: entry["problems"] for name, entry in conditions.items() if not entry["passed"]},
                         {"window-c2": ["the verifier record is malformed (load_s)"],
                          "window-c4": ["the verifier record is malformed (cached)"],
                          "prefix-c1": ["the verifier record is malformed (main_calls)"]})
        self.assertEqual(set(conditions["tail"]["verifier"]), set(native.VERIFIER))
        self.assertEqual(set(conditions["short-c1"]["verifier"]), set(native.VERIFIER) - {"prefix_function"})

    def test_a_non_finite_count_of_the_verifier_that_differs_from_its_file_or_a_huge_number_fails_only_its_condition(self):
        huge = str(10**400)
        result = self.run_compare({"short-c1": {"record": {"non_finite_count": 1}},
                                   "window-c1": {"replace": [['"non_finite_count": 0', '"non_finite_count": ' + huge]]},
                                   "tail": {"replace": [['"load_s": 0.1', '"load_s": ' + huge]]}})
        self.assertEqual(read_json(self.folder / "out/comparison.json"), result)
        conditions = self.conditions(result)
        self.assertEqual({name: entry["problems"] for name, entry in conditions.items() if not entry["passed"]},
                         {"short-c1": ["the verifier counted 1 non-finite values, not the 0 of its file"],
                          "window-c1": ["the verifier record is malformed (non_finite_count)"],
                          "tail": ["the verifier record is malformed (load_s)"]})

    def test_non_finite_logits_or_reference_values_fail_their_conditions_and_the_comparison_is_recorded(self):
        array = np.load(self.arrays["short"])
        array[3, 5] = np.nan
        np.save(self.arrays["short"], array)
        record = read_json(self.reference / "record.json")
        record["inputs"]["short"]["logits"] = layers.file_record(Path(self.arrays["short"]))
        write_json(self.reference / "record.json", record)
        result = self.run_compare({"window-c1": {"set": [[0, 0, float("inf")]]}})
        self.assertEqual(read_json(self.folder / "out/comparison.json"), result)
        conditions = self.conditions(result)
        failing = {name for name, entry in conditions.items() if not entry["passed"]}
        self.assertEqual(failing, {"short-c1", "short-c26", "window-c1"})
        for name in failing:
            self.assertEqual((conditions[name]["max_abs_logit"], conditions[name]["checks"]["finite"]), (None, False))

    def test_a_condition_record_that_is_not_json_is_reset_and_the_other_conditions_are_kept(self):
        measure = native.measure

        def broken(swift, reference, indices, known):
            values = measure(swift, reference, indices, known)
            return {**values, "max_abs_logit": float("nan")} if len(swift) == 300 else values

        with mock.patch.object(native, "measure", side_effect=broken):
            result = self.run_compare(label="nan")
        self.assertEqual(read_json(self.folder / "nan/comparison.json"), result)
        conditions = self.conditions(result)
        self.assertEqual(conditions["prefix-c1"], {"name": "prefix-c1", "input": "context", "tokens": 1300, "start": 1000,
                                                   "chunk": 1, "rows": 300, "checks": {}, "passed": False,
                                                   "problems": ["the condition could not be compared (ValueError)"]})
        self.assertEqual({name for name, entry in conditions.items() if not entry["passed"]}, {"prefix-c1"})

    def test_a_comparison_that_cannot_be_recorded_leaves_a_minimal_failed_record(self):
        calls = []

        def failing_once(path, value):
            calls.append(path)
            if len(calls) == 1:
                raise OSError("no space")
            return write_json(path, value)

        with mock.patch.object(native, "write_json", side_effect=failing_once):
            result = self.run_compare(label="unrecorded")
        expected = {"schema": native.SCHEMA, "passed": False, "conditions": [],
                    "problems": ["the comparison could not be recorded (OSError)"]}
        self.assertEqual(result, expected)
        self.assertEqual(read_json(self.folder / "unrecorded/comparison.json"), expected)

    def test_a_missing_or_not_executable_verifier_still_leaves_a_failed_comparison(self):
        missing = native.compare(self.export, self.reference, self.folder / "absent", self.folder / "missing", self.contract)
        self.assertEqual(read_json(self.folder / "missing/comparison.json"), missing)
        self.assertFalse(missing["passed"])
        self.assertEqual(missing["problems"], ["the verifier is unreadable (FileNotFoundError)"])
        self.run_compare(label="written")
        self.verifier.chmod(0o644)
        denied = native.compare(self.export, self.reference, self.verifier, self.folder / "denied", self.contract)
        self.assertEqual(read_json(self.folder / "denied/comparison.json"), denied)
        self.assertFalse(denied["passed"])
        self.assertEqual(len(denied["conditions"]), 12)
        self.assertEqual({tuple(entry["problems"]) for entry in denied["conditions"]},
                         {("the condition could not be compared (PermissionError)",)})

    def test_any_failure_inside_a_condition_becomes_its_problem_without_the_message(self):
        with mock.patch.object(native, "measure", side_effect=RuntimeError("private-example-sentinel")):
            result = self.run_compare(label="raising")
        self.assertFalse(result["passed"])
        self.assertEqual({tuple(entry["problems"]) for entry in result["conditions"]},
                         {("the condition could not be compared (RuntimeError)",)})
        self.assertNotIn("private-example-sentinel", (self.folder / "raising/comparison.json").read_text())

    def test_unreadable_records_or_logits_leave_a_failed_comparison_without_running_the_verifier(self):
        context = self.reference / "context.npy"
        garbage = b"not an array" * 8
        record = read_json(self.reference / "record.json")
        record["inputs"]["context"]["logits"].update(bytes=len(garbage), sha256=hashlib.sha256(garbage).hexdigest())
        cases = {"a reference record that is a list": (self.reference / "record.json", b"[]", "reference"),
                 "a reference record that is a string": (self.reference / "record.json", b'"text"', "reference"),
                 "a broken reference record": (self.reference / "record.json", b"{", "reference"),
                 "an export record that is a list": (self.export / "export.json", b"[]", "export"),
                 "an export record whose bundle is a list": (self.export / "export.json",
                                                             json.dumps({**read_json(self.export / "export.json"),
                                                                         "bundle": ["bundle"]}).encode(), "export"),
                 "a broken export record": (self.export / "export.json", b"{", "export")}
        for label, (target, content, kind) in cases.items():
            saved = target.read_bytes()
            target.write_bytes(content)
            self.calls.unlink(missing_ok=True)
            with self.subTest(case=label):
                result = self.run_compare(label=label.replace(" ", "-"))
                self.assertEqual(read_json(self.folder / label.replace(" ", "-") / "comparison.json"), result)
                self.assertFalse(result["passed"])
                self.assertTrue(any(problem.startswith(f"the {kind} record is unreadable (") for problem in result["problems"]))
                self.assertFalse(self.calls.exists())
            target.write_bytes(saved)
        saved = {path: path.read_bytes() for path in (context, self.reference / "record.json")}
        context.write_bytes(garbage)
        write_json(self.reference / "record.json", record)
        result = self.run_compare(label="garbage-logits")
        for path, data in saved.items():
            path.write_bytes(data)
        self.assertFalse(result["passed"])
        self.assertIn("the reference record is unreadable (ValueError)", result["problems"])
        self.assertFalse(self.calls.exists())

    def test_a_verifier_that_does_not_finish_in_time_is_stopped_and_its_condition_fails(self):
        with mock.patch.object(native, "TIMEOUT_S", 1.0):
            started = time.perf_counter()
            result = self.run_compare({"short-c1": {"sleep": 30}})
        self.assertLess(time.perf_counter() - started, 20)
        entry = self.conditions(result)["short-c1"]
        self.assertFalse(entry["passed"])
        self.assertTrue(any("1.0 s" in problem for problem in entry["problems"]))
        listed = subprocess.run(["/bin/ps", "-ax", "-o", "command="], capture_output=True, text=True, check=True).stdout
        self.assertNotIn(str(self.verifier), listed)

    def test_records_that_must_not_pass_stop_before_the_verifier_runs(self):
        def other_layers(record):
            record.update(layers=12, layer_types=layers.check_layers(self.config, 12))

        cases = {"a reference of another layer count": ("reference", other_layers),
                 "references without a layer count": ("reference", lambda record: record.pop("layers")),
                 "both records without a layer count": ("both", lambda record: record.pop("layers")),
                 "both records without layer types": ("both", lambda record: record.pop("layer_types")),
                 "a reference of another input": ("reference", lambda record: record["inputs"]["context"].update(
                     tokens_sha256="0" * 64)),
                 "a reference of other rows": ("reference", lambda record: record["inputs"]["context"].update(
                     rows=[row + 1 for row in record["inputs"]["context"]["rows"]])),
                 "a reference of another contract": ("reference", lambda record: record["model"].update(
                     contract_sha256="0" * 64)),
                 "a reference of the model class": ("reference", lambda record: record.update(implementation="class")),
                 "an export of another model": ("export", lambda record: record["model"].update(id="other/model")),
                 "an export that did not pass": ("export", lambda record: record.update(passed=False)),
                 "an export of another main hash": ("export", lambda record: record["bundle"].update(main_hash="cd" * 32)),
                 "an export of other functions": ("export", lambda record: record.update(functions="main,prefill")),
                 "an export that passed as the number one": ("export", lambda record: record.update(passed=1)),
                 "an export with a problem": ("export", lambda record: record.update(problems=["asset"])),
                 "an export without tap layers": ("export", lambda record: record["metadata"].pop("tap_layers")),
                 "an export with tap layers beyond its layers": ("export", lambda record: record["metadata"].update(
                     tap_layers=[1, 7])),
                 "an export with tap layers as booleans": ("export", lambda record: record["metadata"].update(
                     tap_layers=[True])),
                 "an export of a layer count given as a boolean": ("export", lambda record: record.update(layers=True)),
                 "an export whose model record has a file size of true": ("export", lambda record: next(iter(
                     record["model"]["verified_before_loading"].values())).update(bytes=True))}
        for label, (target, change) in cases.items():
            with self.subTest(case=label):
                backups = {path: path.read_bytes() for path in (self.reference / "record.json", self.export / "export.json")}
                if target in ("reference", "both"):
                    edit(self.reference / "record.json", change)
                if target in ("export", "both"):
                    edit(self.export / "export.json", change)
                self.calls.unlink(missing_ok=True)
                result = self.run_compare(label=label.replace(" ", "-"))
                for path, data in backups.items():
                    path.write_bytes(data)
                self.assertFalse(result["passed"])
                self.assertTrue(result["problems"])
                self.assertEqual(result["conditions"], [])
                self.assertFalse(self.calls.exists())

    def quantized(self, **changes):
        shutil.rmtree(self.export)
        self.export.mkdir()
        tensors = self.export / layers.QUANTIZED
        tensors.write_bytes(b"quantized tensors")
        entry = {"bytes": tensors.stat().st_size, "sha256": sha256(tensors)}
        names = sorted(export.quantized_names(self.kinds))
        weights = [{"name": name, "class": "Dequantize", "shape": [1, 32], **export.LAYOUT} for name in names]
        digests = {f"{name}.{part}": "1" * 64 for name in names for part in layers.PARTS}
        complete_export(self.export, self.contract, "8bit", quantized={
            "layers": weights, "skipped": 0, "warnings_enabled": True, "file": entry, "tensors": digests}, **changes)
        edit(self.reference / "record.json", lambda record: record.update(
            weights=layers.RESTORED, quantized={"export_sha256": sha256(self.export / "export.json"), "file": entry,
                                                "tensors": 3 * len(names)}))

    def weights(self, change):
        def edited():
            edit(self.export / "export.json", lambda record: change(record["quantized"]))
            edit(self.reference / "record.json", lambda record: record["quantized"].update(
                export_sha256=sha256(self.export / "export.json")))
        return edited

    def test_the_reader_names_an_8bit_export_of_main_only_when_it_cannot_inspect_the_asset_again(self):
        self.quantized(functions=["main"])
        edit(self.export / "export.json", lambda record: record["asset"].pop("storage_types"))
        self.assertEqual(native.load_export(self.export, load_contract(self.contract), self.contract)[3],
                         ["the export record does not hold a complete inspection of its asset",
                          "an 8-bit export recorded other functions than main and prefill"])

    def test_an_8bit_export_is_compared_with_the_reference_restored_from_its_quantized_tensors(self):
        self.quantized()
        result = self.run_compare()
        self.assertTrue(result["passed"], result["problems"])
        self.assertEqual(len(result["conditions"]), 12)

    def test_an_export_and_a_reference_of_other_weights_stop_before_the_verifier_runs(self):
        tensors = self.export / layers.QUANTIZED

        def link():
            copy = self.folder / "copy.safetensors"
            copy.write_bytes(tensors.read_bytes())
            tensors.unlink()
            tensors.symlink_to(copy)

        cases = {"an FP16 export with a restored reference": (False, lambda: edit(
                     self.reference / "record.json", lambda record: record.update(weights=layers.RESTORED))),
                 "an FP16 export with a reference without weights": (False, lambda: edit(
                     self.reference / "record.json", lambda record: record.pop("weights"))),
                 "an export without a precision": (False, lambda: edit(
                     self.export / "export.json", lambda record: record.pop("precision"))),
                 "an export of another precision": (False, lambda: edit(
                     self.export / "export.json", lambda record: record.update(precision="4bit"))),
                 "an 8-bit export with the rounded reference": (True, lambda: edit(
                     self.reference / "record.json", lambda record: record.update(weights=layers.WEIGHTS))),
                 "a reference restored for another export record": (True, lambda: edit(
                     self.reference / "record.json", lambda record: record["quantized"].update(export_sha256="0" * 64))),
                 "a reference restored from another file": (True, lambda: edit(
                     self.reference / "record.json", lambda record: record["quantized"]["file"].update(sha256="0" * 64))),
                 "a reference of another tensor count": (True, lambda: edit(
                     self.reference / "record.json", lambda record: record["quantized"].update(tensors=3))),
                 "an 8-bit export with main only": (False, lambda: self.quantized(functions=["main"])),
                 "quantized tensors changed after the export": (True, lambda: tensors.write_bytes(b"quantized tensorz")),
                 "no quantized tensors": (True, tensors.unlink),
                 "quantized tensors behind a link": (True, link),
                 "no quantized tensors and a list as their record": (True, lambda: (tensors.unlink(), edit(
                     self.export / "export.json", lambda record: record.update(quantized=["file"])))),
                 "a record of the tensors without the file": (True, lambda: edit(
                     self.export / "export.json", lambda record: record["quantized"].pop("file"))),
                 "a size of the tensors as text": (True, lambda: edit(
                     self.export / "export.json", lambda record: record["quantized"]["file"].update(bytes="17"))),
                 "a weight recorded in blocks of 16": (True, self.weights(
                     lambda quantized: quantized["layers"][0].update(block=[1, 16]))),
                 "a weight recorded with a zero point": (True, self.weights(
                     lambda quantized: quantized["layers"][0].update(zero_point="torch.uint8"))),
                 "a weight left out of the record of the weights": (True, self.weights(
                     lambda quantized: quantized["layers"].pop())),
                 "tensors of another weight": (True, self.weights(lambda quantized: quantized["tensors"].update(
                     {"embed_tokens.weight.scale": "1" * 64}))),
                 "a reference whose tensor count is a float": (True, lambda: edit(
                     self.reference / "record.json", lambda record: record["quantized"].update(
                         tensors=float(record["quantized"]["tensors"]))))}
        for label, (eight_bit, change) in cases.items():
            with self.subTest(case=label):
                backup = (self.reference / "record.json").read_bytes()
                if eight_bit:
                    self.quantized()
                change()
                self.calls.unlink(missing_ok=True)
                result = self.run_compare(label=label.replace(" ", "-"))
                (self.reference / "record.json").write_bytes(backup)
                shutil.rmtree(self.export)
                complete_export(self.export, self.contract)
                self.assertEqual(read_json(self.folder / label.replace(" ", "-") / "comparison.json"), result)
                self.assertFalse(result["passed"])
                self.assertTrue(result["problems"])
                self.assertEqual(result["conditions"], [])
                self.assertFalse(self.calls.exists())

    def test_an_8bit_export_whose_record_has_a_problem_is_not_compared_with_the_weights_of_the_reference(self):
        self.quantized()
        (self.export / layers.QUANTIZED).unlink()
        edit(self.export / "export.json", lambda record: record.update(quantized=["file"]))
        result = self.run_compare(label="problem-only")
        self.assertEqual(read_json(self.folder / "problem-only/comparison.json"), result)
        self.assertEqual((result["problems"], result["conditions"]),
                         (["the export record does not hold a complete inspection of its asset",
                           "the export did not record the tensors of exactly the quantized Linear weights of its layers",
                           "the quantized tensors of the export differ from its record"], []))

    def test_an_export_whose_bundle_or_record_is_not_complete_or_not_of_the_contract_is_not_compared(self):
        original = self.export
        for index, (label, change) in enumerate(BUNDLE_DEFECTS.items()):
            with self.subTest(case=label):
                self.export = complete_export(self.folder / f"defect-{index}", self.contract)
                change(self.export)
                self.calls.unlink(missing_ok=True)
                result = self.run_compare(label=f"defect-{index}-out")
                self.export = original
                self.assertEqual(read_json(self.folder / f"defect-{index}-out" / "comparison.json"), result)
                self.assertFalse(result["passed"])
                self.assertTrue(result["problems"])
                self.assertEqual(result["conditions"], [])
                self.assertFalse(self.calls.exists())

    def test_the_reader_does_not_wait_on_a_pipe_and_does_not_read_a_file_of_another_size(self):
        contract = load_contract(self.contract)
        name = read_json(self.export / "export.json")["bundle"]["name"]

        def piped(relative):
            def change(folder):
                path = folder / relative
                shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
                os.mkfifo(path)
                return path
            return change

        def folder_instead(folder):
            (folder / "export.json").unlink()
            (folder / "export.json").mkdir()
            return folder / "export.json"

        def resized(change):
            def apply(folder):
                edit(folder / "export.json", change)
                return None
            return apply

        def quantized_resized(folder):
            edit(folder / "export.json", lambda record: record["quantized"]["file"].update(
                bytes=record["quantized"]["file"]["bytes"] + 1))
            return None

        unreadable = {"export.json as a pipe": (False, piped("export.json")), "export.json as a folder": (False, folder_instead)}
        refused = {"main.mlirb as a pipe": (False, piped(f"bundle/{name}.aimodel/main.mlirb"), None),
                   "main.mlirb of another size than its record": (False, resized(lambda record: record["bundle"]["files"][
                       f"{name}.aimodel/main.mlirb"].update(bytes=len(MAIN_ASSET) + 1)), f"bundle/{name}.aimodel/main.mlirb"),
                   "main.hash as a pipe": (False, piped(f"bundle/{name}.aimodel/main.hash"), None),
                   "quantized tensors as a pipe": (True, piped(layers.QUANTIZED), None),
                   "quantized tensors of another size than their record": (True, quantized_resized, layers.QUANTIZED)}
        for index, (label, (eight_bit, change)) in enumerate(unreadable.items()):
            folder = self.folder / f"unreadable-{index}"
            shutil.copytree(self.export, folder)
            path = change(folder)
            with self.subTest(case=label):
                thread, outcome = bounded(lambda: native.load_export(folder, contract, self.contract))
                self.assertFalse(release(path, thread))
                self.assertIsInstance(outcome.get("error"), (OSError, ValueError))
        for index, (label, (eight_bit, change, unread)) in enumerate(refused.items()):
            if eight_bit:
                self.quantized()
            folder = self.folder / f"refused-{index}"
            shutil.copytree(self.export, folder)
            path = change(folder)
            with self.subTest(case=label), watched() as reads:
                thread, outcome = bounded(lambda: native.load_export(folder, contract, self.contract))
                self.assertFalse(release(path, thread) if path else thread.is_alive())
                self.assertTrue(outcome["value"][3])
                if unread:
                    self.assertNotIn(inode(folder / unread), reads)
            if eight_bit:
                shutil.rmtree(self.export)
                complete_export(self.export, self.contract)

    def test_the_reader_names_each_number_of_the_record_that_does_not_follow_from_the_others(self):
        limit, contract = native.SWAP_INCREASE_LIMIT_BYTES, load_contract(self.contract)

        def swap(**values):
            return lambda record: record["resources"]["swap"].update(values)

        def monitored(part, **values):
            return lambda record: record["resources"]["monitor"][part].update(values)

        def review(record):
            monitored("first", swap_used_bytes=0)(record)
            monitored("peak", swap_used_bytes=limit + 1)(record)
            swap(first_bytes=0, peak_bytes=limit + 1, increase_bytes=0)(record)

        def single(swap_peak, free):
            def change(record):
                monitored("first", swap_used_bytes=7)(record)
                monitored("peak", swap_used_bytes=swap_peak)(record)
                swap(first_bytes=7, peak_bytes=swap_peak, increase_bytes=swap_peak - 7)(record)
                record["resources"]["monitor"]["samples"] = 1
                lowest = record["resources"]["monitor"]["first"]["free_bytes"]["volume"] if free is None else free
                record["resources"]["monitor"]["lowest"]["free_bytes"]["volume"] = lowest
                record["disk"]["lowest_free_bytes"] = lowest
            return change

        def float16(record):
            record["weights"]["elements_by_dtype"]["torch.float16"] += 2
            record["weights"]["elements"] += 2
            record["asset"]["float16_minus_weights"] -= 2

        increase, first, peak = ("the swap increase is not the peak less the first swap",
                                 "the first swap of the record is not that of the monitor",
                                 "the peak swap of the record is not that of the monitor")
        weights = "the weight elements do not follow from the checkpoint size and the quantized elements"
        one_swap, one_free = ("the monitor recorded one sample and another peak swap than that sample",
                              "the monitor recorded one sample and another lowest free disk than that sample")
        cases = {"a swap above its limit recorded as no increase": (False, review, increase),
                 "a swap increase other than the peak less the first swap": (False, swap(increase_bytes=1), increase),
                 "a first swap other than that of the monitor": (False, swap(first_bytes=4, increase_bytes=1), first),
                 "a peak swap other than that of the monitor": (False, swap(peak_bytes=6, increase_bytes=1), peak),
                 "another limit of the swap increase": (False, swap(limit_increase_bytes=limit + 1),
                                                        "the swap record names another limit than the fixed one"),
                 "a lowest free disk other than that of the monitor": (False, lambda record: record["disk"].update(
                     lowest_free_bytes=export.MARGIN_BYTES + 1), "the lowest free disk is not that of the monitor"),
                 "a first free disk of the monitor below its lowest": (False, lambda record: record["resources"][
                     "monitor"]["first"]["free_bytes"].update(volume=export.MARGIN_BYTES - 1),
                     "the lowest free disk of the monitor is above its first free disk"),
                 "a peak resident memory below that of the monitor": (False, lambda record: record["resources"].update(
                     max_rss_bytes=8), "the peak resident memory is below that of the monitor"),
                 "a monitor that stopped the export": (False, lambda record: record["resources"]["monitor"].update(
                     aborts=1), "the monitor did not run under the fixed limits without stopping"),
                 "a monitor with first and peak readings and no sample": (False, lambda record: record["resources"][
                     "monitor"].update(samples=0), "the monitor recorded no sample"),
                 "more samples than the interval of the monitor allows in the time of the export": (
                     False, lambda record: record["resources"]["monitor"].update(samples=100),
                     "the monitor recorded more samples than its interval allows in the time of the export"),
                 "a monitor record with a key that the monitor does not write": (
                     False, lambda record: record["resources"]["monitor"].update(extra=1),
                     "the monitor record does not have the keys that the monitor writes"),
                 "a peak resident memory of the monitor of zero": (False, monitored("peak", rss_bytes=0),
                                                                  "the monitor recorded no positive resident memory"),
                 "one sample with a peak swap above it and a lowest free disk below it": (
                     False, single(9, export.MARGIN_BYTES), [one_swap, one_free]),
                 "one sample with a peak swap above it": (False, single(9, None), one_swap),
                 "one sample with a lowest free disk below it": (False, single(7, export.MARGIN_BYTES), one_free),
                 "a resident memory of the monitor that is not a whole number of KiB": (
                     False, monitored("peak", rss_bytes=8193), "the resident memory that ps gave the monitor is not in KiB"),
                 "a lowest free memory percentage above 100": (
                     False, lambda record: record["resources"]["monitor"]["lowest"].update(free_percent=101),
                     "the lowest free memory percentage of the monitor is not a percentage"),
                 "the time of a step that the export does not take": (False, lambda record: record["timing"].update(
                     quantize_s=1.0), "the times are not those of the steps of the export"),
                 "a total time shorter than its steps": (False, lambda record: record["timing"].update(total_s=5.0),
                                                         "the total time is shorter than its steps"),
                 "float16 elements other than the checkpoint holds": (False, float16, weights),
                 "8-bit float16 elements other than the checkpoint and the quantized weights give": (True, float16,
                                                                                                    weights),
                 "a quantized shape other than the uint8 elements": (True, lambda record: record["quantized"]["layers"][
                     0].update(shape=[2, 32]), "the uint8 elements are not those of the quantized shapes")}
        for eight_bit in (False, True):
            if eight_bit:
                self.quantized()
            self.assertEqual(native.load_export(self.export, contract, self.contract)[3], [])
            for index, (label, (case_bit, change, expected)) in enumerate(cases.items()):
                if case_bit != eight_bit:
                    continue
                folder = self.folder / f"numbers-{int(eight_bit)}-{index}"
                shutil.copytree(self.export, folder)
                edit(folder / "export.json", change)
                with self.subTest(case=label):
                    self.assertEqual(native.load_export(folder, contract, self.contract)[3],
                                     expected if isinstance(expected, list) else [expected])

    def test_the_export_reader_names_each_defect_of_the_record_and_of_its_folder(self):
        contract = load_contract(self.contract)
        self.assertEqual(native.load_export(self.export, contract, self.contract)[3], [])

        def entry(record):
            next(iter(record["model"]["verified_before_loading"].values())).update(bytes=True)

        def failure(folder):
            write_json(folder / "failure.json", {"schema": export.SCHEMA, "passed": False, "error": "OSError"})

        cases = {"passed as the number one": (lambda record: record.update(passed=1), None,
                                              "the export is not a passed export"),
                 "a problem in the record": (lambda record: record.update(problems=["asset"]), None,
                                             "the export is not a passed export"),
                 "an unknown precision": (lambda record: record.update(precision="4bit"), None,
                                          "the export recorded no known precision"),
                 "no tap layers": (lambda record: record.update(metadata={}), None,
                                   "the export recorded no valid tap layers"),
                 "a model record with a size of true": (entry, None, "the export recorded another model or contract"),
                 "a failure record": (None, failure, "the export folder holds a failure record"),
                 "an unfinished bundle": (None, lambda folder: (folder / export.INCOMPLETE).mkdir(),
                                          "the export folder holds an unfinished bundle")}
        for index, (label, (change, arrange, expected)) in enumerate(cases.items()):
            folder = self.folder / f"read-{index}"
            shutil.copytree(self.export, folder)
            if change:
                edit(folder / "export.json", change)
            if arrange:
                arrange(folder)
            with self.subTest(case=label):
                self.assertEqual(native.load_export(folder, contract, self.contract)[3], [expected])

    def test_an_export_folder_that_did_not_finish_stops_before_the_verifier_runs(self):
        def recorded_as_true():
            edit(self.export / "export.json", lambda record: record["bundle"]["files"]["tokenizer/tokenizer.json"].update(
                bytes=True))

        failure = {"schema": export.SCHEMA, "passed": False, "error": "OSError"}
        cases = {"the last rename failed": lambda: (
                     (self.export / export.BUNDLE).rename(self.export / export.INCOMPLETE),
                     write_json(self.export / "failure.json", failure)),
                 "a failure record next to the bundle": lambda: write_json(self.export / "failure.json", failure),
                 "an unfinished bundle next to the bundle": lambda: (self.export / export.INCOMPLETE).mkdir(),
                 "a tokenizer file recorded with a size of true": recorded_as_true}
        original = self.export
        for index, (label, change) in enumerate(cases.items()):
            with self.subTest(case=label):
                self.export = self.folder / f"export-{index}"
                shutil.copytree(original, self.export)
                change()
                self.calls.unlink(missing_ok=True)
                result = self.run_compare(label=label.replace(" ", "-"))
                self.export = original
                self.assertFalse(result["passed"])
                self.assertTrue(result["problems"])
                self.assertEqual(result["conditions"], [])
                self.assertFalse(self.calls.exists())

    def test_files_that_differ_from_their_records_or_are_one_file_stop_before_the_verifier_runs(self):
        context = self.reference / "context.npy"
        original = context.read_bytes()
        changes = {"changed reference logits": lambda: context.write_bytes(original[:-4] + b"\0\0\0\0"),
                   "missing reference logits": context.unlink,
                   "a changed bundle file": lambda: (self.export / "bundle/metadata.json").write_text("{ }"),
                   "an extra bundle file": lambda: (self.export / "bundle/extra.txt").write_text("x"),
                   "short logits linked to the context logits": lambda: (
                       (self.reference / "short.npy").unlink(), os.link(context, self.reference / "short.npy"))}
        for label, change in changes.items():
            with self.subTest(case=label):
                saved = {path: path.read_bytes() for path in (self.reference / "short.npy", context,
                                                              self.export / "bundle/metadata.json")}
                change()
                self.calls.unlink(missing_ok=True)
                result = self.run_compare(label=label.replace(" ", "-"))
                (self.export / "bundle/extra.txt").unlink(missing_ok=True)
                for path, data in saved.items():
                    path.unlink(missing_ok=True)
                    path.write_bytes(data)
                self.assertFalse(result["passed"])
                self.assertTrue(result["problems"])
                self.assertFalse(self.calls.exists())

    def test_an_existing_output_is_refused_before_anything_is_read(self):
        target = self.folder / "target"
        target.write_text("target")
        broken = self.folder / "broken.json"
        broken.write_text("{")
        for label, make in {"file": lambda path: path.write_text("x"), "folder": lambda path: path.mkdir(),
                            "dangling link": lambda path: path.symlink_to(self.folder / "missing"),
                            "link": lambda path: path.symlink_to(target)}.items():
            output = self.folder / label.replace(" ", "-")
            make(output)
            before = (os.readlink(output) if output.is_symlink() else None, output.read_bytes() if output.is_file() else None)
            with self.subTest(output=label), mock.patch.object(native, "read_once") as read, \
                    mock.patch.object(native, "load_contract", wraps=native.load_contract) as load:
                with self.assertRaises(FileExistsError):
                    native.compare(self.export, self.reference, self.verifier, output, broken)
                read.assert_not_called()
                load.assert_not_called()
            self.assertEqual((os.readlink(output) if output.is_symlink() else None,
                              output.read_bytes() if output.is_file() else None), before)

    @unittest.skipIf(os.geteuid() == 0, "root searches a folder without permission")
    def test_a_link_into_a_folder_without_search_permission_is_refused_before_anything_is_read(self):
        closed, output, broken = self.folder / "closed", self.folder / "closed-link", self.folder / "broken.json"
        closed.mkdir()
        output.symlink_to(closed / "inner")
        broken.write_text("{")
        closed.chmod(0)
        self.addCleanup(closed.chmod, 0o700)
        with mock.patch.object(native, "read_once") as read, \
                mock.patch.object(native, "load_contract", wraps=native.load_contract) as load, \
                self.assertRaises(FileExistsError):
            native.compare(self.export, self.reference, self.verifier, output, broken)
        read.assert_not_called()
        load.assert_not_called()
        self.assertEqual(os.readlink(output), str(closed / "inner"))

    def test_the_command_writes_the_result_and_exits_with_one_unless_it_passes(self):
        result = {"passed": False, "problems": [], "conditions": [{"name": "tail", "passed": False, "max_abs_logit": None}]}
        arguments = ["native", "--export", "e", "--reference", "r", "--verifier", "v", "--output", "o"]
        for passed in (True, False):
            with self.subTest(passed=passed), mock.patch.object(sys, "argv", arguments), \
                    mock.patch.object(native, "compare", return_value={**result, "passed": passed}) as compare, \
                    mock.patch("sys.stdout"):
                self.assertEqual(native.main(), 0 if passed else 1)
            self.assertEqual(compare.call_args.args[:4], (Path("e"), Path("r"), Path("v"), Path("o")))


if __name__ == "__main__":
    unittest.main()
