import contextlib
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

from Tools.Conversion.adapters.gemma4_31b import export, layers, native
from Tools.Conversion.adapters.gemma4_31b.prepare import TOKENIZER_FILES, load_contract
from Tools.Conversion.adapters.pinned import sha256
from Tools.Conversion.storage import canonical, digest, read_json, write_json
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
          "non_finite_count": 0, "load_s": 0.001, "prefix_s": 0.0, "scoring_s": 0.001, "specialize_s": 0.0}}
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
    reads, real = [], os.read

    def read_descriptor(descriptor, count):
        status = os.fstat(descriptor)
        reads.append((status.st_dev, status.st_ino))
        return real(descriptor, count)

    def opener(file, *arguments, **options):
        handle = OPEN(file, *arguments, **options)
        status = os.fstat(handle.fileno())
        original = handle.read

        def read(*sizes):
            reads.append((status.st_dev, status.st_ino))
            return original(*sizes)

        handle.read = read
        return handle

    with mock.patch("builtins.open", new=opener), mock.patch("io.open", new=opener), \
            mock.patch.object(os, "read", side_effect=read_descriptor):
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

    def run_compare(self, changes=None, label="out", verifier=None):
        plan = {"conditions": {name: [length, start, chunk] for name, _, length, start, chunk in layers.CONDITIONS},
                "changes": changes or {}, "arrays": self.arrays, "rows": self.rows, "main_hash": MAIN_HASH,
                "calls": str(self.calls), "prefill_chunk": native.prefill_chunk()}
        self.verifier.write_text(FAKE.format(python=sys.executable))
        self.verifier.chmod(0o755)
        self.verifier.with_suffix(".json").write_text(json.dumps(plan))
        return native.compare(self.export, self.reference, verifier or self.verifier, self.folder / label, self.contract)

    def conditions(self, result):
        return {entry["name"]: entry for entry in result["conditions"]}

    def loaded(self):
        return {name: np.load(path) for name, path in self.arrays.items()}

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
        result = self.run_compare({"window-c2": {"replace": [['"load_s": 0.001', '"load_s": 1e999']]},
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
                                   "tail": {"replace": [['"load_s": 0.001', '"load_s": ' + huge]]}})
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

    def test_every_condition_that_the_comparison_writes_on_any_path_holds_what_the_reader_expects(self):
        changes = {"short-c1": {"record": {"non_finite_count": 1}}, "short-c26": {"rows": -1}, "window-c1": {"exit": 1},
                   "window-c2": {"stdout": "not a record"}, "window-c4": {"no_output": True},
                   "window-c8": {"set": [[100, 5, 30.0]]}, "capacity-2048": {"set": [[2, 3, float("nan")]]},
                   "capacity-4096": {"constant": [4]}, "prefix-c1": {"drop": ["cached"]},
                   "capacity-8192": {"record": {"cached": False, "specialize_s": 0.0005}}}
        conditions = self.conditions(self.run_compare(changes))
        for name, entry in conditions.items():
            with self.subTest(condition=name):
                self.assertEqual(native.condition_problems(entry, self.config, MAIN_HASH, True), [])
        self.assertEqual({len(set(entry) - set(native.FIELDS)) for entry in conditions.values()}, {0, 1, 2, 3, 9})
        measured, stopped = conditions["tail"], conditions["window-c1"]
        sized, counted = conditions["short-c26"], conditions["short-c1"]
        verifier = measured["verifier"]
        steps, shorter = verifier["load_s"] + verifier["prefix_s"] + verifier["scoring_s"], {
            key: value for key, value in stopped["verifier"].items() if key != "scoring_s"}
        took = "the condition took less time than the verifier recorded for its steps"
        fields = "the verifier record holds a field or a value that the verifier does not write"
        cases = {
            "a field that the comparison does not write": (measured, {"extra": 1},
                                                           ["the condition holds other fields than the comparison writes on "
                                                            "its paths"]),
            "checks on a path that stops before its numbers": (
                stopped, {"checks": {"finite": True}}, ["the condition records other checks than its numbers give"]),
            "a verdict that its checks and problems do not give": (
                measured, {"passed": False}, ["the condition records another verdict than its checks and problems give"]),
            "no problem on a path that stops before its numbers": (
                stopped, {"problems": []}, ["the condition records no problem on a path that stops before its numbers"]),
            "two problems on the path that stops at the size of the file": (
                sized, {"problems": sized["problems"] * 2},
                ["the condition records other problems than the one of the path that stops at it"]),
            "two problems on the path that stops at an exception": (
                conditions["window-c4"], {"problems": conditions["window-c4"]["problems"] * 2},
                ["the condition records other problems than the one of the path that stops at it"]),
            "another problem after the numbers": (
                counted, {"problems": counted["problems"] * 2},
                ["the condition records more problems than the count of the verifier after its numbers"]),
            "a negative time of the verifier": (stopped, {"seconds": -1.0}, ["the condition records no time of the verifier"]),
            "a field that the verifier does not write": (
                measured, {"verifier": {**verifier, "cache_query_error": "private"}},
                ["the verifier record holds a field or a value that the verifier does not write"]),
            "a verifier record without the cache flag": (
                measured, {"verifier": {key: value for key, value in verifier.items() if key != "cached"}},
                ["the verifier record misses a field that the verifier writes"]),
            "other calls of the verifier": (measured, {"verifier": {**verifier, "main_calls": verifier["main_calls"] + 1}},
                                            ["the verifier record differs from the reads of the condition"]),
            "a prefill chunk that the engine does not use": (
                measured, {"verifier": {**verifier, "prefix_chunk": 1000}},
                ["the verifier record differs from the reads of the condition"]),
            "a prefix function at position 0": (
                conditions["window-c8"], {"verifier": {**conditions["window-c8"]["verifier"], "prefix_function": "main"}},
                ["the verifier record differs from the reads of the condition"]),
            "a specialization of a cached asset": (
                measured, {"verifier": {**verifier, "specialize_s": 0.0005}},
                ["the verifier recorded a specialization of the asset that it found in the cache"]),
            "a specialization longer than the loading": (
                measured, {"verifier": {**verifier, "cached": False, "specialize_s": verifier["load_s"] + 1}},
                ["the verifier recorded a longer specialization than its loading"]),
            "measured logits of another size than the rows": (
                measured, {"output": {**measured["output"], "bytes": measured["output"]["bytes"] - 2}},
                ["the logits file does not hold the rows of the condition exactly when they were measured"]),
            "a file of the size of the rows that was not measured": (
                sized, {"output": {**sized["output"], "bytes": native.logits_bytes(sized["rows"], VOCABULARY)}},
                ["the logits file does not hold the rows of the condition exactly when they were measured"]),
            "another non-finite count of the verifier without its problem": (
                measured, {"verifier": {**verifier, "non_finite_count": 1}},
                ["the verifier counted other non-finite values than the file exactly when the condition records a problem"]),
            "a condition that took less time than its verifier recorded": (
                measured, {"seconds": math.nextafter(steps, 0)}, [took]),
            "a condition that took as long as its verifier recorded": (measured, {"seconds": steps}, []),
            "a verifier that recorded some of its steps within the time of the condition": (
                stopped, {"verifier": shorter, "seconds": shorter["load_s"] + shorter["prefix_s"]}, []),
            "a verifier that recorded some of its steps beyond the time of the condition": (
                stopped, {"verifier": shorter, "seconds": math.nextafter(shorter["load_s"] + shorter["prefix_s"], 0)},
                [took]),
            "a verifier that did not find the asset in its cache and recorded no specialization": (
                measured, {"verifier": {**verifier, "cached": False, "specialize_s": 0.0}},
                ["the verifier recorded no specialization of the asset that it did not find in the cache"]),
            "a verifier that did not find the asset in its cache and recorded the smallest specialization": (
                measured, {"verifier": {**verifier, "cached": False, "specialize_s": math.nextafter(0.0, 1)}}, []),
            "a time of the verifier of another type": (measured, {"verifier": {**verifier, "load_s": "1"}}, [fields]),
            "a verifier record that is no record": (stopped, {"verifier": []}, [fields]),
            "problems of another type": (measured, {"problems": "x"},
                                         ["the problems of the condition are not a list of texts"]),
            "checks of another type": (measured, {"checks": []}, ["the condition records other checks than its numbers give"]),
            "a time of the condition of another type": (stopped, {"seconds": "4"},
                                                        ["the condition records no time of the verifier"]),
            "a size of the logits file of another type": (
                measured, {"output": {**measured["output"], "bytes": "3"}},
                ["the condition records no size of its logits file"]),
            "a record of the logits file that is no record": (
                measured, {"output": []}, ["the condition records no size of its logits file"])}
        for label, (entry, change, expected) in cases.items():
            with self.subTest(case=label):
                self.assertEqual(native.condition_problems({**entry, **change}, self.config, MAIN_HASH, True), expected)
        zero = {**measured, "verifier": {**verifier, "prefix_chunk": 0}}
        with self.subTest(case="a prefill chunk of zero, which the reads of the condition are not computed for"), \
                mock.patch.object(native, "call_lengths", side_effect=AssertionError):
            self.assertEqual(native.condition_problems(zero, self.config, MAIN_HASH, True),
                             ["the verifier record differs from the reads of the condition"])

    def test_every_condition_that_reached_its_numbers_holds_those_that_its_file_and_the_reference_give(self):
        changes = {"window-c8": {"set": [[100, 5, 30.0], [111, 6, 30.0]]}, "capacity-2048": {"set": [[2, 3, float("nan")]]},
                   "capacity-4096": {"constant": [4]}, "capacity-8192": {"set": [[1, 7, 10.0]]},
                   "capacity-16384": {"set": [[0, 9, 40.0]]}, "short-c26": {"rows": -1}, "window-c1": {"exit": 1}}
        result, arrays = self.run_compare(changes), self.loaded()
        measured = [entry for entry in result["conditions"] if "known" in entry]
        self.assertEqual((len(measured), {entry["passed"] for entry in measured}), (10, {True, False}))
        for entry in measured:
            data = (self.folder / "out" / f"{entry['name']}.f16").read_bytes()
            with self.subTest(condition=entry["name"]):
                self.assertEqual(native.measurement_problems(entry, VOCABULARY, data, arrays), [])

    def test_each_number_of_a_condition_that_its_file_and_the_reference_do_not_give_is_named_on_its_own(self):
        result, arrays = self.run_compare({"window-c8": {"set": [[100, 5, 30.0], [111, 6, 30.0]]}}), self.loaded()
        entry, data = self.conditions(result)["window-c8"], (self.folder / "out/window-c8.f16").read_bytes()
        named = ["the numbers of the condition are not those that its logits file and the reference give"]
        known, top, checks = entry["known"], entry["top1_mismatches"], entry["checks"]
        self.assertEqual(native.measurement_problems(entry, VOCABULARY, data, arrays), [])
        self.assertGreater(known["max_abs_logit"], 25)
        cases = {"one more non-finite value": {"non_finite": entry["non_finite"] + 1},
                 "one more non-finite value of the reference": {"reference_non_finite": entry["reference_non_finite"] + 1},
                 "one more constant row": {"constant_rows": entry["constant_rows"] + 1},
                 "the next number below the largest difference": {"max_abs_logit": math.nextafter(entry["max_abs_logit"], 0)},
                 "the next number above the largest difference": {"max_abs_logit": math.nextafter(entry["max_abs_logit"], 9)},
                 "no largest difference": {"max_abs_logit": None},
                 "one more top-1 change near a tie": {"top1_mismatches": {**top, "near_tie": top["near_tie"] + 1}},
                 "one more top-1 change outside the near tie": {"top1_mismatches": {**top, "outside": top["outside"] + 1}},
                 "a third count of top-1 changes": {"top1_mismatches": {**top, "other": 0}},
                 "one known row less": {"known": {**known, "rows": known["rows"] - 1}},
                 "the next number below the largest difference of the known rows": {
                     "known": {**known, "max_abs_logit": math.nextafter(known["max_abs_logit"], 0)}},
                 "no largest difference of the known rows": {"known": {**known, "max_abs_logit": None}},
                 "no known rows": {"known": None},
                 "a check that the numbers do not give": {"checks": {**checks, "finite": not checks["finite"]}},
                 "one more check": {"checks": {**checks, "extra": True}},
                 "no checks": {"checks": {}}}
        for label, change in cases.items():
            with self.subTest(case=label):
                self.assertEqual(native.measurement_problems({**entry, **change}, VOCABULARY, data, arrays), named)
        for key in ("non_finite", "reference_non_finite", "constant_rows", "max_abs_logit", "top1_mismatches", "known", "checks"):
            for value in (None, "x", [], {}, True, -1):
                with self.subTest(key=key, value=value):
                    self.assertEqual(native.measurement_problems({**entry, key: value}, VOCABULARY, data, arrays), named)
        other = bytearray(data)
        other[:2] = np.float16(1000).astype("<f2").tobytes()
        shifted = {name: array + 1 for name, array in arrays.items()}
        sources = {"a logits file with one value changed": (bytes(other), arrays),
                   "the logits file of another condition": ((self.folder / "out/window-c4.f16").read_bytes(), arrays),
                   "a reference of other values": (data, shifted)}
        for label, (content, held) in sources.items():
            with self.subTest(case=label):
                self.assertEqual(native.measurement_problems(entry, VOCABULARY, content, held), named)

    def test_the_comparison_and_the_reader_measure_a_condition_with_the_same_function(self):
        original = native.measure_condition
        with mock.patch.object(native, "measure_condition", wraps=original) as measured:
            result = self.run_compare()
        self.assertEqual([call.args[0]["name"] for call in measured.call_args_list],
                         [name for name, *_ in layers.CONDITIONS])
        entry, data = self.conditions(result)["tail"], (self.folder / "out/tail.f16").read_bytes()
        self.assertEqual(native.measurement_problems(entry, VOCABULARY, data, self.loaded()), [])
        shifted = lambda *arguments: {**original(*arguments), "max_abs_logit": 1.5}
        with mock.patch.object(native, "measure_condition", side_effect=shifted):
            self.assertEqual(native.measurement_problems(entry, VOCABULARY, data, self.loaded()),
                             ["the numbers of the condition are not those that its logits file and the reference give"])

    def test_a_condition_record_with_a_field_of_another_type_is_named_and_never_raises(self):
        changes = {"short-c26": {"rows": -1}, "window-c1": {"exit": 1}, "window-c2": {"stdout": "not a record"},
                   "window-c4": {"no_output": True}}
        conditions, fixed, arrays = self.conditions(self.run_compare(changes)), set(native.FIELDS[:6]), self.loaded()
        for name in ("window-c4", "window-c2", "window-c1", "short-c26", "window-c8", "tail"):
            entry = conditions[name]
            paths = [(key,) for key in entry if key not in fixed]
            paths += [(key, inner) for key, value in entry.items() if key not in fixed and isinstance(value, dict)
                      for inner in value]
            for path in paths:
                for value in (None, "x", [], {}, True, -1):
                    changed = copy.deepcopy(entry)
                    target = changed
                    for part in path[:-1]:
                        target = target[part]
                    if canonical(target[path[-1]]) == canonical(value):
                        continue
                    target[path[-1]] = value
                    with self.subTest(condition=name, path=path, value=value):
                        found = native.condition_problems(changed, self.config, MAIN_HASH, True)
                        if "known" in entry:
                            data = (self.folder / "out" / f"{name}.f16").read_bytes()
                            found += native.measurement_problems(changed, VOCABULARY, data, arrays)
                        if name == "tail" and path != ("output", "sha256"):
                            self.assertTrue(found)

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

    def test_the_verifier_that_is_hashed_is_the_one_that_runs_when_it_is_given_by_its_name_only(self):
        shadow = self.folder / "elsewhere" / "verifier"
        shadow.parent.mkdir()
        shadow.write_text("#!/bin/sh\nexit 3\n")
        shadow.chmod(0o755)
        previous = os.getcwd()
        os.chdir(self.folder)
        self.addCleanup(os.chdir, previous)
        with mock.patch.dict(os.environ, {"PATH": f"{shadow.parent}{os.pathsep}{os.environ['PATH']}"}):
            result = self.run_compare(verifier=Path("verifier"))
        self.assertTrue(result["passed"], result["problems"])
        self.assertEqual(result["verifier"], {"bytes": self.verifier.stat().st_size, "sha256": sha256(self.verifier)})
        self.assertTrue(self.calls.exists())

    def test_a_verifier_path_that_cannot_be_resolved_leaves_a_failed_comparison(self):
        first, second = self.folder / "loop-a", self.folder / "loop-b"
        first.symlink_to(second)
        second.symlink_to(first)
        result = self.run_compare(verifier=first)
        self.assertEqual(read_json(self.folder / "out/comparison.json"), result)
        self.assertFalse(result["passed"])
        self.assertEqual((len(result["problems"]), result["verifier"], result["conditions"]), (1, None, []))
        self.assertRegex(result["problems"][0], r"^the verifier is unreadable \(\w+\)$")
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

    def test_the_reference_reader_does_not_wait_on_a_pipe_and_reads_only_regular_files_of_the_recorded_size(self):
        contract = load_contract(self.contract)

        def piped(path):
            path.unlink()
            os.mkfifo(path)

        def linked(path):
            path.rename(path.with_name(path.name + ".copy"))
            path.symlink_to(path.with_name(path.name + ".copy"))

        def folder(path):
            path.unlink()
            path.mkdir()

        index = 0
        for kind, make in {"a pipe": piped, "a link": linked, "a folder": folder}.items():
            for place in ("short.npy", "context.npy", "record.json"):
                index += 1
                copied = self.folder / f"reference-{index}"
                shutil.copytree(self.reference, copied)
                make(copied / place)
                with self.subTest(case=kind, place=place):
                    thread, outcome = bounded(lambda: native.load_reference(copied, contract, self.contract))
                    self.assertFalse(release(copied / place, thread))
                    self.assertIsInstance(outcome.get("error"), (OSError, ValueError))
        for place in ("short.npy", "context.npy"):
            copied = self.folder / f"reference-{place}"
            shutil.copytree(self.reference, copied)
            (copied / place).write_bytes((copied / place).read_bytes() + b"x")
            with self.subTest(case="a file of another size", place=place), watched() as reads:
                with self.assertRaises(ValueError):
                    native.load_reference(copied, contract, self.contract)
                self.assertNotIn(inode(copied / place), reads)
        self.assertEqual(native.load_reference(self.reference, contract, self.contract)[3], [])

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
                     0].update(shape=[2, 32]), "the uint8 elements are not those of the quantized shapes"),
                 "a disk record with a key that the export does not write": (
                     False, lambda record: record["disk"].update(extra=1),
                     "the disk record does not have the keys that the export writes"),
                 "a swap record with a key that the export does not write": (
                     False, swap(extra=1), "the swap record does not have the keys that the export writes"),
                 "a resident memory that ps gave the monitor above the peak resident memory that it read": (
                     False, monitored("peak", rss_bytes=17408),
                     "the resident memory that ps gave the monitor is above the peak resident memory that it read"),
                 "the version of a package that the export does not record": (
                     False, lambda record: record["versions"].update(numpy="2.0"),
                     "the export recorded the versions of other packages than it records"),
                 "a function of the asset with a part other than its inputs, outputs and states": (
                     False, lambda record: record["asset"]["functions"]["main"].update(extra={}),
                     "the functions of the asset hold other parts than inputs, outputs and states")}
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

    def test_an_export_whose_context_limit_is_below_the_longest_condition_is_named(self):
        contract, message = load_contract(self.contract), "the export recorded a context limit below the longest condition"
        self.assertEqual(native.LONGEST, 16896)
        for context, expected in ((native.LONGEST - 1, [message]), (native.LONGEST, []), (native.LONGEST + 1, [])):
            folder = self.folder / f"context-{context}"
            complete_export(folder, self.contract, context=context)
            with self.subTest(context=context):
                self.assertEqual(native.load_export(folder, contract, self.contract)[3], expected)

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


def stop_leftovers(processes):
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
            process.wait()


@contextlib.contextmanager
def watched_children():
    started, entered, real = [], threading.Event(), subprocess.Popen

    def popen(*arguments, **keywords):
        process = real(*arguments, **keywords)
        started.append(process)
        original = process.communicate

        def communicate(*items, **options):
            entered.set()
            return original(*items, **options)

        process.communicate = communicate
        return process

    with mock.patch.object(native.subprocess, "Popen", side_effect=popen):
        try:
            yield started, entered
        finally:
            stop_leftovers(started)


class VerifierProcessTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        self.sleeper = self.folder / "sleeper"
        self.sleeper.write_text("#!/bin/sh\nexec sleep 600\n")
        self.sleeper.chmod(0o755)
        self.addCleanup(signal.signal, signal.SIGINT, signal.signal(signal.SIGINT, signal.default_int_handler))

    def test_a_verifier_that_is_started_when_an_interruption_arrives_is_stopped_and_collected(self):
        real = subprocess.Popen
        for name, number in (("SIGINT", signal.SIGINT), ("SIGTERM", signal.SIGTERM)):
            started, forced, timers = [], [], []

            def stop(process):
                if process.returncode is None:
                    forced.append(True)
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        process.kill()

            def popen(*arguments, **keywords):
                process = real(*arguments, **keywords)
                started.append(process)
                timers.append(threading.Timer(5, stop, [process]))
                timers[-1].daemon = True
                timers[-1].start()
                signal.raise_signal(number)
                return process

            with self.subTest(signal=name):
                try:
                    with layers.terminated_as_interrupt(), mock.patch.object(
                            native.subprocess, "Popen", side_effect=popen), self.assertRaises(KeyboardInterrupt):
                        native.run_verifier([str(self.sleeper)], 30, self.folder / f"{name}.log")
                    self.assertEqual(forced, [])
                    self.assertIsNotNone(started[0].returncode)
                    with self.assertRaises(ProcessLookupError):
                        os.kill(started[0].pid, 0)
                finally:
                    for timer in timers:
                        timer.cancel()
                        timer.join(2)
                    stop_leftovers(started)
                    self.assertEqual([timer for timer in timers if timer.is_alive()], [])

    def test_a_verifier_that_is_waited_for_when_an_interruption_arrives_is_stopped_and_collected(self):
        for name, number in (("SIGINT", signal.SIGINT), ("SIGTERM", signal.SIGTERM)):
            pidfile, forced = Path(f"{self.sleeper}.pid"), []
            self.sleeper.write_text('#!/bin/sh\necho $$ > "$0.pid"\nexec sleep 600\n')
            pidfile.unlink(missing_ok=True)
            finished = threading.Event()
            with self.subTest(signal=name), watched_children() as (started, entered):
                def interrupt_when_it_waits():
                    entered.wait(30)
                    for _ in range(600):
                        if pidfile.exists() and pidfile.read_text().strip():
                            break
                        time.sleep(0.05)
                    os.kill(os.getpid(), number)
                    while not finished.wait(5):
                        forced.append(True)
                        with contextlib.suppress(ProcessLookupError, ValueError):
                            os.kill(int(pidfile.read_text()), signal.SIGKILL)

                thread = threading.Thread(target=interrupt_when_it_waits, daemon=True)
                thread.start()
                try:
                    with layers.terminated_as_interrupt(), self.assertRaises(KeyboardInterrupt):
                        native.run_verifier([str(self.sleeper)], 60, self.folder / f"wait-{name}.log")
                finally:
                    finished.set()
                thread.join(10)
                self.assertEqual(forced, [])
                self.assertTrue(entered.is_set())
                with self.assertRaises(ProcessLookupError):
                    os.kill(int(pidfile.read_text()), 0)

    def test_a_verifier_that_fails_to_start_leaves_no_process_and_raises_its_error(self):
        with self.assertRaises(OSError):
            native.run_verifier([str(self.folder / "absent")], 30, self.folder / "absent.log")

    def test_a_verifier_can_be_run_from_a_thread_other_than_the_main_thread(self):
        outcome = {}
        thread = threading.Thread(
            target=lambda: outcome.update(result=native.run_verifier(["/usr/bin/true"], 30, self.folder / "thread.log")))
        thread.start()
        thread.join(30)
        self.assertEqual(outcome.get("result"), (0, b""))


class ReadLimitTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        self.path = self.folder / "file"
        self.path.write_bytes(bytes(range(256)) * 160)

    def read_counting(self, call, short=None):
        reads, real = [], os.read

        def counting(descriptor, count):
            block = real(descriptor, count if short is None else min(count, short))
            reads.append((count, len(block), os.lseek(descriptor, 0, os.SEEK_CUR)))
            return block

        with mock.patch.object(native.os, "read", side_effect=counting), \
                mock.patch.object(native, "BLOCK_BYTES", 4096):
            try:
                return call(), reads
            except ValueError as error:
                return error, reads

    def taken(self, reads):
        return sum(length for _, length, _ in reads)

    def assertWithin(self, reads, budget):
        before = 0
        for count, length, position in reads:
            self.assertLessEqual(count, budget - before)
            before += length
            self.assertEqual(position, before)
        self.assertLessEqual(before, budget)

    def test_a_file_above_the_limit_is_refused_before_it_is_read_and_a_file_up_to_it_is_read(self):
        size = self.path.stat().st_size
        refused, reads = self.read_counting(lambda: native.regular_file(self.path, keep=True, limit=size - 1))
        self.assertIsInstance(refused, ValueError)
        self.assertEqual(reads, [])
        for limit in (size, size + 1, 10**9):
            with self.subTest(limit=limit):
                (found, data), reads = self.read_counting(lambda: native.regular_file(self.path, keep=True, limit=limit))
                self.assertEqual((found["bytes"], len(data), self.taken(reads)), (size, size, size))
        (found, data), _ = self.read_counting(lambda: native.regular_file(self.path, keep=True))
        self.assertEqual((found["bytes"], len(data)), (size, size))

    def test_a_file_that_grows_after_its_size_was_read_is_not_read_beyond_the_limit_and_one_byte_from_its_descriptor(self):
        real = os.fstat
        lying = lambda descriptor: types.SimpleNamespace(st_mode=real(descriptor).st_mode, st_size=100)
        for short in (None, 700):
            with self.subTest(short=short), mock.patch.object(native.os, "fstat", side_effect=lying):
                refused, reads = self.read_counting(
                    lambda: native.regular_file(self.path, keep=True, limit=10000), short=short)
            self.assertIsInstance(refused, ValueError)
            self.assertEqual(self.taken(reads), 10001)
            self.assertWithin(reads, 10001)
        self.assertGreater(len(reads), 10)

    def test_a_file_is_read_completely_when_the_descriptor_returns_short_reads(self):
        for limit in (None, 10**9):
            with self.subTest(limit=limit):
                (found, data), reads = self.read_counting(
                    lambda: native.regular_file(self.path, keep=True, limit=limit), short=333)
                self.assertEqual(data, self.path.read_bytes())
                self.assertEqual(found, {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
                self.assertGreater(len(reads), len(data) // 333)

    def test_a_record_is_read_up_to_its_limit_only(self):
        record = self.folder / "record.json"
        record.write_text(json.dumps({"key": "x" * 200}))
        self.assertEqual(native.read_record(record, 1000)[0], {"key": "x" * 200})
        self.assertEqual(native.read_record(record)[0], {"key": "x" * 200})
        with self.assertRaises(ValueError):
            native.read_record(record, 100)

    def exported(self):
        (self.folder / "snapshot").mkdir()
        for index, item in enumerate(TOKENIZER_FILES):
            (self.folder / "snapshot" / item).write_text("abc"[index])
        files = {item: {"bytes": 1, "sha256": sha256(self.folder / "snapshot" / item)} for item in TOKENIZER_FILES}
        contract_path = write_contract(self.folder, files)
        return complete_export(self.folder / "export", contract_path), load_contract(contract_path), contract_path

    def test_an_export_record_above_the_limit_is_refused_and_not_read_beyond_the_limit_and_one_block(self):
        export, contract, contract_path = self.exported()
        size = (export / "export.json").stat().st_size
        with mock.patch.object(native, "EXPORT_LIMIT", size - 1):
            refused, reads = self.read_counting(lambda: native.load_export(export, contract, contract_path))
        self.assertIsInstance(refused, ValueError)
        self.assertEqual(reads, [])
        real = os.fstat
        lying = lambda descriptor: types.SimpleNamespace(st_mode=real(descriptor).st_mode, st_size=100)
        with mock.patch.object(native, "EXPORT_LIMIT", size - 1), mock.patch.object(native.os, "fstat", side_effect=lying):
            refused, reads = self.read_counting(lambda: native.load_export(export, contract, contract_path))
        self.assertIsInstance(refused, ValueError)
        self.assertEqual(self.taken(reads), size)
        with mock.patch.object(native, "EXPORT_LIMIT", size):
            self.assertEqual(native.load_export(export, contract, contract_path)[3], [])

    def test_an_export_record_that_is_sparse_and_above_the_limit_is_refused_without_reading_it(self):
        export, contract, contract_path = self.exported()
        os.truncate(export / "export.json", native.EXPORT_LIMIT + 1)
        refused, reads = self.read_counting(lambda: native.load_export(export, contract, contract_path))
        self.assertIsInstance(refused, ValueError)
        self.assertIn("is larger than", str(refused))
        self.assertEqual(reads, [])

    def test_every_file_that_the_export_loader_keeps_is_read_under_the_limit_and_the_others_are_only_hashed(self):
        export, contract, contract_path = self.exported()
        calls, real = [], native.regular_file

        def spy(path, size=None, keep=False, limit=None):
            calls.append((Path(path).name, keep, limit))
            return real(path, size, keep, limit)

        with mock.patch.object(native, "regular_file", side_effect=spy):
            self.assertEqual(native.load_export(export, contract, contract_path)[3], [])
        kept = [(name, limit) for name, keep, limit in calls if keep]
        self.assertEqual(sorted(kept), [("export.json", native.EXPORT_LIMIT), ("main.hash", native.EXPORT_LIMIT),
                                        ("metadata.json", native.EXPORT_LIMIT)])
        hashed = [limit for name, keep, limit in calls if not keep]
        self.assertEqual((len(hashed) > 0, set(hashed)), (True, {None}))


if __name__ == "__main__":
    unittest.main()
