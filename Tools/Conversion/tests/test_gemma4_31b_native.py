import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
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
from Tools.Conversion.tests.test_gemma4_31b_layers import TEXT, write_contract

ENGINE = Path(__file__).resolve().parents[3] / "CoreAISamplesKit/Sources/CoreAIBackend/Engine.swift"
LOGITS = Path(__file__).resolve().parents[3] / "Tools/Verify/Sources/coreai-verify/PositionLogits.swift"
VOCABULARY = TEXT["vocab_size"]
MAIN_HASH = "ab" * 32
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
        files = dict.fromkeys(TOKENIZER_FILES, {"bytes": 1, "sha256": "0" * 64})
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
                                                     "model": model_record(self.contract), "inputs": inputs})
        self.export = self.folder / "export"
        bundle = self.export / export.BUNDLE
        (bundle / "asset.aimodel").mkdir(parents=True)
        (bundle / "asset.aimodel/main.hash").write_bytes(bytes.fromhex(MAIN_HASH))
        (bundle / "metadata.json").write_text("{}")
        recorded = {path.relative_to(bundle).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                    for path in sorted(bundle.rglob("*")) if path.is_file()}
        write_json(self.export / "export.json", {"schema": export.SCHEMA, "passed": True, "layers": 6,
                                                 "layer_types": self.kinds, "functions": ["main", "prefill"],
                                                 "model": model_record(self.contract),
                                                 "bundle": {"folder": export.BUNDLE, "name": "asset",
                                                            "main_hash": MAIN_HASH, "files": recorded}})
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
                 "an export of other functions": ("export", lambda record: record.update(functions="main,prefill"))}
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
