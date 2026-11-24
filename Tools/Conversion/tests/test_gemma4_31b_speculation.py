import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import threading
import time
import tracemalloc
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

from Tools.Conversion.adapters.gemma4_31b import layers, native, speculation
from Tools.Conversion.adapters.gemma4_31b.prepare import TOKENIZER_FILES, load_contract
from Tools.Conversion.adapters.pinned import sha256
from Tools.Conversion.storage import digest, read_json, write_json
from Tools.Conversion.tests.test_gemma4_31b_layers import TEXT, complete_export, write_contract
from Tools.Conversion.tests.test_gemma4_31b_native import MAIN_HASH, bounded, edit, model_record

HIDDEN = TEXT["hidden_size"]


def npz(array):
    buffer = io.BytesIO()
    np.savez(buffer, array=array)
    return buffer.getvalue()


def array_bytes(shape, dtype="<f4", fortran=False, values=0):
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(buffer, {"descr": dtype, "fortran_order": fortran, "shape": shape})
    return buffer.getvalue() + bytes(values)


def watched_arrays():
    real = np.ndarray

    class Watching(type):
        def __instancecheck__(cls, item):
            return isinstance(item, real)

        def __subclasscheck__(cls, item):
            return issubclass(item, real)

        def __call__(cls, *arguments, **keywords):
            raise AssertionError("an array was made before its header was checked")

    return Watching("ndarray", (), {})

FAKE = """#!{python}
import hashlib, json, math, sys
from pathlib import Path
import numpy as np

plan = json.loads(Path(__file__).with_suffix(".json").read_text())
pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
single = dict(pairs)
ids = json.loads(Path(single["--tokens"]).read_text())
start, chunk = int(single["--start"]), int(single["--chunk"])
name = next(key for key, value in plan["conditions"].items() if value == [len(ids), start, chunk])
change = plan["changes"].get(name, {{}})
with Path(plan["calls"]).open("a") as calls:
    calls.write(name + "\\n")
source = "short" if name.startswith("short") else "long"
hidden = plan["hidden"]
reference = {{key: np.load(path) for key, path in plan["arrays"][source].items()}}
rows = len(ids) - start
Path(single["--output"]).write_bytes(np.zeros((rows, plan["vocab"]), dtype="<f2").tobytes())
outputs = []
requested = [(option[2:-len("-output")], value) for option, value in pairs
             if option in ("--main-output", "--prefill-output")]
for function, argument in requested:
    output, path = argument.split("=", 1)
    key = function + "." + output
    span = slice(start, len(ids)) if function == "main" else slice(0, start)
    if output == "hidden_states":
        layer = change.get("hidden_from_layer")
        values = reference["hidden_states"][span] if layer is None else \\
            reference["tap_hidden_states"][span, (layer - 1) * hidden:layer * hidden]
    else:
        order = change.get("order", {{}}).get(key, plan["taps"])
        values = np.concatenate([reference["tap_hidden_states"][span, (layer - 1) * hidden:layer * hidden]
                                 for layer in order], axis=1)
    values = np.array(values, dtype=np.float64)
    generator = np.random.default_rng(3)
    for target, section, positions, cosine in change.get("cosine", []):
        if target != key:
            continue
        for row in positions:
            part = values[row, section * hidden:(section + 1) * hidden]
            noise = generator.normal(size=part.shape)
            noise -= noise @ part / (part @ part) * part
            noise *= np.linalg.norm(part) * math.tan(math.acos(cosine)) / np.linalg.norm(noise)
            values[row, section * hidden:(section + 1) * hidden] = part + noise
    for target, row, column in change.get("nan", []):
        if target == key:
            values[row, column] = np.nan
    for target, row in change.get("zero", []):
        if target == key:
            values[row] = 0
    data = values.astype("<f2")
    data = data[:len(data) - change.get("drop_rows", {{}}).get(key, 0)].tobytes()
    Path(path).write_bytes(data)
    entry = {{"function": function, "name": output, "rows": len(values), "width": values.shape[1],
             "sha256": hashlib.sha256(data).hexdigest()}}
    entry.update(change.get("outputs", {{}}).get(key, {{}}))
    if key not in change.get("drop_outputs", []):
        outputs.append(entry)
prefill = plan["prefill_chunk"]
record = {{"record": "logits", "token_count": len(ids), "start": start, "chunk": chunk,
          "tokens_sha256": hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest(),
          "shape": [rows, plan["vocab"]], "dtype": "float16", "byte_order": "little", "prefix_chunk": prefill,
          "prefix_calls": -(-start // prefill), "main_calls": -(-rows // chunk), "main_hash": plan["main_hash"],
          "cached": True, "non_finite_count": 0, "load_s": 0.1, "prefix_s": 0.0, "scoring_s": 0.2,
          "scoring_call_s": 0.1, "specialize_s": 0.0, "outputs": outputs}}
if start:
    record["prefix_function"] = "prefill"
record.update(change.get("record", {{}}))
print(change.get("stdout", json.dumps(record)))
sys.exit(change.get("exit", 0))
"""


@unittest.skipIf(np is None, "NumPy is required for the comparison")
class SectionTests(unittest.TestCase):
    def test_a_section_passes_at_the_limits_and_fails_just_beyond_each_of_them(self):
        cosines = np.array([0.99, 0.999, 0.9995])
        passed = speculation.decided({1: 0.5, 2: 1.0, 3: 4.0}, 1, cosines, 10)
        self.assertEqual(passed["checks"], {"mixup": True, "cosine_smallest": True, "cosine_median": True})
        self.assertEqual((passed["nearest"], passed["ratio"], passed["below"]), ("2", 0.5, []))
        failing = {"mixup": speculation.decided({1: 0.5000001, 2: 1.0}, 1, cosines, 10),
                   "cosine_smallest": speculation.decided({1: 0.1, 2: 1.0}, 1, np.array([0.98999, 0.9995, 0.9995]), 10),
                   "cosine_median": speculation.decided({1: 0.1, 2: 1.0}, 1, np.array([0.99, 0.99899, 0.9995]), 10)}
        for check, result in failing.items():
            with self.subTest(check=check):
                self.assertEqual({key for key, value in result["checks"].items() if not value}, {check})
        self.assertEqual(failing["cosine_smallest"]["below"], [[10, 0.98999]])
        self.assertEqual(failing["cosine_smallest"]["cosine"]["smallest_position"], 10)

    def test_the_ratio_to_every_other_layer_also_makes_the_own_layer_the_nearest(self):
        self.assertTrue(speculation.decided({1: 0.0, 2: 1e-30, 3: 1.0}, 1, np.array([1.0]), 0)["checks"]["mixup"])
        self.assertFalse(speculation.decided({1: 1.0, 2: 0.4, 3: 3.0}, 1, np.array([1.0]), 0)["checks"]["mixup"])

    def test_a_section_fails_the_mixup_check_without_a_ratio_when_another_layer_is_not_apart(self):
        result = speculation.decided({1: 0.0, 2: 0.0, 3: 1.0}, 1, np.array([1.0]), 0)
        self.assertEqual((result["checks"]["mixup"], result["ratio"], result["nearest"]), (False, None, "2"))
        json.dumps(result, allow_nan=False)

    def test_each_tap_section_is_compared_with_every_layer_of_the_reference(self):
        generator = np.random.default_rng(0)
        reference = generator.normal(size=(5, 4 * 3)).astype(np.float32)
        sections = speculation.taps(reference[:, :9].astype(np.float16), reference, [1, 2, 3], [1, 2, 3, 4], 3, 7)
        self.assertEqual([section["layer"] for section in sections], [1, 2, 3])
        self.assertTrue(all(all(section["checks"].values()) for section in sections))
        self.assertEqual(sorted(sections[0]["squares"]), ["1", "2", "3", "4"])
        shifted = speculation.taps(reference[:, 3:12], reference, [1, 2, 3], [1, 2, 3, 4], 3, 7)
        self.assertEqual([section["nearest"] for section in shifted], ["2", "3", "4"])
        self.assertFalse(any(section["checks"]["mixup"] for section in shifted))
        self.assertEqual(shifted[0]["below"][0][0], 7)
        skipping = speculation.taps(np.concatenate([reference[:, 3:6], reference[:, 9:12]], axis=1), reference, [2, 4],
                                    [1, 2, 3, 4], 3, 7)
        self.assertEqual([section["layer"] for section in skipping], [2, 4])
        self.assertTrue(all(all(section["checks"].values()) for section in skipping))

    def test_the_final_hidden_state_is_compared_with_the_final_norm_and_the_last_layer(self):
        generator = np.random.default_rng(1)
        norm, last = generator.normal(size=(2, 6, 4))
        result = speculation.hidden(norm.astype(np.float16), norm, last, 0)
        self.assertEqual(sorted(result["squares"]), ["final_norm", "last_layer"])
        self.assertTrue(all(result["checks"].values()))
        self.assertFalse(speculation.hidden(last, norm, last, 0)["checks"]["mixup"])

    def test_a_row_of_zeros_counts_as_a_cosine_of_zero(self):
        self.assertEqual(speculation.cosines(np.zeros((1, 3)), np.ones((1, 3))).tolist(), [0.0])


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
        inputs, self.arrays = {}, {}
        for name in ("short", "long"):
            ids = layers.tokens(name, self.config)
            arrays = {"hidden_states": generator.normal(size=(len(ids), HIDDEN)),
                      "tap_hidden_states": generator.normal(size=(len(ids), 6 * HIDDEN))}
            inputs[name] = {"positions": len(ids), "tokens_sha256": digest(ids)}
            self.arrays[name] = {}
            for output, array in arrays.items():
                path = self.reference / f"{name}.{output}.npy"
                np.save(path, array.astype(np.float32))
                inputs[name][output] = {**layers.file_record(path), "dtype": "float32", "shape": list(array.shape),
                                        "non_finite": 0}
                self.arrays[name][output] = str(path)
        write_json(self.reference / "record.json", {"schema": layers.SCHEMA, "implementation": "original", "layers": 6,
                                                     "layer_types": self.kinds, "tap_layers": [1, 2, 3, 4, 5, 6],
                                                     "weights": layers.WEIGHTS, "model": model_record(self.contract),
                                                     "inputs": inputs})
        self.export = complete_export(self.folder / "export", self.contract)
        self.verifier = self.folder / "verifier"
        self.calls = self.folder / "calls.txt"

    def write_fake(self, changes=None):
        conditions = {name: [len(layers.tokens(source, self.config)), start, chunk]
                      for name, source, start, chunk in speculation.CONDITIONS}
        plan = {"conditions": conditions, "changes": changes or {}, "arrays": self.arrays, "hidden": HIDDEN,
                "taps": [1, 2, 3, 4, 5], "vocab": TEXT["vocab_size"], "main_hash": MAIN_HASH, "calls": str(self.calls),
                "prefill_chunk": native.prefill_chunk()}
        self.verifier.write_text(FAKE.format(python=sys.executable))
        self.verifier.chmod(0o755)
        self.verifier.with_suffix(".json").write_text(json.dumps(plan))

    def run_compare(self, changes=None, label="out"):
        self.write_fake(changes)
        return speculation.compare(self.export, self.reference, self.verifier, self.folder / label, self.contract)

    def conditions(self, result):
        return {entry["name"]: entry for entry in result["conditions"]}

    def failing(self, entry):
        return {key for key, value in entry["checks"].items() if not value}

    def test_matching_outputs_pass_every_check_and_the_comparison_is_saved(self):
        result = self.run_compare()
        self.assertTrue(result["passed"], result)
        self.assertEqual(read_json(self.folder / "out/comparison.json"), result)
        self.assertEqual(self.calls.read_text().split(), [name for name, *_ in speculation.CONDITIONS])
        conditions = self.conditions(result)
        self.assertEqual(sorted(conditions["long-prefill"]["outputs"]),
                         ["main.hidden_states", "main.tap_hidden_states", "prefill.tap_hidden_states"])
        self.assertEqual(sorted(conditions["short-c1"]["outputs"]), ["main.hidden_states", "main.tap_hidden_states"])
        taps = conditions["long-prefill"]["outputs"]["prefill.tap_hidden_states"]
        self.assertEqual((taps["rows"], taps["width"], len(taps["sections"])), (1100, 5 * HIDDEN, 5))
        self.assertEqual([section["layer"] for section in taps["sections"]], [1, 2, 3, 4, 5])
        self.assertEqual(sorted(taps["sections"][0]["squares"]), ["1", "2", "3", "4", "5", "6"])
        main = conditions["long-main"]["outputs"]["main.hidden_states"]
        self.assertEqual((main["rows"], main["width"], main["non_finite"]), (1400, HIDDEN, 0))
        self.assertLess(main["ratio"], 1e-3)
        self.assertGreater(main["cosine"]["smallest"], 0.999)
        norm, reference = (np.load(self.arrays["long"][output]).astype(np.float64) for output in speculation.OUTPUTS)
        written = norm.astype(np.float16).astype(np.float64)
        expected = float(np.square(written - reference[:, 5 * HIDDEN:]).sum())
        self.assertAlmostEqual(main["squares"]["last_layer"], expected, delta=1e-9 * expected)
        third = reference[:1100, 2 * HIDDEN:3 * HIDDEN].astype(np.float16).astype(np.float64)
        self.assertAlmostEqual(taps["sections"][2]["squares"]["5"],
                               float(np.square(third - reference[:1100, 4 * HIDDEN:5 * HIDDEN]).sum()),
                               delta=1e-9 * taps["sections"][2]["squares"]["5"])
        self.assertEqual(result["criteria"], {"cosine_smallest": 0.99, "cosine_median": 0.999, "mixup_ratio": 0.5})

    def test_mixed_up_values_fail_the_mixup_check_of_their_output_and_section(self):
        cases = {"a tap layer shifted by one": ("short-c1", {"order": {"main.tap_hidden_states": [1, 3, 3, 4, 5]}},
                                                "main.tap_hidden_states.2.mixup"),
                 "tap sections in the reversed order": ("short-c8",
                                                         {"order": {"main.tap_hidden_states": [5, 4, 3, 2, 1]}},
                                                         "main.tap_hidden_states.1.mixup"),
                 "the last layer before the final norm": ("long-main", {"hidden_from_layer": 6},
                                                          "main.hidden_states.mixup"),
                 "a prefill tap layer shifted by one": ("long-prefill",
                                                        {"order": {"prefill.tap_hidden_states": [2, 2, 3, 4, 5]}},
                                                        "prefill.tap_hidden_states.1.mixup")}
        for label, (name, change, check) in cases.items():
            with self.subTest(case=label):
                result = self.run_compare({name: change}, label=label.replace(" ", "-"))
                conditions = self.conditions(result)
                self.assertFalse(result["passed"])
                self.assertIn(check, self.failing(conditions[name]))
                self.assertEqual([entry["name"] for entry in result["conditions"] if not entry["passed"]], [name])
                self.assertTrue((self.folder / label.replace(" ", "-") / "comparison.json").is_file())

    def test_cosines_below_the_limits_fail_and_the_rows_below_the_smallest_cosine_are_named(self):
        result = self.run_compare({"short-c1": {"cosine": [["main.tap_hidden_states", 2, [3, 7], 0.985]]},
                                   "long-prefill": {"cosine": [["main.hidden_states", 0, list(range(300)), 0.995]]},
                                   "short-c8": {"zero": [["main.hidden_states", 5]]}})
        zero = self.conditions(result)["short-c8"]
        self.assertEqual(self.failing(zero), {"main.hidden_states.cosine_smallest"})
        self.assertEqual(zero["outputs"]["main.hidden_states"]["below"], [[5, 0.0]])
        conditions = self.conditions(result)
        section = conditions["short-c1"]["outputs"]["main.tap_hidden_states"]["sections"][2]
        self.assertEqual(self.failing(conditions["short-c1"]), {"main.tap_hidden_states.3.cosine_smallest"})
        self.assertEqual([position for position, _ in section["below"]], [3, 7])
        self.assertTrue(all(abs(value - 0.985) < 1e-3 for _, value in section["below"]))
        self.assertEqual(self.failing(conditions["long-prefill"]), {"main.hidden_states.cosine_median"})
        self.assertTrue(conditions["long-main"]["passed"])

    def test_non_finite_values_fail_their_output_and_name_their_rows(self):
        result = self.run_compare({"long-main": {"nan": [["main.hidden_states", 5, 3], ["main.hidden_states", 9, 0]]},
                                   "long-prefill": {"nan": [["prefill.tap_hidden_states", 1099, 5 * HIDDEN - 1]]}})
        conditions = self.conditions(result)
        hidden = conditions["long-main"]["outputs"]["main.hidden_states"]
        self.assertEqual((hidden["non_finite"], hidden["non_finite_rows"]), (2, [5, 9]))
        self.assertEqual(self.failing(conditions["long-main"]), {"main.hidden_states.finite"})
        taps = conditions["long-prefill"]["outputs"]["prefill.tap_hidden_states"]
        self.assertEqual((taps["non_finite"], taps["non_finite_rows"]), (1, [1099]))
        self.assertEqual(self.failing(conditions["long-prefill"]), {"prefill.tap_hidden_states.finite"})

    def test_arrays_of_another_shape_fail_their_condition_and_the_comparison_is_saved(self):
        cases = {"a file with a row less": ("short-c1", {"drop_rows": {"main.hidden_states": 1}},
                                            "the main.hidden_states file: short-c1.hidden_states.f16 is not a "
                                            "regular file of 2496 B"),
                 "a recorded width of four layers": ("short-c8",
                                                     {"outputs": {"main.tap_hidden_states": {"width": 4 * HIDDEN}}},
                                                     "the verifier recorded main tap_hidden_states of"),
                 "a recorded row count of the whole input": (
                     "long-prefill", {"outputs": {"prefill.tap_hidden_states": {"rows": 1400}}},
                     "the verifier recorded prefill tap_hidden_states of"),
                 "a missing output": ("long-main", {"drop_outputs": ["main.tap_hidden_states"]},
                                      "the verifier recorded the outputs")}
        for label, (name, change, problem) in cases.items():
            with self.subTest(case=label):
                folder = label.replace(" ", "-")
                result = self.run_compare({name: change}, label=folder)
                entry = self.conditions(result)[name]
                self.assertFalse(result["passed"] or entry["passed"])
                self.assertEqual(len(entry["problems"]), 1)
                self.assertTrue(entry["problems"][0].startswith(problem), entry["problems"])
                if label == "a file with a row less":
                    self.assertNotIn("main.hidden_states", entry["outputs"])
                else:
                    self.assertEqual(entry["checks"], {})
                self.assertEqual(read_json(self.folder / folder / "comparison.json"), result)

    def test_an_output_entry_with_a_missing_key_a_flag_or_another_type_is_malformed(self):
        entry = {"function": "main", "name": "hidden_states", "rows": 1, "width": 1, "sha256": "0" * 64}
        record = {"record": "logits", "token_count": 1, "start": 0, "chunk": 1, "tokens_sha256": "x", "shape": [1, 1],
                  "dtype": "float16", "byte_order": "little", "main_calls": 1, "main_hash": "x", "prefix_calls": 0,
                  "prefix_chunk": 4096, "cached": True, "non_finite_count": 0, "load_s": 0.1, "prefix_s": 0,
                  "scoring_s": 0.1, "scoring_call_s": 0.1, "specialize_s": 0}
        kept, malformed = speculation.validated({**record, "outputs": [entry]})
        self.assertEqual((malformed, kept["outputs"]), ([], [entry]))
        for label, outputs in {"a flag as a row count": [{**entry, "rows": True}],
                               "a flag as a width": [{**entry, "width": False}],
                               "a number as a name": [{**entry, "name": 1}],
                               "a missing hash": [{key: value for key, value in entry.items() if key != "sha256"}],
                               "an extra key": [{**entry, "path": "x"}], "an object": {"0": entry}}.items():
            with self.subTest(case=label):
                self.assertEqual(speculation.validated({**record, "outputs": outputs})[1], ["outputs"])
        self.assertEqual(speculation.validated(record)[1], ["outputs"])

    def test_a_verifier_record_or_file_that_differs_fails_only_its_condition(self):
        cases = {"another file hash": {"outputs": {"main.hidden_states": {"sha256": "0" * 64}}},
                 "a flag as a row count": {"outputs": {"main.hidden_states": {"rows": True}}},
                 "a non-finite call time": {"record": {"scoring_call_s": float("nan")}},
                 "another function": {"outputs": {"main.hidden_states": {"function": "prefill"}}},
                 "a non-zero exit": {"exit": 1}, "no record": {"stdout": "not json"},
                 "another prefix function": {"record": {"prefix_function": "main"}}}
        for label, change in cases.items():
            with self.subTest(case=label):
                name = "long-prefill"
                result = self.run_compare({name: change}, label=label.replace(" ", "-"))
                self.assertEqual([entry["name"] for entry in result["conditions"] if not entry["passed"]], [name])
                self.assertTrue(self.conditions(result)[name]["problems"])

    def test_a_condition_that_cannot_be_compared_is_recorded_and_the_others_are_still_compared(self):
        original, calls = speculation.taps, []

        def failing(*arguments):
            calls.append(arguments[-1])
            if len(calls) == 2:
                raise RuntimeError("private-example-sentinel")
            return original(*arguments)

        with mock.patch.object(speculation, "taps", side_effect=failing):
            result = self.run_compare()
        entry = self.conditions(result)["short-c8"]
        self.assertEqual(entry["problems"], ["the condition could not be compared (RuntimeError)"])
        self.assertEqual((entry["outputs"], entry["checks"], entry["passed"]), ({}, {}, False))
        self.assertEqual([item["name"] for item in result["conditions"] if item["passed"]],
                         ["short-c1", "long-main", "long-prefill"])
        self.assertNotIn("private-example-sentinel", (self.folder / "out/comparison.json").read_text())

    def test_files_that_are_not_regular_files_of_the_recorded_size_fail_their_condition(self):
        original = speculation.regular_file

        def swapped(path, size=None, keep=False, limit=None):
            if Path(path).name == "long-main.hidden_states.f16":
                target = Path(path).with_name("long-main.target.f16")
                Path(path).rename(target)
                Path(path).symlink_to(target)
            return original(path, size, keep, limit)

        with mock.patch.object(speculation, "regular_file", side_effect=swapped):
            result = self.run_compare()
        entry = self.conditions(result)["long-main"]
        self.assertEqual(len(entry["problems"]), 1)
        self.assertTrue(entry["problems"][0].startswith(
            "the main.hidden_states file: long-main.hidden_states.f16 is not a regular file of"), entry["problems"])
        self.assertNotIn("main.hidden_states", entry["outputs"])

    def test_every_array_and_the_reference_record_are_read_with_a_limit_that_their_shape_or_the_record_gives(self):
        calls, read_file, read_record = [], speculation.read_file, speculation.read_record

        def recording(path, size, limit=None):
            calls.append((Path(path).name, size, limit))
            return read_file(path, size, limit)

        def recording_record(path, limit=None):
            calls.append((Path(path).name, None, limit))
            return read_record(path, limit)

        with mock.patch.object(speculation, "read_file", side_effect=recording), mock.patch.object(
                speculation, "read_record", side_effect=recording_record):
            self.assertTrue(self.run_compare()["passed"])
        limits = {name: (size, limit) for name, size, limit in calls}
        self.assertEqual(limits["record.json"], (None, speculation.RECORD_LIMIT))
        values = {"short": 26, "long": 1400}
        for name, rows in values.items():
            for output, width in (("hidden_states", HIDDEN), ("tap_hidden_states", 6 * HIDDEN)):
                data = rows * width * 4
                size, limit = limits[f"{name}.{output}.npy"]
                self.assertEqual((limit, size is not None and data <= size <= limit),
                                 (data + speculation.HEADER_LIMIT, True))
        for condition, source, start, chunk in speculation.CONDITIONS:
            rows = values[source] - start
            size, limit = limits[f"{condition}.hidden_states.f16"]
            self.assertEqual((size, limit), (rows * HIDDEN * 2, rows * HIDDEN * 2))
            size, limit = limits[f"{condition}.tap_hidden_states.f16"]
            self.assertEqual((size, limit), (rows * 5 * HIDDEN * 2, rows * 5 * HIDDEN * 2))
            if start:
                size, limit = limits[f"{condition}.prefill.tap_hidden_states.f16"]
                self.assertEqual((size, limit), (start * 5 * HIDDEN * 2, start * 5 * HIDDEN * 2))

    def test_a_header_that_names_far_more_values_than_the_file_holds_makes_no_array_and_allocates_little(self):
        claimed = 2**29 // (HIDDEN * 4)
        content = array_bytes((claimed, HIDDEN), values=26 * HIDDEN * 4)
        tracemalloc.start()
        try:
            with mock.patch("numpy.ndarray", watched_arrays()), mock.patch(
                    "numpy.lib.format.read_array", side_effect=AssertionError("an array was read before its header")), \
                    self.assertRaises(ValueError) as caught:
                speculation.npy(content, (26, HIDDEN))
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertIn("the header gives another shape, type or order", str(caught.exception))
        self.assertLess(peak, 1 << 20)
        made = speculation.npy(array_bytes((26, HIDDEN), values=26 * HIDDEN * 4), (26, HIDDEN))
        self.assertEqual((made.shape, made.dtype), ((26, HIDDEN), np.dtype("<f4")))

    def test_a_failure_after_the_output_is_created_leaves_a_failure_record(self):
        with mock.patch.object(speculation, "run_verifier", side_effect=KeyboardInterrupt), \
                self.assertRaises(KeyboardInterrupt):
            self.run_compare(label="interrupted")
        real, written = speculation.write_json, []

        def full_once(path, value):
            written.append(Path(path).name)
            if len(written) == 1:
                raise OSError("full")
            real(path, value)

        with mock.patch.object(speculation, "write_json", side_effect=full_once), self.assertRaises(OSError):
            self.run_compare(label="full")
        self.assertEqual(written, ["comparison.json", "failure.json"])

        def interrupted_after(path, value):
            real(path, value)
            if Path(path).name == "comparison.json":
                raise KeyboardInterrupt

        with mock.patch.object(speculation, "write_json", side_effect=interrupted_after), \
                self.assertRaises(KeyboardInterrupt):
            self.run_compare(label="late")
        for label, error in (("interrupted", "KeyboardInterrupt"), ("full", "OSError"), ("late", "KeyboardInterrupt")):
            with self.subTest(error=error, label=label):
                self.assertEqual(read_json(self.folder / label / "failure.json"),
                                 {"schema": speculation.SCHEMA, "passed": False, "error": error})
                self.assertFalse((self.folder / label / "comparison.json").exists())

    def test_a_comparison_holds_an_interruption_back_until_its_folder_or_one_of_its_records_is_written(self):
        real_mkdir, real_write = os.mkdir, speculation.write_json
        self.write_fake()

        def interrupted_at(label, point, failure=None):
            output, written = self.folder / label, []

            def after_mkdir(path, *arguments, **keywords):
                real_mkdir(path, *arguments, **keywords)
                if point == "mkdir" and Path(path) == output:
                    signal.raise_signal(signal.SIGINT)

            def around_write(path, value):
                if point == Path(path).name:
                    signal.raise_signal(signal.SIGINT)
                real_write(path, value)
                written.append(Path(path).name)

            with mock.patch("os.mkdir", side_effect=after_mkdir), \
                    mock.patch.object(speculation, "write_json", side_effect=around_write), \
                    mock.patch.object(speculation, "condition", side_effect=failure or speculation.condition), \
                    self.assertRaises(KeyboardInterrupt):
                speculation.compare(self.export, self.reference, self.verifier, output, self.contract)
            return output, written, read_json(output / "failure.json")["error"]

        output, written, error = interrupted_at("folder", "mkdir")
        self.assertEqual((written, error, sorted(path.name for path in output.iterdir())),
                         (["failure.json"], "KeyboardInterrupt", ["failure.json"]))
        self.assertFalse(self.calls.exists())
        output, written, error = interrupted_at("record", "comparison.json")
        self.assertEqual((written, error), (["comparison.json", "failure.json"], "KeyboardInterrupt"))
        self.assertFalse((output / "comparison.json").exists())
        output, written, error = interrupted_at("failure", "failure.json", RuntimeError("private-example-sentinel"))
        self.assertEqual((written, error), (["failure.json"], "RuntimeError"))
        self.assertNotIn("private-example-sentinel", (output / "failure.json").read_text())

    def test_a_comparison_stopped_by_sigterm_after_its_folder_is_made_leaves_a_failure_record(self):
        self.write_fake()
        output = self.folder / "terminated"

        def terminate(*arguments, **keywords):
            self.assertIs(signal.getsignal(signal.SIGTERM), layers.raise_interruption)
            signal.raise_signal(signal.SIGTERM)

        with layers.terminated_as_interrupt(), mock.patch.object(speculation, "condition", side_effect=terminate), \
                self.assertRaises(KeyboardInterrupt):
            speculation.compare(self.export, self.reference, self.verifier, output, self.contract)
        self.assertEqual(read_json(output / "failure.json")["error"], "KeyboardInterrupt")
        self.assertFalse((output / "comparison.json").exists())

    def test_a_comparison_stopped_by_sigterm_while_it_waits_for_the_verifier_stops_the_verifier(self):
        self.verifier.write_text('#!/bin/sh\necho $$ > "$0.pid"\nexec sleep 600\n')
        self.verifier.chmod(0o755)
        pidfile, output, seen, finished = Path(f"{self.verifier}.pid"), self.folder / "terminated", {}, threading.Event()

        def terminate_when_it_waits():
            for _ in range(600):
                if pidfile.exists() and pidfile.read_text().strip():
                    break
                time.sleep(0.05)
            seen["pid"], seen["handler"] = int(pidfile.read_text()), signal.getsignal(signal.SIGTERM)
            if seen["handler"] is layers.raise_interruption:
                os.kill(os.getpid(), signal.SIGTERM)
            while not finished.wait(5):
                seen["forced"] = True
                with contextlib.suppress(ProcessLookupError, ValueError):
                    os.kill(int(pidfile.read_text()), signal.SIGKILL)

        thread = threading.Thread(target=terminate_when_it_waits, daemon=True)
        thread.start()
        try:
            with layers.terminated_as_interrupt(), self.assertRaises(KeyboardInterrupt):
                speculation.compare(self.export, self.reference, self.verifier, output, self.contract)
        finally:
            finished.set()
        thread.join(10)
        self.assertNotIn("forced", seen)
        self.assertIs(seen["handler"], layers.raise_interruption)
        with self.assertRaises(ProcessLookupError):
            os.kill(seen["pid"], 0)
        self.assertEqual(read_json(output / "failure.json")["error"], "KeyboardInterrupt")
        self.assertFalse((output / "comparison.json").exists())

    def test_a_comparison_that_finds_its_folder_made_by_another_process_writes_no_record_into_it(self):
        self.write_fake()
        taken = self.folder / "taken"
        taken.mkdir()
        with mock.patch("os.path.lexists", return_value=False), mock.patch.object(
                speculation, "run_verifier") as run, self.assertRaises(FileExistsError):
            speculation.compare(self.export, self.reference, self.verifier, taken, self.contract)
        run.assert_not_called()
        self.assertEqual(list(taken.iterdir()), [])

    def test_inputs_that_must_not_be_compared_stop_before_anything_is_written(self):
        def reference_of(change):
            return lambda: edit(self.reference / "record.json", change)

        def entry_of(name, output, **change):
            return reference_of(lambda record: record["inputs"][name][output].update(change))

        def array_changed():
            np.save(self.reference / "short.hidden_states.npy", np.zeros((26, HIDDEN), dtype=np.float32))

        def rewritten(path, write):
            write(path)
            edit(self.reference / "record.json", lambda record: record["inputs"]["long"]["tap_hidden_states"].update(
                layers.file_record(path)))

        def non_finite():
            array = np.load(self.reference / "long.tap_hidden_states.npy")
            array[3, 4] = np.inf
            rewritten(self.reference / "long.tap_hidden_states.npy", lambda path: np.save(path, array))

        def compressed():
            array = np.load(self.reference / "long.tap_hidden_states.npy")
            rewritten(self.reference / "long.tap_hidden_states.npy",
                      lambda path: path.write_bytes(npz(array)))

        def linked():
            path = self.reference / "short.hidden_states.npy"
            path.rename(self.reference / "target.npy")
            path.symlink_to(self.reference / "target.npy")

        def piped(path):
            def make():
                path.unlink()
                os.mkfifo(path)
            return make

        def main_only():
            shutil.rmtree(self.export)
            complete_export(self.export, self.contract, functions=("main",))

        def looped():
            self.verifier.unlink()
            self.verifier.symlink_to(self.verifier)

        every_layer, unreadable = "does not hold the output of every layer", "the reference record is unreadable"

        def guarded(*arguments):
            if max(arguments) > 10**6:
                raise MemoryError("a range of a number from a record")
            return range(*arguments)

        read_file, read_record = speculation.read_file, speculation.read_record

        def bounded_file(path, size, limit=None):
            if size is not None and size > 10**8:
                raise MemoryError("a read of a size from a record")
            return read_file(path, size, limit)

        def bounded_record(path, limit=None):
            if os.stat(path).st_size > 10**8:
                raise MemoryError("a read of a file of any size")
            return read_record(path, limit)

        def never(*arguments, **keywords):
            raise AssertionError("an array was read before its header was checked")

        def array_file(shape, dtype="<f4", fortran=False, values=26 * HIDDEN * 4):
            return array_bytes(shape, dtype, fortran, values)

        def headed(content):
            def make():
                (self.reference / "short.hidden_states.npy").write_bytes(content)
                edit(self.reference / "record.json",
                     lambda record: record["inputs"]["short"]["hidden_states"].update(bytes=len(content)))
            return make

        wrong_header = "the header gives another shape, type or order than the reference record"

        def sized(name, output, size):
            def make():
                edit(self.reference / "record.json", lambda record: record["inputs"][name][output].update(bytes=size))
                os.truncate(self.reference / f"{name}.{output}.npy", size)
            return make

        no_size = "the reference record gives a size of the short hidden_states that its shape does not allow"
        data = 26 * HIDDEN * 4

        def padded(path, size):
            with path.open("ab") as handle:
                handle.write(b" " * size)
        entry_problem = "the reference record does not give the short hidden_states"
        cases = {"a reference without the last layer": (
                     reference_of(lambda record: record.update(tap_layers=[1, 2, 3, 4, 5])), every_layer),
                 "a flag among the tap layers": (
                     reference_of(lambda record: record.update(tap_layers=[True, 2, 3, 4, 5, 6])), every_layer),
                 "a reference of the model class": (
                     reference_of(lambda record: record.update(implementation="class")), "not a run of the original"),
                 "a reference of restored weights": (
                     reference_of(lambda record: record.update(weights=layers.RESTORED)), "does not use the weights"),
                 "a reference of other layers": (
                     reference_of(lambda record: record.update(layers=12)), "recorded layer types that differ"),
                 "a reference of a trillion layers": (
                     reference_of(lambda record: record.update(layers=10**12)),
                     "The first 1000000000000 layers must end with a full-attention layer"),
                 "a reference of another input": (
                     reference_of(lambda record: record["inputs"]["long"].update(positions=1399)),
                     "did not read the long input"),
                 "a fractional position count": (
                     reference_of(lambda record: record["inputs"]["short"].update(positions=26.0)),
                     "did not read the short input"),
                 "a fractional byte count": (
                     reference_of(lambda record: record["inputs"]["short"]["hidden_states"].update(
                         bytes=float(record["inputs"]["short"]["hidden_states"]["bytes"]))), entry_problem),
                 "a recorded shape of another width": (
                     entry_of("short", "hidden_states", shape=[26, HIDDEN + 1]), entry_problem),
                 "a recorded dtype of float16": (entry_of("short", "hidden_states", dtype="float16"), entry_problem),
                 "a recorded non-finite value": (entry_of("short", "hidden_states", non_finite=1), entry_problem),
                 "a reference array of a size far above its shape": (sized("short", "hidden_states", 10**10), no_size),
                 "a reference array of a size just above its shape and header": (
                     sized("short", "hidden_states", data + speculation.HEADER_LIMIT + 1), no_size),
                 "a reference array of a size just below its shape": (sized("short", "hidden_states", data - 1), no_size),
                 "a reference record above the size of a record": (
                     lambda: padded(self.reference / "record.json", 2 << 20), unreadable),
                 "a reference array whose header names a trillion rows": (
                     headed(array_file((10**12, HIDDEN))), wrong_header),
                 "a reference array whose header names another type": (
                     headed(array_file((26, HIDDEN), "<f8")), wrong_header),
                 "a reference array whose header names the order of Fortran": (
                     headed(array_file((26, HIDDEN), fortran=True)), wrong_header),
                 "a reference array whose header names a flipped byte order": (
                     headed(array_file((26, HIDDEN), ">f4")), wrong_header),
                 "a reference array with more values than its header gives": (
                     headed(array_file((26, HIDDEN), values=26 * HIDDEN * 4 + 4)),
                     "the values are not those that the header gives"),
                 "a reference array that differs from its record": (
                     array_changed, "the short hidden_states differ from the SHA-256"),
                 "a non-finite reference": (non_finite, "the long tap_hidden_states of the reference hold non-finite"),
                 "a reference array in the NPZ format": (compressed, "not an array in the .npy format"),
                 "a reference array behind a symbolic link": (linked, "short.hidden_states.npy is not a regular file"),
                 "a reference record that is a named pipe": (piped(self.reference / "record.json"), unreadable),
                 "a reference array that is a named pipe": (
                     piped(self.reference / "long.hidden_states.npy"), "long.hidden_states.npy is not a regular file"),
                 "an export that did not pass": (
                     lambda: edit(self.export / "export.json", lambda record: record.update(passed=False)),
                     "the export is not a passed export"),
                 "an export without the prefill function": (main_only, "the export has no prefill function"),
                 "a missing verifier": (lambda: self.verifier.unlink(), "the verifier is unreadable"),
                 "a verifier that is a named pipe": (piped(self.verifier), "the verifier is unreadable"),
                 "a verifier that links to itself": (looped, "the verifier is unreadable")}
        for label, (make, problem) in cases.items():
            with self.subTest(case=label):
                self.write_fake()
                shutil.copytree(self.reference, self.folder / "saved-reference")
                shutil.copytree(self.export, self.folder / "saved-export")
                try:
                    make()
                    output = self.folder / label.replace(" ", "-")
                    with mock.patch.object(speculation, "run_verifier") as run, \
                            mock.patch.object(speculation, "range", guarded, create=True), \
                            mock.patch.object(speculation, "read_file", side_effect=bounded_file), \
                            mock.patch.object(speculation, "read_record", side_effect=bounded_record), \
                            mock.patch("numpy.lib.format.read_array", side_effect=never), \
                            mock.patch("numpy.ndarray", watched_arrays()):
                        thread, outcome = bounded(lambda: speculation.compare(
                            self.export, self.reference, self.verifier, output, self.contract))
                    self.assertFalse(thread.is_alive())
                    self.assertIsInstance(outcome.get("error"), ValueError, outcome)
                    self.assertIn(problem, str(outcome.get("error")))
                    run.assert_not_called()
                    self.assertFalse(os.path.lexists(output))
                    self.assertFalse(self.calls.exists())
                finally:
                    shutil.rmtree(self.reference)
                    shutil.rmtree(self.export)
                    shutil.move(self.folder / "saved-reference", self.reference)
                    shutil.move(self.folder / "saved-export", self.export)
                    self.verifier.unlink(missing_ok=True)

    def test_an_existing_output_is_refused_before_anything_is_read(self):
        target = self.folder / "target"
        target.write_text("target")
        for label, make in {"file": lambda path: path.write_text("x"), "folder": lambda path: path.mkdir(),
                            "dangling link": lambda path: path.symlink_to(self.folder / "missing"),
                            "link": lambda path: path.symlink_to(target)}.items():
            output = self.folder / label.replace(" ", "-")
            make(output)
            with self.subTest(output=label), mock.patch.object(speculation, "regular_file") as read, \
                    mock.patch.object(speculation, "load_contract") as load, self.assertRaises(FileExistsError):
                speculation.compare(self.export, self.reference, self.verifier, output, self.contract)
            read.assert_not_called()
            load.assert_not_called()
            self.assertEqual(target.read_text(), "target")

    def test_the_verifier_that_is_hashed_is_the_one_that_runs_when_it_is_given_by_its_name_only(self):
        shadow = self.folder / "elsewhere" / "verifier"
        shadow.parent.mkdir()
        shadow.write_text("#!/bin/sh\nexit 3\n")
        shadow.chmod(0o755)
        self.write_fake()
        previous = os.getcwd()
        os.chdir(self.folder)
        self.addCleanup(os.chdir, previous)
        with mock.patch.dict(os.environ, {"PATH": f"{shadow.parent}{os.pathsep}{os.environ['PATH']}"}):
            result = speculation.compare(self.export, self.reference, Path("verifier"), self.folder / "out",
                                         self.contract)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["verifier"], {"bytes": self.verifier.stat().st_size, "sha256": sha256(self.verifier)})
        self.assertEqual(self.calls.read_text().split(), [name for name, *_ in speculation.CONDITIONS])

    def test_the_command_turns_sigterm_into_an_interruption_while_it_compares(self):
        seen, previous = [], signal.getsignal(signal.SIGTERM)
        result = {"passed": True, "conditions": []}
        arguments = ["speculation", "--export", "e", "--reference", "r", "--verifier", "v", "--output", "o"]
        with mock.patch.object(sys, "argv", arguments), mock.patch("sys.stdout"), mock.patch.object(
                speculation, "compare", side_effect=lambda *items: seen.append(signal.getsignal(signal.SIGTERM)) or result):
            self.assertEqual(speculation.main(), 0)
        self.assertEqual(seen, [layers.raise_interruption])
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    def test_the_command_prints_a_summary_and_exits_with_one_unless_it_passes(self):
        result = {"passed": False, "conditions": [{"name": "long-main", "passed": False, "checks": {}}]}
        arguments = ["speculation", "--export", "e", "--reference", "r", "--verifier", "v", "--output", "o"]
        for passed in (True, False):
            with self.subTest(passed=passed), mock.patch.object(sys, "argv", arguments), \
                    mock.patch.object(speculation, "compare", return_value={**result, "passed": passed}) as compare, \
                    mock.patch("sys.stdout"):
                self.assertEqual(speculation.main(), 0 if passed else 1)
            self.assertEqual(compare.call_args.args[:4], (Path("e"), Path("r"), Path("v"), Path("o")))


if __name__ == "__main__":
    unittest.main()
