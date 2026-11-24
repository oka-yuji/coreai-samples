import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError:
    torch = None

from Tools.Conversion.adapters.gemma4_31b import mtp, mtp_native
from Tools.Conversion.adapters.gemma4_31b.assistant import SYNTHETIC
from Tools.Conversion.storage import read_json, write_json
from Tools.Conversion.tests.test_gemma4_31b_mtp import (CONFIG, FIXED, FIXTURE_RECORDS, SHARD, assistant_state, decoded,
                                                         opens_counted, reads_counted, reported_small, tensor_reads,
                                                         write_assistant, write_records, write_target)

SEED = 11
STEPS = 3
NAMES = ["input_ids", "hidden_states", "position_ids", "k_sliding", "v_sliding", "k_full", "v_full"]
TYPES = ["int32", "float16", "int32", "float16", "float16", "float16", "float16"]
ROWS = [("alpha-0001", "alpha", 20, 0, 20), ("alpha-0001", "alpha", 23, 3, 23), ("alpha-0001", "alpha", 26, 6, 26),
        ("alpha-0001", "alpha", 28, 8, 28), ("beta-0001", "beta", 1100, 0, 1025), ("beta-0001", "beta", 1102, 2, 1025),
        ("beta-0001", "beta", 1104, 4, 1025), ("beta-0001", "beta", 1105, 5, 1025)]
ARRAYS = {"logits": ("float32", [8, 301]), "drafts": ("int32", [8, 3]), "top2_ids": ("int32", [8, 3, 2]),
          "top2_probabilities": ("float64", [8, 3, 2])}
ROUNDED = ["hidden_states", "k_sliding", "v_sliding", "k_full", "v_full"]
WIDTH = {"int32": 4, "float16": 2, "float32": 4, "float64": 8}
NUMPY = {"int32": "<i4", "float16": "<f2", "float32": "<f4", "float64": "<f8"}
SCHEMA = "gemma4-31b-mtp-native-inputs/1"
LISTED = ["drafts.bin", "logits.bin", "record.json", "rows", "top2_ids.bin", "top2_probabilities.bin"]


def shapes(anchor, sliding):
    return [[1, 1], [1, 1, 48], [1, 1], [1, 1, 2, sliding, 16], [1, 1, 2, sliding, 16], [1, 1, 1, anchor, 32],
            [1, 1, 1, anchor, 32]]


def size(shape, dtype):
    return math.prod(shape) * WIDTH[dtype]


def array_of(path, dtype, shape):
    return np.frombuffer(path.read_bytes(), dtype=NUMPY[dtype]).reshape(shape)


def ranked(values):
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))[:2]
    peak = max(values)
    total = sum(math.exp(value - peak) for value in values)
    return order, [math.exp(values[index] - peak) / total for index in order]


class ToyDrafter:
    def __init__(self, steps):
        generator = torch.Generator().manual_seed(SEED)
        self.steps, width = steps, CONFIG["hidden_size"]
        self.table = torch.randn(CONFIG["vocab_size"], 16, generator=generator)
        self.mixes = [torch.randn(size, 16, generator=generator) for size in (width, 32, 32, 32, 32)]
        self.back = torch.randn(16, width, generator=generator)

    def check(self, input_ids, hidden_states, position_ids, k_sliding, v_sliding, k_full, v_full):
        expected = ((input_ids, (1, 1), torch.int32), (position_ids, (1, 1), torch.int32),
                    (hidden_states, (1, 1, CONFIG["hidden_size"]), torch.float32),
                    (k_sliding, None, torch.float32), (v_sliding, None, torch.float32), (k_full, None, torch.float32),
                    (v_full, None, torch.float32))
        if any(tensor.dtype != dtype or shape is not None and tuple(tensor.shape) != shape
               for tensor, shape, dtype in expected) or k_sliding.shape != v_sliding.shape or k_full.shape != v_full.shape \
                or tuple(k_sliding.shape[:3]) != (1, 1, 2) or k_sliding.shape[4] != 16 \
                or tuple(k_full.shape[:3]) != (1, 1, 1) or k_full.shape[4] != 32:
            raise ValueError("The toy drafter inputs differ from the contract")

    def outputs(self, input_ids, hidden_states, position_ids, k_sliding, v_sliding, k_full, v_full):
        self.check(input_ids, hidden_states, position_ids, k_sliding, v_sliding, k_full, v_full)
        parts = (hidden_states[0, 0], k_sliding[0, 0].mean(dim=1).reshape(-1), v_sliding[0, 0].mean(dim=1).reshape(-1),
                 k_full[0, 0].mean(dim=1).reshape(-1), v_full[0, 0].mean(dim=1).reshape(-1))
        mixed = sum(part @ mix for part, mix in zip(parts, self.mixes)) / 32
        feature = torch.tanh(self.table[input_ids[0, 0]] + mixed + position_ids[0, 0].to(torch.float32) / 2048)
        return (feature @ self.table.T).reshape(1, 1, -1), (feature @ self.back).reshape(1, 1, -1)

    def __call__(self, input_ids, hidden_states, position_ids, k_sliding, v_sliding, k_full, v_full):
        self.check(input_ids, hidden_states, position_ids, k_sliding, v_sliding, k_full, v_full)
        token, hidden, drafted = input_ids, hidden_states, []
        for _ in range(self.steps):
            logits, hidden = self.outputs(token, hidden, position_ids, k_sliding, v_sliding, k_full, v_full)
            token = logits.argmax(dim=-1).to(torch.int32)
            drafted.append(token)
        return torch.cat(drafted, dim=-1)


def toy_drafter(assistant, table, config, steps, size=None):
    return ToyDrafter(steps)


def fixtures(cls):
    return unittest.skipIf(np is None or torch is None, "NumPy, PyTorch and safetensors are required for the folders")(
        mock.patch.object(mtp, "SUBSET", FIXED)(mock.patch.object(mtp, "RECORDS", FIXTURE_RECORDS)(
            mock.patch.object(mtp, "class_drafter", toy_drafter)(cls))))


def headed(records):
    place = mtp.layout(CONFIG)
    for path in sorted((Path(records) / "tf/features").iterdir()):
        tensors = load_file(str(path))
        keys = tensors[f"k{place['sliding']}"].to(torch.float32)
        for head in range(keys.shape[0]):
            keys[head, :, 2] = head + 1
        tensors[f"k{place['sliding']}"], tensors[f"v{place['sliding']}"] = keys.to(torch.bfloat16), (-keys).to(torch.bfloat16)
        save_file({name: tensor.contiguous() for name, tensor in tensors.items()}, str(path))


class Folder:
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        self.records, self.inputs, self.outputs = write_records(self.folder)
        headed(self.records)
        self.target, self.contract = write_target(self.folder, torch.zeros(CONFIG["vocab_size"], CONFIG["hidden_size"]))
        self.assistant, self.assistant_contract = write_assistant(self.folder, SYNTHETIC["assistant"], assistant_state())
        self.output = self.folder / "inputs"

    def write(self, steps=STEPS, output=None):
        return mtp_native.write_inputs(self.contract, self.assistant_contract, self.target, self.assistant, self.records,
                                       self.inputs, self.outputs, steps, output or self.output)

    def names(self, output=None):
        return sorted(path.name for path in (output or self.output).iterdir())

    def model(self):
        return ToyDrafter(STEPS)

    def feature(self):
        return self.records / "tf/features/beta-0001.safetensors"

    def guarded(self):
        def terminated(number, frame):
            raise RuntimeError("SIGTERM reached the handler that was set before the command")

        def interrupted(number, frame):
            raise KeyboardInterrupt("the handler that was set before the command")

        for number, handler in ((signal.SIGTERM, terminated), (signal.SIGINT, interrupted)):
            self.addCleanup(signal.signal, number, signal.signal(number, handler))
        return terminated, interrupted


class Recording:
    def __init__(self, model, altered=None):
        self.model, self.steps, self.altered, self.calls = model, model.steps, altered, []

    def outputs(self, *arguments):
        self.calls.append(arguments)
        return self.model.outputs(*arguments)

    def __call__(self, *arguments):
        self.calls.append(arguments)
        result = self.model(*arguments)
        return self.altered(result) if self.altered else result


class Scripted:
    def __init__(self, logits, projections, drafts, steps):
        self.logits, self.projections, self.drafts, self.steps, self.calls = logits, projections, drafts, steps, []

    def outputs(self, *arguments):
        self.calls.append(arguments)
        index = max(0, len(self.calls) - 2)
        return self.logits[index][None, None], self.projections[index][None, None]

    def __call__(self, *arguments):
        return torch.tensor([self.drafts], dtype=torch.int32)


def recorded_model(altered=None):
    holder, real = [], mtp.class_drafter

    def build(*arguments, **options):
        holder.append(Recording(real(*arguments, **options), altered))
        return holder[0]

    return holder, mock.patch.object(mtp, "class_drafter", side_effect=build)


def fake_record(output, passed=True):
    (output / "rows").mkdir(exist_ok=True)
    return {"schema": SCHEMA, "rows": [{}], "problems": [] if passed else ["a problem"], "passed": passed}


def fake(passed=True):
    return lambda plan, output: fake_record(output, passed)


class TopTwoTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch is required")
    def test_the_two_best_tokens_are_ranked_by_logit_and_then_by_the_smaller_id(self):
        cases = {"no tie": ([0.5, 2.0, -1.0, 1.0], [1, 3]), "a tie for the best": ([1.0, 3.0, 0.0, 3.0, 3.0], [1, 3]),
                 "a tie for the second": ([0.0, 5.0, 2.0, 1.0, 2.0, 2.0], [1, 2]),
                 "a tie of all": ([1.0, 1.0, 1.0], [0, 1]), "negative values": ([-3.0, -1.0, -2.0, -1.0], [1, 3]),
                 "the best last": ([0.0, 0.5, 4.0 - 1e-6, 4.0], [3, 2])}
        for label, (values, expected) in cases.items():
            with self.subTest(case=label):
                tensor = torch.tensor(values, dtype=torch.float32)
                ids, probabilities = mtp_native.top_two(tensor)
                self.assertEqual(ids, expected)
                self.assertEqual(len(probabilities), 2)
                for found, wanted in zip(probabilities, ranked(tensor.tolist())[1]):
                    self.assertAlmostEqual(found, wanted, places=12)
                self.assertTrue(all(type(value) is float for value in probabilities))

    @unittest.skipIf(torch is None, "PyTorch is required")
    def test_the_probabilities_are_those_of_a_softmax_of_temperature_one_in_float64(self):
        values = torch.tensor([10.0, 9.0, 0.0, -5.0], dtype=torch.float32)
        ids, probabilities = mtp_native.top_two(values)
        total = sum(math.exp(value) for value in values.tolist())
        self.assertEqual(ids, [0, 1])
        self.assertAlmostEqual(probabilities[0], math.exp(10.0) / total, places=14)
        self.assertAlmostEqual(probabilities[1], math.exp(9.0) / total, places=14)
        wide = torch.tensor([100.0, 99.5, 0.0], dtype=torch.float32)
        self.assertAlmostEqual(mtp_native.top_two(wide)[1][0], 1 / (1 + math.exp(-0.5) + math.exp(-100.0)), places=14)

    @unittest.skipIf(torch is None, "PyTorch is required")
    def test_logits_that_are_not_finite_are_ranked_without_an_error(self):
        ids, probabilities = mtp_native.top_two(torch.tensor([float("nan"), 1.0, float("inf")], dtype=torch.float32))
        self.assertEqual((len(ids), len(set(ids)), len(probabilities)), (2, 2, 2))


class RowProblemTests(unittest.TestCase):
    def test_a_row_is_a_problem_when_the_unrolled_drafts_differ_from_the_ones_made_step_by_step(self):
        self.assertEqual(mtp_native.row_problems(3, [5, 6, 7], [5, 6, 7], [[5, 1], [6, 2], [7, 3]]), [])
        found = mtp_native.row_problems(3, [5, 6, 8], [5, 6, 7], [[5, 1], [6, 2], [7, 3]])
        self.assertEqual(len(found), 1)
        self.assertIn("row 3", found[0])
        self.assertEqual(len(mtp_native.row_problems(0, [4, 6, 7], [5, 6, 7], [[5, 1], [6, 2], [7, 3]])), 1)

    def test_a_row_is_a_problem_when_a_token_made_step_by_step_is_not_the_best_of_its_step(self):
        found = mtp_native.row_problems(5, [5, 6, 7], [5, 6, 7], [[5, 1], [9, 6], [7, 3]])
        self.assertEqual(len(found), 1)
        self.assertIn("row 5", found[0])
        self.assertEqual(len(mtp_native.row_problems(1, [5, 6, 7], [5, 6, 7], [[9, 5], [6, 2], [7, 3]])), 1)
        self.assertEqual(len(mtp_native.row_problems(1, [5, 6, 7], [5, 6, 7], [[5, 1], [6, 2], [8, 7]])), 1)
        self.assertEqual(len(mtp_native.row_problems(1, [5, 6, 7], [5, 6, 8], [[5, 1], [6, 2], [9, 7]])), 2)
        self.assertEqual(len(mtp_native.row_problems(1, [5, 6, 7], [5, 6, 7], [[9, 5], [9, 6], [8, 3]])), 1)

    def test_the_problems_name_no_token(self):
        found = mtp_native.row_problems(5, [111, 222, 333], [111, 222, 444], [[111, 555], [222, 666], [777, 888]])
        self.assertEqual(len(found), 2)
        text = " ".join(found)
        for token in (111, 222, 333, 444, 555, 666, 777, 888):
            self.assertNotIn(str(token), text)


@unittest.skipIf(torch is None, "PyTorch is required")
class RoundingTests(unittest.TestCase):
    def arguments(self, hidden, sliding):
        return (torch.tensor([[7]], dtype=torch.int32), torch.tensor(hidden, dtype=torch.float32).reshape(1, 1, -1),
                torch.tensor([[9]], dtype=torch.int32), torch.tensor(sliding, dtype=torch.float32).reshape(1, 1, 1, -1, 1),
                torch.tensor(sliding, dtype=torch.float32).reshape(1, 1, 1, -1, 1) * 0.5,
                torch.zeros(1, 1, 1, 2, 1), torch.ones(1, 1, 1, 2, 1))

    def test_a_float_input_is_rounded_to_float16_and_the_reference_gets_the_rounded_values_in_float32(self):
        hidden = [1.0, 1e-8, 3e-8, 1.0009765625, -2.5e-6, 0.1]
        sliding = [0.5, 7.4e-7, 65504.0]
        half, reference, statistics = mtp_native.rounded(self.arguments(hidden, sliding))
        self.assertEqual([str(value.dtype) for value in half],
                         ["torch.int32", "torch.float16", "torch.int32"] + ["torch.float16"] * 4)
        self.assertEqual([str(value.dtype) for value in reference],
                         ["torch.int32", "torch.float32", "torch.int32"] + ["torch.float32"] * 4)
        expected = [float(np.float16(np.float32(value))) for value in hidden]
        self.assertEqual(half[1].reshape(-1).tolist(), expected)
        self.assertEqual(reference[1].reshape(-1).tolist(), expected)
        self.assertEqual(reference[0].tolist(), [[7]])
        self.assertEqual(reference[2].tolist(), [[9]])
        differences = [abs(a - float(np.float32(b))) for a, b in zip(expected, hidden)]
        self.assertEqual(statistics["hidden_states"], {"elements": 6, "changed": sum(1 for d in differences if d),
                                                       "largest_difference": max(differences), "non_finite": 0})
        self.assertEqual(statistics["k_full"], {"elements": 2, "changed": 0, "largest_difference": 0.0, "non_finite": 0})
        self.assertGreater(statistics["hidden_states"]["changed"], 3)
        self.assertGreater(statistics["k_sliding"]["changed"], 0)
        self.assertEqual(sorted(statistics), sorted(ROUNDED))

    def test_a_value_beyond_the_range_of_float16_is_counted_as_not_finite(self):
        half, reference, statistics = mtp_native.rounded(self.arguments([1.0, 70000.0, -1e6, 2.0], [0.5, 1.0]))
        self.assertEqual(statistics["hidden_states"]["non_finite"], 2)
        self.assertEqual(statistics["hidden_states"]["changed"], 2)
        self.assertEqual(statistics["hidden_states"]["largest_difference"], 0.0)
        self.assertEqual(sum(statistics[name]["non_finite"] for name in ROUNDED if name != "hidden_states"), 0)
        self.assertTrue(torch.isinf(half[1]).any())


@fixtures
class SequenceTests(Folder, unittest.TestCase):
    def test_the_steps_follow_the_best_token_and_the_projection_of_the_step_before(self):
        generator = torch.Generator().manual_seed(5)
        logits = [torch.randn(20, generator=generator) for _ in range(3)]
        projections = [torch.randn(6, generator=generator) for _ in range(3)]
        best = [int(logit.argmax()) for logit in logits]
        model = Scripted(logits, projections, [3, 5, 8], 3)
        arguments = (torch.tensor([[4]], dtype=torch.int32), torch.ones(1, 1, 6), torch.tensor([[9]], dtype=torch.int32),
                     torch.zeros(1, 1, 1, 2, 1), torch.zeros(1, 1, 1, 2, 1), torch.zeros(1, 1, 1, 3, 1),
                     torch.zeros(1, 1, 1, 3, 1))
        found = mtp_native.reference_row(model, arguments, 3)
        self.assertEqual(len(model.calls), 4)
        self.assertEqual(found["logits"].tolist(), logits[0].tolist())
        self.assertEqual(found["drafts"], [3, 5, 8])
        self.assertEqual(found["repeated"], best)
        self.assertEqual(model.calls[0][0].tolist(), [[4]])
        self.assertEqual(model.calls[1][0].tolist(), [[4]])
        self.assertTrue(torch.equal(model.calls[1][1], torch.ones(1, 1, 6)))
        for call, previous in ((2, 0), (3, 1)):
            token, hidden = model.calls[call][0], model.calls[call][1]
            self.assertEqual(token.tolist(), [[best[previous]]])
            self.assertEqual(token.dtype, torch.int32)
            self.assertTrue(torch.equal(hidden, projections[previous][None, None]))
        for call in range(4):
            self.assertEqual(model.calls[call][2].tolist(), [[9]])
        for step in range(3):
            ids, probabilities = ranked(logits[step].tolist())
            self.assertEqual(found["ids"][step], ids)
            for value, wanted in zip(found["probabilities"][step], probabilities):
                self.assertAlmostEqual(value, wanted, places=12)


@fixtures
class WrittenTests(Folder, unittest.TestCase):
    def test_the_files_of_every_row_have_the_names_types_shapes_and_sizes_of_the_inputs(self):
        record = self.write()
        self.assertEqual(self.names(), LISTED)
        self.assertEqual(self.names(self.output / "rows"), [f"{number:03d}" for number in range(8)])
        self.assertEqual(len(record["rows"]), 8)
        for number, (_, _, anchor, _, sliding) in enumerate(ROWS):
            folder = self.output / "rows" / f"{number:03d}"
            with self.subTest(row=number):
                self.assertEqual(self.names(folder), sorted(["inputs.json"] + [f"{name}.bin" for name in NAMES]))
                listed = json.loads((folder / "inputs.json").read_text())
                self.assertEqual(list(listed), ["inputs"])
                wanted = [{"name": name, "dtype": dtype, "shape": shape, "file": f"{name}.bin"}
                          for name, dtype, shape in zip(NAMES, TYPES, shapes(anchor, sliding))]
                self.assertEqual(listed["inputs"], wanted)
                for entry in listed["inputs"]:
                    self.assertEqual(sorted(entry), ["dtype", "file", "name", "shape"])
                    self.assertEqual((folder / entry["file"]).stat().st_size, size(entry["shape"], entry["dtype"]))
                    self.assertTrue(all(type(dimension) is int for dimension in entry["shape"]))

    def test_the_values_of_a_row_are_the_token_the_position_the_hidden_state_and_the_keys_and_values_of_its_anchor(self):
        self.write()
        inputs = {entry["id"]: entry for entry in map(json.loads, self.inputs.read_text().splitlines())}
        outputs = {entry["id"]: entry for entry in map(json.loads, self.outputs.read_text().splitlines())}
        sequences = {name: inputs[name]["input_ids"] + outputs[name]["output_ids"]
                     + ([outputs[name]["stop_token"]] if "stop_token" in outputs[name] else [])
                     for name in ("alpha-0001", "beta-0001")}
        for number, (name, _, anchor, _, sliding) in enumerate(ROWS):
            folder = self.output / "rows" / f"{number:03d}"
            found = {entry: array_of(folder / f"{entry}.bin", dtype, shape)
                     for entry, dtype, shape in zip(NAMES, TYPES, shapes(anchor, sliding))}
            floats = {entry: torch.from_numpy(found[entry].astype(np.float32)) for entry in ROUNDED}
            with self.subTest(row=number):
                self.assertEqual(found["input_ids"].tolist(), [[sequences[name][anchor]]])
                self.assertEqual(found["position_ids"].tolist(), [[anchor]])
                self.assertEqual(decoded(floats["hidden_states"][0, 0]), anchor - 1)
                for head in range(2):
                    marker = torch.full((sliding,), float(head + 1))
                    self.assertTrue(decoded(floats["k_sliding"][0, 0, head]) == list(range(anchor - sliding, anchor)))
                    self.assertTrue(decoded(-floats["v_sliding"][0, 0, head]) == list(range(anchor - sliding, anchor)))
                    self.assertTrue(torch.equal(floats["k_sliding"][0, 0, head, :, 2], marker))
                    self.assertTrue(torch.equal(-floats["v_sliding"][0, 0, head, :, 2], marker))
                self.assertTrue(decoded(floats["k_full"][0, 0, 0]) == list(range(anchor)))
                self.assertTrue(decoded(-floats["v_full"][0, 0, 0]) == list(range(anchor)))

    def test_the_arrays_have_the_fixed_types_and_shapes_and_row_i_of_each_belongs_to_row_i(self):
        record = self.write()
        model = self.model()
        loaded = {name: array_of(self.output / f"{name}.bin", dtype, shape) for name, (dtype, shape) in ARRAYS.items()}
        for name, (dtype, shape) in ARRAYS.items():
            self.assertEqual((self.output / f"{name}.bin").stat().st_size, size(shape, dtype))
            self.assertEqual(record["arrays"][name], {"dtype": dtype, "shape": shape, "bytes": size(shape, dtype),
                                                      "sha256": hashlib.sha256((self.output / f"{name}.bin")
                                                                               .read_bytes()).hexdigest()})
        for number, (_, _, anchor, _, sliding) in enumerate(ROWS):
            folder = self.output / "rows" / f"{number:03d}"
            arguments = []
            for name, dtype, shape in zip(NAMES, TYPES, shapes(anchor, sliding)):
                values = torch.from_numpy(array_of(folder / f"{name}.bin", dtype, shape).copy())
                arguments.append(values.to(torch.float32) if dtype == "float16" else values)
            with self.subTest(row=number), torch.no_grad():
                logits, _ = model.outputs(*arguments)
                self.assertTrue(np.array_equal(loaded["logits"][number], logits[0, 0].numpy()))
                self.assertEqual(loaded["drafts"][number].tolist(), model(*arguments)[0].tolist())
                token, hidden = arguments[0], arguments[1]
                for step in range(STEPS):
                    values, hidden = model.outputs(token, hidden, *arguments[2:])
                    ids, probabilities = ranked(values[0, 0].tolist())
                    self.assertEqual(loaded["top2_ids"][number, step].tolist(), ids)
                    for found, wanted in zip(loaded["top2_probabilities"][number, step].tolist(), probabilities):
                        self.assertAlmostEqual(found, wanted, places=12)
                    self.assertEqual(int(loaded["drafts"][number, step]), ids[0])
                    token = torch.tensor([[ids[0]]], dtype=torch.int32)

    def test_the_record_lists_the_rows_the_files_the_arrays_and_the_sources_and_names_no_path_or_token_list(self):
        record = self.write()
        self.assertEqual(read_json(self.output / "record.json"), record)
        self.assertEqual((record["schema"], record["steps"], record["problems"], record["passed"]),
                         (SCHEMA, STEPS, [], True))
        for number, (name, category, anchor, index, sliding) in enumerate(ROWS):
            row, folder = record["rows"][number], self.output / "rows" / f"{number:03d}"
            with self.subTest(row=number):
                self.assertEqual({key: row[key] for key in ("row", "id", "category", "anchor", "output_index", "sliding",
                                                           "full")},
                                 {"row": number, "id": name, "category": category, "anchor": anchor,
                                  "output_index": index, "sliding": sliding, "full": anchor})
                listing = (folder / "inputs.json").read_bytes()
                self.assertEqual(row["inputs_json"], {"bytes": len(listing), "sha256": hashlib.sha256(listing).hexdigest()})
                self.assertEqual(list(row["files"]), NAMES)
                for entry, dtype, shape in zip(NAMES, TYPES, shapes(anchor, sliding)):
                    data = (folder / f"{entry}.bin").read_bytes()
                    self.assertEqual(row["files"][entry], {"dtype": dtype, "shape": shape, "bytes": len(data),
                                                           "sha256": hashlib.sha256(data).hexdigest()})
        self.assertEqual(sorted(record["rounding"]), sorted(ROUNDED))
        for name in ROUNDED:
            self.assertEqual((record["rounding"][name]["changed"], record["rounding"][name]["largest_difference"],
                              record["rounding"][name]["non_finite"]), (0, 0.0, 0))
        self.assertEqual(record["rounding"]["hidden_states"]["elements"], 8 * 48)
        self.assertEqual(record["rounding"]["k_full"]["elements"], sum(anchor * 32 for _, _, anchor, _, _ in ROWS))
        self.assertEqual(record["rounding"]["k_sliding"]["elements"], sum(2 * sliding * 16 for *_, sliding in ROWS))
        sources = mtp.prepared(self.contract, self.assistant_contract, self.target, self.assistant, self.records,
                               self.inputs, self.outputs, mtp.first_per_category)[3]
        self.assertEqual({key: record[key] for key in ("model", "assistant", "records")}, sources)
        self.assertEqual(record["weights"], "every checkpoint tensor rounded to float16, computed in float32 on the CPU")
        self.assertEqual(sorted(record["versions"]), ["numpy", "safetensors", "torch"])
        self.assertEqual(record["versions"]["torch"], importlib.metadata.version("torch"))
        self.assertEqual(sorted(record["timing"]), ["load_s", "rows_s", "total_s"])
        self.assertEqual(sorted(record["memory"]), ["lifetime_peak_footprint_bytes", "max_rss_bytes"])
        text = json.dumps(record)
        for value in (str(self.folder), str(Path.home()), "/Users", "/private", "/var/folders"):
            self.assertNotIn(value, text)

    def test_values_that_change_in_the_rounding_are_counted_and_the_reference_gets_the_rounded_values(self):
        path = self.feature()
        small = {key: (value.to(torch.float32) * 1e-7).to(torch.bfloat16) for key, value in load_file(str(path)).items()}
        save_file({key: value.contiguous() for key, value in small.items()}, str(path))
        holder, patched = recorded_model()
        with patched:
            record = self.write()
        stored = load_file(str(path))
        place = mtp.layout(CONFIG)
        expected_changed, expected_largest = 0, 0.0
        for number, (name, _, anchor, _, sliding) in enumerate(ROWS):
            if name != "beta-0001":
                continue
            call = holder[0].calls[number * (2 + STEPS)]
            original = stored["postnorm"].to(torch.float32)[anchor - 1].reshape(1, 1, -1)
            wanted = original.to(torch.float16).to(torch.float32)
            self.assertFalse(torch.equal(wanted, original))
            self.assertTrue(torch.equal(call[1], wanted))
            self.assertEqual(call[1].dtype, torch.float32)
            keys = stored[f"k{place['sliding']}"].to(torch.float32)[None, None, :, anchor - sliding:anchor]
            self.assertTrue(torch.equal(call[3], keys.to(torch.float16).to(torch.float32)))
            self.assertFalse(torch.equal(call[3], keys))
            values = stored[f"v{place['full']}"].to(torch.float32)[None, None, :, :anchor]
            self.assertTrue(torch.equal(call[6], values.to(torch.float16).to(torch.float32)))
            folder = self.output / "rows" / f"{number:03d}"
            self.assertTrue(np.array_equal(array_of(folder / "hidden_states.bin", "float16", [1, 1, 48]),
                                           wanted.numpy().astype(np.float16)))
            difference = (wanted - original).abs()
            expected_changed += int((difference > 0).sum())
            expected_largest = max(expected_largest, float(difference.max()))
        self.assertEqual(record["rounding"]["hidden_states"],
                         {"elements": 8 * 48, "changed": expected_changed, "largest_difference": expected_largest,
                          "non_finite": 0})
        self.assertGreater(expected_changed, 0)
        self.assertGreater(record["rounding"]["k_sliding"]["changed"], 0)
        self.assertGreater(record["rounding"]["v_full"]["changed"], 0)
        self.assertEqual(record["rounding"]["k_full"]["non_finite"], 0)

    def test_a_value_that_is_not_finite_after_the_rounding_fails_the_run_and_still_leaves_the_record(self):
        path = self.feature()
        tensors = {key: value.to(torch.float32) for key, value in load_file(str(path)).items()}
        tensors["postnorm"][1104, 0] = 1e5
        save_file({key: value.to(torch.bfloat16).contiguous() for key, value in tensors.items()}, str(path))
        record = self.write()
        self.assertFalse(record["passed"])
        self.assertEqual(record["rounding"]["hidden_states"]["non_finite"], 1)
        self.assertTrue(any("hidden_states" in problem for problem in record["problems"]))
        self.assertEqual(self.names(), LISTED)
        self.assertEqual(read_json(self.output / "record.json"), record)

    def test_drafts_that_differ_between_the_unrolled_and_the_step_by_step_run_fail_the_run_and_leave_the_record(self):
        def altered(result):
            changed = result.clone()
            changed[0, 0] = (changed[0, 0] + 1) % CONFIG["vocab_size"]
            return changed

        holder, patched = recorded_model(altered)
        with patched:
            record = self.write()
        self.assertFalse(record["passed"])
        self.assertEqual(len(record["problems"]), 8)
        self.assertTrue(all(problem.startswith("row ") for problem in record["problems"]))
        self.assertEqual(read_json(self.output / "record.json"), record)

    def test_a_reference_that_is_not_finite_fails_the_run(self):
        real = mtp_native.reference_row

        def not_finite(model, arguments, steps):
            found = real(model, arguments, steps)
            found["logits"] = found["logits"].clone()
            found["logits"][3] = float("nan")
            return found

        with mock.patch.object(mtp_native, "reference_row", side_effect=not_finite):
            record = self.write()
        self.assertFalse(record["passed"])
        self.assertTrue(any("logits" in problem for problem in record["problems"]))
        self.assertEqual(read_json(self.output / "record.json"), record)

    def test_the_model_is_built_with_the_steps_and_the_pinned_size_of_the_configuration(self):
        calls = []
        real = mtp.class_drafter

        def watch(*arguments, **options):
            calls.append((arguments, options))
            return real(*arguments, **options)

        with mock.patch.object(mtp, "class_drafter", side_effect=watch):
            self.write()
        self.assertEqual(len(calls), 1)
        arguments, options = calls[0]
        self.assertEqual(arguments[3], STEPS)
        pinned = read_json(self.assistant_contract)["files"]["config.json"]["bytes"]
        self.assertEqual(pinned, (self.assistant / "config.json").stat().st_size)
        self.assertIn(pinned, list(arguments[4:]) + list(options.values()))


@fixtures
class RefusalTests(Folder, unittest.TestCase):
    def test_an_existing_output_or_a_bad_number_of_steps_is_refused_before_anything_is_read_or_made(self):
        (self.folder / "file").write_text("x")
        (self.folder / "dangling").symlink_to(self.folder / "missing")
        (self.folder / "folder").mkdir()
        (self.folder / "folder/kept").write_text("x")
        events = []
        with opens_counted(events, [self.contract, self.assistant_contract]), \
                mock.patch.object(mtp, "prepared_from") as prepare, mock.patch.object(mtp_native, "compute") as work:
            for name in ("file", "dangling", "folder"):
                with self.subTest(output=name), self.assertRaises(FileExistsError):
                    self.write(output=self.folder / name)
            for steps in (0, -1, True, 1.0, None, "7", 16, 2 ** 40):
                with self.subTest(steps=steps), self.assertRaises(ValueError):
                    self.write(steps=steps)
            prepare.assert_not_called()
            work.assert_not_called()
        self.assertEqual(events, [])
        self.assertEqual(sorted(path.name for path in (self.folder / "folder").iterdir()), ["kept"])
        self.assertFalse(self.output.exists())

    def test_a_file_that_differs_from_its_pinned_size_or_hash_is_refused_before_the_output_and_any_tensor(self):
        files = {"the assistant weights": self.assistant / "model.safetensors",
                 "the assistant configuration": self.assistant / "config.json",
                 "the shard of the table": self.target / SHARD, "the index of the target": self.target / mtp.INDEX}
        for label, path in files.items():
            saved, middle = path.read_bytes(), len(path.read_bytes()) // 2
            changes = {"a byte flipped": (saved[:middle] + bytes([saved[middle] ^ 1]) + saved[middle + 1:],
                                          "differs from its contract"),
                       "a byte appended": (saved + b" ", "B, not")}
            for change, (data, message) in changes.items():
                path.write_bytes(data)
                events, watched = tensor_reads()
                with self.subTest(file=label, change=change), watched(), self.assertRaisesRegex(ValueError, message):
                    self.write()
                self.assertEqual(events, [])
                self.assertFalse(self.output.exists())
            path.write_bytes(saved)
        with mock.patch.object(mtp_native, "compute", side_effect=fake()):
            self.write()
        self.assertEqual(self.names(), ["record.json", "rows"])

    def test_records_and_features_that_differ_from_their_pinned_form_are_refused_before_the_output_and_any_tensor(self):
        subset = read_json(self.records / "subset.json")
        feature = self.feature()
        saved_feature, saved_outputs, saved_inputs = feature.read_bytes(), self.outputs.read_bytes(), self.inputs.read_bytes()

        def outputs_changed():
            self.outputs.write_bytes(saved_outputs + b"\n")

        def inputs_changed():
            self.inputs.write_bytes(saved_inputs + b"\n")

        def subset_counts():
            write_json(self.records / "subset.json", {**subset, "counts": {"alpha": 2, "beta": 2}})

        def feature_in_float32():
            tensors = {key: value.to(torch.float32) for key, value in load_file(str(feature)).items()}
            save_file({key: value.contiguous() for key, value in tensors.items()}, str(feature))

        def feature_missing():
            feature.unlink()

        cases = {"an output record that subset.json does not pin": (outputs_changed, ValueError),
                 "an input record that subset.json does not pin": (inputs_changed, ValueError),
                 "counts that are not the fixed ones": (subset_counts, ValueError),
                 "a feature file in float32": (feature_in_float32, ValueError),
                 "a feature file that is missing": (feature_missing, FileNotFoundError)}
        for label, (change, error) in cases.items():
            change()
            events, watched = tensor_reads()
            with self.subTest(case=label), watched(), self.assertRaises(error):
                self.write()
            self.assertEqual(events, [])
            self.assertFalse(self.output.exists())
            feature.write_bytes(saved_feature)
            self.outputs.write_bytes(saved_outputs)
            self.inputs.write_bytes(saved_inputs)
            write_json(self.records / "subset.json", subset)

    def contracts(self):
        return (("the target contract", self.contract), ("the assistant contract", self.assistant_contract))

    def test_a_contract_that_is_a_link_to_a_valid_contract_is_refused(self):
        for label, path in self.contracts():
            saved, copy_of = path.read_bytes(), self.folder / f"{path.name}.copy"
            copy_of.write_bytes(saved)
            path.unlink()
            path.symlink_to(copy_of)
            with self.subTest(file=label), mock.patch.object(mtp_native, "compute", side_effect=fake()), \
                    self.assertRaisesRegex(ValueError, "is a link"):
                self.write()
            self.assertFalse(self.output.exists())
            path.unlink()
            path.write_bytes(saved)

    def test_a_contract_above_the_size_limit_is_refused_by_what_is_read_whatever_size_is_reported(self):
        limit = 1048576
        for label, path in self.contracts():
            saved = path.read_bytes()
            before = 0 if path == self.contract else len(self.contract.read_bytes())
            path.write_bytes(saved + b" " * (limit + 100000 - len(saved)))
            total = []
            with self.subTest(file=label), reads_counted(total), mock.patch.object(os, "stat", reported_small(os.stat)), \
                    mock.patch.object(os, "fstat", reported_small(os.fstat)), \
                    mock.patch.object(mtp_native, "compute", side_effect=fake()), \
                    self.assertRaisesRegex(ValueError, "holds more than 1048576 B"):
                self.write()
            self.assertEqual(sum(total), before + limit + 1)
            self.assertFalse(self.output.exists())
            path.write_bytes(saved + b" " * (limit - len(saved)))
            with self.subTest(file=label, size="at the limit"), mock.patch.object(mtp_native, "compute", side_effect=fake()):
                self.write(output=self.folder / f"at-the-limit-{path.name}")
            path.write_bytes(saved)

    def test_a_contract_is_read_to_its_end_and_its_limit_is_judged_when_a_read_returns_less_than_asked(self):
        limit = 1048576
        for label, path in self.contracts():
            saved = path.read_bytes()
            with self.subTest(file=label, size="as written"), reads_counted([], 7), \
                    mock.patch.object(mtp_native, "compute", side_effect=fake()):
                self.write(output=self.folder / f"short-{path.name}")
            path.write_bytes(saved + b" " * (limit - len(saved)))
            with self.subTest(file=label, size="at the limit"), reads_counted([], 4099), \
                    mock.patch.object(mtp_native, "compute", side_effect=fake()):
                self.write(output=self.folder / f"short-at-the-limit-{path.name}")
            path.write_bytes(saved + b" " * (limit - len(saved) + 1))
            with self.subTest(file=label, size="above the limit"), reads_counted([], 4099), \
                    mock.patch.object(mtp_native, "compute", side_effect=fake()), \
                    self.assertRaisesRegex(ValueError, "holds more than 1048576 B"):
                self.write()
            path.write_bytes(saved)

    def test_each_contract_is_opened_once_and_the_hashes_in_the_record_come_from_the_bytes_that_were_read(self):
        events, plans = [], []

        def keeping(plan, output):
            plans.append(plan)
            return fake_record(output)

        with opens_counted(events, [self.contract, self.assistant_contract]), \
                mock.patch.object(mtp, "load_contract") as reread, mock.patch.object(mtp, "target_contract") as target, \
                mock.patch.object(mtp, "load_assistant") as assistant, \
                mock.patch.object(mtp_native, "compute", side_effect=keeping):
            record = self.write()
        self.assertEqual(sorted(events), sorted(os.path.realpath(path) for _, path in self.contracts()))
        for call in (reread, target, assistant):
            call.assert_not_called()
        self.assertNotEqual(self.contract.read_bytes(), self.assistant_contract.read_bytes())
        self.assertEqual(len(plans), 1)
        self.assertEqual({name: plans[0]["sources"][name]["contract_sha256"] for name in ("model", "assistant")},
                         {"model": hashlib.sha256(self.contract.read_bytes()).hexdigest(),
                          "assistant": hashlib.sha256(self.assistant_contract.read_bytes()).hexdigest()})
        self.assertEqual(self.names(), ["record.json", "rows"])
        self.assertTrue(record["passed"])

    def test_contracts_that_are_not_in_the_pinned_form_are_refused_before_the_output(self):
        pinned = read_json(self.assistant_contract)
        entries = pinned["files"]
        changes = {"a revision that is not a full hash": {**pinned, "revision": "627c5ec"},
                   "a missing file": {**pinned, "files": {"config.json": entries["config.json"]}},
                   "a size as a float": {**pinned, "files": {**entries, "config.json": {
                       **entries["config.json"], "bytes": float(entries["config.json"]["bytes"])}}},
                   "a hash in capitals": {**pinned, "files": {**entries, "config.json": {
                       **entries["config.json"], "sha256": entries["config.json"]["sha256"].upper()}}}}
        for label, change in changes.items():
            write_json(self.assistant_contract, change)
            with self.subTest(change=label), self.assertRaises(ValueError):
                self.write()
            self.assertFalse(self.output.exists())
        write_json(self.assistant_contract, pinned)
        good = read_json(self.contract)
        documents = {"a width as a float": {**good, "config": {**good["config"], "hidden_size": 48.0}},
                     "a window as a boolean": {**good, "config": {**good["config"], "sliding_window": True}},
                     "a field that the contract does not have": {**good, "extra": 1},
                     "another reference": {**good, "reference": {**good["reference"], "compute": "float32"}}}
        for label, document in documents.items():
            write_json(self.contract, document)
            with self.subTest(change=label), self.assertRaises(ValueError):
                self.write()
            self.assertFalse(self.output.exists())


@fixtures
class FailureTests(Folder, unittest.TestCase):
    def failure(self, name, output=None):
        return read_json((output or self.output) / "failure.json") == {"schema": SCHEMA, "failure": name}

    def test_a_run_that_stops_after_creating_its_output_leaves_only_the_name_of_the_error(self):
        self.guarded()

        def signalled(*arguments):
            signal.raise_signal(signal.SIGINT)

        def after_rows(plan, output):
            fake_record(output)
            raise RuntimeError("a message with details")

        cases = [("RuntimeError", RuntimeError("a message with details"), RuntimeError, ["failure.json"]),
                 ("KeyboardInterrupt", KeyboardInterrupt(), KeyboardInterrupt, ["failure.json"]),
                 ("SystemExit", SystemExit(3), SystemExit, ["failure.json"]),
                 ("MemoryError", MemoryError(), MemoryError, ["failure.json"]),
                 ("a signal", signalled, KeyboardInterrupt, ["failure.json"]),
                 ("a failure after the rows were written", after_rows, RuntimeError, ["failure.json", "rows"])]
        for label, effect, error, names in cases:
            output = self.folder / label.replace(" ", "-")
            with self.subTest(error=label), mock.patch.object(mtp_native, "compute", side_effect=effect), \
                    self.assertRaises(error):
                self.write(output=output)
            self.assertEqual(self.names(output), names)
            self.assertTrue(self.failure(error.__name__, output))

    def test_a_signal_while_the_output_is_made_lets_the_creation_finish_and_leaves_a_failure_record(self):
        self.guarded()
        real = Path.mkdir
        for label, number in (("SIGINT", signal.SIGINT), ("SIGTERM", signal.SIGTERM)):
            output, fired = self.folder / label, []

            def made_and_signalled(path, *arguments, output=output, fired=fired, number=number, **options):
                real(path, *arguments, **options)
                if path == output and not fired:
                    fired.append(path)
                    signal.raise_signal(number)

            with self.subTest(signal=label), mock.patch.object(Path, "mkdir", made_and_signalled), \
                    mock.patch.object(mtp_native, "compute") as work, self.assertRaises(KeyboardInterrupt):
                self.write(output=output)
            work.assert_not_called()
            self.assertEqual(self.names(output), ["failure.json"])
            self.assertTrue(self.failure("KeyboardInterrupt", output))

    def test_a_signal_while_the_record_is_written_lets_the_write_finish_and_replaces_it_by_a_failure_record(self):
        self.guarded()
        real = mtp_native.write_json
        for label, number in (("SIGINT", signal.SIGINT), ("SIGTERM", signal.SIGTERM)):
            events, output = [], self.folder / label

            def logged(path, value, events=events, number=number):
                if Path(path).name == "record.json":
                    events.append("signal")
                    signal.raise_signal(number)
                real(path, value)
                events.append(Path(path).name)

            with self.subTest(signal=label), mock.patch.object(mtp_native, "write_json", logged), \
                    mock.patch.object(mtp_native, "compute", side_effect=fake()), self.assertRaises(KeyboardInterrupt):
                self.write(output=output)
            self.assertEqual(events, ["signal", "record.json", "failure.json"])
            self.assertEqual(self.names(output), ["failure.json", "rows"])
            self.assertTrue(self.failure("KeyboardInterrupt", output))

    def test_a_signal_while_the_failure_record_is_written_lets_the_write_finish(self):
        self.guarded()
        real = mtp_native.write_json
        for label, number in (("SIGINT", signal.SIGINT), ("SIGTERM", signal.SIGTERM)):
            events, output = [], self.folder / label

            def logged(path, value, events=events, number=number):
                if Path(path).name == "failure.json":
                    events.append("signal")
                    signal.raise_signal(number)
                real(path, value)
                events.append(Path(path).name)

            with self.subTest(signal=label), mock.patch.object(mtp_native, "write_json", logged), \
                    mock.patch.object(mtp_native, "compute", side_effect=RuntimeError("a message")), \
                    self.assertRaises(KeyboardInterrupt):
                self.write(output=output)
            self.assertEqual(events, ["signal", "failure.json"])
            self.assertEqual(self.names(output), ["failure.json"])
            self.assertTrue(self.failure("RuntimeError", output))

    def test_a_sigterm_after_the_output_is_made_is_an_interruption_that_leaves_a_failure_record(self):
        terminated, interrupted = self.guarded()
        seen = {}

        def terminate(*arguments):
            seen.update(term=signal.getsignal(signal.SIGTERM), interrupt=signal.getsignal(signal.SIGINT))
            signal.raise_signal(signal.SIGTERM)

        with mock.patch.object(mtp_native, "compute", side_effect=terminate), self.assertRaises(KeyboardInterrupt):
            self.write()
        self.assertEqual(self.names(), ["failure.json"])
        self.assertTrue(self.failure("KeyboardInterrupt"))
        self.assertIsNot(seen["term"], terminated)
        self.assertIs(seen["interrupt"], interrupted)

    def test_a_sigterm_while_the_inputs_are_checked_is_an_interruption_and_makes_no_output(self):
        terminated, interrupted = self.guarded()

        def terminate(*arguments):
            signal.raise_signal(signal.SIGTERM)

        with mock.patch.object(mtp, "prepared_from", side_effect=terminate), self.assertRaises(KeyboardInterrupt):
            self.write()
        self.assertFalse(self.output.exists())
        self.assertEqual((signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)), (terminated, interrupted))

    def test_the_handlers_that_were_set_before_are_set_again_after_a_success_a_failure_and_a_refusal(self):
        terminated, interrupted = self.guarded()
        with mock.patch.object(mtp_native, "compute", side_effect=fake()):
            self.write(output=self.folder / "succeeded")
        self.assertEqual((signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)), (terminated, interrupted))
        with mock.patch.object(mtp_native, "compute", side_effect=RuntimeError("a message")), self.assertRaises(RuntimeError):
            self.write(output=self.folder / "failed")
        self.assertEqual((signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)), (terminated, interrupted))
        with self.assertRaises(FileExistsError):
            self.write(output=self.folder / "succeeded")
        self.assertEqual((signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)), (terminated, interrupted))

    def signalled_while_the_handlers_are_set_again(self, number, terminated, seen, output):
        real = signal.signal

        def setting(signum, handler):
            if signum == signal.SIGTERM and handler is terminated and not seen:
                seen.append((output / "record.json").exists())
                signal.raise_signal(number)
            return real(signum, handler)

        return mock.patch.object(signal, "signal", setting)

    def test_a_signal_while_the_handlers_are_set_again_after_the_record_is_written_replaces_it_by_a_failure_record(self):
        for label, number in (("SIGINT", signal.SIGINT), ("SIGTERM", signal.SIGTERM)):
            terminated, _ = self.guarded()
            output, seen = self.folder / label, []
            with self.subTest(signal=label), self.signalled_while_the_handlers_are_set_again(
                    number, terminated, seen, output), mock.patch.object(mtp_native, "compute", side_effect=fake()), \
                    self.assertRaises(KeyboardInterrupt):
                self.write(output=output)
            self.assertEqual(seen, [True])
            self.assertEqual(self.names(output), ["failure.json", "rows"])
            self.assertTrue(self.failure("KeyboardInterrupt", output))

    def test_a_signal_while_the_handlers_are_set_again_after_a_refusal_before_the_output_is_made_writes_nothing(self):
        (self.folder / "folder").mkdir()
        (self.folder / "folder/kept").write_text("x")
        for label, number in (("SIGINT", signal.SIGINT), ("SIGTERM", signal.SIGTERM)):
            for case, arguments in (("a bad number of steps", {"steps": 0, "output": self.folder / label}),
                                    ("an existing output", {"output": self.folder / "folder"})):
                terminated, _ = self.guarded()
                seen = []
                with self.subTest(signal=label, case=case), self.signalled_while_the_handlers_are_set_again(
                        number, terminated, seen, arguments["output"]), \
                        mock.patch.object(mtp_native, "compute") as work, self.assertRaises(KeyboardInterrupt):
                    self.write(**arguments)
                work.assert_not_called()
                self.assertEqual(seen, [False])
                self.assertFalse((self.folder / label).exists())
                self.assertEqual(sorted(path.name for path in (self.folder / "folder").iterdir()), ["kept"])

    def test_an_output_that_another_process_made_after_the_check_is_refused_without_a_file_written_into_it(self):
        real, fired = Path.mkdir, []

        def raced(path, *arguments, **options):
            if path == self.output and not fired:
                fired.append(path)
                real(path)
                (path / "theirs").write_text("x")
            return real(path, *arguments, **options)

        with mock.patch.object(Path, "mkdir", raced), mock.patch.object(mtp_native, "compute") as work, \
                self.assertRaises(FileExistsError):
            self.write()
        work.assert_not_called()
        self.assertEqual(self.names(), ["theirs"])

    def test_the_record_is_the_last_file_written_and_a_run_that_did_not_pass_is_recorded_with_passed_false(self):
        real = mtp_native.write_json
        for passed in (True, False):
            output, writes = self.folder / str(passed), []

            def logged(path, value, output=output, writes=writes):
                writes.append((Path(path).name, sorted(item.name for item in output.iterdir())))
                real(path, value)

            with self.subTest(passed=passed), mock.patch.object(mtp_native, "write_json", logged), \
                    mock.patch.object(mtp_native, "compute", side_effect=fake(passed)):
                record = self.write(output=output)
            self.assertEqual(writes, [("record.json", ["rows"])])
            self.assertEqual(read_json(output / "record.json"), record)
            self.assertEqual(record["passed"], passed)
            self.assertEqual(self.names(output), ["record.json", "rows"])

    def test_a_file_that_changes_after_it_was_checked_and_while_the_rows_are_made_leaves_a_failure_record_only(self):
        files = {"the assistant weights": self.assistant / "model.safetensors",
                 "the assistant configuration": self.assistant / "config.json",
                 "the shard of the table": self.target / SHARD, "the index of the target": self.target / mtp.INDEX,
                 "a feature file": self.feature()}
        for label, path in files.items():
            saved, middle, output = path.read_bytes(), len(path.read_bytes()) // 2, self.folder / label.replace(" ", "-")

            def changing(plan, folder, path=path, saved=saved, middle=middle):
                record = fake_record(folder)
                path.write_bytes(saved[:middle] + bytes([saved[middle] ^ 1]) + saved[middle + 1:])
                return record

            with self.subTest(file=label), mock.patch.object(mtp_native, "compute", side_effect=changing), \
                    self.assertRaises(ValueError):
                self.write(output=output)
            self.assertEqual(self.names(output), ["failure.json", "rows"])
            self.assertTrue(self.failure("ValueError", output))
            path.write_bytes(saved)

    def test_the_results_of_the_check_before_the_rows_and_the_check_after_them_must_be_the_same(self):
        real, calls = mtp.prepared_from, []

        def changing(*arguments):
            calls.append(arguments)
            found = real(*arguments)
            if len(calls) == 1:
                return found
            sources = json.loads(json.dumps(found[3]))
            sources["records"]["subset_sha256"] = "0" * 64
            return (*found[:3], sources, found[4])

        with mock.patch.object(mtp, "prepared_from", changing), mock.patch.object(mtp_native, "compute", side_effect=fake()), \
                self.assertRaisesRegex(ValueError, "changed while the rows were made"):
            self.write()
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.names(), ["failure.json", "rows"])

    def test_the_files_are_checked_before_the_rows_are_made_and_again_before_the_record_is_written(self):
        events, real_prepared, real_write = [], mtp.prepared_from, mtp_native.write_json

        def prepared(*arguments):
            events.append("check")
            return real_prepared(*arguments)

        def rows(plan, output):
            events.append("rows")
            return fake_record(output)

        def written(path, value):
            events.append(Path(path).name)
            real_write(path, value)

        with mock.patch.object(mtp, "prepared_from", prepared), mock.patch.object(mtp_native, "compute", side_effect=rows), \
                mock.patch.object(mtp_native, "write_json", written):
            self.write()
        self.assertEqual(events, ["check", "rows", "check", "record.json"])


COMMAND = ["inputs", "--model", "a.json", "--assistant", "b.json", "--snapshot", "target", "--assistant-snapshot",
           "assistant", "--records", "records", "--inputs", "i.jsonl", "--outputs", "o.jsonl", "--steps", "7", "--output",
           "out"]


class CommandTests(unittest.TestCase):
    def test_the_command_passes_its_arguments_prints_the_summary_and_exits_with_one_unless_the_run_passed(self):
        for passed in (True, False):
            record = {"rows": [{}, {}, {}], "passed": passed}
            with self.subTest(passed=passed), mock.patch.object(mtp_native, "write_inputs", return_value=record) as write, \
                    mock.patch("sys.argv", ["mtp_native", *COMMAND]), \
                    mock.patch("sys.stdout", new_callable=io.StringIO) as printed:
                code = mtp_native.main()
            write.assert_called_once_with(Path("a.json"), Path("b.json"), Path("target"), Path("assistant"),
                                          Path("records"), Path("i.jsonl"), Path("o.jsonl"), 7, Path("out"))
            self.assertEqual(json.loads(printed.getvalue()), {"passed": passed, "rows": 3})
            self.assertEqual(code, 0 if passed else 1)

    def test_the_command_has_one_subcommand(self):
        for arguments in (["run"], ["acceptance"], ["export-weights"], []):
            with self.subTest(arguments=arguments), mock.patch.object(mtp_native, "write_inputs") as write, \
                    mock.patch("sys.argv", ["mtp_native", *arguments]), \
                    mock.patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as raised:
                mtp_native.main()
            write.assert_not_called()
            self.assertEqual(raised.exception.code, 2)

    def test_the_command_refuses_an_option_that_chooses_an_implementation_when_every_other_argument_is_valid(self):
        with mock.patch.object(mtp_native, "write_inputs", return_value={"rows": [], "passed": True}) as write, \
                mock.patch("sys.argv", ["mtp_native", *COMMAND, "--implementation", "class"]), \
                mock.patch("sys.stderr", new_callable=io.StringIO) as printed, self.assertRaises(SystemExit) as raised:
            mtp_native.main()
        write.assert_not_called()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unrecognized arguments: --implementation class", printed.getvalue())
