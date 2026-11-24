import builtins
import contextlib
import copy
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    torch = None

from Tools.LLMEvaluation import layer1
from Tools.Conversion.adapters.gemma4_31b import mtp
from Tools.Conversion.adapters.gemma4_31b.assistant import SYNTHETIC, assistant_config, checkpoint_shapes
from Tools.Conversion.adapters.gemma4_31b.prepare import TOKENIZER_FILES, load_contract
from Tools.Conversion.adapters.pinned import sha256
from Tools.Conversion.storage import read_json, write_json

RECIPE = Path(__file__).resolve().parents[1] / "recipes/gemma4_31b"
SLIDING, FULL = "sliding_attention", "full_attention"
STOPS = {1, 106, 50}
CONFIG = {"num_hidden_layers": 12, "layer_pattern": [SLIDING] * 5 + [FULL], "hidden_size": 48, "intermediate_size": 96,
          "vocab_size": 301, "sliding_window": 1024, "num_key_value_heads": 2, "num_global_key_value_heads": 1,
          "head_dim": 16, "global_head_dim": 32, "attention_k_eq_v": True, "final_logit_softcapping": 30.0,
          "tie_word_embeddings": True, "bos_token_id": 2, "eos_token_id": [1, 106, 50]}
ITEMS = (("alpha-0001", 20, 9, 106), ("alpha-0002", 30, 4, 106), ("beta-0001", 1100, 6, None))
FIXED = {"alpha": 2, "beta": 1}
SHARD = "model-00001-of-00001.safetensors"


def tokens_sha256(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode("utf-8")).hexdigest()


def line(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def sequences(number, prompt, output, stop):
    generator = np.random.default_rng(number)
    ids = [2] + generator.integers(3, CONFIG["vocab_size"], prompt - 1).tolist()
    drafts = generator.integers(3, CONFIG["vocab_size"], output).tolist()
    while set(drafts) & set(CONFIG["eos_token_id"]):
        drafts = generator.integers(3, CONFIG["vocab_size"], output).tolist()
    return ids, drafts, drafts + ([stop] if stop else [])


def coded(length, width):
    position = torch.arange(length, dtype=torch.float32)
    return torch.cat([(position // 32)[:, None], (position % 32)[:, None], torch.zeros(length, width - 2)], dim=-1)


def decoded(tensor):
    return (tensor[..., 0] * 32 + tensor[..., 1]).to(torch.int64).tolist()


def write_records(folder):
    folder = Path(folder)
    inputs, outputs, items, features = [], [], [], folder / "records/tf/features"
    features.mkdir(parents=True)
    place = mtp.layout(CONFIG)
    for number, (name, prompt, output, stop) in enumerate(ITEMS):
        ids, drafts, teacher = sequences(number, prompt, output, stop)
        inputs.append({"id": name, "input_ids": ids, "input_sha256": tokens_sha256(ids), "max_new_tokens": 16})
        outputs.append({"id": name, "input_sha256": tokens_sha256(ids), "input_tokens": prompt, "non_finite": 0,
                        "output_ids": drafts, "stop": "stop_token" if stop else "limit",
                        "top2": [[token, 0.6, 5 if token != 5 else 6, 0.3] for token in teacher],
                        **({"stop_token": stop} if stop else {})})
        items.append({"id": name, "category": name.rsplit("-", 1)[0], "max_new_tokens": 16,
                      "stop": "stop_token" if stop else "limit", "stop_token": stop, "input_tokens": prompt,
                      "output_tokens": output, "teacher_tokens": len(teacher)})
        end = prompt + len(teacher) - 1
        first, start = max(0, prompt - 2048), max(0, prompt - 1101)
        tensors = {"postnorm": coded(end, CONFIG["hidden_size"])[first:],
                   f"k{place['sliding']}": coded(end, CONFIG["head_dim"])[None, start:].repeat(
                       CONFIG["num_key_value_heads"], 1, 1),
                   f"k{place['full']}": coded(end, CONFIG["global_head_dim"])[None].repeat(
                       CONFIG["num_global_key_value_heads"], 1, 1),
                   "L0": torch.zeros(2, 2)}
        tensors[f"v{place['sliding']}"] = -tensors[f"k{place['sliding']}"]
        tensors[f"v{place['full']}"] = -tensors[f"k{place['full']}"]
        save_file({key: value.to(torch.bfloat16).contiguous() for key, value in tensors.items()},
                  str(features / f"{name}.safetensors"))
    evaluation = folder / "evaluation"
    evaluation.mkdir()
    (evaluation / "inputs.jsonl").write_text("".join(map(line, inputs)))
    (evaluation / "outputs.jsonl").write_text("".join(map(line, outputs)))
    counts = {}
    for entry in items:
        counts[entry["category"]] = counts.get(entry["category"], 0) + 1
    write_json(folder / "records/subset.json", {
        "inputs": {"path": "inputs.jsonl", "sha256": sha256(evaluation / "inputs.jsonl")},
        "outputs": {"path": "outputs.jsonl", "sha256": sha256(evaluation / "outputs.jsonl")},
        "rule": "first N ids per category in numeric id order", "counts": counts, "items": items,
        "total": {"items": len(items), "input_tokens": sum(entry["input_tokens"] for entry in items),
                  "output_tokens": sum(entry["output_tokens"] for entry in items),
                  "teacher_tokens": sum(entry["teacher_tokens"] for entry in items)}})
    return folder / "records", evaluation / "inputs.jsonl", evaluation / "outputs.jsonl"


def write_contract(folder, files):
    contract = read_json(RECIPE / "model.json")
    contract.update(files=files, config=dict(CONFIG),
                    tokenizer={"policy": "keep-distributed", "files": {name: files[name]["sha256"]
                                                                       for name in TOKENIZER_FILES}})
    path = Path(folder) / "model.json"
    write_json(path, contract)
    return path


def write_target(folder, table):
    snapshot = Path(folder) / "target"
    snapshot.mkdir()
    save_file({mtp.TABLE: table.to(torch.bfloat16).contiguous()}, str(snapshot / SHARD))
    write_json(snapshot / mtp.INDEX, {"metadata": {}, "weight_map": {mtp.TABLE: SHARD}})
    for name in TOKENIZER_FILES:
        (snapshot / name).write_text(name)
    files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(snapshot.iterdir())}
    return snapshot, write_contract(folder, files)


def write_assistant(folder, document, state):
    snapshot = Path(folder) / "assistant"
    snapshot.mkdir()
    write_json(snapshot / "config.json", document)
    save_file({name: tensor.to(torch.bfloat16).contiguous() for name, tensor in state.items()},
              str(snapshot / "model.safetensors"))
    contract = Path(folder) / "assistant.json"
    write_json(contract, {"model_id": "google/gemma-4-31B-it-assistant", "revision": "0" * 40,
                          "files": {name: {"bytes": (snapshot / name).stat().st_size, "sha256": sha256(snapshot / name)}
                                    for name in mtp.ASSISTANT_FILES}})
    return snapshot, contract


def fixture_records():
    if np is None or torch is None:
        return {}
    with tempfile.TemporaryDirectory() as folder:
        _, inputs, outputs = write_records(folder)
        return {"inputs": sha256(inputs), "outputs": sha256(outputs)}


FIXTURE_RECORDS = fixture_records()


def assistant_state():
    shapes = checkpoint_shapes(assistant_config(SYNTHETIC["assistant"], SYNTHETIC["target"]))
    return {name: torch.zeros(shape) for name, shape in shapes.items()}


def tensor_reads():
    import safetensors

    events, real = [], safetensors.safe_open

    class Watched:
        def __init__(self, path, *arguments, **options):
            status = os.stat(path)
            self.name = status.st_dev, status.st_ino
            if not stat.S_ISREG(status.st_mode):
                events.append((self.name, None))
                raise OSError(f"the watched safetensors open was refused for the pipe or folder {Path(path).name}")
            self.opened = real(path, *arguments, **options)

        def __enter__(self):
            self.handle = self.opened.__enter__()
            return self

        def __exit__(self, *details):
            return self.opened.__exit__(*details)

        def keys(self):
            return self.handle.keys()

        def get_slice(self, name):
            return self.handle.get_slice(name)

        def get_tensor(self, name):
            events.append((self.name, name))
            return self.handle.get_tensor(name)

    return events, lambda: mock.patch.object(safetensors, "safe_open", Watched)


def inode(path):
    status = os.stat(path)
    return status.st_dev, status.st_ino


def bounded(call, pipe, seconds=3):
    outcome = {}

    def work():
        try:
            outcome["value"] = call()
        except BaseException as error:
            outcome["error"] = error

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(seconds)
    waited = thread.is_alive()
    if waited:
        try:
            os.close(os.open(pipe, os.O_WRONLY | os.O_NONBLOCK))
        except OSError:
            pass
        thread.join(5)
    return waited, outcome


def buffered_reads():
    events = []

    def opener(file, *arguments, **options):
        status = os.fstat(file) if isinstance(file, int) else os.stat(file)
        events.append((status.st_dev, status.st_ino))
        return builtins.open(file, *arguments, **options)

    return events, lambda: mock.patch.object(layer1, "open", new=opener, create=True)


def piped(path):
    saved = path.read_bytes()
    path.unlink()
    os.mkfifo(path)
    return saved, inode(path)


def drafts_of_691(choices, teacher, minimum, steps):
    found = []

    def draft(tokens):
        index = len(tokens)
        for value in choices(index):
            drafted = tokens + [value]
            if value in STOPS:
                found.append(drafted)
            elif index + 1 >= minimum and (index >= len(teacher) or value != teacher[index]):
                found.append(drafted)
            elif index == steps - 1:
                found.append(drafted)
            else:
                draft(drafted)

    draft([])
    return found


def drafter_rows(folder, teacher, tokens):
    rows = []
    for j in range(len(teacher) - 1):
        drafts, rest = tokens if j == 0 else [106], teacher[j + 1:]
        accepted = next((index for index, (draft, token) in enumerate(zip(drafts, rest)) if draft != token),
                        min(len(drafts), len(rest)))
        rows.append({"a": 2 + j, "j": j, "available": min(15, len(rest)),
                     "fixed": {"accepted": accepted, "tokens": drafts, "margins": [0.5] * len(drafts)}})
    path = Path(folder) / "drafter.json"
    path.write_text(json.dumps({"id": "alpha-0001", "P": 2, "T": len(teacher), "mtp": rows}))
    return mtp.draft_rows(path, {"id": "alpha-0001", "input_ids": [2, 3], "teacher": teacher}, CONFIG["vocab_size"],
                          STOPS)


class PureTests(unittest.TestCase):
    def test_the_cut_keeps_the_drafts_up_to_and_with_the_first_stop_token(self):
        cases = {"no stop token": ([4, 5, 6], (1,), [4, 5, 6]), "a stop token first": ([1, 5, 6], (1,), [1]),
                 "a stop token in the middle": ([4, 106, 6, 50], (1, 106, 50), [4, 106]),
                 "the earliest of two stop tokens": ([4, 50, 106], (106, 50), [4, 50]), "no drafts": ([], (1,), [])}
        for label, (tokens, stops, expected) in cases.items():
            with self.subTest(case=label):
                self.assertEqual(mtp.cut(tokens, stops), expected)

    def test_the_matching_prefix_stops_at_the_first_difference_or_the_end_of_the_teacher(self):
        self.assertEqual([mtp.matches([1, 2, 3], [1, 2, 3, 4]), mtp.matches([1, 2, 3], [1, 9, 3]),
                          mtp.matches([1, 2, 3], [1, 2]), mtp.matches([], [1]), mtp.matches([5], [6])], [3, 1, 2, 0, 0])

    def test_the_rounds_follow_the_teacher_path_with_the_accepted_drafts_capped_at_the_steps(self):
        accepted = [2, 0, 5, 0, 1, 0, 0]
        self.assertEqual(mtp.simulate(accepted, 8, 3), (4, 7))
        self.assertEqual(mtp.simulate(accepted, 8, 1), (4, 7))
        self.assertEqual(mtp.simulate([0] * 7, 8, 7), (7, 7))
        self.assertEqual(mtp.simulate([6, 0], 3, 7), (1, 7))
        self.assertEqual(mtp.simulate([3, 0, 0, 2, 0, 0, 0, 0], 9, 2), (4, 8))
        self.assertEqual(mtp.simulate([], 1, 7), (0, 0))

    def test_the_anchors_are_the_first_output_the_thirds_and_the_last_output_without_repeats(self):
        self.assertEqual(mtp.anchors(10, 1), [10])
        self.assertEqual(mtp.anchors(10, 2), [10, 11])
        self.assertEqual(mtp.anchors(10, 9), [10, 13, 16, 18])
        self.assertEqual(mtp.anchors(28, 256), [28, 113, 198, 283])

    def test_the_limits_include_their_boundary_and_the_ratio_is_the_largest_difference_over_the_largest_value(self):
        self.assertTrue(mtp.within(1e-3, mtp.LOGIT_LIMIT))
        self.assertFalse(mtp.within(math.nextafter(1e-3, 1.0), mtp.LOGIT_LIMIT))
        self.assertTrue(mtp.within(mtp.RATIO, mtp.RATIO))
        self.assertFalse(mtp.within(math.nextafter(mtp.RATIO, 1.0), mtp.RATIO))
        self.assertFalse(mtp.within(None, mtp.RATIO))
        self.assertEqual(mtp.ratio([1.0, -4.5, 2.0], [1.5, -4.0, 2.0]), 0.5 / 4.0)
        self.assertEqual(mtp.ratio([0.0, -0.0], [0.0, 0.0]), 0.0)
        for label, (model_class, original) in {"a zero original": ([1.0], [0.0]), "a NaN": ([float("nan")], [1.0]),
                                               "an infinity": ([1.0], [float("inf")])}.items():
            with self.subTest(case=label):
                self.assertIsNone(mtp.ratio(model_class, original))

    def test_assistant_values_of_another_kind_are_refused_even_when_they_equal_a_supported_value(self):
        good = assistant_config(SYNTHETIC["assistant"], SYNTHETIC["target"])
        self.assertEqual((good.hidden_size, good.intermediate_size, good.rms_norm_eps), (32, 64, 1e-6))
        top = {"backbone_hidden_size": [48.0], "use_ordered_embeddings": [0, None], "tie_word_embeddings": [1],
               "text_config": [[]]}
        inner = {"vocab_size": [301.0], "hidden_size": [32.0, "32", 0], "intermediate_size": [64.0, -64, None],
                 "num_hidden_layers": [4.0], "num_attention_heads": [4.0], "num_key_value_heads": [2.0],
                 "num_global_key_value_heads": [True], "head_dim": [16.0], "global_head_dim": [32.0],
                 "sliding_window": [1024.0], "num_kv_shared_layers": [4.0], "hidden_size_per_layer_input": [0.0, False],
                 "enable_moe_block": [0], "use_double_wide_mlp": [0], "attention_bias": [0], "hidden_activation": [None],
                 "rms_norm_eps": ["1e-06", 0.0, -1e-6, 1, float("nan"), float("inf")], "layer_types": ["full_attention"],
                 "rope_parameters": [[]]}
        cases = [(name, value, {**SYNTHETIC["assistant"], name: value}, SYNTHETIC["target"])
                 for name, values in top.items() for value in values]
        cases += [(name, value, {**SYNTHETIC["assistant"], "text_config": {**SYNTHETIC["assistant"]["text_config"],
                                                                           name: value}}, SYNTHETIC["target"])
                  for name, values in inner.items() for value in values]
        cases += [(name, value, SYNTHETIC["assistant"], {**SYNTHETIC["target"], name: value})
                  for name, value in (("hidden_size", 48.0), ("vocab_size", True), ("sliding_window", "1024"))]
        for name, value, document, target in cases:
            with self.subTest(key=name, value=value), self.assertRaisesRegex(ValueError, name if target is
                                                                             SYNTHETIC["target"] else "target"):
                assistant_config(document, target)
        rope = json.loads(json.dumps(SYNTHETIC["assistant"]["text_config"]["rope_parameters"]))
        rope[SLIDING]["rope_theta"] = 10000
        with self.assertRaisesRegex(ValueError, "another RoPE"):
            assistant_config({**SYNTHETIC["assistant"], "text_config": {**SYNTHETIC["assistant"]["text_config"],
                                                                        "rope_parameters": rope}}, SYNTHETIC["target"])
        for name, value in (("head_dim", 17), ("global_head_dim", 33), ("head_dim", 1)):
            document = {**SYNTHETIC["assistant"], "text_config": {**SYNTHETIC["assistant"]["text_config"], name: value}}
            with self.subTest(key=name, odd=value), self.assertRaisesRegex(ValueError, "head widths that RoPE cannot"):
                assistant_config(document, {**SYNTHETIC["target"], name: value})
        missing = {key: value for key, value in SYNTHETIC["assistant"]["text_config"].items() if key != "rms_norm_eps"}
        with self.assertRaisesRegex(ValueError, "rms_norm_eps"):
            assistant_config({**SYNTHETIC["assistant"], "text_config": missing}, SYNTHETIC["target"])

    def test_a_step_count_that_is_not_a_whole_number_from_one_is_refused(self):
        self.assertEqual(mtp.check_steps(7), 7)
        for steps in (0, -1, True, 1.0, None, "7"):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "whole number"):
                mtp.check_steps(steps)

    def test_a_drafter_record_passes_exactly_the_rows_that_the_drafting_of_691_can_give(self):
        def choices(index):
            return 11 + index, 8, 106

        given, refused, wrong = 0, 0, []
        with tempfile.TemporaryDirectory() as folder:
            for remaining in range(1, 9):
                teacher = list(range(10, 11 + remaining))
                rows = {tuple(tokens) for minimum in (0, 6)
                        for tokens in drafts_of_691(choices, teacher[1:], minimum, mtp.DRAFTS)}
                for tokens in rows:
                    try:
                        drafter_rows(folder, teacher, list(tokens))
                    except ValueError as error:
                        wrong.append((remaining, tokens, str(error)))
                given += len(rows)
                for length in range(1, 9):
                    for kinds in itertools.product(range(3), repeat=length):
                        tokens = tuple(choices(index)[kind] for index, kind in enumerate(kinds))
                        if tokens in rows:
                            continue
                        refused += 1
                        try:
                            drafter_rows(folder, teacher, list(tokens))
                            wrong.append((remaining, tokens, "passed"))
                        except ValueError as error:
                            if "cannot give" not in str(error):
                                wrong.append((remaining, tokens, str(error)))
        self.assertEqual((len(wrong), wrong[:3]), (0, []))
        self.assertEqual((given, refused), (1438, 77378))

    def test_a_drafter_record_passes_15_drafts_only_when_the_last_is_the_fifteenth_of_691(self):
        teacher = list(range(10, 27))
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(drafter_rows(folder, teacher, teacher[1:16])[0]["accepted"], 15)
            self.assertEqual(drafter_rows(folder, teacher, [8] * 5 + teacher[6:16])[0]["accepted"], 0)
            for tokens in (teacher[1:17], [8] * 5 + teacher[6:17]):
                with self.subTest(drafts=tokens), self.assertRaisesRegex(ValueError, "cannot give"):
                    drafter_rows(folder, teacher, tokens)

    def test_the_layout_names_the_last_layer_of_each_kind_and_the_window_of_the_assistant(self):
        self.assertEqual(mtp.layout(load_contract(RECIPE / "model.json")["config"]),
                         {"sliding": 58, "full": 59, "window": 1025})
        self.assertEqual(mtp.layout(CONFIG), {"sliding": 10, "full": 11, "window": 1025})

    def test_the_assistant_contract_pins_the_two_files_that_are_read(self):
        contract, found = mtp.load_assistant(RECIPE / "assistant.json")
        self.assertEqual(found, sha256(RECIPE / "assistant.json"))
        self.assertEqual((contract["model_id"], contract["revision"], sorted(contract["files"])),
                         ("google/gemma-4-31B-it-assistant", "627c5ec1458b9086b841a91e0512fd31fd2fbbf1",
                          ["config.json", "model.safetensors"]))
        self.assertEqual(contract["files"]["model.safetensors"]["bytes"], 939042560)
        entry = {"bytes": 1, "sha256": "0" * 64}
        good = {"model_id": "a/b", "revision": "0" * 40, "files": {"config.json": entry, "model.safetensors": entry}}
        changes = {"another file": {**good, "files": {"config.json": entry, "tokenizer.json": entry}},
                   "one file": {**good, "files": {"config.json": entry}},
                   "an extra field": {**good, "config": {}}, "a short revision": {**good, "revision": "0" * 39},
                   "an ID without an owner": {**good, "model_id": "b"},
                   "a size as a boolean": {**good, "files": {"config.json": {"bytes": True, "sha256": "0" * 64},
                                                            "model.safetensors": entry}},
                   "a size of zero": {**good, "files": {"config.json": {"bytes": 0, "sha256": "0" * 64},
                                                       "model.safetensors": entry}},
                   "a size as a float": {**good, "files": {"config.json": {"bytes": 1.0, "sha256": "0" * 64},
                                                          "model.safetensors": entry}},
                   "an uppercase hash": {**good, "files": {"config.json": {"bytes": 1, "sha256": "A" * 64},
                                                          "model.safetensors": entry}}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "assistant.json"
            write_json(path, good)
            self.assertEqual(mtp.load_assistant(path), (good, sha256(path)))
            for label, contract in changes.items():
                write_json(path, contract)
                with self.subTest(change=label), self.assertRaisesRegex(ValueError, "assistant contract"):
                    mtp.load_assistant(path)

    def test_the_fixed_subset_is_the_48_items_of_the_13_categories_of_691(self):
        self.assertEqual(mtp.SUBSET, {"ja-choice": 4, "ja-reading": 4, "code-gen-he": 4, "code-gen-mbpp": 4, "code-fix": 4,
                                      "extract-ja": 6, "long-ja-4k": 2, "long-ja-16k": 1, "synth-extract": 4,
                                      "synth-format": 6, "free-ja": 4, "synth-long-4k": 3, "synth-long-16k": 2})
        self.assertEqual((len(mtp.SUBSET), sum(mtp.SUBSET.values())), (13, 48))
        self.assertEqual(mtp.RECORDS, {"inputs": "a5606837d5cf300ea46fcdbb51e9380fbd69a5d0e0d5d6c027b39d4e4a260aa6",
                                       "outputs": "1d84643d3d623fabc4ad66645ce3f862994a4afc8a7f40d6091ca6984ad5f5a9"})

    def test_the_first_item_of_each_category_follows_the_numbers_of_the_ids(self):
        record = {"counts": {"a": 3, "b": 1}, "items": [{"id": "a-10", "category": "a"}, {"id": "a-9", "category": "a"},
                                                        {"id": "b-0002", "category": "b"}, {"id": "a-0011", "category": "a"}]}
        self.assertEqual([entry["id"] for entry in mtp.first_per_category(record)], ["a-9", "b-0002"])


@unittest.skipIf(np is None or torch is None, "NumPy, PyTorch and safetensors are required for the records")
@mock.patch.object(mtp, "SUBSET", FIXED)
@mock.patch.object(mtp, "RECORDS", FIXTURE_RECORDS)
class RecordTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        self.records, self.inputs, self.outputs = write_records(self.folder)
        self.contract = {"config": dict(CONFIG)}

    def items(self, choose=mtp.first_per_category):
        return mtp.items_of(self.contract, self.records, self.inputs, self.outputs, choose)[2]

    def test_the_items_take_the_tokens_and_the_feature_offsets_of_the_records(self):
        items = self.items()
        self.assertEqual([item["id"] for item in items], ["alpha-0001", "beta-0001"])
        self.assertEqual([(len(item["input_ids"]), len(item["teacher"]), item["outputs"], item["first"], item["start"])
                          for item in items], [(20, 10, 9, 0, 0), (1100, 6, 6, 0, 0)])
        self.assertEqual(len(self.items(lambda record: record["items"])), 3)

    def test_the_inputs_of_a_row_are_the_hidden_state_before_the_anchor_and_the_keys_of_the_confirmed_positions(self):
        item = self.items()[1]
        features = mtp.loaded_features(item)
        sequence = item["input_ids"] + item["teacher"]
        for anchor in (1100, 1102, 1105):
            inputs = mtp.row_inputs(item, features, anchor, 1025)
            token, hidden, position, k_sliding, v_sliding, k_full, v_full = inputs["arguments"]
            with self.subTest(anchor=anchor):
                self.assertEqual((int(token), inputs["token"], int(position), inputs["position"]),
                                 (sequence[anchor], sequence[anchor], anchor, anchor))
                self.assertEqual(decoded(hidden[0, 0]), anchor - 1)
                self.assertEqual(tuple(k_sliding.shape), (1, 1, 2, 1025, 16))
                self.assertEqual(decoded(k_sliding[0, 0, 1]), list(range(anchor - 1025, anchor)))
                self.assertEqual(decoded(-v_sliding[0, 0, 1]), list(range(anchor - 1025, anchor)))
                self.assertEqual(decoded(k_full[0, 0, 0]), list(range(anchor)))
                self.assertEqual(decoded(-v_full[0, 0, 0]), list(range(anchor)))
                self.assertEqual(decoded(inputs["every_sliding"][0][0, 1]), list(range(anchor)))
                self.assertEqual(inputs["window"], 1025)
        short = self.items()[0]
        inputs = mtp.row_inputs(short, mtp.loaded_features(short), 21, 1025)
        self.assertEqual((inputs["window"], inputs["arguments"][3].shape[3], inputs["arguments"][5].shape[3]), (21, 21, 21))
        self.assertEqual(decoded(inputs["hidden"][0, 0]), 20)

    def test_records_that_differ_from_each_other_or_from_the_contract_are_refused(self):
        def outputs_with(change):
            rows = [json.loads(row) for row in self.outputs.read_text().splitlines()]
            change(rows)
            self.outputs.write_text("".join(map(line, rows)))
            subset = read_json(self.records / "subset.json")
            subset["outputs"]["sha256"] = sha256(self.outputs)
            write_json(self.records / "subset.json", subset)

        place, features = mtp.layout(CONFIG), self.records / "tf/features"

        def as_float32():
            from safetensors.torch import load_file

            path = features / "beta-0001.safetensors"
            save_file({key: value.float() for key, value in load_file(str(path)).items()}, str(path))

        def starting_inside_the_window():
            tensors = {"postnorm": torch.zeros(1105, 48), **{f"{kind}{place['full']}": torch.zeros(1, 1105, 32)
                                                            for kind in "kv"},
                       **{f"{kind}{place['sliding']}": torch.zeros(2, 1105 - 200, 16) for kind in "kv"}}
            save_file({key: value.to(torch.bfloat16) for key, value in tensors.items()},
                      str(features / "beta-0001.safetensors"))

        def linked():
            (features / "alpha-0001.safetensors").unlink()
            (features / "alpha-0001.safetensors").symlink_to(features / "alpha-0002.safetensors")

        cases = {
            "inputs changed after the subset": lambda: self.inputs.write_text(self.inputs.read_text() + "\n"),
            "a stop token inside the output": lambda: outputs_with(lambda rows: rows[0]["output_ids"].__setitem__(0, 50)),
            "an output with another length": lambda: outputs_with(lambda rows: rows[0]["output_ids"].append(7)),
            "an output from another input": lambda: outputs_with(lambda rows: rows[0].update(input_tokens=21)),
            "an output input length as a float": lambda: outputs_with(lambda rows: rows[0].update(input_tokens=20.0)),
            "an output without non-finite values as a float": lambda: outputs_with(
                lambda rows: rows[0].update(non_finite=0.0)),
            "features in float32": as_float32,
            "features of another shape": lambda: save_file({key: torch.zeros(2, 2, dtype=torch.bfloat16) for key in (
                "postnorm", f"k{place['sliding']}")}, str(features / "alpha-0001.safetensors")),
            "features that start inside the window": starting_inside_the_window,
            "features linked from elsewhere": linked}
        backup = {path: path.read_bytes() for path in [self.inputs, self.outputs, self.records / "subset.json",
                                                       *(self.records / "tf/features").iterdir()]}
        for label, change in cases.items():
            for path, data in backup.items():
                if path.is_symlink():
                    path.unlink()
                path.write_bytes(data)
            change()
            with self.subTest(change=label), self.assertRaises(ValueError):
                self.items()

    def test_input_and_output_records_of_other_files_are_refused_even_when_they_hold_the_same_items(self):
        def extended(path, line_of_record):
            path.write_text(path.read_text() + line_of_record)
            subset = read_json(self.records / "subset.json")
            subset[path.stem]["sha256"] = sha256(path)
            write_json(self.records / "subset.json", subset)

        records = {"inputs": (self.inputs, {"id": "gamma-0001", "input_ids": [2, 3], "input_sha256": tokens_sha256([2, 3]),
                                            "max_new_tokens": 16}),
                   "outputs": (self.outputs, {"id": "gamma-0001", "input_sha256": tokens_sha256([2, 3]), "input_tokens": 2,
                                              "non_finite": 0, "output_ids": [4], "stop": "limit", "top2": []})}
        backup = {path: path.read_bytes() for path in (self.inputs, self.outputs, self.records / "subset.json")}
        for name, (path, record) in records.items():
            for kept, data in backup.items():
                kept.write_bytes(data)
            extended(path, line(record))
            with self.subTest(records=name), self.assertRaisesRegex(ValueError, "other input or output records"):
                self.items()

    def test_records_that_are_named_pipes_are_refused_without_waiting_for_a_writer(self):
        item = self.items()[0]
        paths = {"subset.json": (self.records / "subset.json", lambda path: mtp.subset_record(path)),
                 "a drafter record": (self.folder / "drafter.json", lambda path: mtp.draft_rows(path, item, CONFIG["vocab_size"], STOPS)),
                 "trajectories.json": (self.folder / "trajectories.json", lambda path: mtp.published(path, 7, ["alpha"])),
                 "the assistant contract": (self.folder / "assistant.json", mtp.load_assistant)}
        for label, (path, read) in paths.items():
            if path.exists():
                path.unlink()
            os.mkfifo(path)
            with self.subTest(record=label), self.assertRaisesRegex(ValueError, "not a regular file"):
                read(path)

    def test_the_subset_record_holds_its_counts_and_unique_ids_of_its_categories(self):
        subset = read_json(self.records / "subset.json")
        changes = {"a repeated id": lambda record: record["items"].append(dict(record["items"][0])),
                   "another count": lambda record: record["counts"].update(alpha=3),
                   "an id of another category": lambda record: record["items"][0].update(category="beta"),
                   "an id without a number": lambda record: record["items"][0].update(id="alpha-x"),
                   "a missing field": lambda record: record["items"][0].pop("teacher_tokens"),
                   "a count as a boolean": lambda record: record["items"][0].update(input_tokens=True),
                   "a length as a float": lambda record: record["items"][0].update(input_tokens=30.0),
                   "a category count as a float": lambda record: record["counts"].update(alpha=2.0),
                   "a stop token as a float": lambda record: record["items"][0].update(stop_token=106.0),
                   "a stop token for a limit": lambda record: record["items"][2].update(stop_token=0),
                   "no stop token for a stop": lambda record: record["items"][0].update(stop_token=None),
                   "an empty subset": lambda record: record.update(counts={}, items=[]),
                   "another id of a category": lambda record: record["items"][1].update(id="alpha-0003"),
                   "the fixed ids in another order": lambda record: record.update(items=[record["items"][1],
                                                                                         record["items"][0],
                                                                                         record["items"][2]]),
                   "another subset than the fixed one": lambda record: record.update(
                       counts={"alpha": 2}, items=[entry for entry in record["items"] if entry["category"] == "alpha"]),
                   "an item without outputs": lambda record: record["items"][0].update(output_tokens=0, teacher_tokens=1),
                   "an item without inputs": lambda record: record["items"][0].update(input_tokens=0),
                   "a teacher length unlike the outputs": lambda record: record["items"][0].update(teacher_tokens=4),
                   "an extra field": lambda record: record.update(note="x")}
        for label, change in changes.items():
            record = json.loads(json.dumps(subset))
            change(record)
            write_json(self.records / "subset.json", record)
            with self.subTest(change=label), self.assertRaisesRegex(ValueError, "subset.json"):
                mtp.subset_record(self.records / "subset.json")

    def test_the_drafter_records_and_the_rounds_of_691_are_read_only_in_their_form(self):
        item = self.items()[0]
        rows = [{"a": 20 + j, "j": j, "available": 9 - j, "fixed": {"accepted": 0, "tokens": [0], "margins": [0.5]}}
                for j in range(9)]
        rows[0]["fixed"].update(accepted=2, tokens=item["teacher"][1:3] + [0], margins=[0.5, 0.0, 0.25])
        path = self.folder / "drafter.json"
        write_json(path, {"id": item["id"], "P": 20, "T": 10, "mtp": rows})
        self.assertEqual([row["accepted"] for row in mtp.draft_rows(path, item, CONFIG["vocab_size"], STOPS).values()], [2] + [0] * 8)
        changes = {"a missing anchor": lambda record: record["mtp"].pop(),
                   "an anchor off its output index": lambda record: record["mtp"][0].update(a=21),
                   "a boolean accepted count": lambda record: record["mtp"][0]["fixed"].update(accepted=True),
                   "margins of another length": lambda record: record["mtp"][0]["fixed"]["margins"].pop(),
                   "a negative margin": lambda record: record["mtp"][0]["fixed"]["margins"].__setitem__(0, -0.5),
                   "a draft beyond the vocabulary": lambda record: record["mtp"][1]["fixed"]["tokens"].__setitem__(
                       0, CONFIG["vocab_size"]),
                   "another input length": lambda record: record.update(P=21),
                   "an input length as a float": lambda record: record.update(P=20.0),
                   "a teacher length as a float": lambda record: record.update(T=10.0),
                   "an input length as text": lambda record: record.update(P="20"),
                   "an anchor as a float": lambda record: record["mtp"][0].update(a=20.0),
                   "an output index as a float": lambda record: record["mtp"][0].update(j=0.0),
                   "an accepted count as a float": lambda record: record["mtp"][1]["fixed"].update(accepted=0.0),
                   "a draft as a float": lambda record: record["mtp"][1]["fixed"]["tokens"].__setitem__(0, 0.0),
                   "no drafts": lambda record: record["mtp"][1]["fixed"].update(tokens=[], margins=[]),
                   "more drafts than #691 made": lambda record: record["mtp"][1]["fixed"].update(tokens=[0] * 16,
                                                                                                margins=[0.5] * 16),
                   "another count of available drafts": lambda record: record["mtp"][1].update(available=3),
                   "a count of available drafts as a float": lambda record: record["mtp"][1].update(available=8.0),
                   "a margin rounded to more places": lambda record: record["mtp"][1]["fixed"]["margins"].__setitem__(
                       0, 0.12345),
                   "a draft after a stop token": lambda record: record["mtp"][1]["fixed"].update(tokens=[106, 0],
                                                                                                 margins=[0.5, 0.5]),
                   "a difference before the last of fewer than six drafts": lambda record: record["mtp"][1]["fixed"].update(
                       tokens=[0, 0], margins=[0.5, 0.5]),
                   "a draft past a difference after the sixth": lambda record: record["mtp"][1]["fixed"].update(
                       accepted=5, tokens=item["teacher"][2:7] + [0, 0], margins=[0.5] * 7),
                   "a last draft that does not end the drafting": lambda record: record["mtp"][1]["fixed"].update(
                       accepted=2, tokens=item["teacher"][2:4], margins=[0.5, 0.5])}
        for label, change in changes.items():
            record = {"id": item["id"], "P": 20, "T": 10, "mtp": json.loads(json.dumps(rows))}
            change(record)
            write_json(path, record)
            with self.subTest(change=label), self.assertRaises(ValueError):
                mtp.draft_rows(path, item, CONFIG["vocab_size"], STOPS)
        skipped = json.loads(json.dumps(rows))
        skipped[1]["fixed"]["accepted"] = 1
        self.assertEqual(mtp.simulate([row["fixed"]["accepted"] for row in skipped], 10, 7),
                         mtp.simulate([row["fixed"]["accepted"] for row in rows], 10, 7))
        write_json(path, {"id": item["id"], "P": 20, "T": 10, "mtp": skipped})
        with self.assertRaisesRegex(ValueError, "accepted counts that its drafts and the teacher do not give at 1 anchors"):
            mtp.draft_rows(path, item, CONFIG["vocab_size"], STOPS)
        level = {"alpha": {"rounds": 3, "tokens": 7, "tokens_per_round": 2.3333},
                 "beta": {"rounds": 0, "tokens": 0, "tokens_per_round": None},
                 "pooled": {"rounds": 3, "tokens": 7, "tokens_per_round": 2.3333}}
        write_json(path, {"per_round_on_teacher_path": {"mtp_fixed": {"7": level}}})
        self.assertEqual(mtp.published(path, 7, ["alpha", "beta"]), level)
        for steps, categories in ((6, ["alpha", "beta"]), (7, ["alpha"])):
            with self.subTest(steps=steps, categories=categories), self.assertRaisesRegex(ValueError, "trajectories"):
                mtp.published(path, steps, categories)
        rates = {"a rate rounded otherwise": ("alpha", 2.333), "a rate unlike its counts": ("pooled", 2.5),
                 "a rate as a boolean": ("alpha", True), "no rate for rounds": ("alpha", None),
                 "a rate without rounds": ("beta", 0.0), "a rate as text": ("alpha", "2.3333")}
        for label, (name, rate) in rates.items():
            changed = json.loads(json.dumps(level))
            changed[name]["tokens_per_round"] = rate
            write_json(path, {"per_round_on_teacher_path": {"mtp_fixed": {"7": changed}}})
            with self.subTest(change=label), self.assertRaisesRegex(ValueError, f"tokens per round .*{name}"):
                mtp.published(path, 7, ["alpha", "beta"])
        changed = json.loads(json.dumps(level))
        changed["alpha"].update(rounds=2, tokens=2, tokens_per_round=True)
        write_json(path, {"per_round_on_teacher_path": {"mtp_fixed": {"7": changed}}})
        with self.assertRaisesRegex(ValueError, "tokens per round .*alpha"):
            mtp.published(path, 7, ["alpha", "beta"])
        for rounds in (True, 3.0):
            changed = json.loads(json.dumps(level))
            changed["alpha"].update(rounds=rounds)
            write_json(path, {"per_round_on_teacher_path": {"mtp_fixed": {"7": changed}}})
            with self.subTest(rounds=rounds), self.assertRaisesRegex(ValueError, "no per-round values"):
                mtp.published(path, 7, ["alpha", "beta"])
        changed = json.loads(json.dumps(level))
        changed["alpha"].update(tokens=7.0)
        write_json(path, {"per_round_on_teacher_path": {"mtp_fixed": {"7": changed}}})
        with self.assertRaisesRegex(ValueError, "no per-round values"):
            mtp.published(path, 7, ["alpha", "beta"])


ROWS = [{"id": name, "category": name.rsplit("-", 1)[0], "anchor": anchor, "output_index": anchor - prompt,
         "window": min(anchor, 1025)}
        for name, prompt, positions in (("alpha-0001", 20, (20, 23, 26, 28)), ("beta-0001", 1100, (1100, 1102, 1104, 1105)))
        for anchor in positions]


def pins(place, names):
    return {name: {"bytes": (place / name).stat().st_size, "sha256": sha256(place / name)} for name in names}


def run_record(folder, kind):
    contract, features = read_json(folder / "model.json"), folder / "records/tf/features"
    return {"schema": mtp.SCHEMA, "implementation": kind, "steps": 3, "stops": [1, 106, 50], "window": 1025,
            "layers": {"sliding": 10, "full": 11},
            "model": {"id": contract["model_id"], "revision": contract["revision"],
                      "contract_sha256": sha256(folder / "model.json"),
                      "verified_before_loading": pins(folder / "target", (mtp.INDEX, SHARD))},
            "assistant": {"id": "google/gemma-4-31B-it-assistant", "revision": "0" * 40,
                          "contract_sha256": sha256(folder / "assistant.json"),
                          "verified_before_loading": pins(folder / "assistant", ("config.json", "model.safetensors"))},
            "records": {"subset_sha256": sha256(folder / "records/subset.json"),
                        "inputs_sha256": sha256(folder / "evaluation/inputs.jsonl"),
                        "outputs_sha256": sha256(folder / "evaluation/outputs.jsonl"),
                        "features": {name: pins(features, [f"{name}.safetensors"])[f"{name}.safetensors"]
                                     for name in ("alpha-0001", "beta-0001")}},
            "weights": mtp.WEIGHTS, "embed_scale": 6.9375, "rows": [dict(row) for row in ROWS],
            "versions": {"torch": "0"}, "timing": {"total_s": 1.0}, "memory": {"max_rss_bytes": 1}}


def write_run(folder, record, arrays):
    folder.mkdir(parents=True)
    record = dict(record, arrays={})
    for name, array in arrays.items():
        np.save(folder / f"{name}.npy", array)
        record["arrays"][name] = {**pins(folder, [f"{name}.npy"])[f"{name}.npy"], "dtype": str(array.dtype),
                                  "shape": list(array.shape),
                                  "non_finite": int(array.size - np.isfinite(array).sum())}
    write_json(folder / "record.json", record)


@unittest.skipIf(np is None or torch is None, "NumPy, PyTorch and safetensors are required for the comparison")
@mock.patch.object(mtp, "SUBSET", FIXED)
@mock.patch.object(mtp, "RECORDS", FIXTURE_RECORDS)
class CompareTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        write_records(self.folder)
        write_target(self.folder, torch.zeros(CONFIG["vocab_size"], CONFIG["hidden_size"]))
        write_assistant(self.folder, SYNTHETIC["assistant"], assistant_state())
        generator = np.random.default_rng(3)
        logits = generator.normal(size=(8, CONFIG["vocab_size"])).astype(np.float32)
        projected = generator.normal(size=(8, CONFIG["hidden_size"])).astype(np.float32)
        drafts = generator.integers(51, 106, size=(8, 3)).astype(np.int32)
        drafts[1], drafts[2] = [7, 106, 8], [9, 9, 50]
        original = drafts.copy()
        original[1] = [7, 106, -1]
        self.arrays = {"original": {"logits": logits, "projected": projected, "drafts": original},
                       "class": {"logits": logits.copy(), "projected": projected.copy(), "drafts": drafts,
                                 "repeated": drafts.copy()}}

    def compare(self, label, original="original", model_class="class", contract="model.json"):
        return mtp.compare(self.folder / label / original, self.folder / label / model_class, self.folder / contract,
                           self.folder / "assistant.json", self.folder / "target", self.folder / "records",
                           self.folder / "evaluation/inputs.jsonl", self.folder / "evaluation/outputs.jsonl", 3)

    def pair(self, label, change=None, record=None, both=None):
        arrays = {kind: {name: array.copy() for name, array in found.items()} for kind, found in self.arrays.items()}
        if change:
            change(arrays)
        for kind in mtp.IMPLEMENTATIONS:
            entry = run_record(self.folder, kind)
            for edit in (both, record if kind == "class" else None):
                if edit:
                    edit(entry)
            write_run(self.folder / label / kind, entry, arrays[kind])
        return self.compare(label)

    def test_runs_within_the_criteria_pass_and_record_the_differences_from_the_original_drafts(self):
        result = self.pair("same")
        self.assertTrue(result["passed"], result["problems"])
        self.assertEqual(result["criteria"], {"max_abs_logit": 1e-3, "projection_ratio": 1e-3})
        self.assertEqual(result["summary"], {"rows": 8, "items": 2, "max_abs_logit": 0.0, "largest_projection_ratio": 0.0,
                                             "argmax_equal": 8, "unrolled_equals_repeated": 8, "cut_equals_original": 8})
        self.assertEqual([{key: row[key] for key in ROWS[0]} for row in result["rows"]], ROWS)
        self.assertEqual(result["records_sha256"], {kind: sha256(self.folder / "same" / kind / "record.json")
                                                    for kind in mtp.IMPLEMENTATIONS})

        def drafts(arrays):
            arrays["original"]["drafts"][2] = [9, 8, 50]

        result = self.pair("other-drafts", drafts)
        self.assertTrue(result["passed"], result["problems"])
        self.assertEqual([(row["cut_equals_original"], row["first_step_unlike_original"]) for row in result["rows"]],
                         [(True, None)] * 2 + [(False, 1)] + [(True, None)] * 5)

    def replaced(self, label, kind, name, data):
        folder = self.folder / label / kind
        (folder / f"{name}.npy").write_bytes(data)
        record = read_json(folder / "record.json")
        record["arrays"][name].update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        write_json(folder / "record.json", record)

    def test_arrays_of_another_type_shape_or_count_of_non_finite_values_than_their_record_are_problems(self):
        for name, value in {"dtype": "float64", "shape": [8, 300], "non_finite": 1}.items():
            label = f"record-{name}"
            self.pair(label)
            record = read_json(self.folder / label / "class/record.json")
            record["arrays"]["logits"][name] = value
            write_json(self.folder / label / "class/record.json", record)
            with self.subTest(field=name):
                self.assertIn("the class run records another type, shape or count of non-finite values of its logits",
                              self.compare(label)["problems"])

    def test_record_values_of_another_type_than_the_fixed_inputs_give_are_problems(self):
        def feature(record):
            record["records"]["features"]["alpha-0001"]["bytes"] = float(record["records"]["features"]["alpha-0001"]
                                                                         ["bytes"])

        changes = {"steps": lambda record: record.update(steps=3.0), "window": lambda record: record.update(window=1025.0),
                   "layers": lambda record: record["layers"].update(sliding=10.0),
                   "records": feature, "rows": lambda record: record["rows"][0].update(anchor=20.0),
                   "stops": lambda record: record.update(stops=[True, 106, 50])}
        for key, change in changes.items():
            result = self.pair(f"typed-{key}", both=change)
            with self.subTest(field=key):
                self.assertIn(f"the class run holds another {key} than its fixed inputs give", result["problems"])
        for field, value in {"non_finite": 0.0, "bytes": None}.items():
            label = f"typed-array-{field}"
            self.pair(label)
            record = read_json(self.folder / label / "class/record.json")
            record["arrays"]["logits"][field] = value if value is not None else float(record["arrays"]["logits"]["bytes"])
            write_json(self.folder / label / "class/record.json", record)
            with self.subTest(array_field=field):
                self.assertTrue(any(problem.startswith("the class run records") for problem in self.compare(label)
                                    ["problems"]))

    def test_an_archive_of_arrays_in_place_of_an_array_is_a_problem_of_that_array(self):
        self.pair("archive")
        buffer = io.BytesIO()
        np.savez(buffer, logits=self.arrays["class"]["logits"])
        self.replaced("archive", "class", "logits", buffer.getvalue())
        result = self.compare("archive")
        self.assertFalse(result["passed"])
        self.assertEqual(result["problems"], ["the class logits are not one array"])
        json.dumps(result, allow_nan=False)

    def test_any_content_of_the_runs_gives_a_result_with_its_problems_that_json_can_hold(self):
        def saved(array):
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=True)
            return buffer.getvalue()

        logits, vocabulary = self.arrays["class"]["logits"], CONFIG["vocab_size"]
        files = {"a pickled object array": ("logits", saved(np.array([{"a": 1}], dtype=object)),
                                            "the class logits cannot be read (ValueError)"),
                 "a zero-dimensional array": ("logits", saved(np.float32(1.0)),
                                              f"the class logits are float32 [], not float32 [8, {vocabulary}]"),
                 "an empty array": ("logits", saved(logits[:0]),
                                    f"the class logits are float32 [0, {vocabulary}], not float32 [8, {vocabulary}]"),
                 "an array of another row count": ("logits", saved(logits[:7]), f"the class logits are float32 "
                                                   f"[7, {vocabulary}], not float32 [8, {vocabulary}]"),
                 "projections that are all NaN": ("projected", saved(np.full((8, CONFIG["hidden_size"]), np.nan,
                                                                             np.float32)),
                                                  "the class projected hold non-finite values"),
                 "drafts as text": ("drafts", saved(np.full((8, 3), "a")), "the class drafts are <U1 [8, 3], not int32 "
                                    "[8, 3]")}
        for label, (name, data, problem) in files.items():
            name_of = label.replace(" ", "-")
            self.pair(name_of)
            self.replaced(name_of, "class", name, data)
            result = self.compare(name_of)
            with self.subTest(case=label):
                self.assertFalse(result["passed"])
                self.assertIn(problem, result["problems"])
                json.dumps(result, allow_nan=False)

        def huge(arrays):
            arrays["original"]["logits"][0] = -np.finfo(np.float32).max
            arrays["class"]["logits"][0] = np.finfo(np.float32).max

        def zero(arrays):
            for kind in mtp.IMPLEMENTATIONS:
                arrays[kind]["projected"][:] = 0.0

        for label, (change, passed) in {"logits at the largest float32 values": (huge, False),
                                        "projections that are all zero": (zero, True)}.items():
            result = self.pair(label.replace(" ", "-"), change)
            with self.subTest(case=label):
                self.assertEqual((result["passed"], result["problems"]), (passed, []))
                self.assertTrue(math.isfinite(result["summary"]["max_abs_logit"]))
                json.dumps(result, allow_nan=False)
        self.pair("listed")
        write_json(self.folder / "listed/class/record.json", [])
        self.assertEqual(self.compare("listed")["problems"], ["the class run holds another record"])
        self.pair("unrecorded")
        record = read_json(self.folder / "unrecorded/class/record.json")
        record["arrays"]["repeated"] = {"bytes": True, "sha256": None}
        write_json(self.folder / "unrecorded/class/record.json", record)
        result = self.compare("unrecorded")
        self.assertIn("the class run records no size and SHA-256 of its repeated", result["problems"])
        json.dumps(result, allow_nan=False)
        self.pair("unexpected")
        with mock.patch.object(mtp, "cut", side_effect=RuntimeError("a message with details")):
            result = self.compare("unexpected")
            arguments = ["mtp", "compare", "--original", str(self.folder / "unexpected/original"), "--model-class",
                         str(self.folder / "unexpected/class"), "--model", str(self.folder / "model.json"),
                         "--assistant", str(self.folder / "assistant.json"), "--snapshot", str(self.folder / "target"),
                         "--records", str(self.folder / "records"), "--inputs", str(self.folder / "evaluation/inputs.jsonl"),
                         "--outputs", str(self.folder / "evaluation/outputs.jsonl"), "--steps", "3"]
            with mock.patch("sys.argv", arguments), mock.patch("sys.stdout", new_callable=io.StringIO) as printed:
                code = mtp.main()
        self.assertEqual(result, {"schema": mtp.COMPARISON, "passed": False,
                                  "problems": ["the comparison stopped with RuntimeError"]})
        self.assertEqual((json.loads(printed.getvalue()), code), (result, 1))

    def test_a_zero_original_projection_passes_only_with_a_zero_class_projection_and_the_summary_holds_either(self):
        def zero(value):
            def change(arrays):
                arrays["original"]["projected"][3] = 0.0
                arrays["class"]["projected"][3] = 0.0
                arrays["class"]["projected"][3, 4] = value
            return change

        for label, (value, passed, row, largest) in {"a zero class projection": (0.0, True, 0.0, 0.0),
                                                     "a class projection that is not zero": (1e-30, False, None, None)
                                                     }.items():
            result = self.pair(label.replace(" ", "-"), zero(value))
            with self.subTest(case=label):
                self.assertEqual(result["problems"], [])
                self.assertEqual(result["passed"], passed)
                self.assertEqual(result["rows"][3]["projection_ratio"], row)
                self.assertEqual([entry["checks"]["projection"] for entry in result["rows"]],
                                 [True] * 3 + [passed] + [True] * 4)
                self.assertEqual(result["summary"]["largest_projection_ratio"], largest)
                json.dumps(result, allow_nan=False)

    def test_each_criterion_fails_just_beyond_its_limit(self):
        def logits(value):
            return lambda arrays: arrays["class"]["logits"].__setitem__((1, 7), arrays["class"]["logits"][1, 7] + value)

        def projected(ratio):
            def change(arrays):
                arrays["original"]["projected"][2] = 1.0
                arrays["original"]["projected"][2, 0] = 10.0
                arrays["class"]["projected"][2] = arrays["original"]["projected"][2]
                arrays["class"]["projected"][2, 5] += ratio * 10.0
            return change

        def repeated(arrays):
            arrays["class"]["repeated"][0, 2] = 7

        def logits_at(value):
            def change(arrays):
                arrays["original"]["logits"][1, 7] = 1.0
                arrays["class"]["logits"][1, 7] = 1.0 + value
            return change

        def projected_at(value):
            def change(arrays):
                arrays["original"]["projected"][2] = 1.0
                arrays["class"]["projected"][2] = 1.0
                arrays["class"]["projected"][2, 5] = 1.0 + value
            return change

        edge, beyond = 2.0 ** -10, 2.0 ** -10 + 2.0 ** -23
        cases = {"logits below": (logits(0.0009), set()), "logits above": (logits(0.0011), {"max_abs_logit"}),
                 "projection below": (projected(0.00099), set()), "projection above": (projected(0.00101), {"projection"}),
                 "unrolled unlike repeated": (repeated, {"unrolled_equals_repeated"}),
                 "logits exactly at a limit": (logits_at(edge), set()),
                 "logits just beyond a limit": (logits_at(beyond), {"max_abs_logit"}),
                 "projection exactly at a limit": (projected_at(edge), set()),
                 "projection just beyond a limit": (projected_at(beyond), {"projection"})}
        for label, (change, failing) in cases.items():
            limits = (edge, edge) if "a limit" in label else (mtp.LOGIT_LIMIT, mtp.RATIO)
            with mock.patch.object(mtp, "LOGIT_LIMIT", limits[0]), mock.patch.object(mtp, "RATIO", limits[1]):
                result = self.pair(label.replace(" ", "-"), change)
            with self.subTest(case=label):
                self.assertEqual(result["problems"], [])
                self.assertEqual(result["passed"], not failing)
                self.assertEqual({key for row in result["rows"] for key, passed in row["checks"].items() if not passed},
                                 failing)
                self.assertEqual(result["criteria"], {"max_abs_logit": limits[0], "projection_ratio": limits[1]})
        self.assertEqual(np.float32(1.0 + beyond) - np.float32(1.0), beyond)

    def test_runs_that_must_not_pass(self):
        def value(kind, name, index, number):
            return lambda arrays: arrays[kind][name].__setitem__(index, number)

        cases = {"a NaN logit in both runs": (lambda arrays: [arrays[kind]["logits"].__setitem__((0, 3), np.nan)
                                                              for kind in mtp.IMPLEMENTATIONS], None),
                 "an infinite projection": (value("class", "projected", (1, 2), np.inf), None),
                 "logits of another vocabulary": (lambda arrays: arrays["class"].update(
                     logits=np.zeros((8, 300), np.float32)), None),
                 "drafts as int64": (lambda arrays: arrays["class"].update(
                     drafts=arrays["class"]["drafts"].astype(np.int64)), None),
                 "a missing row": (lambda arrays: arrays["original"].update(
                     projected=arrays["original"]["projected"][:7]), None),
                 "another step count": (None, lambda record: record.update(steps=4)),
                 "other rows": (None, lambda record: record["rows"][0].update(anchor=99)),
                 "a row without its window": (None, lambda record: [row.pop("window") for row in record["rows"]]),
                 "the original implementation twice": (None, lambda record: record.update(implementation="original")),
                 "another weight rounding": (None, lambda record: record.update(weights="other")),
                 "a padding inside the original drafts": (value("original", "drafts", (0, 1), -1), None),
                 "a padding in the drafts of the class": (value("class", "drafts", (1, 2), -1), None),
                 "a token beyond the vocabulary": (value("class", "repeated", (2, 0), CONFIG["vocab_size"]), None)}
        for label, (change, record) in cases.items():
            result = self.pair(label.replace(" ", "-"), change, record)
            with self.subTest(case=label):
                self.assertFalse(result["passed"])
                self.assertTrue(result["problems"])

        def sliced(rows):
            return lambda arrays: [found.update({name: array[rows] for name, array in found.items()})
                                   for found in arrays.values()]

        def target(record):
            files = record["model"]["verified_before_loading"]
            files["model-00002-of-00002.safetensors"] = files.pop(SHARD)

        both = {"stop tokens": (None, lambda record: record.update(stops=[1, 106])),
                "a window": (None, lambda record: record.update(window=1024)),
                "an embedding scale": (None, lambda record: record.update(embed_scale=6.928)),
                "layers": (None, lambda record: record.update(layers={"sliding": 9, "full": 11})),
                "no assistant": (None, lambda record: record.pop("assistant")),
                "a null assistant": (None, lambda record: record.update(assistant=None)),
                "an assistant contract of zeros": (None, lambda record: record["assistant"].update(contract_sha256="0" * 64)),
                "another revision of the assistant": (None, lambda record: record["assistant"].update(revision="1" * 40)),
                "assistant files without their sizes": (None, lambda record: [
                    entry.pop("bytes") for entry in record["assistant"]["verified_before_loading"].values()]),
                "empty records": (None, lambda record: record.update(records={})),
                "no features": (None, lambda record: record["records"].pop("features")),
                "features of another file": (None, lambda record: record["records"]["features"]["alpha-0001"].update(
                    sha256="1" * 64)),
                "the features of one item": (None, lambda record: record["records"]["features"].pop("beta-0001")),
                "a null subset": (None, lambda record: record["records"].update(subset_sha256=None)),
                "other input records": (None, lambda record: record["records"].update(inputs_sha256="1" * 64)),
                "no verified files of the target": (None, lambda record: record["model"].pop("verified_before_loading")),
                "another shard of the target": (None, target),
                "a missing row": (sliced(slice(0, 7)), lambda record: record["rows"].pop()),
                "a repeated row": (sliced([0, 1, 2, 3, 4, 5, 6, 7, 7]), lambda record: record["rows"].append(
                    dict(record["rows"][-1]))),
                "rows of another selection": (sliced(slice(0, 3)), lambda record: record.update(rows=[
                    {"id": "alpha-0001", "category": "alpha", "anchor": 20 + index, "output_index": index,
                     "window": 20 + index} for index in range(3)])),
                "an extra field": (None, lambda record: record.update(note="x")),
                "no versions": (None, lambda record: record.pop("versions"))}
        for label, (change, record) in both.items():
            result = self.pair(f"both-{label.replace(' ', '-')}", change, both=record)
            with self.subTest(case=f"{label} in both runs"):
                self.assertFalse(result["passed"])
                for kind in mtp.IMPLEMENTATIONS:
                    self.assertTrue(any(problem.startswith(f"the {kind} run holds") for problem in result["problems"]),
                                    result["problems"])
        self.pair("files")
        for name in ("logits", "repeated"):
            np.save(self.folder / "files/class" / f"{name}.npy", self.arrays["class"][name] + 1)
            with self.subTest(case=f"{name} changed after the record"):
                result = self.compare("files")
                self.assertFalse(result["passed"])
                self.assertIn(f"the class {name} differ from the SHA-256 in their record", result["problems"])
        self.pair("again")
        with self.subTest(case="a run compared with itself"):
            self.assertFalse(self.compare("again", model_class="original")["passed"])
        other = read_json(self.folder / "model.json")
        other["revision"] = "1" * 40
        write_json(self.folder / "other.json", other)
        with self.subTest(case="another contract"):
            result = self.compare("again", contract="other.json")
            self.assertFalse(result["passed"])
            self.assertIn("the original run holds another model than its fixed inputs give", result["problems"])
            self.assertIn("the class run holds another model than its fixed inputs give", result["problems"])
        saved = (self.folder / "again/class/logits.npy").read_bytes()
        (self.folder / "again/class/logits.npy").write_bytes(saved + b"0")
        with self.subTest(case="an array of another size than its record"):
            self.assertIn("the class logits cannot be read (ValueError)", self.compare("again")["problems"])
        (self.folder / "again/class/logits.npy").write_bytes(saved)
        (self.folder / "logits-copy.npy").write_bytes(saved)
        (self.folder / "again/class/logits.npy").unlink()
        (self.folder / "again/class/logits.npy").symlink_to(self.folder / "logits-copy.npy")
        with self.subTest(case="an array that links to an identical copy"):
            self.assertIn("the class logits cannot be read (ValueError)", self.compare("again")["problems"])
        (self.folder / "again/class/logits.npy").unlink()
        (self.folder / "again/class/logits.npy").write_bytes(saved)
        for name, problem in {"record.json": "the class run has no readable record.json (ValueError)",
                              "logits.npy": "the class logits cannot be read (ValueError)"}.items():
            saved, _ = piped(self.folder / "again/class" / name)
            with self.subTest(case=f"{name} as a named pipe"):
                waited, outcome = bounded(lambda: self.compare("again"), self.folder / "again/class" / name)
                self.assertFalse(waited)
                self.assertFalse(outcome["value"]["passed"])
                self.assertIn(problem, outcome["value"]["problems"])
            (self.folder / "again/class" / name).unlink()
            (self.folder / "again/class" / name).write_bytes(saved)
        self.assertTrue(self.compare("again")["passed"])
        inputs = {"the target contract": "model.json", "the assistant contract": "assistant.json",
                  "the index of the target": f"target/{mtp.INDEX}", "subset.json": "records/subset.json",
                  "the input records": "evaluation/inputs.jsonl", "the output records": "evaluation/outputs.jsonl",
                  "the features": "records/tf/features/beta-0001.safetensors"}
        for label, name in inputs.items():
            saved, _ = piped(self.folder / name)
            with self.subTest(case=f"{label} as a named pipe"), \
                    mock.patch.object(mtp, "load_contract", wraps=load_contract) as load:
                waited, outcome = bounded(lambda: self.compare("again"), self.folder / name)
                self.assertFalse(waited)
                self.assertIsInstance(outcome.get("error"), ValueError)
                self.assertRegex(str(outcome["error"]), "is not a regular file")
                self.assertEqual(load.call_count, 0 if name == "model.json" else 1)
            (self.folder / name).unlink()
            (self.folder / name).write_bytes(saved)

    def test_compare_writes_only_its_result_as_json_and_exits_with_one_unless_it_passes(self):
        for label, change in {"passing": None, "failing": lambda arrays: arrays["class"]["repeated"].__setitem__(
                (0, 0), 3)}.items():
            result = self.pair(f"main-{label}", change)
            arguments = ["mtp", "compare", "--original", str(self.folder / f"main-{label}/original"), "--model-class",
                         str(self.folder / f"main-{label}/class"), "--model", str(self.folder / "model.json"),
                         "--assistant", str(self.folder / "assistant.json"), "--snapshot", str(self.folder / "target"),
                         "--records", str(self.folder / "records"), "--inputs", str(self.folder / "evaluation/inputs.jsonl"),
                         "--outputs", str(self.folder / "evaluation/outputs.jsonl"), "--steps", "3"]
            with self.subTest(run=label), mock.patch("sys.argv", arguments), \
                    mock.patch("sys.stdout", new_callable=io.StringIO) as printed:
                code = mtp.main()
            self.assertEqual(json.loads(printed.getvalue()), result)
            self.assertEqual(code, 0 if label == "passing" else 1)


@unittest.skipIf(np is None or torch is None, "NumPy, PyTorch and safetensors are required for the records")
@mock.patch.object(mtp, "SUBSET", FIXED)
@mock.patch.object(mtp, "RECORDS", FIXTURE_RECORDS)
class ReadTests(unittest.TestCase):
    def test_a_file_without_a_known_size_is_hashed_only_when_it_is_a_regular_file(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "file").write_bytes(b"abc")
            self.assertEqual(mtp.digest(folder / "file"), {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()})
            os.mkfifo(folder / "pipe")
            (folder / "folder").mkdir()
            for name in ("pipe", "folder"):
                with self.subTest(kind=name):
                    waited, outcome = bounded(lambda: mtp.digest(folder / name), folder / name)
                    self.assertFalse(waited)
                    self.assertIsInstance(outcome.get("error"), ValueError)
                    self.assertRegex(str(outcome["error"]), "is not a regular file")

    def test_a_target_contract_that_changes_between_its_two_reads_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_contract(folder, {name: {"bytes": 1, "sha256": "0" * 64} for name in TOKENIZER_FILES})
            self.assertEqual(mtp.target_contract(path), (load_contract(path), sha256(path)))

            def changed(name):
                found = load_contract(name)
                found["revision"] = "1" * 40
                return found

            with mock.patch.object(mtp, "load_contract", side_effect=changed), \
                    self.assertRaisesRegex(ValueError, "model.json changed while it was read"):
                mtp.target_contract(path)
            good = read_json(path)
            changes = {"a width as a float": {"hidden_size": 48.0}, "a head dimension as a boolean": {"head_dim": True},
                       "a vocabulary as text": {"vocab_size": "301"}, "layers as a float": {"num_hidden_layers": 12.0},
                       "a window of zero": {"sliding_window": 0},
                       "a stop token beyond the vocabulary": {"eos_token_id": [1, 106, CONFIG["vocab_size"]]},
                       "a stop token as a float": {"eos_token_id": [1.0, 106, 50]},
                       "a stop token as a boolean": {"eos_token_id": [True, 106, 50]},
                       "a layer pattern without a full-attention layer": {"layer_pattern": [SLIDING] * 6},
                       "a layer pattern as text": {"layer_pattern": "sliding_attention"}}
            for label, change in changes.items():
                write_json(path, {**good, "config": {**good["config"], **change}})
                with self.subTest(change=label), self.assertRaises(ValueError):
                    mtp.target_contract(path)

    def test_inputs_that_are_named_pipes_are_refused_without_waiting_or_reading_them(self):
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder)
        records, inputs, outputs = write_records(folder)
        target, contract = write_target(folder, torch.zeros(CONFIG["vocab_size"], CONFIG["hidden_size"]))
        assistant, assistant_contract = write_assistant(folder, SYNTHETIC["assistant"], assistant_state())
        level = {"rounds": 1, "tokens": 1, "tokens_per_round": 1.0}
        write_json(records / "trajectories.json", {"per_round_on_teacher_path": {"mtp_fixed": {"7": {
            name: level for name in ("alpha", "beta", "pooled")}}}})
        (records / "drafters").mkdir()
        write_json(records / "drafters/alpha-0001.json", {})
        arguments = (contract, assistant_contract, target, assistant, records, inputs, outputs, 7)
        commands = {"run": lambda output: mtp.run("class", *arguments, output),
                    "acceptance": lambda output: mtp.acceptance(*arguments),
                    "export_weights": lambda output: mtp.export_weights(contract, assistant_contract, target, assistant,
                                                                        7, 16896, output)}
        cases = {"the target contract": (contract, ("run", "acceptance", "export_weights")),
                 "the assistant contract": (assistant_contract, ("run", "acceptance", "export_weights")),
                 "the index of the target": (target / mtp.INDEX, ("run", "export_weights")),
                 "the shard of the table": (target / SHARD, ("run", "export_weights")),
                 "the assistant configuration": (assistant / "config.json", ("run", "export_weights")),
                 "the assistant weights": (assistant / "model.safetensors", ("run", "export_weights")),
                 "subset.json": (records / "subset.json", ("run",)), "the input records": (inputs, ("run",)),
                 "the output records": (outputs, ("run",)),
                 "the features": (records / "tf/features/alpha-0001.safetensors", ("run",)),
                 "trajectories.json": (records / "trajectories.json", ("acceptance",)),
                 "a drafter record": (records / "drafters/alpha-0001.json", ("acceptance",))}
        for label, (path, names) in cases.items():
            saved, pipe = piped(path)
            for name in names:
                output = folder / f"out-{name}"
                (events, watched), (tensors, opened) = buffered_reads(), tensor_reads()
                with self.subTest(input=label, command=name), \
                        mock.patch.object(mtp, "load_contract", wraps=load_contract) as load:
                    with watched(), opened():
                        waited, outcome = bounded(lambda: commands[name](output), path)
                    self.assertFalse(waited)
                    self.assertIsInstance(outcome.get("error"), ValueError)
                    self.assertRegex(str(outcome["error"]), "is not a regular file")
                    self.assertNotIn(pipe, events)
                    self.assertEqual(tensors, [])
                    self.assertFalse(output.exists())
                    self.assertEqual(load.call_count, 0 if label == "the target contract" or name == "export_weights" else 1)
            path.unlink()
            path.write_bytes(saved)
        item = mtp.items_of(load_contract(contract), records, inputs, outputs, mtp.first_per_category)[2][0]
        piped(item["features"])
        tensors, opened = tensor_reads()
        with opened():
            waited, outcome = bounded(lambda: mtp.loaded_features(item), item["features"])
        self.assertFalse(waited)
        self.assertIsInstance(outcome.get("error"), ValueError)
        self.assertRegex(str(outcome["error"]), "is not a regular file")
        self.assertEqual(tensors, [])


@mock.patch.object(mtp, "SUBSET", FIXED)
@mock.patch.object(mtp, "RECORDS", FIXTURE_RECORDS)
class RefusalTests(unittest.TestCase):
    def test_a_run_refuses_an_existing_output_or_a_bad_argument_before_it_reads_anything(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "file").write_text("x")
            (folder / "dangling").symlink_to(folder / "missing")
            with mock.patch.object(mtp, "load_contract") as load:
                for name in ("file", "dangling"):
                    with self.subTest(output=name), self.assertRaises(FileExistsError):
                        mtp.run("class", folder / "model.json", folder / "assistant.json", folder, folder, folder,
                                folder / "inputs.jsonl", folder / "outputs.jsonl", 7, folder / name)
                for implementation, steps in (("other", 7), ("class", 0), ("original", True)):
                    with self.subTest(implementation=implementation, steps=steps), self.assertRaises(ValueError):
                        mtp.run(implementation, folder / "model.json", folder / "assistant.json", folder, folder, folder,
                                folder / "inputs.jsonl", folder / "outputs.jsonl", steps, folder / "new")
            load.assert_not_called()
            self.assertFalse((folder / "new").exists())

    @unittest.skipIf(np is None or torch is None, "NumPy, PyTorch and safetensors are required for the records")
    def test_a_run_that_stops_after_creating_its_output_leaves_only_the_name_of_the_error(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            records, inputs, outputs = write_records(folder)
            target, contract = write_target(folder, torch.zeros(CONFIG["vocab_size"], CONFIG["hidden_size"]))
            assistant, assistant_contract = write_assistant(folder, SYNTHETIC["assistant"], assistant_state())
            for error in (RuntimeError("a message with details"), KeyboardInterrupt()):
                output = folder / type(error).__name__
                with self.subTest(error=type(error).__name__), mock.patch.object(mtp, "run_rows", side_effect=error), \
                        self.assertRaises(type(error)):
                    mtp.run("class", contract, assistant_contract, target, assistant, records, inputs, outputs, 7, output)
                self.assertEqual(sorted(path.name for path in output.iterdir()), ["failure.json"])
                self.assertEqual(read_json(output / "failure.json"),
                                 {"schema": mtp.SCHEMA, "implementation": "class", "failure": type(error).__name__})

    @unittest.skipIf(np is None or torch is None, "NumPy, PyTorch and safetensors are required for the records")
    def test_a_run_creates_no_output_when_a_pinned_file_differs_from_its_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            records, inputs, outputs = write_records(folder)
            target, contract = write_target(folder, torch.zeros(CONFIG["vocab_size"], CONFIG["hidden_size"]))
            assistant, assistant_contract = write_assistant(folder, SYNTHETIC["assistant"], assistant_state())
            shutil.copy(target / mtp.INDEX, folder / "index-copy.json")
            for name, (change, message) in {
                    "the shard of the table": (lambda: (target / SHARD).write_bytes(b"0"), SHARD),
                    "the assistant weights": (lambda: (assistant / "model.safetensors").write_bytes(b"0"),
                                              "model.safetensors"),
                    "the index as a link to an identical copy": (lambda: (target / mtp.INDEX).unlink()
                                                                 or (target / mtp.INDEX).symlink_to(
                                                                     folder / "index-copy.json"), "is a link")}.items():
                saved = {path: path.read_bytes() for path in (target / SHARD, assistant / "model.safetensors")}
                change()
                events, watched = tensor_reads()
                arguments = (contract, assistant_contract, target, assistant, records, inputs, outputs, 7)
                for command, call in {"run": lambda: mtp.run("class", *arguments, folder / "out"),
                                      "acceptance": lambda: mtp.acceptance(*arguments),
                                      "export_weights": lambda: mtp.export_weights(
                                          contract, assistant_contract, target, assistant, 7, 16896,
                                          folder / "weights")}.items():
                    with self.subTest(change=name, command=command), watched(), \
                            self.assertRaisesRegex(ValueError, message):
                        call()
                self.assertEqual(events, [])
                self.assertFalse((folder / "out").exists())
                self.assertFalse((folder / "weights").exists())
                for path, data in saved.items():
                    path.write_bytes(data)
                if (target / mtp.INDEX).is_symlink():
                    (target / mtp.INDEX).unlink()
                    write_json(target / mtp.INDEX, {"metadata": {}, "weight_map": {mtp.TABLE: SHARD}})

    @unittest.skipIf(np is None or torch is None, "NumPy, PyTorch and safetensors are required for the records")
    def test_records_of_691_that_differ_from_their_drafts_or_counts_are_refused_before_any_tensor_is_read(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            records, inputs, outputs = write_records(folder)
            target, contract = write_target(folder, torch.zeros(CONFIG["vocab_size"], CONFIG["hidden_size"]))
            assistant, assistant_contract = write_assistant(folder, SYNTHETIC["assistant"], assistant_state())
            arguments = (contract, assistant_contract, target, assistant, records, inputs, outputs, 7)
            items = mtp.items_of(load_contract(contract), records, inputs, outputs, lambda record: record["items"])[2]
            (records / "drafters").mkdir()
            for item in items:
                rows = [{"a": len(item["input_ids"]) + j, "j": j, "available": min(15, len(item["teacher"]) - 1 - j),
                         "fixed": {"accepted": 0, "tokens": [0], "margins": [0.5]}} for j in range(len(item["teacher"]) - 1)]
                write_json(records / "drafters" / f"{item['id']}.json",
                           {"id": item["id"], "P": len(item["input_ids"]), "T": len(item["teacher"]), "mtp": rows})
            level = {"rounds": 1, "tokens": 1, "tokens_per_round": 1.0}
            good = {"per_round_on_teacher_path": {"mtp_fixed": {"7": {name: level for name in ("alpha", "beta", "pooled")}}}}
            first = records / "drafters" / f"{items[0]['id']}.json"
            kept = read_json(first)
            changes = {"an accepted count unlike the drafts": (lambda record: record["mtp"][0]["fixed"].update(accepted=1),
                                                               good, "accepted counts"),
                       "a draft beyond the vocabulary": (lambda record: record["mtp"][0]["fixed"].update(
                           tokens=[CONFIG["vocab_size"]]), good, "malformed row"),
                       "a rate unlike its counts": (None, {"per_round_on_teacher_path": {"mtp_fixed": {"7": {
                           **good["per_round_on_teacher_path"]["mtp_fixed"]["7"],
                           "pooled": {"rounds": 1, "tokens": 1, "tokens_per_round": 2.0}}}}}, "tokens per round")}
            events, watched = tensor_reads()
            for label, (change, trajectories, message) in changes.items():
                record = json.loads(json.dumps(kept))
                if change:
                    change(record)
                write_json(first, record)
                write_json(records / "trajectories.json", trajectories)
                with self.subTest(change=label), watched(), self.assertRaisesRegex(ValueError, message):
                    mtp.acceptance(*arguments)
            self.assertEqual(events, [])

    @unittest.skipIf(np is None or torch is None, "NumPy, PyTorch and safetensors are required for the records")
    def test_checkpoint_tensors_are_rounded_to_float16_and_returned_in_float32(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.safetensors"
            values = torch.tensor([1e-8, 70000.0, 1.0 + 2 ** -9, -3.0])
            save_file({"weight": values.to(torch.bfloat16)}, str(path))
            loaded = mtp.checkpoint(path, {"weight": [4]}, True)["weight"]
            self.assertEqual(loaded.dtype, torch.float32)
            self.assertEqual(loaded.tolist(), values.to(torch.bfloat16).to(torch.float16).to(torch.float32).tolist())
            self.assertNotEqual(loaded.tolist(), values.to(torch.bfloat16).to(torch.float32).tolist())

    @unittest.skipIf(np is None or torch is None, "NumPy, PyTorch and safetensors are required for the records")
    def test_empty_inputs_of_each_command_are_refused_or_give_a_result_that_did_not_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            records, inputs, outputs = write_records(folder)
            target, contract = write_target(folder, torch.zeros(CONFIG["vocab_size"], CONFIG["hidden_size"]))
            assistant, assistant_contract = write_assistant(folder, SYNTHETIC["assistant"], assistant_state())
            arguments = (contract, assistant_contract, target, assistant, records, inputs, outputs, 7)
            write_json(records / "trajectories.json", {})
            with self.subTest(input="an empty trajectories.json"), self.assertRaisesRegex(ValueError, "trajectories.json"):
                mtp.acceptance(*arguments)
            level = {"rounds": 1, "tokens": 1, "tokens_per_round": 1.0}
            write_json(records / "trajectories.json", {"per_round_on_teacher_path": {"mtp_fixed": {"7": {
                name: level for name in ("alpha", "beta", "pooled")}}}})
            (records / "drafters").mkdir()
            with self.subTest(input="an empty folder of drafter records"), self.assertRaises(FileNotFoundError):
                mtp.acceptance(*arguments)
            for name in ("original", "class"):
                (folder / name).mkdir()
            for label, record in {"empty run folders": None, "empty records": {}}.items():
                if record is not None:
                    for name in ("original", "class"):
                        write_json(folder / name / "record.json", record)
                result = mtp.compare(folder / "original", folder / "class", contract, assistant_contract, target, records,
                                     inputs, outputs, 7)
                with self.subTest(input=label):
                    self.assertFalse(result["passed"])
                    self.assertTrue(result["problems"])
            for path in (inputs, outputs):
                saved = path.read_bytes()
                path.write_bytes(b"")
                with self.subTest(input=f"an empty {path.name}"), self.assertRaisesRegex(ValueError, "SHA-256"):
                    mtp.run("class", *arguments, folder / "out")
                path.write_bytes(saved)
            (records / "tf/features").rename(records / "tf/kept")
            (records / "tf/features").mkdir()
            with self.subTest(input="an empty folder of features", command="run"), self.assertRaises(FileNotFoundError):
                mtp.run("class", *arguments, folder / "out")
            with self.subTest(input="an empty folder of features", command="compare"), \
                    self.assertRaises(FileNotFoundError):
                mtp.compare(folder / "original", folder / "class", contract, assistant_contract, target, records, inputs,
                            outputs, 7)
            self.assertFalse((folder / "out").exists())

    @unittest.skipIf(np is None or torch is None, "NumPy, PyTorch and safetensors are required for the records")
    def test_an_empty_or_another_subset_is_refused_by_every_command_before_any_tensor_is_read(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            records, inputs, outputs = write_records(folder)
            target, contract = write_target(folder, torch.zeros(CONFIG["vocab_size"], CONFIG["hidden_size"]))
            assistant, assistant_contract = write_assistant(folder, SYNTHETIC["assistant"], assistant_state())
            subset = read_json(records / "subset.json")
            write_json(records / "trajectories.json", {"per_round_on_teacher_path": {"mtp_fixed": {"7": {
                "pooled": {"rounds": 0, "tokens": 0, "tokens_per_round": None}}}}})
            (records / "drafters").mkdir()
            events, watched = tensor_reads()
            other = json.loads(json.dumps(subset["items"]))
            other[1]["id"] = "alpha-0003"
            for label, change, message in (
                    ("an empty subset", {"counts": {}, "items": []}, "fixed subset of #691: 3 items in 2 categories"),
                    ("a subset of one category", {"counts": {"alpha": 2}, "items": [
                        entry for entry in subset["items"] if entry["category"] == "alpha"]},
                     "fixed subset of #691: 3 items in 2 categories"),
                    ("a subset with another id", {"items": other}, "fixed ids of #691")):
                write_json(records / "subset.json", {**subset, **change})
                arguments = (contract, assistant_contract, target, assistant, records, inputs, outputs, 7)
                commands = {"run": lambda: mtp.run("class", *arguments, folder / "out"),
                            "acceptance": lambda: mtp.acceptance(*arguments),
                            "compare": lambda: mtp.compare(folder / "a", folder / "b", contract, assistant_contract,
                                                           target, records, inputs, outputs, 7)}
                for name, command in commands.items():
                    with self.subTest(subset=label, command=name), watched(), self.assertRaisesRegex(ValueError, message):
                        command()
                self.assertFalse((folder / "out").exists())
            self.assertEqual(events, [])

    @unittest.skipIf(np is None or torch is None, "NumPy, PyTorch and safetensors are required for the records")
    def test_an_unsupported_assistant_or_tensors_of_other_shapes_are_refused_before_the_output_or_any_tensor_is_read(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            records, inputs, outputs = write_records(folder)
            vocabulary, hidden = CONFIG["vocab_size"], CONFIG["hidden_size"]
            place = mtp.layout(CONFIG)

            def tensor(name, shape):
                return lambda state: state.update({name: torch.zeros(shape)})

            def features():
                save_file({key: torch.zeros(2, 2, dtype=torch.bfloat16) for key in ("postnorm", f"k{place['sliding']}")},
                          str(records / "tf/features/alpha-0001.safetensors"))

            unsupported, weights = "Unsupported Gemma 4 assistant", "model.safetensors differs from the expected"
            cases = {"ordered embeddings": (lambda document: document.update(use_ordered_embeddings=True), None, None,
                                            unsupported),
                     "a width as a float": (lambda document: document["text_config"].update(hidden_size=32.0), None, None,
                                            "hidden_size"),
                     "a target width as a float": (None, None, "target width", "model.json holds configuration values"),
                     "another width of the target": (lambda document: document.update(backbone_hidden_size=64), None,
                                                     None, unsupported),
                     "a sliding-window layer in place of the full-attention layer": (
                         lambda document: document["text_config"].update(layer_types=[SLIDING] * 4), None, None,
                         unsupported),
                     "a query projection of another shape": (None, tensor("model.layers.0.self_attn.q_proj.weight",
                                                                          (64, 33)), None, weights),
                     "a missing tensor": (None, lambda state: state.pop("model.norm.weight"), None, weights),
                     "an extra tensor": (None, tensor("model.layers.4.layer_scalar", (1,)), None, weights),
                     "a table of another width": (None, None, (vocabulary, hidden + 2),
                                                  f"{SHARD} differs from the expected"),
                     "features of another shape": (None, None, features, "alpha-0001.safetensors does not hold")}
            backup = {path: path.read_bytes() for path in (records / "tf/features").iterdir()}
            events, watched = tensor_reads()
            for label, (configured, changed, other, message) in cases.items():
                for path, data in backup.items():
                    path.write_bytes(data)
                case = folder / label.replace(" ", "-")
                case.mkdir()
                document, state = json.loads(json.dumps(SYNTHETIC["assistant"])), assistant_state()
                if configured:
                    configured(document)
                if changed:
                    changed(state)
                if callable(other):
                    other()
                table = torch.zeros(*(other if isinstance(other, tuple) else (vocabulary, hidden)))
                target, contract = write_target(case, table)
                if other == "target width":
                    pinned = read_json(contract)
                    write_json(contract, {**pinned, "config": {**pinned["config"], "hidden_size": float(hidden)}})
                assistant, assistant_contract = write_assistant(case, document, state)
                arguments = (contract, assistant_contract, target, assistant, records, inputs, outputs, 7)
                for implementation in mtp.IMPLEMENTATIONS:
                    output = case / implementation
                    with self.subTest(case=label, implementation=implementation), watched(), \
                            self.assertRaisesRegex(ValueError, message):
                        mtp.run(implementation, *arguments, output)
                    self.assertFalse(output.exists())
                with self.subTest(case=label, command="acceptance"), watched(), \
                        self.assertRaisesRegex(ValueError, message):
                    mtp.acceptance(*arguments)
                if not callable(other):
                    with self.subTest(case=label, command="export_weights"), watched(), \
                            self.assertRaisesRegex(ValueError, message):
                        mtp.export_weights(contract, assistant_contract, target, assistant, 7, 16896, case / "weights")
                    self.assertFalse((case / "weights").exists())
            self.assertEqual(events, [])
            with watched():
                mtp.table(target / SHARD, CONFIG)
            self.assertEqual(events, [(inode(target / SHARD), mtp.TABLE)])


REAL = SimpleNamespace(backbone_hidden_size=5376, vocab_size=262144, num_key_value_heads=16, head_dim=256,
                       num_global_key_value_heads=4, global_head_dim=512)
HELD = {"tensors": 49, "elements": 74372, "elements_by_dtype": {"torch.float16": 74372}, "quantized_layouts": 0}


def shaped(dtype, *shape):
    return {"dtype": dtype, "shape": list(shape)}


def found_of(name, steps=7):
    outputs = ({"logits": shaped("float16", 1, 1, 262144), "post_projection": shaped("float16", 1, 1, 5376)}
               if name == "verification" else {"draft_tokens": shaped("int32", 1, steps)})
    function = "verify_step" if name == "verification" else f"draft_{steps}"
    sliding, full = shaped("float16", 1, 1, 16, None, 256), shaped("float16", 1, 1, 4, None, 512)
    inputs = {"input_ids": shaped("int32", 1, 1), "hidden_states": shaped("float16", 1, 1, 5376),
              "position_ids": shaped("int32", 1, 1), "k_sliding": sliding, "v_sliding": copy.deepcopy(sliding),
              "k_full": full, "v_full": copy.deepcopy(full)}
    return {"functions": {function: {"inputs": inputs, "outputs": outputs, "states": {}}},
            "output_order": {function: list(outputs)}}


def function_of(found):
    return next(iter(found["functions"].values()))


def inputs_of(found):
    return function_of(found)["inputs"]


def outputs_of(found):
    return function_of(found)["outputs"]


def first_output(found):
    return next(iter(outputs_of(found).values()))


def inspected(name, change=None, steps=7, expected_steps=7):
    found, held = found_of(name, steps), copy.deepcopy(HELD)
    if change:
        change(found, held)
    return mtp.asset_problems(found, mtp.expected_assets(REAL, expected_steps)[name], held)


def renamed_function(found, held):
    old = next(iter(found["functions"]))
    found["functions"]["other"] = found["functions"].pop(old)
    found["output_order"]["other"] = found["output_order"].pop(old)


def extra_function(found, held):
    found["functions"]["extra"] = copy.deepcopy(function_of(found))
    found["output_order"]["extra"] = list(outputs_of(found))


def no_function(found, held):
    found["functions"].clear()
    found["output_order"].clear()


def set_field(getter, key, field, value):
    def change(found, held):
        getter(found)[key][field] = value
    return change


def set_dimension(getter, key, index, value):
    def change(found, held):
        getter(found)[key]["shape"][index] = value
    return change


def drop(getter, key):
    return lambda found, held: getter(found).pop(key)


def rename(getter, old, new):
    return lambda found, held: getter(found).update({new: getter(found).pop(old)})


def add(getter, key, value):
    return lambda found, held: getter(found).update({key: value})


def element_types(types, layouts=0):
    def change(found, held):
        held.update(elements_by_dtype=types, quantized_layouts=layouts)
    return change


def set_output(index, **fields):
    def change(found, held):
        list(outputs_of(found).values())[index].update(fields)
    return change


class AssetInspectionTests(unittest.TestCase):
    def test_the_expected_functions_are_the_one_step_logits_and_the_unrolled_tokens_of_the_assistant_and_the_steps(self):
        for steps in (1, 7, 15):
            expected = mtp.expected_assets(REAL, steps)
            with self.subTest(steps=steps):
                self.assertEqual(list(expected), ["verification", "distribution"])
                self.assertEqual(expected["verification"], found_of("verification")["functions"])
                self.assertEqual(expected["distribution"], found_of("distribution", steps)["functions"])
                self.assertEqual(list(expected["verification"]["verify_step"]["outputs"]), ["logits", "post_projection"])
                self.assertEqual(list(expected["distribution"][f"draft_{steps}"]["outputs"]), ["draft_tokens"])
        self.assertEqual(list(inputs_of(found_of("distribution"))), ["input_ids", "hidden_states", "position_ids",
                                                                       "k_sliding", "v_sliding", "k_full", "v_full"])
        self.assertEqual(mtp.expected_assets(REAL, 7)["verification"]["verify_step"]["inputs"],
                         mtp.expected_function(REAL, 7)["inputs"])

    def test_an_asset_that_matches_the_expected_functions_and_float16_weights_passes(self):
        for name in ("verification", "distribution"):
            with self.subTest(asset=name):
                self.assertEqual(inspected(name), [])

    def test_an_asset_whose_functions_inputs_outputs_states_or_weights_differ_from_the_expected_fails(self):
        cases = {"another function name": renamed_function, "an extra function": extra_function, "no function": no_function,
                 "an input with another name": rename(inputs_of, "input_ids", "token_ids"),
                 "an input that is missing": drop(inputs_of, "position_ids"),
                 "an extra input": add(inputs_of, "mask", shaped("int32", 1, 1)),
                 "hidden_states in float32": set_field(inputs_of, "hidden_states", "dtype", "float32"),
                 "input_ids in int64": set_field(inputs_of, "input_ids", "dtype", "int64"),
                 "position_ids in float32": set_field(inputs_of, "position_ids", "dtype", "float32"),
                 "k_full in float32": set_field(inputs_of, "k_full", "dtype", "float32"),
                 "v_sliding in int32": set_field(inputs_of, "v_sliding", "dtype", "int32"),
                 "an input of one dimension less": set_field(inputs_of, "k_sliding", "shape", [1, 16, None, 256]),
                 "an input of one dimension more": set_field(inputs_of, "k_sliding", "shape", [1, 1, 1, 16, None, 256]),
                 "a fixed dimension of another size": set_dimension(inputs_of, "k_full", 2, 8),
                 "another width of the hidden state": set_dimension(inputs_of, "hidden_states", 2, 5375),
                 "a window of fixed length": set_dimension(inputs_of, "k_sliding", 3, 1025),
                 "full keys of fixed length": set_dimension(inputs_of, "v_full", 3, 16896),
                 "a hidden state of dynamic width": set_dimension(inputs_of, "hidden_states", 2, None),
                 "a dynamic dimension of the anchor token": set_dimension(inputs_of, "input_ids", 1, None),
                 "a head width that is dynamic": set_dimension(inputs_of, "v_sliding", 4, None),
                 "an output that is missing": lambda found, held: outputs_of(found).pop(next(reversed(outputs_of(found)))),
                 "an extra output": add(outputs_of, "extra", shaped("float16", 1, 1, 5376)),
                 "an output in float32": set_output(0, dtype="float32"),
                 "an output in int64": set_output(0, dtype="int64"),
                 "an output of one dimension less": lambda found, held: first_output(found)["shape"].pop(0),
                 "an output of one dimension more": lambda found, held: first_output(found)["shape"].insert(0, 1),
                 "a state": add(lambda found: function_of(found)["states"], "keyCache",
                                shaped("float16", 1, 1, 4, None, 512)),
                 "float32 elements next to the float16 elements": element_types(
                     {"torch.float16": 74371, "torch.float32": 1}),
                 "only float32 elements": element_types({"torch.float32": 74372}),
                 "bfloat16 elements": element_types({"torch.bfloat16": 74372}),
                 "uint8 elements of a quantization": element_types({"torch.float16": 10, "torch.uint8": 74362}, 7),
                 "a quantized layout among float16 elements": element_types({"torch.float16": 74372}, 1),
                 "a quantized layout count of 0.0": element_types({"torch.float16": 74372}, 0.0),
                 "a quantized layout count of false": element_types({"torch.float16": 74372}, False),
                 "no weights": element_types({})}
        for name in ("verification", "distribution"):
            for label, change in cases.items():
                with self.subTest(asset=name, case=label):
                    self.assertTrue(inspected(name, change))

    def test_the_outputs_must_be_returned_in_the_fixed_order(self):
        def swapped(found, held):
            found["output_order"]["verify_step"] = ["post_projection", "logits"]

        def one_missing(found, held):
            found["output_order"]["verify_step"] = ["logits"]

        def other_function(found, held):
            found["output_order"] = {"other": ["logits", "post_projection"]}

        def empty(found, held):
            found["output_order"] = {}

        for label, change in {"swapped": swapped, "one missing": one_missing, "listed for another function": other_function,
                              "not listed": empty}.items():
            with self.subTest(order=label):
                self.assertTrue(inspected("verification", change))
        self.assertTrue(inspected("distribution", lambda found, held: found["output_order"].update({"draft_7": []})))
        self.assertTrue(inspected("distribution", lambda found, held: found["output_order"].update(
            {"draft_7": ["draft_tokens", "extra"]})))

    def test_an_asset_of_another_number_of_steps_fails_even_when_its_name_and_its_output_agree(self):
        for steps in (1, 6, 8, 15):
            with self.subTest(steps=steps):
                problems = inspected("distribution", steps=steps)
                self.assertTrue(problems)
                self.assertIn("draft_7", " ".join(problems))
        self.assertTrue(inspected("distribution", set_output(0, shape=[1, 6])))
        self.assertTrue(inspected("distribution", set_output(0, shape=[1, 8])))
        self.assertTrue(inspected("distribution", expected_steps=6))
        self.assertEqual(inspected("verification", expected_steps=6), [])

    def test_values_that_equal_the_expected_numbers_in_python_but_are_not_the_same_type_fail(self):
        cases = {"a width given as a float": ("verification", set_dimension(inputs_of, "hidden_states", 2, 5376.0)),
                 "a width given as a boolean": ("verification", set_dimension(inputs_of, "input_ids", 0, True)),
                 "an output width given as a float": ("verification", set_output(0, shape=[1, 1, 262144.0])),
                 "steps given as a float": ("distribution", set_output(0, shape=[1, 7.0]))}
        for label, (name, change) in cases.items():
            with self.subTest(case=label):
                self.assertTrue(inspected(name, change))

    def test_a_folder_of_an_asset_holds_exactly_its_three_regular_files(self):
        self.assertEqual(mtp.ASSET_ITEMS, ("main.hash", "main.mlirb", "metadata.json"))
        self.assertEqual(list(mtp.ASSET_FOLDERS.items()), [("verification", "verification.aimodel"),
                                                           ("distribution", "distribution.aimodel")])
        wanted = {name: "file" for name in mtp.ASSET_ITEMS}
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            good = folder / "good"
            good.mkdir()
            for name in mtp.ASSET_ITEMS:
                (good / name).write_bytes(b"data")
            self.assertEqual(mtp.listing(good), wanted)
            self.assertEqual(mtp.layout_problems(mtp.listing(good), wanted), [])
            (folder / "outside.bin").write_bytes(b"data")

            def link(name):
                return lambda path: (path / name).unlink() or (path / name).symlink_to(folder / "outside.bin")

            def folder_in_place(name):
                return lambda path: (path / name).unlink() or (path / name).mkdir()

            def pipe_in_place(name):
                return lambda path: (path / name).unlink() or os.mkfifo(path / name)

            cases = {"no main.hash": lambda path: (path / "main.hash").unlink(),
                     "no main.mlirb": lambda path: (path / "main.mlirb").unlink(),
                     "no metadata.json": lambda path: (path / "metadata.json").unlink(),
                     "no file": lambda path: [(path / name).unlink() for name in mtp.ASSET_ITEMS],
                     "an extra file": lambda path: (path / "notes.txt").write_bytes(b"data"),
                     "an extra hidden file": lambda path: (path / ".DS_Store").write_bytes(b"data"),
                     "an extra folder": lambda path: (path / "derived").mkdir(),
                     "a link in place of main.mlirb": link("main.mlirb"),
                     "a link in place of main.hash": link("main.hash"),
                     "a link in place of metadata.json": link("metadata.json"),
                     "a folder in place of metadata.json": folder_in_place("metadata.json"),
                     "a named pipe in place of main.hash": pipe_in_place("main.hash")}
            for label, change in cases.items():
                changed = folder / "changed"
                shutil.copytree(good, changed)
                change(changed)
                with self.subTest(case=label):
                    self.assertTrue(mtp.layout_problems(mtp.listing(changed), wanted))
                shutil.rmtree(changed)

    def test_main_hash_is_read_once_with_its_limit_before_any_file_of_the_asset_is_hashed(self):
        found = found_of("verification")
        expected = mtp.expected_assets(REAL, 7)["verification"]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            asset = output / "verification.aimodel"
            asset.mkdir()
            (asset / "main.mlirb").write_bytes(b"m" * 100)
            (asset / "metadata.json").write_bytes(b"{}")
            (asset / "main.hash").write_bytes(b"h" * (4 << 20))
            total, events = [], []
            with mock.patch("Tools.Conversion.adapters.gemma4_31b.export.inspect_asset", return_value=found), \
                    reads_counted(total), opens_counted(events, [asset / "main.hash"]), \
                    self.assertRaisesRegex(ValueError, "main.hash holds more than 64 B"):
                mtp.inspected(output, "verification", expected, copy.deepcopy(HELD), {})
            self.assertEqual((sum(total), len(events)), (65, 1))
            (asset / "main.hash").write_bytes(b"h" * 32)
            total, events = [], []
            with mock.patch("Tools.Conversion.adapters.gemma4_31b.export.inspect_asset", return_value=found), \
                    reads_counted(total), opens_counted(events, [asset / "main.hash"]):
                record = mtp.inspected(output, "verification", expected, copy.deepcopy(HELD), {})
            self.assertEqual(len(events), 1)
            self.assertEqual(record["main_hash"], "68" * 32)
            self.assertEqual(record["files"]["verification.aimodel/main.hash"],
                             {"bytes": 32, "sha256": hashlib.sha256(b"h" * 32).hexdigest()})
            self.assertEqual(sorted(record["files"]), [f"verification.aimodel/{item}" for item in mtp.ASSET_ITEMS])
            self.assertEqual(record["bytes"], 32 + 100 + 2)

    def test_the_output_folder_holds_exactly_the_two_asset_folders(self):
        wanted = {folder: "folder" for folder in mtp.ASSET_FOLDERS.values()}
        cases = {"another file": lambda path: (path / "export.json").write_bytes(b"{}"),
                 "a failure record": lambda path: (path / "failure.json").write_bytes(b"{}"),
                 "another folder": lambda path: (path / "other.aimodel").mkdir(),
                 "a missing asset": lambda path: (path / "verification.aimodel").rmdir(),
                 "both assets missing": lambda path: [(path / name).rmdir() for name in wanted],
                 "a file in place of an asset": lambda path: (path / "distribution.aimodel").rmdir() or (
                     path / "distribution.aimodel").write_bytes(b"data"),
                 "a link in place of an asset": lambda path: (path / "distribution.aimodel").rmdir() or (
                     path / "distribution.aimodel").symlink_to(path / "verification.aimodel")}
        with tempfile.TemporaryDirectory() as folder:
            for label, change in [("nothing", None), *cases.items()]:
                path = Path(folder) / label.replace(" ", "-")
                path.mkdir()
                for name in wanted:
                    (path / name).mkdir()
                if change:
                    change(path)
                with self.subTest(case=label):
                    self.assertEqual(bool(mtp.layout_problems(mtp.listing(path), wanted)), change is not None)


def reported_small(real):
    def call(*arguments, **options):
        status = real(*arguments, **options)
        return os.stat_result((*tuple(status)[:6], 10, *tuple(status)[7:]))

    return call


@contextlib.contextmanager
def reads_counted(total, limit=None):
    real = os.read

    def call(descriptor, length):
        block = real(descriptor, length if limit is None else min(length, limit))
        total.append(len(block))
        return block

    with mock.patch.object(os, "read", call):
        yield


@contextlib.contextmanager
def opens_counted(events, paths):
    names, real_os, real_io = {os.path.realpath(path) for path in paths}, os.open, io.open

    def count(file):
        if isinstance(file, (str, os.PathLike)) and os.path.realpath(file) in names:
            events.append(os.path.realpath(file))

    def by_os(path, *arguments, **options):
        count(path)
        return real_os(path, *arguments, **options)

    def by_io(file, *arguments, **options):
        count(file)
        return real_io(file, *arguments, **options)

    with mock.patch.object(os, "open", by_os), mock.patch.object(io, "open", by_io):
        yield


def fake_assets(output, passed=True, partly=False):
    for name in mtp.ASSET_FOLDERS.values():
        (output / name).mkdir(parents=True, exist_ok=True)
        (output / name / "main.mlirb").write_bytes(b"data")
        if partly:
            break
    return {"schema": mtp.EXPORT_WEIGHTS, "assets": {"verification": {"bytes": 4, "problems": []},
                                                      "distribution": {"bytes": 4, "problems": []}},
            "problems": [] if passed else ["a problem"], "passed": passed}


def fake(passed=True):
    return lambda *arguments: fake_assets(arguments[-1], passed)


@unittest.skipIf(np is None or torch is None, "NumPy, PyTorch and safetensors are required for the folders")
class ExportWeightsTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        self.target, self.contract = write_target(self.folder, torch.zeros(CONFIG["vocab_size"], CONFIG["hidden_size"]))
        self.assistant, self.assistant_contract = write_assistant(self.folder, SYNTHETIC["assistant"], assistant_state())
        self.output = self.folder / "export"

    def export(self, steps=7, context=16896, output=None):
        return mtp.export_weights(self.contract, self.assistant_contract, self.target, self.assistant, steps, context,
                                  output or self.output)

    def names(self, output=None):
        return sorted(path.name for path in (output or self.output).iterdir())

    def test_an_existing_output_or_a_bad_number_of_steps_is_refused_before_anything_is_read_or_made(self):
        (self.folder / "file").write_text("x")
        (self.folder / "dangling").symlink_to(self.folder / "missing")
        (self.folder / "folder").mkdir()
        (self.folder / "folder/kept").write_text("x")
        with mock.patch.object(mtp, "target_contract") as read, mock.patch.object(mtp, "export_assets") as work:
            for name in ("file", "dangling", "folder"):
                with self.subTest(output=name), self.assertRaises(FileExistsError):
                    self.export(output=self.folder / name)
            for steps in (0, -1, True, 1.0, None, "7", mtp.DRAFTS + 1, 2 ** 40):
                with self.subTest(steps=steps), self.assertRaises(ValueError):
                    self.export(steps=steps)
            read.assert_not_called()
            work.assert_not_called()
        self.assertEqual(self.names(self.folder / "folder"), ["kept"])
        self.assertFalse(self.output.exists())

    def test_a_context_that_is_not_a_whole_number_above_the_window_and_within_int32_is_refused_before_the_table(self):
        window = CONFIG["sliding_window"] + 1
        with mock.patch.object(mtp, "table_shard") as shard, mock.patch.object(mtp, "export_assets") as work:
            for context in (True, 1.0, "16896", None, -1, 0, 1, window - 1, window, 2 ** 31):
                with self.subTest(context=context), self.assertRaises(ValueError):
                    self.export(context=context)
            shard.assert_not_called()
            work.assert_not_called()
        self.assertFalse(self.output.exists())
        with mock.patch.object(mtp, "export_assets", side_effect=fake()) as work:
            for index, (steps, context) in enumerate(((1, window + 1), (mtp.DRAFTS, 2 ** 31 - 1))):
                output = self.folder / f"accepted-{index}"
                with self.subTest(steps=steps, context=context):
                    self.export(steps=steps, context=context, output=output)
                    self.assertEqual(self.names(output), ["distribution.aimodel", "export.json", "verification.aimodel"])
            self.assertEqual([(call.args[0]["steps"], call.args[0]["context"]) for call in work.call_args_list],
                             [(1, window + 1), (mtp.DRAFTS, 2 ** 31 - 1)])

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
                    self.export()
                self.assertEqual(events, [])
                self.assertFalse(self.output.exists())
            path.write_bytes(saved)
        with mock.patch.object(mtp, "export_assets", side_effect=fake()):
            self.export()
        self.assertEqual(self.names(), ["distribution.aimodel", "export.json", "verification.aimodel"])

    def test_the_configuration_of_the_assistant_is_read_with_its_pinned_size_even_when_the_check_before_is_skipped(self):
        path = self.assistant / "config.json"
        path.write_bytes(path.read_bytes() + b" ")
        unchecked = {"side_effect": lambda folder, files, names: {name: dict(files[name]) for name in names}}
        events, watched = tensor_reads()
        with mock.patch.object(mtp, "verified", **unchecked), watched(), self.assertRaisesRegex(ValueError, "B, not"):
            self.export()
        self.assertEqual(events, [])
        self.assertFalse(self.output.exists())

    def test_contracts_that_are_not_in_the_pinned_form_are_refused_before_the_output(self):
        pinned = read_json(self.assistant_contract)
        entries = pinned["files"]
        changes = {"a revision that is not a full hash": {**pinned, "revision": "627c5ec"},
                   "a missing file": {**pinned, "files": {"config.json": entries["config.json"]}},
                   "an extra file": {**pinned, "files": {**entries, "tokenizer.json": entries["config.json"]}},
                   "a size as a float": {**pinned, "files": {**entries, "config.json": {
                       **entries["config.json"], "bytes": float(entries["config.json"]["bytes"])}}},
                   "a hash in capitals": {**pinned, "files": {**entries, "config.json": {
                       **entries["config.json"], "sha256": entries["config.json"]["sha256"].upper()}}},
                   "another model": {**pinned, "model_id": "not a repository"}}
        for label, change in changes.items():
            write_json(self.assistant_contract, change)
            with self.subTest(change=label), self.assertRaises(ValueError):
                self.export()
            self.assertFalse(self.output.exists())
        write_json(self.assistant_contract, pinned)
        good = read_json(self.contract)
        documents = {"a width as a float": {**good, "config": {**good["config"], "hidden_size": 48.0}},
                     "a window as a boolean": {**good, "config": {**good["config"], "sliding_window": True}},
                     "a stop token beyond the vocabulary": {**good, "config": {**good["config"], "eos_token_id": [1, 301]}},
                     "a field that the contract does not have": {**good, "extra": 1},
                     "a revision that is not a full hash": {**good, "revision": "main"},
                     "another reference": {**good, "reference": {**good["reference"], "compute": "float32"}},
                     "another tokenizer policy": {**good, "tokenizer": {**good["tokenizer"], "policy": "resave"}}}
        for label, document in documents.items():
            write_json(self.contract, document)
            with self.subTest(change=label), self.assertRaises(ValueError):
                self.export()
            self.assertFalse(self.output.exists())

    def contracts(self):
        return (("the target contract", self.contract), ("the assistant contract", self.assistant_contract))

    def test_a_contract_that_is_a_link_to_a_valid_contract_is_refused(self):
        for label, path in self.contracts():
            saved, copy_of = path.read_bytes(), self.folder / f"{path.name}.copy"
            copy_of.write_bytes(saved)
            path.unlink()
            path.symlink_to(copy_of)
            with self.subTest(file=label), mock.patch.object(mtp, "export_assets", side_effect=fake()), \
                    self.assertRaisesRegex(ValueError, "is a link"):
                self.export()
            self.assertFalse(self.output.exists())
            path.unlink()
            path.write_bytes(saved)

    def test_a_contract_above_the_size_limit_is_refused_by_what_is_read_whatever_size_is_reported(self):
        limit = mtp.CONTRACT_LIMIT
        self.assertEqual(limit, 2 ** 20)
        for label, path in self.contracts():
            saved = path.read_bytes()
            before = 0 if path == self.contract else len(self.contract.read_bytes())
            path.write_bytes(saved + b" " * (limit + 100000 - len(saved)))
            total = []
            with self.subTest(file=label), reads_counted(total), mock.patch.object(os, "stat", reported_small(os.stat)), \
                    mock.patch.object(os, "fstat", reported_small(os.fstat)), \
                    mock.patch.object(mtp, "export_assets", side_effect=fake()), \
                    self.assertRaisesRegex(ValueError, "holds more than 1048576 B"):
                self.export()
            self.assertEqual(sum(total), before + limit + 1)
            self.assertFalse(self.output.exists())
            path.write_bytes(saved + b" " * (limit - len(saved)))
            with self.subTest(file=label, size="at the limit"), mock.patch.object(mtp, "export_assets", side_effect=fake()):
                self.export(output=self.folder / f"at-the-limit-{path.name}")
            path.write_bytes(saved)

    def test_a_contract_is_read_to_its_end_and_its_limit_is_judged_when_a_read_returns_less_than_asked(self):
        limit = mtp.CONTRACT_LIMIT
        for label, path in self.contracts():
            saved = path.read_bytes()
            with self.subTest(file=label, size="as written"), reads_counted([], 7), \
                    mock.patch.object(mtp, "export_assets", side_effect=fake()):
                self.export(output=self.folder / f"short-{path.name}")
            path.write_bytes(saved + b" " * (limit - len(saved)))
            with self.subTest(file=label, size="at the limit"), reads_counted([], 4099), \
                    mock.patch.object(mtp, "export_assets", side_effect=fake()):
                self.export(output=self.folder / f"short-at-the-limit-{path.name}")
            path.write_bytes(saved + b" " * (limit - len(saved) + 1))
            with self.subTest(file=label, size="above the limit"), reads_counted([], 4099), \
                    mock.patch.object(mtp, "export_assets", side_effect=fake()), \
                    self.assertRaisesRegex(ValueError, "holds more than 1048576 B"):
                self.export()
            path.write_bytes(saved)

    def test_each_contract_is_opened_once_by_its_path_and_the_checks_use_the_bytes_that_were_read(self):
        events = []
        with opens_counted(events, [self.contract, self.assistant_contract]), \
                mock.patch.object(mtp, "load_contract") as reread, \
                mock.patch.object(mtp, "export_assets", side_effect=fake()):
            record = self.export()
        self.assertEqual(sorted(events), sorted(os.path.realpath(path) for _, path in self.contracts()))
        reread.assert_not_called()
        self.assertEqual((record["passed"], self.names()), (True, ["distribution.aimodel", "export.json", "verification.aimodel"]))

    def test_an_export_that_stops_after_creating_its_output_leaves_only_the_name_of_the_error(self):
        def signalled(*arguments):
            signal.raise_signal(signal.SIGINT)

        def after_one_asset(*arguments):
            fake_assets(arguments[-1], partly=True)
            raise RuntimeError("a message with details")

        cases = [("RuntimeError", RuntimeError("a message with details"), RuntimeError, ["failure.json"]),
                 ("KeyboardInterrupt", KeyboardInterrupt(), KeyboardInterrupt, ["failure.json"]),
                 ("SystemExit", SystemExit(3), SystemExit, ["failure.json"]),
                 ("MemoryError", MemoryError(), MemoryError, ["failure.json"]),
                 ("a signal", signalled, KeyboardInterrupt, ["failure.json"]),
                 ("a failure after one asset was saved", after_one_asset, RuntimeError,
                  ["failure.json", "verification.aimodel"])]
        for label, effect, error, names in cases:
            output = self.folder / label.replace(" ", "-")
            with self.subTest(error=label), mock.patch.object(mtp, "export_assets", side_effect=effect), \
                    self.assertRaises(error):
                self.export(output=output)
            self.assertEqual(self.names(output), names)
            self.assertEqual(read_json(output / "failure.json"), {"schema": mtp.EXPORT_WEIGHTS, "failure": error.__name__})

    def test_a_signal_while_the_output_is_made_lets_the_creation_finish_and_leaves_a_failure_record(self):
        real = Path.mkdir
        for label, number in (("SIGINT", signal.SIGINT), ("SIGTERM", signal.SIGTERM)):
            output, fired = self.folder / label, []

            def made_and_signalled(path, *arguments, output=output, fired=fired, number=number, **options):
                real(path, *arguments, **options)
                if path == output and not fired:
                    fired.append(path)
                    signal.raise_signal(number)

            with self.subTest(signal=label), mock.patch.object(Path, "mkdir", made_and_signalled), \
                    mock.patch.object(mtp, "export_assets") as work, self.assertRaises(KeyboardInterrupt):
                self.export(output=output)
            work.assert_not_called()
            self.assertEqual(self.names(output), ["failure.json"])
            self.assertEqual(read_json(output / "failure.json"), {"schema": mtp.EXPORT_WEIGHTS,
                                                                  "failure": "KeyboardInterrupt"})

    def test_a_signal_while_the_success_record_is_written_lets_the_write_finish_and_replaces_it_by_a_failure_record(self):
        real = mtp.write_json
        for label, number in (("SIGINT", signal.SIGINT), ("SIGTERM", signal.SIGTERM)):
            events, output = [], self.folder / label

            def logged(path, value, events=events, number=number):
                if Path(path).name == "export.json":
                    events.append("signal")
                    signal.raise_signal(number)
                real(path, value)
                events.append(Path(path).name)

            with self.subTest(signal=label), mock.patch.object(mtp, "write_json", logged), \
                    mock.patch.object(mtp, "export_assets", side_effect=fake()), self.assertRaises(KeyboardInterrupt):
                self.export(output=output)
            self.assertEqual(events, ["signal", "export.json", "failure.json"])
            self.assertEqual(self.names(output), ["distribution.aimodel", "failure.json", "verification.aimodel"])
            self.assertEqual(read_json(output / "failure.json"), {"schema": mtp.EXPORT_WEIGHTS,
                                                                  "failure": "KeyboardInterrupt"})

    def test_a_signal_while_the_failure_record_is_written_lets_the_write_finish(self):
        real = mtp.write_json
        for label, number in (("SIGINT", signal.SIGINT), ("SIGTERM", signal.SIGTERM)):
            events, output = [], self.folder / label

            def logged(path, value, events=events, number=number):
                if Path(path).name == "failure.json":
                    events.append("signal")
                    signal.raise_signal(number)
                real(path, value)
                events.append(Path(path).name)

            with self.subTest(signal=label), mock.patch.object(mtp, "write_json", logged), \
                    mock.patch.object(mtp, "export_assets", side_effect=RuntimeError("a message")), \
                    self.assertRaises(KeyboardInterrupt):
                self.export(output=output)
            self.assertEqual(events, ["signal", "failure.json"])
            self.assertEqual(self.names(output), ["failure.json"])
            self.assertEqual(read_json(output / "failure.json"), {"schema": mtp.EXPORT_WEIGHTS, "failure": "RuntimeError"})

    def guarded(self):
        def terminated(number, frame):
            raise RuntimeError("SIGTERM reached the handler that was set before the export")

        def interrupted(number, frame):
            raise KeyboardInterrupt("the handler that was set before the export")

        for number, handler in ((signal.SIGTERM, terminated), (signal.SIGINT, interrupted)):
            self.addCleanup(signal.signal, number, signal.signal(number, handler))
        return terminated, interrupted

    def test_a_sigterm_after_the_output_is_made_is_an_interruption_that_leaves_a_failure_record(self):
        terminated, interrupted = self.guarded()
        seen = {}

        def terminate(*arguments):
            seen.update(term=signal.getsignal(signal.SIGTERM), interrupt=signal.getsignal(signal.SIGINT))
            signal.raise_signal(signal.SIGTERM)

        with mock.patch.object(mtp, "export_assets", side_effect=terminate), self.assertRaises(KeyboardInterrupt):
            self.export()
        self.assertEqual(self.names(), ["failure.json"])
        self.assertEqual(read_json(self.output / "failure.json"), {"schema": mtp.EXPORT_WEIGHTS,
                                                                   "failure": "KeyboardInterrupt"})
        self.assertIsNot(seen["term"], terminated)
        self.assertIs(seen["interrupt"], interrupted)

    def test_a_sigterm_while_the_inputs_are_checked_is_an_interruption_and_makes_no_output(self):
        terminated, interrupted = self.guarded()

        def terminate(*arguments):
            signal.raise_signal(signal.SIGTERM)

        with mock.patch.object(mtp, "table_shard", side_effect=terminate), self.assertRaises(KeyboardInterrupt):
            self.export()
        self.assertFalse(self.output.exists())
        self.assertEqual((signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)), (terminated, interrupted))

    def test_the_handlers_that_were_set_before_are_set_again_after_a_success_a_failure_and_a_refusal(self):
        terminated, interrupted = self.guarded()
        with mock.patch.object(mtp, "export_assets", side_effect=fake()):
            self.export(output=self.folder / "succeeded")
        self.assertEqual((signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)), (terminated, interrupted))
        with mock.patch.object(mtp, "export_assets", side_effect=RuntimeError("a message")), self.assertRaises(RuntimeError):
            self.export(output=self.folder / "failed")
        self.assertEqual((signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)), (terminated, interrupted))
        with mock.patch.object(mtp, "export_assets", side_effect=fake()), self.assertRaises(FileExistsError):
            self.export(output=self.folder / "succeeded")
        self.assertEqual((signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)), (terminated, interrupted))

    def test_an_output_that_another_process_made_after_the_check_is_refused_without_a_file_written_into_it(self):
        real, fired = Path.mkdir, []

        def raced(path, *arguments, **options):
            if path == self.output and not fired:
                fired.append(path)
                real(path)
                (path / "theirs").write_text("x")
            return real(path, *arguments, **options)

        with mock.patch.object(Path, "mkdir", raced), mock.patch.object(mtp, "export_assets") as work, \
                self.assertRaises(FileExistsError):
            self.export()
        work.assert_not_called()
        self.assertEqual(self.names(), ["theirs"])

    def test_the_record_is_the_last_file_written_and_a_failed_inspection_is_recorded_with_passed_false(self):
        real = mtp.write_json
        for passed in (True, False):
            output, writes = self.folder / str(passed), []

            def logged(path, value, output=output, writes=writes):
                writes.append((Path(path).name, sorted(item.name for item in output.iterdir())))
                real(path, value)

            with self.subTest(passed=passed), mock.patch.object(mtp, "write_json", logged), \
                    mock.patch.object(mtp, "export_assets", side_effect=fake(passed)):
                record = self.export(output=output)
            self.assertEqual(writes, [("export.json", ["distribution.aimodel", "verification.aimodel"])])
            self.assertEqual(read_json(output / "export.json"), record)
            self.assertEqual(record["passed"], passed)
            self.assertEqual(self.names(output), ["distribution.aimodel", "export.json", "verification.aimodel"])

    def test_the_command_passes_its_arguments_prints_the_sizes_and_exits_with_one_unless_the_inspection_passed(self):
        for passed in (True, False):
            record = {"assets": {"verification": {"bytes": 11}, "distribution": {"bytes": 22}}, "passed": passed}
            with self.subTest(passed=passed), mock.patch.object(mtp, "export_weights", return_value=record) as export, \
                    mock.patch("sys.argv", ["mtp", "export-weights", "--model", "a.json", "--assistant", "b.json",
                                            "--snapshot", "target", "--assistant-snapshot", "assistant", "--steps", "7",
                                            "--context", "16896", "--output", "out"]), \
                    mock.patch("sys.stdout", new_callable=io.StringIO) as printed:
                code = mtp.main()
            export.assert_called_once_with(Path("a.json"), Path("b.json"), Path("target"), Path("assistant"), 7, 16896,
                                           Path("out"))
            self.assertEqual(json.loads(printed.getvalue()), {"bytes": {"verification": 11, "distribution": 22},
                                                              "passed": passed})
            self.assertEqual(code, 0 if passed else 1)


if __name__ == "__main__":
    unittest.main()
