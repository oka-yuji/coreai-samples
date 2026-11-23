import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
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

from Tools.Conversion.adapters.gemma4_31b import mtp
from Tools.Conversion.adapters.gemma4_31b.assistant import SYNTHETIC, assistant_config, checkpoint_shapes
from Tools.Conversion.adapters.gemma4_31b.prepare import TOKENIZER_FILES, load_contract
from Tools.Conversion.adapters.pinned import sha256
from Tools.Conversion.storage import read_json, write_json

RECIPE = Path(__file__).resolve().parents[1] / "recipes/gemma4_31b"
SLIDING, FULL = "sliding_attention", "full_attention"
CONFIG = {"num_hidden_layers": 12, "layer_pattern": [SLIDING] * 5 + [FULL], "hidden_size": 48, "intermediate_size": 96,
          "vocab_size": 301, "sliding_window": 1024, "num_key_value_heads": 2, "num_global_key_value_heads": 1,
          "head_dim": 16, "global_head_dim": 32, "attention_k_eq_v": True, "final_logit_softcapping": 30.0,
          "tie_word_embeddings": True, "bos_token_id": 2, "eos_token_id": [1, 106, 50]}
ITEMS = (("alpha-0010", 30, 4, 106), ("alpha-0002", 20, 9, 106), ("beta-0001", 1100, 6, None))
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


def assistant_state():
    shapes = checkpoint_shapes(assistant_config(SYNTHETIC["assistant"], SYNTHETIC["target"]))
    return {name: torch.zeros(shape) for name, shape in shapes.items()}


def tensor_reads():
    import safetensors

    events, real = [], safetensors.safe_open

    class Watched:
        def __init__(self, path, *arguments, **options):
            status = os.stat(path)
            self.name, self.opened = (status.st_dev, status.st_ino), real(path, *arguments, **options)

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
        for label, (model_class, original) in {"a zero original": ([1.0], [0.0]), "a NaN": ([float("nan")], [1.0]),
                                               "an infinity": ([1.0], [float("inf")])}.items():
            with self.subTest(case=label):
                self.assertIsNone(mtp.ratio(model_class, original))

    def test_a_step_count_that_is_not_a_whole_number_from_one_is_refused(self):
        self.assertEqual(mtp.check_steps(7), 7)
        for steps in (0, -1, True, 1.0, None, "7"):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "whole number"):
                mtp.check_steps(steps)

    def test_the_layout_names_the_last_layer_of_each_kind_and_the_window_of_the_assistant(self):
        self.assertEqual(mtp.layout(load_contract(RECIPE / "model.json")["config"]),
                         {"sliding": 58, "full": 59, "window": 1025})
        self.assertEqual(mtp.layout(CONFIG), {"sliding": 10, "full": 11, "window": 1025})

    def test_the_assistant_contract_pins_the_two_files_that_are_read(self):
        contract = mtp.load_assistant(RECIPE / "assistant.json")
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
                   "an uppercase hash": {**good, "files": {"config.json": {"bytes": 1, "sha256": "A" * 64},
                                                          "model.safetensors": entry}}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "assistant.json"
            write_json(path, good)
            self.assertEqual(mtp.load_assistant(path), good)
            for label, contract in changes.items():
                write_json(path, contract)
                with self.subTest(change=label), self.assertRaisesRegex(ValueError, "assistant contract"):
                    mtp.load_assistant(path)

    def test_the_first_item_of_each_category_follows_the_numbers_of_the_ids(self):
        record = {"counts": {"a": 3, "b": 1}, "items": [{"id": "a-10", "category": "a"}, {"id": "a-9", "category": "a"},
                                                        {"id": "b-0002", "category": "b"}, {"id": "a-0011", "category": "a"}]}
        self.assertEqual([entry["id"] for entry in mtp.first_per_category(record)], ["a-9", "b-0002"])


@unittest.skipIf(np is None or torch is None, "NumPy, PyTorch and safetensors are required for the records")
class RecordTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        self.records, self.inputs, self.outputs = write_records(self.folder)
        self.contract = {"config": dict(CONFIG)}

    def items(self, choose=mtp.first_per_category):
        return mtp.items_of(self.contract, self.records, self.inputs, self.outputs, choose)[1]

    def test_the_items_take_the_tokens_and_the_feature_offsets_of_the_records(self):
        items = self.items()
        self.assertEqual([item["id"] for item in items], ["alpha-0002", "beta-0001"])
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
            (features / "alpha-0002.safetensors").unlink()
            (features / "alpha-0002.safetensors").symlink_to(features / "alpha-0010.safetensors")

        cases = {
            "inputs changed after the subset": lambda: self.inputs.write_text(self.inputs.read_text() + "\n"),
            "a stop token inside the output": lambda: outputs_with(lambda rows: rows[1]["output_ids"].__setitem__(0, 50)),
            "an output with another length": lambda: outputs_with(lambda rows: rows[1]["output_ids"].append(7)),
            "an output from another input": lambda: outputs_with(lambda rows: rows[1].update(input_tokens=21)),
            "features in float32": as_float32,
            "features of another shape": lambda: save_file({key: torch.zeros(2, 2, dtype=torch.bfloat16) for key in (
                "postnorm", f"k{place['sliding']}")}, str(features / "alpha-0002.safetensors")),
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

    def test_records_that_are_named_pipes_are_refused_without_waiting_for_a_writer(self):
        item = self.items()[0]
        paths = {"subset.json": (self.records / "subset.json", lambda path: mtp.subset_record(path)),
                 "a drafter record": (self.folder / "drafter.json", lambda path: mtp.draft_rows(path, item)),
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
                   "an extra field": lambda record: record.update(note="x")}
        for label, change in changes.items():
            record = json.loads(json.dumps(subset))
            change(record)
            write_json(self.records / "subset.json", record)
            with self.subTest(change=label), self.assertRaisesRegex(ValueError, "subset.json"):
                mtp.subset_record(self.records / "subset.json")

    def test_the_drafter_records_and_the_rounds_of_691_are_read_only_in_their_form(self):
        item = self.items()[0]
        rows = [{"a": 20 + j, "j": j, "fixed": {"accepted": 0, "tokens": [0, 0], "margins": [0.5, 0.0]}} for j in range(9)]
        rows[0]["fixed"].update(accepted=2, tokens=item["teacher"][1:3])
        path = self.folder / "drafter.json"
        write_json(path, {"id": item["id"], "P": 20, "T": 10, "mtp": rows})
        self.assertEqual([row["accepted"] for row in mtp.draft_rows(path, item).values()], [2] + [0] * 8)
        changes = {"a missing anchor": lambda record: record["mtp"].pop(),
                   "an anchor off its output index": lambda record: record["mtp"][0].update(a=21),
                   "a boolean accepted count": lambda record: record["mtp"][0]["fixed"].update(accepted=True),
                   "margins of another length": lambda record: record["mtp"][0]["fixed"]["margins"].pop(),
                   "a negative margin": lambda record: record["mtp"][0]["fixed"]["margins"].__setitem__(0, -0.5),
                   "another input length": lambda record: record.update(P=21)}
        for label, change in changes.items():
            record = {"id": item["id"], "P": 20, "T": 10, "mtp": json.loads(json.dumps(rows))}
            change(record)
            write_json(path, record)
            with self.subTest(change=label), self.assertRaises(ValueError):
                mtp.draft_rows(path, item)
        skipped = json.loads(json.dumps(rows))
        skipped[1]["fixed"]["accepted"] = 1
        self.assertEqual(mtp.simulate([row["fixed"]["accepted"] for row in skipped], 10, 7),
                         mtp.simulate([row["fixed"]["accepted"] for row in rows], 10, 7))
        write_json(path, {"id": item["id"], "P": 20, "T": 10, "mtp": skipped})
        with self.assertRaisesRegex(ValueError, "accepted counts that its drafts and the teacher do not give at 1 anchors"):
            mtp.draft_rows(path, item)
        level = {"alpha": {"rounds": 2, "tokens": 5, "tokens_per_round": 2.5},
                 "pooled": {"rounds": 2, "tokens": 5, "tokens_per_round": 2.5}}
        write_json(path, {"per_round_on_teacher_path": {"mtp_fixed": {"7": level}}})
        self.assertEqual(mtp.published(path, 7, ["alpha"]), level)
        for steps, categories in ((6, ["alpha"]), (7, ["alpha", "beta"])):
            with self.subTest(steps=steps, categories=categories), self.assertRaisesRegex(ValueError, "trajectories"):
                mtp.published(path, steps, categories)


def write_run(folder, kind, rows, arrays, contract, steps=3, change=None):
    folder.mkdir(parents=True)
    files = {}
    for name, array in arrays.items():
        np.save(folder / f"{name}.npy", array)
        files[name] = {"bytes": (folder / f"{name}.npy").stat().st_size, "sha256": sha256(folder / f"{name}.npy")}
    record = {"schema": mtp.SCHEMA, "implementation": kind, "steps": steps, "stops": [1, 106, 50], "window": 1025,
              "layers": {"sliding": 10, "full": 11}, "model": {"contract_sha256": sha256(contract)},
              "assistant": {"contract_sha256": "0" * 64}, "records": {}, "weights": mtp.WEIGHTS, "embed_scale": 6.9375,
              "rows": [dict(row) for row in rows], "arrays": files}
    if change:
        change(record)
    write_json(folder / "record.json", record)


@unittest.skipIf(np is None, "NumPy is required for the comparison")
class CompareTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        self.contract = write_contract(self.folder, {name: {"bytes": 1, "sha256": "0" * 64} for name in TOKENIZER_FILES})
        generator = np.random.default_rng(3)
        self.rows = [{"id": "alpha-0002", "category": "alpha", "anchor": 20 + index, "output_index": index,
                      "window": 20 + index} for index in range(3)]
        logits = generator.normal(size=(3, CONFIG["vocab_size"])).astype(np.float32)
        projected = generator.normal(size=(3, CONFIG["hidden_size"])).astype(np.float32)
        drafts = np.array([[4, 5, 6], [7, 106, 8], [9, 9, 50]], dtype=np.int32)
        self.arrays = {"original": {"logits": logits, "projected": projected,
                                    "drafts": np.array([[4, 5, 6], [7, 106, -1], [9, 9, 50]], dtype=np.int32)},
                       "class": {"logits": logits.copy(), "projected": projected.copy(), "drafts": drafts,
                                 "repeated": drafts.copy()}}

    def pair(self, label, change=None, record=None, both=None):
        arrays = {kind: {name: array.copy() for name, array in found.items()} for kind, found in self.arrays.items()}
        if change:
            change(arrays)
        for kind in mtp.IMPLEMENTATIONS:
            write_run(self.folder / label / kind, kind, self.rows, arrays[kind], self.contract,
                      change=both or (record if kind == "class" else None))
        return mtp.compare(self.folder / label / "original", self.folder / label / "class", self.contract)

    def test_runs_within_the_criteria_pass_and_record_the_differences_from_the_original_drafts(self):
        result = self.pair("same")
        self.assertTrue(result["passed"], result["problems"])
        self.assertEqual(result["criteria"], {"max_abs_logit": 1e-3, "projection_ratio": 1e-3})
        self.assertEqual(result["summary"], {"rows": 3, "items": 1, "max_abs_logit": 0.0, "largest_projection_ratio": 0.0,
                                             "argmax_equal": 3, "unrolled_equals_repeated": 3, "cut_equals_original": 3})

        def drafts(arrays):
            arrays["original"]["drafts"][2] = [9, 8, 50]

        result = self.pair("other-drafts", drafts)
        self.assertTrue(result["passed"], result["problems"])
        self.assertEqual([(row["cut_equals_original"], row["first_step_unlike_original"]) for row in result["rows"]],
                         [(True, None), (True, None), (False, 1)])

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
                     logits=np.zeros((3, 300), np.float32)), None),
                 "drafts as int64": (lambda arrays: arrays["class"].update(
                     drafts=arrays["class"]["drafts"].astype(np.int64)), None),
                 "a missing row": (lambda arrays: arrays["original"].update(
                     projected=arrays["original"]["projected"][:2]), None),
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
        both = {"stop tokens": lambda record: record.update(stops=[1, 106]), "a window": lambda record: record.update(
                    window=1024), "an embedding scale": lambda record: record.update(embed_scale=6.928),
                "layers": lambda record: record.update(layers={"sliding": 9, "full": 11})}
        for label, record in both.items():
            result = self.pair(f"both-{label.replace(' ', '-')}", both=record)
            with self.subTest(case=f"{label} unlike the contract in both runs"):
                self.assertFalse(result["passed"])
                self.assertEqual(len(result["problems"]), 1)
                self.assertIn("than the contract", result["problems"][0])
        self.pair("files")
        for name in ("logits", "repeated"):
            np.save(self.folder / "files/class" / f"{name}.npy", self.arrays["class"][name] + 1)
            with self.subTest(case=f"{name} changed after the record"):
                result = mtp.compare(self.folder / "files/original", self.folder / "files/class", self.contract)
                self.assertFalse(result["passed"])
                self.assertIn(f"the class {name} differ from the size and SHA-256 in their record", result["problems"])
        self.pair("again")
        with self.subTest(case="a run compared with itself"):
            self.assertFalse(mtp.compare(self.folder / "again/original", self.folder / "again/original",
                                         self.contract)["passed"])
        (self.folder / "other").mkdir()
        other = write_contract(self.folder / "other", {name: {"bytes": 2, "sha256": "1" * 64} for name in TOKENIZER_FILES})
        with self.subTest(case="another contract"):
            result = mtp.compare(self.folder / "again/original", self.folder / "again/class", other)
            self.assertFalse(result["passed"])
            self.assertIn("the runs were made under another contract", result["problems"])
        (self.folder / "again/class/record.json").unlink()
        os.mkfifo(self.folder / "again/class/record.json")
        with self.subTest(case="a record that is a named pipe"):
            result = mtp.compare(self.folder / "again/original", self.folder / "again/class", self.contract)
            self.assertFalse(result["passed"])
            self.assertIn("the class run has no readable record.json (ValueError)", result["problems"])

    def test_compare_writes_only_its_result_as_json_and_exits_with_one_unless_it_passes(self):
        for label, change in {"passing": None, "failing": lambda arrays: arrays["class"]["repeated"].__setitem__(
                (0, 0), 3)}.items():
            result = self.pair(f"main-{label}", change)
            arguments = ["mtp", "compare", "--original", str(self.folder / f"main-{label}/original"), "--model-class",
                         str(self.folder / f"main-{label}/class"), "--model", str(self.contract)]
            with self.subTest(run=label), mock.patch("sys.argv", arguments), \
                    mock.patch("sys.stdout", new_callable=io.StringIO) as printed:
                code = mtp.main()
            self.assertEqual(json.loads(printed.getvalue()), result)
            self.assertEqual(code, 0 if label == "passing" else 1)


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
            for name, change in {"the shard of the table": lambda: (target / SHARD).write_bytes(b"0"),
                                 "the assistant weights": lambda: (assistant / "model.safetensors").write_bytes(b"0"),
                                 "the index as a link": lambda: (target / mtp.INDEX).unlink()
                                 or (target / mtp.INDEX).symlink_to(target / "tokenizer.json")}.items():
                saved = {path: path.read_bytes() for path in (target / SHARD, assistant / "model.safetensors")}
                change()
                with self.subTest(change=name), self.assertRaises(ValueError):
                    mtp.run("class", contract, assistant_contract, target, assistant, records, inputs, outputs, 7,
                            folder / "out")
                self.assertFalse((folder / "out").exists())
                for path, data in saved.items():
                    path.write_bytes(data)
                if (target / mtp.INDEX).is_symlink():
                    (target / mtp.INDEX).unlink()
                    write_json(target / mtp.INDEX, {"metadata": {}, "weight_map": {mtp.TABLE: SHARD}})

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
                          str(records / "tf/features/alpha-0002.safetensors"))

            unsupported, weights = "Unsupported Gemma 4 assistant", "model.safetensors differs from the expected"
            cases = {"ordered embeddings": (lambda document: document.update(use_ordered_embeddings=True), None, None,
                                            unsupported),
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
                     "features of another shape": (None, None, features, "alpha-0002.safetensors does not hold")}
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
            self.assertEqual(events, [])
            with watched():
                mtp.table(target / SHARD, CONFIG)
            self.assertEqual(events, [(inode(target / SHARD), mtp.TABLE)])


if __name__ == "__main__":
    unittest.main()
