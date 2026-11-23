import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

from Tools.Conversion.adapters.gemma4_31b import anchor, layer1_anchor, layers, reference
from Tools.Conversion.adapters.gemma4_31b.prepare import load_contract
from Tools.Conversion.adapters.pinned import sha256
from Tools.Conversion.storage import read_json, write_json
from Tools.Conversion.tests.test_gemma4_31b_anchor import (GEMMA4, MEMORY, OBSERVED, RECIPE, VOCABULARY, WINDOW,
                                                            ReadyMonitor, tiny_model)
from Tools.Conversion.tests.test_gemma4_31b_layers import TEXT

EVALUATION = Path(__file__).resolve().parents[2] / "LLMEvaluation/contract.json"
LENGTHS = ((5, 1), (9, 3), (23, 10), (40, 30))


def tokens_sha256(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def forced_record(name, inputs, generated):
    tokens = inputs + generated[:-1]
    return {"id": name, "input_sha256": tokens_sha256(inputs), "tokens": tokens, "tokens_sha256": tokens_sha256(tokens),
            "start": len(inputs) - 1, "rows": len(generated), "input_tokens": len(inputs), "targets": generated}


def made(vocabulary, lengths=LENGTHS, seed=7):
    rng = random.Random(seed)
    return [forced_record(f"ja-choice-{index:04d}", [2] + [rng.randrange(3, vocabulary) for _ in range(inputs - 1)],
                          [rng.randrange(3, vocabulary) for _ in range(rows)])
            for index, (inputs, rows) in enumerate(lengths, 1)]


def identity():
    return {"version": json.loads(EVALUATION.read_text())["version"], "sha256": sha256(EVALUATION)}


def write_forced(folder, records, summary=None):
    folder = Path(folder)
    folder.mkdir(parents=True)
    content = b"".join(json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n" for record in records)
    (folder / "forced.jsonl").write_bytes(content)
    value = {"contract": identity(), "set": "primary", "bands": {}, "categories": {},
             "forced": {"count": len(records), "sha256": hashlib.sha256(content).hexdigest()}}
    (folder / "forced.json").write_text(json.dumps(value if summary is None else summary(value)))
    return folder


def rounded_copy(model):
    plain = copy.deepcopy(model)
    with torch.no_grad():
        for name, tensor in [*plain.named_parameters(), *plain.named_buffers()]:
            if tensor.is_floating_point() and "rotary_emb" not in name:
                tensor.data = tensor.data.to(torch.float16).float()
    return plain.float()


def plain_rows(model, record):
    with torch.no_grad():
        values, _ = reference.forced(model, record["tokens"], [], "cpu")
    return values[record["start"]:record["start"] + record["rows"]]


class ForcedInputTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)

    def refused(self, label, records, summary=None, text="forced input is refused"):
        path = write_forced(self.folder / label, records, summary)
        with self.subTest(case=label), self.assertRaisesRegex(ValueError, text):
            layer1_anchor.forced(path, TEXT["vocab_size"])

    def test_the_forced_records_and_the_sha256_of_forced_jsonl_are_read_in_their_order(self):
        records = made(TEXT["vocab_size"])
        path = write_forced(self.folder / "forced", records)
        hashes, found = layer1_anchor.forced(path, TEXT["vocab_size"])
        self.assertEqual(found, records)
        self.assertEqual(hashes, {"sha256": sha256(path / "forced.jsonl"), "summary_sha256": sha256(path / "forced.json")})

    def test_a_forced_input_whose_summary_is_of_another_forced_input_contract_or_set_is_refused(self):
        records = made(TEXT["vocab_size"])
        changes = {"another forced.jsonl": lambda value: {**value, "forced": {**value["forced"], "sha256": "0" * 64}},
                   "another count": lambda value: {**value, "forced": {**value["forced"], "count": len(records) + 1}},
                   "another contract version": lambda value: {**value, "contract": {**value["contract"], "version": "x"}},
                   "another contract file": lambda value: {**value, "contract": {**value["contract"], "sha256": "0" * 64}},
                   "another set": lambda value: {**value, "set": "reserve"}, "no summary": lambda value: [value]}
        for label, change in changes.items():
            self.refused(label, records, change, "forced.json is not the summary")
        self.refused("a count of true", records[:1], lambda value: {**value, "forced": {**value["forced"], "count": True}},
                     "forced.json is not the summary")
        path, fewer = write_forced(self.folder / "replaced", records), write_forced(self.folder / "fewer", records[:3])
        (path / "forced.jsonl").write_bytes((fewer / "forced.jsonl").read_bytes())
        with self.assertRaisesRegex(ValueError, "forced.json is not the summary"):
            layer1_anchor.forced(path, TEXT["vocab_size"])

    def test_records_of_other_types_or_with_fields_that_disagree_are_refused(self):
        vocabulary = TEXT["vocab_size"]

        def tokens(record, value):
            record.update(tokens=value, tokens_sha256=tokens_sha256(value))

        changes = {"a start of true": lambda record: record.update(start=True),
                   "rows as a float": lambda record: record.update(rows=float(record["rows"])),
                   "a token outside the vocabulary": lambda record: tokens(record, record["tokens"][:-1] + [vocabulary]),
                   "a negative token": lambda record: tokens(record, [-1] + record["tokens"][1:]),
                   "a token of true": lambda record: tokens(record, [True] + record["tokens"][1:]),
                   "no targets": lambda record: record.pop("targets"),
                   "an id with a path": lambda record: record.update(id="../ja-choice-0001"),
                   "an id in upper case": lambda record: record.update(id="JA-choice-0001"),
                   "the tokens_sha256 of other tokens": lambda record: record.update(
                       tokens_sha256=tokens_sha256(record["tokens"][:-1])),
                   "rows past the last token": lambda record: record.update(rows=record["rows"] + 1),
                   "a start one row early": lambda record: record.update(start=record["start"] - 1),
                   "no scored row": lambda record: record.update(start=len(record["tokens"]), rows=0)}
        for label, change in changes.items():
            records = made(vocabulary)
            change(records[2])
            self.refused(label, records)
        records = made(vocabulary)
        self.refused("a repeated item", records + records[:1])
        self.refused("no record", [])
        self.refused("not an object", [records[0], [1, 2]])
        for label, line in (("not json", b"{"), ("a repeated key", b'{"id": "a", "id": "b"}'), ("nan", b'{"id": NaN}')):
            path = write_forced(self.folder / label, records)
            (path / "forced.jsonl").write_bytes(line + b"\n")
            with self.subTest(case=label), self.assertRaises(ValueError):
                layer1_anchor.forced(path, vocabulary)

    def test_the_chosen_items_are_distinct_items_of_the_forced_input_in_the_given_order(self):
        records = made(TEXT["vocab_size"])
        self.assertEqual(layer1_anchor.selected(records, None), records)
        self.assertEqual(layer1_anchor.selected(records, "ja-choice-0003,ja-choice-0001"), [records[2], records[0]])
        for names in ("ja-choice-0001,ja-choice-0001", "ja-choice-0009", "", "ja-choice-0001,"):
            with self.subTest(items=names), self.assertRaisesRegex(ValueError, "distinct items"):
                layer1_anchor.selected(records, names)


@unittest.skipIf(not GEMMA4 or np is None, "Transformers with Gemma 4, PyTorch and NumPy are required for the model checks")
class ScoredRowsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            cls.model = tiny_model(folder)

    def test_the_scored_rows_with_float16_rounded_weights_match_the_plain_float32_forward_of_the_rounded_weights(self):
        model, plain = copy.deepcopy(self.model), rounded_copy(self.model)
        for record in made(VOCABULARY):
            with self.subTest(tokens=len(record["tokens"])), torch.no_grad():
                found = layer1_anchor.scored(model, record, layer1_anchor.rounded(model), "cpu", 5)
                self.assertEqual((found.dtype.name, found.shape), ("float32", (record["rows"], VOCABULARY)))
                self.assertLessEqual(float(np.abs(found - plain_rows(plain, record)).max()), anchor.LIMIT)
        self.assertEqual({str(parameter.dtype) for parameter in model.parameters()}, {"torch.bfloat16"})

    def test_the_weights_are_the_resident_values_rounded_to_float16_with_the_bf16_embedding_scale(self):
        model = copy.deepcopy(self.model)
        name = "model.language_model.layers.0.mlp.down_proj.weight"
        weight = model.get_parameter(name)
        with torch.no_grad():
            weight.view(-1)[:2] = torch.tensor([3e-8, 1e-6]).to(torch.bfloat16)
        weights = layer1_anchor.rounded(model)
        self.assertTrue(torch.equal(weights(name), weight.to(torch.float16).float()))
        self.assertFalse(torch.equal(weights(name), weight.float()))
        self.assertEqual(float(weights(anchor.SCALE)), 6.9375)


def tiny_contract(folder):
    contract = read_json(RECIPE / "model.json")
    contract["config"].update(num_hidden_layers=6, vocab_size=VOCABULARY, sliding_window=WINDOW)
    path = Path(folder) / "model.json"
    write_json(path, contract)
    return path


@unittest.skipIf(not GEMMA4 or np is None, "Transformers with Gemma 4, PyTorch and NumPy are required for the runs")
class AnchorRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            cls.model = tiny_model(folder)

    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        self.contract = tiny_contract(self.folder)
        prepare = self.folder / "prepare"
        prepare.mkdir()
        write_json(prepare / "ready.json", {"ready": True, "contract_sha256": sha256(self.contract),
                                            "environment": {"python": "3.11.9", "packages": {}, "os_build": "26A1"}})
        self.records = made(VOCABULARY)
        self.forced = write_forced(self.folder / "forced", self.records)

    def main(self, command, *extra, loader=None, monitor=None, original=None, verify=None, output="out"):
        argv = ["layer1_anchor", command, "--model", str(self.contract), "--snapshot", str(self.folder / "snapshot"),
                "--prepare", str(self.folder / "prepare"), "--forced", str(self.forced), "--layers", "6",
                "--output", str(self.folder / output), *extra]
        loaded = lambda *arguments: (copy.deepcopy(self.model), {"after_load": dict(MEMORY)}, True)
        observed = lambda path, volumes, patterns, on_abort: ReadyMonitor(on_abort, monitor or OBSERVED)
        on_cpu = lambda path: {**load_contract(path), "reference": {**load_contract(path)["reference"], "device": "cpu"}}
        with mock.patch.object(sys, "argv", argv), mock.patch.object(reference, "ReferenceMonitor", observed), \
                mock.patch.object(layer1_anchor, "load_contract", side_effect=on_cpu), \
                mock.patch.object(reference, "memory", return_value=dict(MEMORY)), \
                mock.patch.object(reference, "verify_inputs", side_effect=verify, return_value={}) as verified, \
                mock.patch.object(layer1_anchor, "model_for", side_effect=loader or loaded) as model_for, \
                mock.patch.object(layers, "original_model", side_effect=original), \
                mock.patch.object(layer1_anchor, "swap_used", return_value=5), \
                contextlib.redirect_stdout(io.StringIO()):
            code = layer1_anchor.main()
        return code, verified, model_for

    def test_a_run_writes_the_float32_scored_rows_of_every_item_with_its_record_and_the_handoff(self):
        code, _, _ = self.main("run", "--weights", "float16")
        out, plain = self.folder / "out", rounded_copy(self.model)
        handoff, record = read_json(out / "logits.json"), read_json(out / "run.json")
        self.assertEqual((code, record["passed"], record["error"]), (0, True, None))
        self.assertEqual(record["checks"], dict.fromkeys(("loading_clean", "shapes", "finite", "items", "monitor_clean",
                                                          "resources_recorded"), True))
        self.assertEqual({key: handoff[key] for key in ("schema", "role", "forced_sha256")},
                         {"schema": layer1_anchor.LOGITS, "role": "anchor", "forced_sha256": sha256(self.forced / "forced.jsonl")})
        self.assertIn("the first 6 layers", handoff["weights"])
        self.assertIn("rounded to float16", handoff["weights"])
        self.assertEqual([entry["id"] for entry in handoff["items"]], [item["id"] for item in self.records])
        for entry, item in zip(handoff["items"], self.records):
            with self.subTest(item=entry["id"]):
                data = (out / entry["file"]).read_bytes()
                self.assertEqual({key: entry[key] for key in ("file", "dtype", "shape", "sha256", "tokens_sha256")},
                                 {"file": f"{item['id']}.f32", "dtype": "float32", "shape": [item["rows"], VOCABULARY],
                                  "sha256": hashlib.sha256(data).hexdigest(), "tokens_sha256": item["tokens_sha256"]})
                values = np.frombuffer(data, dtype="<f4").reshape(entry["shape"])
                self.assertLessEqual(float(np.abs(values - plain_rows(plain, item)).max()), anchor.LIMIT)
        self.assertEqual(record["compute"], {"weights": "float16", "compute": "float32", "attention_rows": anchor.ROWS,
                                             "embed_scale": 6.9375})
        self.assertEqual(record["forced"]["items"], [item["id"] for item in self.records])
        self.assertEqual({item["id"]: item["swap_used_bytes"] for item in record["items"]},
                         {item["id"]: {"before": 5, "after": 5} for item in self.records})

    def test_the_chosen_items_run_in_their_order(self):
        code, _, _ = self.main("run", "--weights", "float16", "--items", "ja-choice-0004,ja-choice-0002")
        handoff = read_json(self.folder / "out/logits.json")
        self.assertEqual((code, [entry["id"] for entry in handoff["items"]]), (0, ["ja-choice-0004", "ja-choice-0002"]))
        self.assertEqual(sorted(path.name for path in (self.folder / "out").glob("*.f32")),
                         ["ja-choice-0002.f32", "ja-choice-0004.f32"])

    def test_a_failure_after_the_folder_is_made_is_recorded_and_no_handoff_is_written(self):
        def broken(*arguments):
            raise RuntimeError("no device")

        unclean = {**OBSERVED, "unreadable_samples": 1}
        for label, options in (("load", {"loader": broken}), ("monitor", {"monitor": unclean})):
            with self.subTest(case=label):
                code, _, _ = self.main("run", "--weights", "float16", output=label, **options)
                record = read_json(self.folder / label / "run.json")
                self.assertEqual((code, record["passed"]), (1, False))
                self.assertFalse((self.folder / label / "logits.json").exists())
        self.assertEqual(read_json(self.folder / "load/run.json")["error"], "RuntimeError")
        self.assertFalse(read_json(self.folder / "monitor/run.json")["checks"]["monitor_clean"])

    def test_inputs_that_do_not_fit_are_refused_before_the_folder_is_made_or_the_model_is_loaded(self):
        def summary(value):
            return {**value, "forced": {**value["forced"], "sha256": "0" * 64}}

        def stale(path, contract):
            raise ValueError("A file to load differs from the contract")

        cases = {"layers": (("--layers", "5"), ValueError), "items": (("--items", "ja-choice-0009"), ValueError)}
        for label, (extra, error) in cases.items():
            with self.subTest(case=label), self.assertRaises(error):
                self.main("run", "--weights", "float16", *extra, output=label)
            self.assertFalse((self.folder / label).exists())
        self.forced = write_forced(self.folder / "other", self.records, summary)
        with self.assertRaisesRegex(ValueError, "forced.json is not the summary"):
            self.main("run", "--weights", "float16", output="other-forced")
        self.forced = self.folder / "forced"
        with self.assertRaisesRegex(ValueError, "differs"):
            self.main("agreement", "--items", "ja-choice-0001", verify=stale, output="snapshot")
        (self.folder / "taken").mkdir()
        with self.assertRaises(FileExistsError):
            self.main("run", "--weights", "float16", output="taken")
        write_json(self.folder / "prepare/ready.json", {"ready": True, "contract_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "prepare"):
            self.main("run", "--weights", "float16", output="unprepared")
        self.assertEqual(sorted(path.name for path in self.folder.iterdir()),
                         ["forced", "model.json", "other", "prepare", "taken"])

    def test_the_agreement_passes_only_when_the_original_gives_the_same_rows_with_an_item_beyond_the_window(self):
        plain, changed = rounded_copy(self.model), rounded_copy(self.model)
        changed.model.language_model.norm.weight.data.mul_(1.01)
        kinds = layers.check_layers(load_contract(self.contract)["config"], 6)
        results = {}
        for label, model, items in (("same", plain, "ja-choice-0001,ja-choice-0003"),
                                    ("changed", changed, "ja-choice-0001,ja-choice-0003"),
                                    ("short", plain, "ja-choice-0001"), ("long", plain, "ja-choice-0003")):
            code, _, _ = self.main("agreement", "--items", items, output=label,
                                   original=lambda *arguments, model=model: (model, 6.9375, kinds))
            record = read_json(self.folder / label / "run.json")
            results[label] = (code, record["checks"]["agreement"], record["checks"]["beyond_window"])
            self.assertFalse((self.folder / label / "logits.json").exists())
        self.assertEqual(results, {"same": (0, True, True), "changed": (1, False, True), "short": (1, True, False),
                                   "long": (0, True, True)})
        record = read_json(self.folder / "same/run.json")
        self.assertEqual([item["shapes"] for item in record["items"]], [[[1, VOCABULARY]] * 2, [[10, VOCABULARY]] * 2])
        self.assertLess(max(item["max_abs_difference"] for item in record["items"]), anchor.LIMIT)


if __name__ == "__main__":
    unittest.main()
