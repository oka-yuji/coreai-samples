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

from Tools.Conversion.adapters.gemma4_31b import anchor, export, layer1_anchor, layer1_candidate, layers, native, reference
from Tools.Conversion.adapters.gemma4_31b.prepare import TOKENIZER_FILES, load_contract
from Tools.Conversion.adapters.pinned import sha256
from Tools.Conversion.storage import file_hash, read_json, write_json
from Tools.Conversion.tests.test_gemma4_31b_anchor import (GEMMA4, MEMORY, OBSERVED, RECIPE, VOCABULARY, WINDOW,
                                                            ReadyMonitor, tiny_model)
from Tools.Conversion.tests.test_gemma4_31b_layers import (TEXT, float16_restore, quantized_tensors, write_contract,
                                                            write_export, write_snapshot)
from Tools.Conversion.tests.test_gemma4_31b_native import MAIN_HASH, model_record
from Tools.LLMEvaluation.tests.test_layer1 import bounded, inode, release, replaced, touched, watched

EVALUATION = Path(__file__).resolve().parents[2] / "LLMEvaluation/contract.json"
LENGTHS = ((5, 1), (9, 3), (23, 10), (40, 30))
FAKE = """#!{python}
import hashlib, json, os, sys
from pathlib import Path

plan = json.loads(Path(__file__).with_suffix(".json").read_text())
arguments = dict(zip(sys.argv[2::2], sys.argv[3::2]))
ids = json.loads(Path(arguments["--tokens"]).read_text())
start, chunk = int(arguments["--start"]), int(arguments["--chunk"])
name = Path(arguments["--output"]).stem
change = plan["changes"].get(name, {{}})
with Path(plan["calls"]).open("a") as calls:
    calls.write(name + "\\n")
rows = len(ids) - start
if change.get("fifo"):
    os.mkfifo(arguments["--output"])
elif not change.get("no_output"):
    size = (rows + change.get("rows", 0)) * plan["vocabulary"] * 2
    Path(arguments["--output"]).write_bytes((name.encode() * size)[:size])
record = {{"record": "logits", "token_count": len(ids), "start": start, "chunk": chunk,
          "tokens_sha256": hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest(),
          "shape": [rows, plan["vocabulary"]], "dtype": "float16", "byte_order": "little",
          "main_calls": -(-rows // chunk), "main_hash": plan["main_hash"], "cached": True, "non_finite_count": 0,
          "load_s": 0.1, "prefix_s": 0.0, "scoring_s": 0.2, "specialize_s": 0.0, **plan["prefix"][name]}}
record.update(change.get("record", {{}}))
for key in change.get("drop", []):
    record.pop(key)
print(change.get("stdout", json.dumps(record)))
sys.exit(change.get("exit", 0))
"""


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
            record.update(tokens=value, tokens_sha256=tokens_sha256(value),
                          input_sha256=tokens_sha256(value[:record["input_tokens"]]))

        changes = {"a start of true": lambda record: record.update(start=True),
                   "rows as a float": lambda record: record.update(rows=float(record["rows"])),
                   "a token outside the vocabulary": lambda record: tokens(record, [2, vocabulary] + record["tokens"][2:]),
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

    def test_a_forced_file_that_is_not_a_regular_file_is_refused_without_waiting_or_reading(self):
        for name in ("forced.jsonl", "forced.json"):
            for kind, error in (("fifo", ValueError), ("folder", ValueError), ("missing", FileNotFoundError)):
                with self.subTest(file=name, kind=kind):
                    path = write_forced(Path(tempfile.mkdtemp(dir=self.folder)) / "forced", made(TEXT["vocab_size"]))
                    replaced(path / name, kind) if kind != "missing" else (path / name).unlink()
                    events, patched = watched()
                    with patched:
                        thread, outcome = bounded(lambda: layer1_anchor.forced(path, TEXT["vocab_size"]))
                        self.assertFalse(release(path / name, thread), "the run waited on the forced input")
                    self.assertIsInstance(outcome.get("error"), error)
                    if kind != "missing":
                        self.assertEqual(touched(events, inode(path / name)), [])
                        self.assertNotIn(str(path), str(outcome["error"]))

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

    def main(self, command, *extra, loader=None, monitor=None, original=None, verify=None, output="out", device="cpu"):
        argv = ["layer1_anchor", command, "--model", str(self.contract), "--snapshot", str(self.folder / "snapshot"),
                "--prepare", str(self.folder / "prepare"), "--forced", str(self.forced), "--layers", "6",
                "--output", str(self.folder / output), *extra]
        loaded = lambda *arguments: (copy.deepcopy(self.model), {"after_load": dict(MEMORY)}, True)
        observed = lambda path, volumes, patterns, on_abort: ReadyMonitor(on_abort, monitor or OBSERVED)
        placed = lambda path: {**load_contract(path), "reference": {**load_contract(path)["reference"], "device": device}}
        with mock.patch.object(sys, "argv", argv), mock.patch.object(reference, "ReferenceMonitor", observed), \
                mock.patch.object(layer1_anchor, "load_contract", side_effect=placed), \
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

        def unclean_loading(*arguments):
            return copy.deepcopy(self.model), {"after_load": dict(MEMORY)}, False

        unclean = {**OBSERVED, "unreadable_samples": 1}
        for label, options in (("load", {"loader": broken}), ("monitor", {"monitor": unclean}),
                               ("loading", {"loader": unclean_loading})):
            with self.subTest(case=label):
                code, _, _ = self.main("run", "--weights", "float16", output=label, **options)
                record = read_json(self.folder / label / "run.json")
                self.assertEqual((code, record["passed"]), (1, False))
                self.assertFalse((self.folder / label / "logits.json").exists())
        self.assertEqual(read_json(self.folder / "load/run.json")["error"], "RuntimeError")
        self.assertFalse(read_json(self.folder / "monitor/run.json")["checks"]["monitor_clean"])
        self.assertFalse(read_json(self.folder / "loading/run.json")["checks"]["loading_clean"])

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
        replaced(self.folder / "prepare/ready.json", "fifo")
        thread, outcome = bounded(lambda: self.main("run", "--weights", "float16", output="no-ready"))
        self.assertFalse(release(self.folder / "prepare/ready.json", thread), "the run waited on ready.json")
        self.assertEqual(str(outcome.get("error")), "ready.json is not a regular file")
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
                                   original=lambda *arguments, model=model: (copy.deepcopy(model), 6.9375, kinds))
            record = read_json(self.folder / label / "run.json")
            results[label] = (code, record["checks"]["agreement"], record["checks"]["beyond_window"])
            self.assertFalse((self.folder / label / "logits.json").exists())
        self.assertEqual(results, {"same": (0, True, True), "changed": (1, False, True), "short": (1, True, False),
                                   "long": (0, True, True)})
        record = read_json(self.folder / "same/run.json")
        self.assertEqual([item["shapes"] for item in record["items"]],
                         [dict.fromkeys(record["variants"], [rows, VOCABULARY]) for rows in (1, 10)])
        self.assertLess(max(item["max_abs_difference"] for item in record["items"]), anchor.LIMIT)
        self.assertEqual([sorted(item["recorded"]) for item in record["items"]],
                         [sorted(" - ".join(pair) for pair in layer1_anchor.RECORDED)] * 2)

    def test_the_rows_of_the_original_use_the_requested_attention_implementation(self):
        model, plain = rounded_copy(self.model), rounded_copy(self.model)
        model.set_attn_implementation("eager")
        for attention in ("sdpa", "eager"):
            with self.subTest(attention=attention):
                found = layer1_anchor.original_rows(model, self.records, "cpu", attention)
                self.assertEqual(model.model.language_model.config._attn_implementation, attention)
                for values, record in zip(found, self.records, strict=True):
                    self.assertLessEqual(float(np.abs(values - plain_rows(plain, record)).max()), anchor.LIMIT)

    def test_the_agreement_compares_the_cpu_anchor_with_the_sdpa_original_on_the_cpu_and_records_the_other_pairs(self):
        offsets = {("anchor", "mps"): 0.25, ("anchor", "cpu"): 0.0, ("original", "cpu", "sdpa"): 0.0,
                   ("original", "cpu", "eager"): 0.5, ("original", "mps", "sdpa"): 0.75}
        calls = []

        def rows(key, records):
            calls.append(key)
            return [np.full((item["rows"], VOCABULARY), 1.0 + offsets[key], dtype=np.float32) for item in records]

        anchored = lambda args, records, device: (rows(("anchor", device), records), {"after_load": dict(MEMORY)}, True)
        originals = lambda model, records, device, attention: rows(("original", device, attention), records)
        kinds = layers.check_layers(load_contract(self.contract)["config"], 6)
        with mock.patch.object(layer1_anchor, "anchor_rows", side_effect=anchored), \
                mock.patch.object(layer1_anchor, "original_rows", side_effect=originals):
            code, _, _ = self.main("agreement", "--items", "ja-choice-0001,ja-choice-0003", device="mps",
                                   original=lambda *arguments: (None, 6.9375, kinds))
        record = read_json(self.folder / "out/run.json")
        self.assertEqual((code, record["checks"]["agreement"]), (0, True))
        self.assertEqual(calls, [("anchor", "mps"), ("anchor", "cpu"), ("original", "cpu", "sdpa"),
                                 ("original", "cpu", "eager"), ("original", "mps", "sdpa")])
        self.assertEqual(record["compared"], ["anchor_cpu", "original_cpu"])
        self.assertEqual(record["variants"], {"anchor_cpu": {"device": "cpu"}, "anchor_run": {"device": "mps"},
                                              "original_cpu": {"device": "cpu", "attention": "sdpa"},
                                              "original_eager_cpu": {"device": "cpu", "attention": "eager"},
                                              "original_run": {"device": "mps", "attention": "sdpa"}})
        self.assertEqual([item["max_abs_difference"] for item in record["items"]], [0.0, 0.0])
        self.assertEqual([item["recorded"] for item in record["items"]],
                         [{"anchor_run - anchor_cpu": 0.25, "original_eager_cpu - original_cpu": 0.5,
                           "original_run - original_cpu": 0.75, "anchor_run - original_run": 0.5}] * 2)


class CandidateRunTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        self.contract = write_contract(self.folder, dict.fromkeys(TOKENIZER_FILES, {"bytes": 1, "sha256": "0" * 64}))
        self.vocabulary = TEXT["vocab_size"]
        self.records = made(self.vocabulary, [(index % 7 + 3, index % 5 + 1) for index in range(12)])
        self.forced = write_forced(self.folder / "forced", self.records)
        self.export = self.folder / "export"
        bundle = self.export / export.BUNDLE
        (bundle / "asset.aimodel").mkdir(parents=True)
        (bundle / "asset.aimodel/main.hash").write_bytes(bytes.fromhex(MAIN_HASH))
        (bundle / "metadata.json").write_text("{}")
        files = {path.relative_to(bundle).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                 for path in sorted(bundle.rglob("*")) if path.is_file()}
        self.exported = {"schema": export.SCHEMA, "passed": True, "problems": [], "layers": 6, "precision": "float16",
                         "context": 64, "layer_types": layers.check_layers(load_contract(self.contract)["config"], 6),
                         "functions": ["main", "prefill"], "model": model_record(self.contract),
                         "metadata": {"tap_layers": [1, 2, 3, 4, 5]},
                         "bundle": {"folder": export.BUNDLE, "name": "asset", "main_hash": MAIN_HASH, "files": files}}
        write_json(self.export / "export.json", self.exported)
        self.verifier, self.calls = self.folder / "verifier", self.folder / "calls.txt"

    def run_candidate(self, changes=None, output="out", main_hash=MAIN_HASH):
        prefix = {record["id"]: native.prefix_record(record["start"], native.prefill_chunk(), True)
                  for record in self.records}
        plan = {"changes": changes or {}, "calls": str(self.calls), "vocabulary": self.vocabulary, "main_hash": main_hash,
                "prefix": prefix}
        self.verifier.write_text(FAKE.format(python=sys.executable))
        self.verifier.chmod(0o755)
        self.verifier.with_suffix(".json").write_text(json.dumps(plan))
        with contextlib.redirect_stdout(io.StringIO()):
            return layer1_candidate.run(self.export, self.forced, self.verifier, self.folder / output, self.contract)

    def called(self):
        return self.calls.read_text().split() if self.calls.exists() else []

    def test_every_item_is_read_by_one_launch_and_the_handoff_lists_its_file(self):
        result = self.run_candidate()
        out = self.folder / "out"
        self.assertTrue(result["passed"], result["problems"])
        self.assertEqual(read_json(out / "run.json"), result)
        self.assertEqual(self.called(), [record["id"] for record in self.records])
        handoff = read_json(out / "logits.json")
        self.assertEqual({key: handoff[key] for key in ("schema", "role", "weights", "forced_sha256")},
                         {"schema": layer1_anchor.LOGITS, "role": "fp16", "weights": f"asset main.hash {MAIN_HASH}",
                          "forced_sha256": sha256(self.forced / "forced.jsonl")})
        self.assertEqual(len(handoff["items"]), len(self.records))
        for entry, record in zip(handoff["items"], self.records):
            with self.subTest(item=record["id"]):
                self.assertEqual(entry, {"id": record["id"], "file": f"{record['id']}.f16", "dtype": "float16",
                                         "shape": [record["rows"], self.vocabulary], "sha256": sha256(out / entry["file"]),
                                         "tokens_sha256": record["tokens_sha256"]})
                self.assertEqual(json.loads((out / f"{record['id']}.tokens.json").read_text()), record["tokens"])
        self.assertEqual({(entry["verifier"]["chunk"], entry["verifier"]["main_calls"] == entry["rows"])
                          for entry in result["items"]}, {(1, True)})
        with mock.patch.object(sys, "argv", ["layer1_candidate", "--export", "e", "--forced", "f", "--verifier", "v",
                                             "--output", "o"]), \
                mock.patch.object(layer1_candidate, "run", return_value={**result, "passed": False}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(layer1_candidate.main(), 1)

    def test_a_failing_item_is_recorded_as_a_problem_and_the_other_items_are_still_read(self):
        ids = [record["id"] for record in self.records]
        other = 2048 if native.prefill_chunk() == 4096 else 4096
        changes = {ids[0]: {"exit": 1}, ids[1]: {"record": {"tokens_sha256": "0" * 64}},
                   ids[2]: {"record": {"shape": [1, self.vocabulary]}}, ids[3]: {"rows": -1},
                   ids[4]: {"stdout": "not a record"}, ids[5]: {"drop": ["cached"]},
                   ids[6]: {"record": {"main_hash": "cd" * 32}}, ids[7]: {"record": {"prefix_chunk": other}},
                   ids[8]: {"no_output": True}, ids[9]: {"fifo": True}}
        thread, outcome = bounded(lambda: self.run_candidate(changes), 60)
        out = self.folder / "out"
        self.assertFalse(release(out / f"{ids[9]}.f16", thread), "the run waited on the logits file of a launch")
        result = outcome["value"]
        self.assertFalse(result["passed"])
        self.assertEqual(read_json(out / "run.json"), result)
        self.assertEqual(self.called(), ids)
        self.assertEqual([entry["id"] for entry in read_json(out / "logits.json")["items"]], ids[10:])
        failing = {entry["id"]: entry["problems"] for entry in result["items"] if entry["problems"]}
        self.assertEqual(set(failing), set(ids[:10]))
        self.assertEqual(failing[ids[0]], ["the verifier exited with 1"])
        self.assertEqual(failing[ids[4]], ["the verifier did not print one JSON object"])
        self.assertEqual(failing[ids[5]], ["the verifier record is malformed (cached)"])
        self.assertEqual(failing[ids[8]], ["the item could not be run (FileNotFoundError)"])
        self.assertEqual(failing[ids[3]], [f"the logits file is refused: {ids[3]}.f16 holds "
                                           f"{(self.records[3]['rows'] - 1) * self.vocabulary * 2} B, not "
                                           f"{self.records[3]['rows'] * self.vocabulary * 2} B"])
        self.assertEqual(failing[ids[9]], [f"the logits file is refused: {ids[9]}.f16 is not a regular file"])
        self.assertEqual(len(result["problems"]), sum(map(len, failing.values())))

    def test_an_export_or_forced_input_that_does_not_fit_is_refused_before_anything_is_written_or_launched(self):
        def summary(value):
            return {**value, "forced": {**value["forced"], "sha256": "0" * 64}}

        cases = {"not passed": {"passed": False}, "another precision": {"precision": "4bit"},
                 "a shorter context": {"context": 8}, "a context of true": {"context": True},
                 "another main hash": {"bundle": {**self.exported["bundle"], "main_hash": "cd" * 32}}}
        for label, change in cases.items():
            write_json(self.export / "export.json", {**self.exported, **change})
            with self.subTest(case=label), self.assertRaisesRegex(ValueError, "export is refused"):
                self.run_candidate(output=label)
            self.assertFalse((self.folder / label).exists())
        write_json(self.export / "export.json", self.exported)
        self.forced = write_forced(self.folder / "other", self.records, summary)
        with self.assertRaisesRegex(ValueError, "forced.json is not the summary"):
            self.run_candidate(output="other-forced")
        self.forced = self.folder / "forced"
        (self.folder / "taken").mkdir()
        with self.assertRaises(FileExistsError):
            self.run_candidate(output="taken")
        with self.assertRaises(FileNotFoundError):
            layer1_candidate.run(self.export, self.forced, self.folder / "absent", self.folder / "unlaunched", self.contract)
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            layer1_candidate.run(self.export, self.forced, self.folder, self.folder / "unlaunched", self.contract)
        self.assertFalse((self.folder / "other-forced").exists() or (self.folder / "unlaunched").exists())
        self.assertEqual(self.called(), [])

    @unittest.skipIf(not GEMMA4, "Transformers with Gemma 4 and PyTorch are required to write the quantized tensors")
    def test_an_8bit_export_gives_the_role_w8(self):
        self.export = write_export(self.folder / "export-8bit", self.contract,
                                   quantized_tensors({**TEXT, "hidden_size": 64}), context=64)
        result = self.run_candidate(main_hash=bytes(32).hex(), output="w8")
        self.assertTrue(result["passed"], result["problems"])
        self.assertEqual((result["role"], read_json(self.folder / "w8/logits.json")["role"]), ("w8", "w8"))

    def test_a_stopped_run_still_records_the_items_that_were_read(self):
        attempted = layer1_candidate.attempted
        calls = []

        def stopping(*arguments):
            calls.append(arguments[0]["id"])
            if len(calls) == 3:
                raise KeyboardInterrupt
            return attempted(*arguments)

        with mock.patch.object(layer1_candidate, "attempted", side_effect=stopping), self.assertRaises(KeyboardInterrupt):
            self.run_candidate()
        record, handoff = read_json(self.folder / "out/run.json"), read_json(self.folder / "out/logits.json")
        self.assertEqual((record["passed"], [entry["id"] for entry in record["items"]]), (False, calls[:2]))
        self.assertEqual([entry["id"] for entry in handoff["items"]], calls[:2])



@unittest.skipIf(not GEMMA4 or np is None, "Transformers with Gemma 4, PyTorch and NumPy are required for the runs")
class RestoredRunTests(unittest.TestCase):
    TEXT = {**TEXT, "hidden_size": 64}

    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        self.snapshot, self.contract = write_snapshot(self.folder, hidden_size=64)
        self.tensors = quantized_tensors(self.TEXT)
        self.export = write_export(self.folder / "export", self.contract, self.tensors)
        (self.folder / "prepare").mkdir()
        write_json(self.folder / "prepare/ready.json", {"ready": True, "contract_sha256": sha256(self.contract),
                                                        "environment": {"python": "3.11.9", "packages": {},
                                                                        "os_build": "26A1"}})
        self.records = made(TEXT["vocab_size"])
        self.forced = write_forced(self.folder / "forced", self.records)

    def main(self, *extra, output="out"):
        argv = ["layer1_anchor", "run", "--model", str(self.contract), "--snapshot", str(self.snapshot), "--prepare",
                str(self.folder / "prepare"), "--forced", str(self.forced), "--layers", "6",
                "--output", str(self.folder / output), *extra]
        observed = lambda path, volumes, patterns, on_abort: ReadyMonitor(on_abort, OBSERVED)
        placed = lambda path: {**load_contract(path), "reference": {**load_contract(path)["reference"], "device": "cpu"}}
        with mock.patch.object(sys, "argv", argv), mock.patch.object(reference, "ReferenceMonitor", observed), \
                mock.patch.object(layer1_anchor, "load_contract", side_effect=placed), \
                mock.patch.object(reference, "memory", return_value=dict(MEMORY)), \
                mock.patch.object(reference, "verify_inputs", return_value={}), \
                mock.patch.object(layer1_anchor, "swap_used", return_value=5), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return layer1_anchor.main()

    def test_the_scored_rows_with_weights_restored_from_the_quantized_tensors_match_the_plain_float32_forward(self):
        self.assertEqual((self.main("--weights", "8bit", "--export", str(self.export)),
                          self.main("--weights", "float16", output="rounded")), (0, 0))
        out = self.folder / "out"
        handoff, record = read_json(out / "logits.json"), read_json(out / "run.json")
        self.assertEqual(record["quantized"], {"export_sha256": sha256(self.export / "export.json"),
                                               "file": file_hash(self.export / layers.QUANTIZED),
                                               "tensors": len(self.tensors)})
        self.assertIn(f"quantized tensors {file_hash(self.export / layers.QUANTIZED)['sha256']} of the export "
                      f"{sha256(self.export / 'export.json')}", handoff["weights"])
        model, _, _ = layers.original_model(self.snapshot, 6)
        model.set_attn_implementation("sdpa")
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if f"{name}.quantized_data" in self.tensors:
                    parameter.copy_(float16_restore(*(self.tensors[f"{name}.{part}"]
                                                      for part in ("quantized_data", "scale", "minval"))).float())
            rounded = read_json(self.folder / "rounded/logits.json")
            for entry, other, item in zip(handoff["items"], rounded["items"], self.records, strict=True):
                values = np.frombuffer((out / entry["file"]).read_bytes(), dtype="<f4").reshape(entry["shape"])
                expected = model(torch.tensor([item["tokens"]]), use_cache=False).logits[0].numpy()
                start = item["start"]
                self.assertLessEqual(float(np.abs(values - expected[start:start + item["rows"]]).max()), anchor.LIMIT)
                kept = np.frombuffer((self.folder / "rounded" / other["file"]).read_bytes(), dtype="<f4")
                self.assertGreater(float(np.abs(values.reshape(-1) - kept).max()), 0.1)

    def test_an_export_or_quantized_tensors_that_do_not_fit_are_refused_before_the_folder_is_made(self):
        def edited(label, change):
            folder = write_export(self.folder / label, self.contract, self.tensors)
            change(folder)
            return folder

        def digest(folder):
            record = read_json(folder / "export.json")
            name = sorted(record["quantized"]["tensors"])[0]
            record["quantized"]["tensors"][name] = "0" * 64
            write_json(folder / "export.json", record)

        def flipped(folder):
            data = bytearray((folder / layers.QUANTIZED).read_bytes())
            data[-1] ^= 1
            (folder / layers.QUANTIZED).write_bytes(bytes(data))

        cases = {"not passed": write_export(self.folder / "failed", self.contract, self.tensors, passed=False),
                 "not 8bit": write_export(self.folder / "half", self.contract, {}, precision="float16",
                                          quantized=None),
                 "another tensor digest": edited("digest", digest), "a changed tensor file": edited("flipped", flipped)}
        with mock.patch.object(layer1_anchor, "model_for") as model_for:
            for label, folder in cases.items():
                with self.subTest(case=label), self.assertRaises(ValueError):
                    self.main("--weights", "8bit", "--export", str(folder), output=label)
                self.assertFalse((self.folder / f"{label}").is_dir() and (self.folder / label / "run.json").exists())
            for label, extra in (("no export", ("--weights", "8bit")),
                                 ("an export of rounded weights", ("--weights", "float16", "--export", str(self.export)))):
                with self.subTest(case=label), self.assertRaisesRegex(ValueError, "--export exactly"):
                    self.main(*extra, output=label)
            model_for.assert_not_called()
        self.assertEqual(sorted(path.name for path in self.folder.iterdir() if (path / "run.json").exists()), [])


if __name__ == "__main__":
    unittest.main()
