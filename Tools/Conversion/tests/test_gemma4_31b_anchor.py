import contextlib
import copy
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
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
    import transformers
except ImportError:
    torch = transformers = None

GEMMA4 = transformers is not None and hasattr(transformers, "Gemma4ForConditionalGeneration")

from Tools.Conversion.adapters.gemma4_31b import anchor, anchor_compare, anchor_summary, reference
from Tools.Conversion.adapters.pinned import sha256
from Tools.Conversion.storage import PipelineError

RECIPE = Path(__file__).resolve().parents[1] / "recipes/gemma4_31b"
VOCABULARY = 97
WINDOW = 8
LENGTHS = (1, 5, 8, 9, 23, 40)
MEMORY = {"max_rss_bytes": 10, "footprint_bytes": 5, "lifetime_peak_footprint_bytes": 12,
          "mps_driver_allocated_bytes": 8, "mps_current_allocated_bytes": 7}
OBSERVED = {"samples": 3, "interval_s": 5, "aborts": 0, "failure": None, "unreadable_samples": 0,
            "first": {"swap_used_bytes": 5}, "peak": {"swap_used_bytes": 5, "rss_bytes": 11},
            "lowest": {"free_percent": 40, "free_bytes": {"volume/folder": 1}}}
EXPECTED = anchor_compare.expectations()


def expected(names, vocabulary=4, window=4, **changes):
    return {**EXPECTED, "cases": list(names), "vocabulary": vocabulary, "window": window, **changes}


def tiny_model(folder):
    text = {"hidden_size": 48, "intermediate_size": 96, "num_hidden_layers": 6, "num_attention_heads": 4,
            "num_key_value_heads": 2, "num_global_key_value_heads": 1, "head_dim": 16, "global_head_dim": 32,
            "sliding_window": WINDOW, "vocab_size": VOCABULARY,
            "layer_types": ["sliding_attention"] * 5 + ["full_attention"], "attention_k_eq_v": True,
            "final_logit_softcapping": 30.0, "tie_word_embeddings": True, "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": 0, "max_position_embeddings": 256, "pad_token_id": 0, "bos_token_id": 2,
            "eos_token_id": 1,
            "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6,
                                                   "rope_type": "proportional"},
                                "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"}}}
    config = transformers.Gemma4Config(text_config=text, vision_config=None, audio_config=None, tie_word_embeddings=True)
    torch.manual_seed(0)
    model = transformers.Gemma4ForConditionalGeneration(config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.normal_(1.0, 0.3)
        for index, layer in enumerate(model.model.language_model.layers):
            layer.layer_scalar.fill_(0.5 + 0.1 * index)
    with contextlib.redirect_stderr(io.StringIO()):
        model.to(torch.bfloat16).save_pretrained(folder)
        loaded, _ = transformers.Gemma4ForConditionalGeneration.from_pretrained(folder, dtype=torch.bfloat16,
                                                                                 output_loading_info=True)
    return loaded.eval()


def sequences():
    generator = torch.Generator().manual_seed(1)
    return [torch.randint(3, VOCABULARY, (length,), generator=generator).tolist() for length in LENGTHS]


@unittest.skipIf(not GEMMA4 or np is None, "Transformers with Gemma 4, PyTorch and NumPy are required for the model checks")
class LayerwiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            cls.model = tiny_model(folder)

    def test_the_layerwise_fp32_computation_with_split_attention_matches_the_plain_fp32_forward(self):
        with torch.no_grad():
            results = anchor.agreement(copy.deepcopy(self.model), sequences(), "cpu", rows=5)
        self.assertEqual([row["positions"] for row in results], list(LENGTHS))
        for row in results:
            self.assertEqual(row["non_finite"], {"layerwise": 0, "plain": 0})
            self.assertLessEqual(row["max_abs_difference"], anchor.LIMIT)

    def test_the_computation_uses_every_value_that_the_weights_function_returns(self):
        model, plain = copy.deepcopy(self.model), copy.deepcopy(self.model).float()
        base = anchor.resident(model, "bfloat16")
        for name, tensor in [*plain.named_parameters(), *plain.named_buffers()]:
            if "rotary_emb" not in name:
                tensor.data.mul_(1.25)
        sequence = sequences()[-1]
        with torch.no_grad():
            changed = anchor.logits(model, sequence, lambda name: base(name) * 1.25, "cpu", rows=5)
            original = anchor.logits(model, sequence, base, "cpu", rows=5)
            expected, _ = reference.forced(plain, sequence, [], "cpu")
        self.assertEqual((changed.dtype.name, {str(p.dtype) for p in model.parameters()}), ("float32", {"torch.bfloat16"}))
        self.assertLessEqual(float(np.abs(changed - expected).max()), anchor.LIMIT)
        self.assertGreater(float(np.abs(original - expected).max()), 1e-2)

    def test_the_embedding_scale_and_the_float16_weights_replace_the_resident_values(self):
        model = copy.deepcopy(self.model)
        self.assertEqual([anchor.embed_scale(model, dtype) for dtype in ("bfloat16", "float16")], [6.9375, 6.9296875])
        self.assertAlmostEqual(anchor.embed_scale(model, "float32"), 48 ** 0.5, places=6)
        scaled = anchor.scaled(anchor.resident(model, "float16"), 6.9296875)
        self.assertEqual(float(scaled(anchor.SCALE)), 6.9296875)
        name = "model.language_model.layers.0.self_attn.q_proj.weight"
        weight = model.get_parameter(name)
        self.assertTrue(torch.equal(scaled(name), weight.to(torch.float16).float()))

    def test_the_float16_loading_keeps_the_bf16_embedding_scale_and_the_state_of_the_float16_loading(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            tiny_model(folder)
            model, info = anchor.load(folder, "float16", "cpu")
            expected, expected_info = transformers.Gemma4ForConditionalGeneration.from_pretrained(
                folder, dtype=torch.float16, output_loading_info=True)
        state, wanted = ({**dict(item.named_buffers(remove_duplicate=False)), **dict(item.named_parameters(remove_duplicate=False))}
                         for item in (model, expected))
        self.assertEqual((sorted(state), info), (sorted(wanted), expected_info))
        for name, tensor in wanted.items():
            with self.subTest(name=name):
                self.assertEqual(state[name].dtype, tensor.dtype)
                self.assertEqual(name == anchor.SCALE, not torch.equal(state[name], tensor))
        bf16 = self.model.model.language_model.embed_tokens.embed_scale
        self.assertEqual((str(bf16.dtype), float(state[anchor.SCALE]), float(wanted[anchor.SCALE])),
                         ("torch.bfloat16", float(bf16.to(torch.float16)), 6.9296875))
        self.assertEqual(str(state["model.language_model.layers.0.layer_scalar"].dtype), "torch.float16")
        self.assertIs(model.lm_head.weight, model.model.language_model.embed_tokens.weight)

    def test_ranges_are_recorded_for_every_layer_and_the_hooks_are_removed(self):
        model, watch = copy.deepcopy(self.model), anchor.Ranges()
        with torch.no_grad():
            with watch.watching(model):
                watch.case = "first"
                anchor.logits(model, sequences()[-1], anchor.resident(model, "bfloat16"), "cpu")
            rows = copy.deepcopy(watch.rows)
            anchor.logits(model, sequences()[-1], anchor.resident(model, "bfloat16"), "cpu")
        self.assertEqual(watch.rows, rows)
        self.assertEqual(len(rows), 6 * 4 + 3)
        self.assertEqual({row["case"] for row in rows.values()}, {"first"})
        recorder = anchor.Ranges()
        recorder.case, hook = "case", recorder.hook("point", 0)
        hook(None, (torch.tensor([70000.0, -40000.0, 1.0]),))
        hook(None, (torch.tensor([0.0]),), torch.tensor([-50000.0]))
        self.assertEqual(recorder.rows[("point", 0)], {"point": "point", "layer": 0, "max_abs": 70000.0, "case": "case",
                                                       "above_float16_max": 1, "above_half_float16_max": 3})

    def test_cast_statistics_count_changes_underflow_and_overflow_once_per_tensor(self):
        values = torch.tensor([1.0, 3e-8, 2.9e-8, 1e-6, 7e4, 0.0]).to(torch.bfloat16)
        module = torch.nn.Module()
        module.first = torch.nn.Linear(6, 1, bias=False)
        module.first.weight.data = values.reshape(1, 6).clone()
        module.second = torch.nn.Linear(6, 1, bias=False)
        module.second.weight = module.first.weight
        result = anchor.cast_statistics(module)
        exact = values.float()
        error = float((exact[1].to(torch.float16).float() - exact[1]).abs())
        self.assertEqual({key: result[key] for key in ("tensors", "elements", "changed", "underflow", "non_finite",
                                                        "above_float16_max")},
                         {"tensors": 1, "elements": 6, "changed": 4, "underflow": 1, "non_finite": 1, "above_float16_max": 1})
        self.assertAlmostEqual(result["max_abs_error"], error, places=12)

    def test_the_loading_check_requires_exactly_the_skipped_layers_and_one_dtype(self):
        loading = {"info": {"missing_keys": [], "unexpected_keys": ["model.language_model.layers.6.x"],
                            "mismatched_keys": [], "error_msgs": []},
                   "parameter_dtypes": ["torch.bfloat16"], "parameter_devices": ["mps:0"], "training": False}
        self.assertTrue(anchor.clean(loading, "bfloat16", "mps", ["model.language_model.layers.6.x"]))
        for change in ({"unexpected": []}, {"missing": ["x"]}, {"dtype": "float16"}):
            data = copy.deepcopy(loading)
            if "missing" in change:
                data["info"]["missing_keys"] = change["missing"]
            with self.subTest(change=change):
                self.assertFalse(anchor.clean(data, change.get("dtype", "bfloat16"), "mps",
                                              change.get("unexpected", ["model.language_model.layers.6.x"])))


def write_reference(folder, cases, passed=True, model=None, device="mps"):
    folder = Path(folder)
    (folder / "cases").mkdir(parents=True)
    records = []
    for name, kind, inputs, outputs, logits in cases:
        tokens, array = folder / f"cases/{name}.tokens.json", folder / f"cases/{name}.logits.npy"
        tokens.write_text(json.dumps({"input_tokens": inputs, "output_tokens": outputs}))
        np.save(array, np.asarray(logits, dtype=np.float32))
        records.append({"id": name, "kind": kind, "input_tokens": len(inputs), "output_tokens": len(outputs),
                        "saved_positions": len(inputs) + len(outputs), "first_scoring_row": len(inputs) - 1,
                        "scoring_positions": len(outputs),
                        "files": {"tokens": reference.file_record(tokens), "logits": reference.file_record(array)}})
    (folder / "reference.json").write_text(json.dumps({"schema": reference.SCHEMA, "passed": passed,
                                                       "dtype": anchor_compare.REFERENCE_DTYPE, "device": device,
                                                       "model": model or EXPECTED["model"], "cases": records,
                                                       "source": {"reference_sha256": "a" * 64}, "run": folder.name}))
    return folder


def edit(folder, change):
    path = Path(folder) / "reference.json"
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def rows(count, vocabulary=4, shift=0.0):
    return [[float((row * 3 + column) % 5) + (shift if column == 0 else 0.0) for column in range(vocabulary)]
            for row in range(count)]


@unittest.skipIf(np is None, "NumPy is required for the comparison")
class CompareTests(unittest.TestCase):
    def pair(self, folder, first, second, **options):
        a = write_reference(Path(folder) / "a", first)
        b = write_reference(Path(folder) / "b", second)
        return anchor_compare.compare(a, b, expected([case[0] for case in first], **options))

    def test_the_metrics_follow_their_definitions(self):
        baseline = [[0.0, 0.0, -9.0, -9.0], [2.0, 1.99, -9.0, -9.0], [3.0, 0.0, 2.0, -9.0], [1.0, 0.0, 0.0, 0.0]]
        candidate = [[0.0, 0.0, -9.0, -9.0], [1.99, 2.0, -9.0, -9.0], [0.0, 0.0, 3.5, -9.0], [1.0, 0.5, 0.0, 0.0]]
        with tempfile.TemporaryDirectory() as folder:
            result = self.pair(folder, [("case", "chat", [2, 1], [3, 1], baseline)],
                               [("case", "chat", [2, 1], [3, 1], candidate)])
        stored = [[float(np.float32(value)) for value in row] for row in (*baseline, *candidate)]
        softmax = [[math.exp(value) / sum(math.exp(item) for item in row) for value in row] for row in stored]
        p, q = softmax[:4], softmax[4:]
        errors = [max(abs(a - b) for a, b in zip(x, y)) for x, y in zip(p, q)]
        divergence = [sum(a * math.log(a / b) for a, b in zip(x, y)) for x, y in zip(p, q)]
        case = result["cases"][0]
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(case["all"]["max_probability_error_max"], max(errors), places=12)
        self.assertAlmostEqual(case["all"]["max_probability_error_mean"], sum(errors) / 4, places=12)
        self.assertAlmostEqual(case["all"]["kl_mean"], sum(divergence) / 4, places=12)
        self.assertAlmostEqual(case["all"]["kl_max"], max(divergence), places=12)
        self.assertEqual(case["all"]["max_abs_logit_difference"], 3.0)
        self.assertEqual((case["all"]["top1_matches"], case["all"]["near_tie_mismatches"], case["all"]["other_mismatches"]),
                         (2, 1, 1))
        self.assertEqual([(item["row"], item["scoring"], item["near_tie"]) for item in case["mismatches"]],
                         [(1, True, True), (2, True, False)])
        self.assertAlmostEqual(case["mismatches"][0]["probability_gap"], p[1][0] - p[1][1], places=12)
        self.assertEqual((case["scoring"]["rows"], case["scoring"]["top1_matches"]), (2, 0))

    def test_groups_separate_kinds_bands_scoring_rows_and_the_window(self):
        with tempfile.TemporaryDirectory() as folder:
            first = [("short", "chat", [2, 3], [1], rows(3)), ("raw", "raw", [2, 3], [1], rows(3)),
                     ("long", "chat", [2] * 600, [1], rows(601))]
            second = [(name, kind, inputs, outputs, np.asarray(logits) + 0.5) for name, kind, inputs, outputs, logits in first]
            result = self.pair(folder, first, second, window=40)
        self.assertTrue(result["passed"])
        self.assertEqual([(group["kind"], group["band"], group["scope"], group["cases"], group["rows"])
                          for group in result["groups"]],
                         [("chat", "1-512", "all", ["short"], 3), ("chat", "1-512", "scoring", ["short"], 1),
                          ("raw", "1-512", "all", ["raw"], 3), ("raw", "1-512", "scoring", ["raw"], 1),
                          ("chat", "513-2048", "all", ["long"], 601), ("chat", "513-2048", "scoring", ["long"], 1),
                          ("chat", "513-2048", "window", ["long"], 65)])
        self.assertEqual((result["groups"][-1]["first_row"], result["groups"][-1]["last_row"]), (8, 72))

    def test_inputs_that_must_not_pass(self):
        clean = [("case", "chat", [2, 3], [4, 1], rows(4))]

        def nan(value):
            return [("case", "chat", [2, 3], [4, 1], [rows(4)[0], [value, 0.0, 0.0, 0.0], *rows(4)[2:]])]

        variants = {"nan in the candidate": (clean, nan(math.nan), {}), "inf in both": (nan(math.inf), nan(math.inf), {}),
                    "-inf in the baseline": (nan(-math.inf), clean, {}),
                    "missing rows": (clean, [("case", "chat", [2, 3], [4, 1], rows(3))], {}),
                    "another vocabulary": (clean, clean, {"vocabulary": 5}),
                    "other tokens": (clean, [("case", "chat", [2, 3], [5, 1], rows(4))], {}),
                    "another case": (clean, [("other", "chat", [2, 3], [4, 1], rows(4))], {})}
        for label, (first, second, options) in variants.items():
            with tempfile.TemporaryDirectory() as folder, self.subTest(label=label), np.errstate(invalid="ignore"):
                result = self.pair(folder, first, second, **options)
                self.assertFalse(result["passed"])
                self.assertTrue(result["problems"])
                self.assertEqual(result["groups"], [])
                json.dumps(result, allow_nan=False)

    def test_records_that_differ_from_their_files_or_did_not_pass_never_pass(self):
        clean = [("case", "chat", [2, 3], [4, 1], rows(4))]
        changes = {"same size logits": lambda b: np.save(b / "cases/case.logits.npy", np.asarray(rows(4, shift=0.5),
                                                                                                 dtype=np.float32)),
                   "same size tokens": lambda b: (b / "cases/case.tokens.json").write_text(
                       json.dumps({"input_tokens": [2, 3], "output_tokens": [4, 2]})),
                   "failed record": lambda b: edit(b, lambda data: data.update(passed=False)),
                   "first scoring row": lambda b: edit(b, lambda data: data["cases"][0].update(first_scoring_row=0)),
                   "other model": lambda b: edit(b, lambda data: data.update(model={"id": "owner/other"})),
                   "path in the case ID": lambda b: edit(b, lambda data: data["cases"][0].update(id="../case"))}
        for label, change in changes.items():
            with tempfile.TemporaryDirectory() as folder, self.subTest(label=label):
                a = write_reference(Path(folder) / "a", clean)
                b = write_reference(Path(folder) / "b", clean)
                change(b)
                if label == "path in the case ID":
                    change(a)
                names = ["../case"] if label == "path in the case ID" else ["case"]
                result = anchor_compare.compare(a, b, expected(names))
                self.assertFalse(result["passed"])
                self.assertTrue(result["problems"])

    def test_an_artifact_compared_with_itself_or_with_a_copy_never_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            first = write_reference(Path(folder) / "a", [("case", "chat", [2, 3], [4, 1], rows(4))])
            shutil.copytree(first, Path(folder) / "copy")
            for second in (first, Path(folder) / "copy"):
                with self.subTest(second=second.name):
                    result = anchor_compare.compare(first, second, expected(["case"]))
                    self.assertFalse(result["passed"])
                    self.assertIn("both artifacts are the same run", result["problems"])

    def test_records_that_agree_with_each_other_but_not_with_the_contract_never_pass(self):
        clean = [("case", "chat", [2, 3], [4, 1], rows(4))]
        anchor_record = {"schema": anchor.SCHEMA, "stage": "anchor",
                         "compute": {"weights": "bfloat16", "compute": "float32", "embed_scale": 73.5, "attention_rows": 512}}
        changes = {"no model": (lambda data: data.pop("model"), ["case"], "the model differs from the contract"),
                   "another revision": (lambda data: data["model"].update(revision="0" * 40), ["case"],
                                        "the model differs from the contract"),
                   "a case missing in both": (lambda data: None, ["case", "other"], "the cases differ from the fixtures"),
                   "no dtype in a reference record": (lambda data: data.pop("dtype"), ["case"], anchor_compare.UNKNOWN),
                   "another device": (lambda data: data.update(device="cpu"), ["case"], anchor_compare.UNKNOWN),
                   "an unknown embedding scale": (lambda data: data.update(
                       anchor_record, compute={**anchor_record["compute"], "embed_scale": 73.4}), ["case"], anchor_compare.UNKNOWN),
                   "no compute": (lambda data: data.update(schema=anchor.SCHEMA, stage="anchor"), ["case"], anchor_compare.UNKNOWN)}
        for label, (change, names, message) in changes.items():
            with tempfile.TemporaryDirectory() as folder, self.subTest(label=label):
                a, b = (write_reference(Path(folder) / side, clean) for side in ("a", "b"))
                for side in (a, b):
                    edit(side, change)
                result = anchor_compare.compare(a, b, expected(names))
                self.assertFalse(result["passed"])
                self.assertEqual([problem for problem in result["problems"] if problem.endswith(message)],
                                 [f"baseline: {message}", f"candidate: {message}"])
        with tempfile.TemporaryDirectory() as folder:
            a, b = (write_reference(Path(folder) / side, clean) for side in ("a", "b"))
            edit(a, lambda data: data.update(anchor_record))
            result = anchor_compare.compare(a, b, expected(["case"]))
        self.assertTrue(result["passed"])
        self.assertEqual(result["identities"], {"baseline": {"stage": "anchor", **anchor_record["compute"]},
                                                "candidate": anchor_compare.definition("reference", EXPECTED)})

    def test_inputs_that_are_the_same_file_never_pass(self):
        clean = [("case", "chat", [2, 3], [4, 1], rows(4))]
        with tempfile.TemporaryDirectory() as folder:
            a = write_reference(Path(folder) / "a", clean)
            linked = Path(folder) / "linked"
            shutil.copytree(a, linked)
            (linked / "reference.json").unlink()
            os.link(a / "reference.json", linked / "reference.json")
            (Path(folder) / "alias").symlink_to(a)
            for second in (linked, Path(folder) / "alias"):
                with self.subTest(second=second.name):
                    result = anchor_compare.compare(a, second, expected(["case"]))
                    self.assertFalse(result["passed"])
                    self.assertIn("both records are the same file", result["problems"])
            other = write_reference(Path(folder) / "other", clean)
            (other / "cases/case.logits.npy").unlink()
            os.link(a / "cases/case.logits.npy", other / "cases/case.logits.npy")
            result = anchor_compare.compare(a, other, expected(["case"]))
        self.assertFalse(result["passed"])
        self.assertEqual(result["problems"], ["case: both logits files are the same file"])

    def test_the_output_is_created_once_and_never_replaces_a_file_or_a_link(self):
        with tempfile.TemporaryDirectory() as folder:
            target, link, broken = (Path(folder) / name for name in ("out.json", "link.json", "broken.json"))
            anchor_compare.write_new(target, {"value": 1})
            link.symlink_to(Path(folder) / "missing.json")
            for path in (target, link):
                with self.subTest(path=path.name), self.assertRaises(FileExistsError):
                    anchor_compare.write_new(path, {"value": 2})
            with mock.patch.object(anchor_compare.os, "fsync", side_effect=OSError("disk")), self.assertRaises(OSError):
                anchor_compare.write_new(broken, {"value": 3})
            self.assertEqual((json.loads(target.read_text()), link.is_symlink(), (Path(folder) / "missing.json").exists(),
                              broken.exists()), ({"value": 1}, True, False, False))


def write_forced(folder):
    folder = Path(folder)
    (folder / "cases").mkdir(parents=True)
    records = []
    for name, inputs, outputs in (("first", [2, 5, 6], [7, 1]), ("second", [2, 8], [1])):
        path = folder / f"cases/{name}.tokens.json"
        path.write_text(json.dumps({"input_tokens": inputs, "output_tokens": outputs, "steps": []}))
        records.append({"id": name, "kind": "chat", "input_tokens": len(inputs), "output_tokens": len(outputs),
                        "files": {"tokens": reference.file_record(path)}})
    (folder / "reference.json").write_text(json.dumps({"schema": reference.SCHEMA, "passed": True, "cases": records}))
    return folder


class ForcedInputTests(unittest.TestCase):
    def test_the_forced_tokens_come_from_the_recorded_files(self):
        with tempfile.TemporaryDirectory() as folder:
            write_forced(folder)
            digest, cases = anchor.forced_inputs(folder, 9)
            self.assertEqual(digest, sha256(Path(folder) / "reference.json"))
        self.assertEqual([(case["id"], case["input_tokens"], case["output_tokens"]) for case in cases],
                         [("first", [2, 5, 6], [7, 1]), ("second", [2, 8], [1])])

    def test_changed_files_out_of_range_tokens_and_other_records_are_rejected(self):
        def tokens(folder, value):
            (Path(folder) / "cases/first.tokens.json").write_text(json.dumps(value))

        changes = {"same size tokens": lambda f: tokens(f, {"input_tokens": [2, 5, 7], "output_tokens": [7, 1], "steps": []}),
                   "failed record": lambda f: edit(f, lambda data: data.update(passed=False)),
                   "other schema": lambda f: edit(f, lambda data: data.update(schema="other/1")),
                   "path in the case ID": lambda f: edit(f, lambda data: data["cases"][0].update(id="../first")),
                   "duplicate key": lambda f: (Path(f) / "reference.json").write_text(
                       (Path(f) / "reference.json").read_text()[:-1] + ', "passed": true}')}
        for label, change in changes.items():
            with tempfile.TemporaryDirectory() as folder, self.subTest(label=label):
                write_forced(folder)
                change(folder)
                with self.assertRaises(ValueError):
                    anchor.forced_inputs(folder, 9)
        for label, value in (("vocabulary", 9), ("negative", -1), ("boolean", True), ("float", 2.0)):
            with tempfile.TemporaryDirectory() as folder, self.subTest(label=label):
                write_forced(folder)
                tokens(folder, {"input_tokens": [2, 5, value], "output_tokens": [7, 1], "steps": []})
                path = Path(folder) / "cases/first.tokens.json"
                edit(folder, lambda data: data["cases"][0]["files"].update(tokens=reference.file_record(path)))
                with self.assertRaises(ValueError):
                    anchor.forced_inputs(folder, 9)


@unittest.skipIf(not GEMMA4 or np is None, "Transformers with Gemma 4, PyTorch and NumPy are required for the stage")
class StageTests(unittest.TestCase):
    def arguments(self, folder, stage, **options):
        return SimpleNamespace(stage=stage, snapshot=Path(folder) / "snapshot", reference=Path(folder) / "reference",
                               output=Path(folder) / stage, **options)

    def ready(self):
        return {"contract_sha256": "c" * 64, "environment": {"python": "3.11.9", "packages": {}, "os_build": "26A428"}}

    def contract(self):
        return {"model_id": "owner/model", "revision": "r" * 40, "config": {"vocab_size": VOCABULARY},
                "reference": {"device": "cpu"}}

    def test_the_anchor_and_half_records_compare_with_the_bf16_reference_files(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            model = tiny_model(Path(folder) / "model")
            cases = []
            for index, sequence in enumerate(sequences()[2:]):
                with torch.no_grad():
                    values, _ = reference.forced(model, sequence[:-1], sequence[-1:], "cpu")
                cases.append((f"case{index}", "chat" if index else "raw", sequence[:-1], sequence[-1:], values))
            identity = {"id": "owner/model", "revision": "r" * 40, "contract_sha256": "c" * 64, "weights_sha256": {},
                        "verified_before_loading": {}}
            write_reference(Path(folder) / "reference", cases, model=identity, device="cpu")
            outputs, records = {}, {}
            for stage, options in (("anchor", {"weights": "bfloat16", "embed_scale": "bfloat16"}), ("half", {})):
                with mock.patch.object(reference, "verify_inputs", return_value={}), \
                        mock.patch.object(anchor, "load", side_effect=lambda *a, **k: (copy.deepcopy(model), {})), \
                        mock.patch.object(anchor, "clean", return_value=True):
                    args = self.arguments(folder, stage, **options)
                    record = anchor.logits_stage(args, self.contract(), self.ready())
                record.update(passed=True)
                (args.output / "reference.json").write_text(json.dumps(record))
                outputs[stage], records[stage] = args.output, record
            result = anchor_compare.compare(outputs["anchor"], Path(folder) / "reference",
                                            expected([case[0] for case in cases], VOCABULARY, WINDOW, model=identity,
                                                     device="cpu", scales={**EXPECTED["scales"], "bfloat16": 6.9375}))
        self.assertEqual(result["problems"], [])
        self.assertTrue(result["passed"])
        self.assertGreater(result["cases"][0]["all"]["max_abs_logit_difference"], 0.0)
        self.assertEqual((len(records["anchor"]["ranges"]), "ranges" in records["half"]), (6 * 4 + 3, False))
        self.assertEqual((records["anchor"]["compute"]["embed_scale"], records["half"]["compute"]["compute"]),
                         (6.9375, "float16"))

    def test_the_inputs_are_checked_before_the_model_is_loaded(self):
        with tempfile.TemporaryDirectory() as folder:
            write_forced(Path(folder) / "reference")
            for stage, work, options in (("anchor", anchor.logits_stage, {"weights": "bfloat16", "embed_scale": "bfloat16"}),
                                         ("half", anchor.logits_stage, {}), ("layers", anchor.layers_stage, {})):
                with self.subTest(stage=stage), mock.patch.object(anchor, "load") as load, \
                        mock.patch.object(reference, "verify_inputs", side_effect=ValueError("changed")):
                    with self.assertRaises(ValueError):
                        work(self.arguments(folder, stage, **options), self.contract(), self.ready())
                    load.assert_not_called()


class ReadyMonitor:
    def __init__(self, on_abort, observed=None):
        self.on_abort, self.observed, self.ready = on_abort, observed, threading.Event()
        self.ready.set()

    def start(self):
        pass

    def finish(self):
        return copy.deepcopy(self.observed or {"samples": 1, "aborts": 0})


def finished_record():
    return {"checks": {"finite": True}, "loading": {"after_load": dict(MEMORY)}, "resources": {"cases": [dict(MEMORY)]},
            "timing": {}}


class SupervisedTests(unittest.TestCase):
    def run_main(self, folder, stage, monitor, work, extra=()):
        prepare = Path(folder) / "prepare"
        prepare.mkdir(exist_ok=True)
        (prepare / "ready.json").write_text(json.dumps({"ready": True, "contract_sha256": sha256(RECIPE / "model.json")}))
        argv = ["anchor", stage, *extra, "--model", str(RECIPE / "model.json"), "--snapshot", folder, "--prepare",
                str(prepare), "--reference", folder, "--output", folder]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(reference, "ReferenceMonitor", monitor), \
                mock.patch.object(reference, "memory", return_value=dict(MEMORY)), \
                mock.patch.object(anchor, "layers_stage", side_effect=lambda *a: work()), \
                mock.patch.object(anchor, "logits_stage", side_effect=lambda *a: work()):
            return anchor.main()

    def test_each_stage_writes_its_own_finished_record_with_the_monitor_checks(self):
        names = {}
        for stage, extra in (("layers", ()), ("anchor", ("--weights", "float16", "--embed-scale", "float32")), ("half", ())):
            with tempfile.TemporaryDirectory() as folder:
                code = self.run_main(folder, stage, lambda path, volumes, patterns, on_abort: ReadyMonitor(on_abort, OBSERVED),
                                     finished_record, extra)
                names[stage] = sorted(path.name for path in Path(folder).glob("*.json"))
                record = json.loads((Path(folder) / names[stage][0]).read_text())
            self.assertEqual((code, record["passed"], record["checks"]["monitor_clean"]), (0, True, True))
        self.assertEqual(names, {"layers": ["layers.json"], "anchor": ["reference.json"], "half": ["reference.json"]})

    def test_a_stop_is_recorded_with_the_schema_of_the_stage(self):
        def monitor(path, volumes, patterns, on_abort):
            on_abort("swap rose by 2000000001 B, above 2000000000 B")
            return ReadyMonitor(on_abort)

        def interrupted():
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as folder:
            code = self.run_main(folder, "half", monitor, interrupted)
            failure = json.loads((Path(folder) / "failure.json").read_text())
            self.assertFalse((Path(folder) / "reference.json").exists())
        self.assertEqual((code, failure["schema"], failure["abort"][:9]), (1, anchor.SCHEMA, "swap rose"))

    def test_an_existing_record_is_refused_before_the_contract_is_read(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "layers.json").write_text("{}")
            with mock.patch.object(anchor, "load_contract") as contract, self.assertRaises(FileExistsError):
                self.run_main(folder, "layers", mock.Mock(), mock.Mock())
            contract.assert_not_called()


def stage_record(name, digest, **changes):
    case = {"kind": "chat", "input_tokens": 2, "output_tokens": 1, "saved_positions": 3, "first_scoring_row": 1,
            "scoring_positions": 1, "non_finite": 0, "seconds": 0.5, "files": {"logits": {"sha256": "b" * 64}}}
    record = {"schema": anchor.LAYERS_SCHEMA if name == "layers" else anchor.SCHEMA, "passed": True,
              "checks": {"finite": True}, "source": {"reference_sha256": digest}, "device": "mps",
              "model": copy.deepcopy(EXPECTED["model"]),
              "loading": {"class": "Gemma4ForConditionalGeneration", "attn_implementation": "sdpa", "parameters": 9,
                          "parameter_dtypes": ["torch.bfloat16"], "parameter_devices": ["mps:0"],
                          "info": {"missing_keys": []}, "embed_scale": 73.5, "after_load": dict(MEMORY)},
              "resources": {"cases": [dict(MEMORY)], "final": dict(MEMORY),
                            "swap": {"first_bytes": 1, "peak_bytes": 1, "increase_bytes": 0, "limit_increase_bytes": 2},
                            "monitor": copy.deepcopy(OBSERVED)},
              "timing": {"total_s": 2.0}}
    if name == "layers":
        record.update(stage="layers", layers=6, layer_types=["sliding_attention"], limit=1e-4, attention_rows=512,
                      cases=[{"id": case_id, "positions": 3, "max_abs_difference": 1e-6,
                              "non_finite": {"layerwise": 0, "plain": 0}} for case_id in EXPECTED["cases"]])
    else:
        definition = dict(anchor_summary.definitions(EXPECTED)[name])
        record.update(stage=definition.pop("stage"), compute=definition, cases=[{"id": case_id, **case} for case_id in EXPECTED["cases"]],
                      ranges=[{"point": "residual_input", "layer": 0, "max_abs": 3.0, "case": "ja_long",
                               "above_float16_max": 0, "above_half_float16_max": 0}], cast=None)
    record.update(changes)
    return record


def write_summary_run(folder, run="run", reused=(), changes=None):
    folder = Path(folder)
    stages = []
    artifacts = folder / "reference"
    artifacts.mkdir(parents=True)
    (artifacts / "reference.json").write_text(json.dumps({
        "schema": reference.SCHEMA, "passed": True, "dtype": anchor_compare.REFERENCE_DTYPE, "device": "mps",
        "model": EXPECTED["model"], "cases": [{"id": case_id} for case_id in EXPECTED["cases"]]}))
    digest = sha256(artifacts / "reference.json")
    stages.append({"id": "reference", "role": "reference", "status": "reused", "origin": "earlier",
                   "artifacts": str(artifacts), "execute_s": 0})
    for name, file in anchor_summary.STAGES.items():
        artifacts = folder / name
        artifacts.mkdir()
        (artifacts / file).write_text(json.dumps(stage_record(name, digest, **(changes or {}).get(name, {}))))
        stages.append({"id": name, "role": "verify" if name == "layers" else "reference",
                       "status": "reused" if name in reused else "executed", "origin": "earlier" if name in reused else run,
                       "artifacts": str(artifacts), "execute_s": 1.0})
    report = folder / "report.json"
    report.write_text(json.dumps({"version": 1, "run_id": run, "status": "completed", "verification": "passed",
                                  "stages": stages}))
    return report


def comparison_record(hashes):
    metrics = {"rows": 3, "max_probability_error_max": 0.01, "max_probability_error_mean": 0.001, "kl_mean": -1e-18,
               "kl_p99": 1e-4, "kl_max": 1e-4, "max_abs_logit_difference": 0.5, "top1_matches": 3,
               "near_tie_mismatches": 0, "other_mismatches": 0}
    return {"schema": anchor_compare.SCHEMA, "passed": True, "problems": [], "vocabulary": 262144, "window": 1024,
            "near_tie_margin": 0.01, "records_sha256": hashes,
            "cases": [{"id": "case", "problems": [], "non_finite": {"baseline": 0, "candidate": 0}, "all": metrics,
                       "scoring": metrics, "mismatches": []}],
            "groups": [{"kind": "chat", "band": "1-512", "scope": "all", "cases": ["case"], **metrics}]}


class SummaryTests(unittest.TestCase):
    def summarize(self, report, comparisons):
        paths = []
        for index, data in enumerate(comparisons):
            path = report.parent / f"comparison{index}.json"
            path.write_text(json.dumps(data))
            paths.append(path)
        argv = ["summary", "--report", str(report), *(item for path in paths for item in ("--comparison", str(path)))]
        output = io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(output), \
                contextlib.redirect_stderr(io.StringIO()):
            return anchor_summary.main(), output.getvalue()

    def hashes(self, report, *names):
        data = json.loads(report.read_text())
        artifacts = {stage["id"]: Path(stage["artifacts"]) for stage in data["stages"]}
        return [sha256(artifacts[name] / anchor_summary.STAGES.get(name, "reference.json")) for name in names]

    def test_the_projection_names_the_compared_stages_and_drops_local_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            report = write_summary_run(folder)
            code, output = self.summarize(report, [comparison_record(self.hashes(report, "anchor", "reference"))])
        result = json.loads(output)
        self.assertEqual(code, 0)
        self.assertNotIn(folder, output)
        self.assertNotIn("volume/folder", output)
        self.assertEqual((result["comparisons"][0]["baseline"], result["comparisons"][0]["candidate"],
                          result["reference"]["origin"], len(result["records"])), ("anchor", "reference", "earlier", 5))

    def test_stages_from_another_run_and_comparisons_of_other_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for label, (reused, hashes) in {"reused stage": (("anchor",), None),
                                            "other records": ((), ["a" * 64, "b" * 64])}.items():
                with self.subTest(label=label):
                    report = write_summary_run(Path(folder) / label, reused=reused)
                    code, output = self.summarize(report, [comparison_record(hashes or self.hashes(report, "anchor",
                                                                                                  "half"))])
                    self.assertEqual((code, output), (1, ""))

    def test_a_record_that_does_not_match_its_stage_definition_is_rejected(self):
        anchor_definition = {key: value for key, value in anchor_summary.definitions(EXPECTED)["anchor"].items() if key != "stage"}
        variants = {"float16 weights recorded as bfloat16": {"anchor_w16": {"compute": anchor_definition}},
                    "a half record named anchor": {"half": {"stage": "anchor"}},
                    "layers of another revision": {"layers": {"model": {**EXPECTED["model"], "revision": "0" * 40}}},
                    "a case missing": {"anchor": {"cases": []}}}
        with tempfile.TemporaryDirectory() as folder:
            for label, changes in variants.items():
                with self.subTest(label=label):
                    report = write_summary_run(Path(folder) / label.replace(" ", "-"), changes=changes)
                    code, output = self.summarize(report, [comparison_record(self.hashes(report, "anchor", "reference"))])
                    self.assertEqual((code, output), (1, ""))

    def test_a_comparison_given_twice_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            report = write_summary_run(folder)
            comparison = comparison_record(self.hashes(report, "anchor", "reference"))
            self.assertEqual(self.summarize(report, [comparison, comparison]), (1, ""))

    def test_free_text_in_a_shareable_field_is_rejected(self):
        for change in (lambda r: r["cases"][0].update(kind="free text"), lambda r: r["compute"].update(weights="a/b"),
                       lambda r: r.update(passed="yes")):
            record = stage_record("anchor", "a" * 64)
            change(record)
            with self.subTest(change=change), self.assertRaises(PipelineError):
                anchor_summary.logits_record({"record": record, "sha256": "c" * 64}, "anchor")


if __name__ == "__main__":
    unittest.main()
