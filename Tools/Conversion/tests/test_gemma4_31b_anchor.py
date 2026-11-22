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
from Tools.Conversion import runner
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
LIMIT = 64
ANCHOR_RECORD = {"schema": anchor.SCHEMA, "stage": "anchor",
                 "compute": {"weights": "bfloat16", "compute": "float32", "embed_scale": 73.5, "attention_rows": 512}}


def expected(cases, vocabulary=4, window=4, **changes):
    return {**EXPECTED, "cases": [{"id": name, "kind": kind, "max_new_tokens": LIMIT} for name, kind in cases],
            "vocabulary": vocabulary, "window": window, **changes}


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
        records.append({"id": name, "kind": kind, "max_new_tokens": LIMIT, "input_tokens": len(inputs), "output_tokens": len(outputs),
                        "saved_positions": len(inputs) + len(outputs), "first_scoring_row": len(inputs) - 1,
                        "scoring_positions": len(outputs),
                        "files": {"tokens": reference.file_record(tokens), "logits": reference.file_record(array)}})
    (folder / "reference.json").write_text(json.dumps({"schema": reference.SCHEMA, "passed": passed,
                                                       "dtype": anchor_compare.REFERENCE_DTYPE, "device": device,
                                                       "model": model or EXPECTED["model"], "cases": records,
                                                       "fixtures_sha256": EXPECTED["fixtures_sha256"],
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
        return anchor_compare.compare(a, b, expected([case[:2] for case in first], **options))

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
                   "an archive as the logits file": lambda b: (
                       np.savez(b / "cases/case.logits.npy", logits=np.asarray(rows(4), dtype=np.float32)),
                       (b / "cases/case.logits.npy.npz").replace(b / "cases/case.logits.npy"),
                       edit(b, lambda data: data["cases"][0]["files"].update(logits=reference.file_record(b / "cases/case.logits.npy")))),
                   "token lists as text": lambda b: (
                       (b / "cases/case.tokens.json").write_text(json.dumps({"input_tokens": "23", "output_tokens": "41"})),
                       edit(b, lambda data: data["cases"][0]["files"].update(tokens=reference.file_record(b / "cases/case.tokens.json")))),
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
                result = anchor_compare.compare(a, b, expected([(name, "chat") for name in names]))
                self.assertFalse(result["passed"])
                self.assertTrue(result["problems"])

    def test_an_artifact_compared_with_itself_or_with_a_copy_never_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            first = write_reference(Path(folder) / "a", [("case", "chat", [2, 3], [4, 1], rows(4))])
            shutil.copytree(first, Path(folder) / "copy")
            for second in (first, Path(folder) / "copy"):
                with self.subTest(second=second.name):
                    result = anchor_compare.compare(first, second, expected([("case", "chat")]))
                    self.assertFalse(result["passed"])
                    self.assertIn("both artifacts are the same run", result["problems"])

    def test_records_that_agree_with_each_other_but_not_with_the_contract_never_pass(self):
        clean = [("case", "chat", [2, 3], [4, 1], rows(4))]
        anchor_record = ANCHOR_RECORD
        changes = {"no model": (lambda data: data.pop("model"), ["case"], anchor_compare.SHAPE),
                   "another revision": (lambda data: data["model"].update(revision="0" * 40), ["case"],
                                        "the model differs from the contract"),
                   "a case missing in both": (lambda data: None, ["case", "other"], "the cases differ from the fixtures"),
                   "another kind in both": (lambda data: data["cases"][0].update(kind="raw"), ["case"],
                                            "the cases differ from the fixtures"),
                   "another output limit in both": (lambda data: data["cases"][0].update(max_new_tokens=LIMIT + 1), ["case"],
                                                    "the cases differ from the fixtures"),
                   "another kind in two anchor records": (lambda data: (data.update(anchor_record),
                                                                        data["cases"][0].update(kind="raw")), ["case"],
                                                          "the cases differ from the fixtures"),
                   "other fixtures in both": (lambda data: data.update(fixtures_sha256="0" * 64), ["case"],
                                              "the fixtures differ from the fixture file"),
                   "no dtype in a reference record": (lambda data: data.pop("dtype"), ["case"], anchor_compare.SHAPE),
                   "another dtype in a reference record": (lambda data: data.update(dtype={}), ["case"], anchor_compare.UNKNOWN),
                   "another device": (lambda data: data.update(device="cpu"), ["case"], anchor_compare.UNKNOWN),
                   "an unknown embedding scale": (lambda data: data.update(
                       anchor_record, compute={**anchor_record["compute"], "embed_scale": 73.4}), ["case"], anchor_compare.UNKNOWN),
                   "no compute": (lambda data: data.update(schema=anchor.SCHEMA, stage="anchor"), ["case"], anchor_compare.SHAPE),
                   "a stage inside compute": (lambda data: data.update(
                       anchor_record, stage="other", compute={**anchor_record["compute"], "stage": "anchor"}), ["case"],
                       anchor_compare.UNKNOWN),
                   "an extra entry in compute": (lambda data: data.update(
                       anchor_record, compute={**anchor_record["compute"], "stage": "anchor"}), ["case"],
                       anchor_compare.UNKNOWN)}
        for label, (change, names, message) in changes.items():
            with tempfile.TemporaryDirectory() as folder, self.subTest(label=label):
                a, b = (write_reference(Path(folder) / side, clean) for side in ("a", "b"))
                for side in (a, b):
                    edit(side, change)
                result = anchor_compare.compare(a, b, expected([(name, "chat") for name in names]))
                self.assertFalse(result["passed"])
                self.assertEqual([problem for problem in result["problems"] if problem.endswith(message)],
                                 [f"baseline: {message}", f"candidate: {message}"])
        with tempfile.TemporaryDirectory() as folder:
            a, b = (write_reference(Path(folder) / side, clean) for side in ("a", "b"))
            edit(a, lambda data: data.update(anchor_record, source={"reference_sha256": sha256(b / "reference.json")}))
            result = anchor_compare.compare(a, b, expected([("case", "chat")]))
        self.assertTrue(result["passed"])
        self.assertEqual(result["identities"], {"baseline": {"stage": "anchor", **anchor_record["compute"]},
                                                "candidate": anchor_compare.definition("reference", EXPECTED)})

    def test_a_diagnostic_record_must_come_from_the_compared_or_the_same_bf16_record(self):
        clean = [("case", "chat", [2, 3], [4, 1], rows(4))]
        message = "the forced inputs do not come from the same BF16 reference record"
        with tempfile.TemporaryDirectory() as folder:
            a, b, c, d = (write_reference(Path(folder) / side, clean) for side in "abcd")
            named = {"another BF16 record": c / "reference.json", "the compared BF16 record": b / "reference.json"}
            for label, source in named.items():
                edit(a, lambda data: data.update(ANCHOR_RECORD, source={"reference_sha256": sha256(source)}))
                with self.subTest(label=label):
                    result = anchor_compare.compare(a, b, expected([("case", "chat")]))
                    self.assertEqual([problem for problem in result["problems"] if problem.endswith(message)],
                                     [] if source.parent == b else [f"baseline: {message}"])
            for label, sources in {"other BF16 records": ("c" * 64, "d" * 64), "a name that is not a SHA-256": ("x", "x")}.items():
                for side, value in zip((c, d), sources):
                    edit(side, lambda data: data.update(ANCHOR_RECORD, source={"reference_sha256": value}))
                with self.subTest(label=label):
                    result = anchor_compare.compare(c, d, expected([("case", "chat")]))
                    self.assertFalse(result["passed"])
                    self.assertEqual([problem for problem in result["problems"] if problem.endswith(message)],
                                     [f"baseline: {message}", f"candidate: {message}"])

    def test_a_record_of_another_shape_gives_a_failed_result_with_the_reason(self):
        clean = [("case", "chat", [2, 3], [4, 1], rows(4))]
        variants = {"a list instead of a record": lambda data: [],
                    "the schema as a list": lambda data: {**data, "schema": [reference.SCHEMA]},
                    "the schema as an object": lambda data: {**data, "schema": {"name": reference.SCHEMA}},
                    "the schema as a number": lambda data: {**data, "schema": 1},
                    "the model as a list": lambda data: {**data, "model": []},
                    "the cases as an object": lambda data: {**data, "cases": {}},
                    "a case as a list": lambda data: {**data, "cases": [[]]},
                    "the files of a case as a list": lambda data: {**data, "cases": [{**data["cases"][0], "files": []}]},
                    "a token count as text": lambda data: {**data, "cases": [{**data["cases"][0], "input_tokens": "2"}]},
                    "passed as text": lambda data: {**data, "passed": "yes"},
                    "the compute of an anchor record as a list": lambda data: {**data, **ANCHOR_RECORD, "compute": []},
                    "the source of an anchor record as text": lambda data: {**data, **ANCHOR_RECORD, "source": "text"}}
        for label, change in variants.items():
            with tempfile.TemporaryDirectory() as folder, self.subTest(label=label):
                a, b = (write_reference(Path(folder) / side, clean) for side in ("a", "b"))
                (b / "reference.json").write_text(json.dumps(change(json.loads((b / "reference.json").read_text()))))
                result = anchor_compare.compare(a, b, expected([("case", "chat")]))
                self.assertFalse(result["passed"])
                self.assertIn(f"candidate: {anchor_compare.SHAPE}", result["problems"])
                self.assertEqual((result["identities"]["candidate"], result["cases"], result["groups"]), (None, [], []))
                json.dumps(result, allow_nan=False)
        with tempfile.TemporaryDirectory() as folder:
            a, b = (write_reference(Path(folder) / side, clean) for side in ("a", "b"))
            edit(b, lambda data: data.update(schema=[reference.SCHEMA]))
            output = io.StringIO()
            with mock.patch.object(sys, "argv", ["compare", "--baseline", str(a), "--candidate", str(b)]), \
                    contextlib.redirect_stdout(output):
                code = anchor_compare.main()
        self.assertEqual(code, 1)
        self.assertIn(f"candidate: {anchor_compare.SHAPE}", json.loads(output.getvalue())["problems"])

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
                    result = anchor_compare.compare(a, second, expected([("case", "chat")]))
                    self.assertFalse(result["passed"])
                    self.assertIn("both records are the same file", result["problems"])
            other = write_reference(Path(folder) / "other", clean)
            (other / "cases/case.logits.npy").unlink()
            os.link(a / "cases/case.logits.npy", other / "cases/case.logits.npy")
            result = anchor_compare.compare(a, other, expected([("case", "chat")]))
        self.assertFalse(result["passed"])
        self.assertEqual(result["problems"], ["case: both logits files are the same file"])

    def test_the_result_is_the_only_output_and_reads_back_as_json(self):
        for passed, code in ((True, 0), (False, 1)):
            result = {"passed": passed, "problems": [] if passed else ["x"], "value": 0.5}
            output = io.StringIO()
            with self.subTest(passed=passed), mock.patch.object(sys, "argv", ["compare", "--baseline", "a", "--candidate", "b"]), \
                    mock.patch.object(anchor_compare, "compare", return_value=result), contextlib.redirect_stdout(output):
                self.assertEqual(anchor_compare.main(), code)
            self.assertEqual(output.getvalue(), json.dumps(result, indent=2, sort_keys=True) + "\n")
            self.assertEqual(json.loads(output.getvalue()), result)


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
                                            expected([case[:2] for case in cases], VOCABULARY, WINDOW, model=identity,
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


FILES = {"tokens": {"bytes": 1, "sha256": "b" * 64}, "logits": {"bytes": 1, "sha256": "b" * 64}}
COUNTS = {"input_tokens": 2, "output_tokens": 1, "saved_positions": 3, "first_scoring_row": 1, "scoring_positions": 1}


def stage_record(name, digest, **changes):
    case = {"kind": "chat", **COUNTS, "non_finite": 0, "seconds": 0.5, "files": copy.deepcopy(FILES)}
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
                              "non_finite": {"layerwise": 0, "plain": 0}} for case_id in (item["id"] for item in EXPECTED["cases"])])
    else:
        definition = dict(anchor_summary.definitions(EXPECTED)[name])
        record.update(stage=definition.pop("stage"), compute=definition,
                      cases=[{**case, "id": item["id"], "kind": item["kind"]} for item in EXPECTED["cases"]],
                      ranges=[{"point": "residual_input", "layer": 0, "max_abs": 3.0, "case": "ja_long",
                               "above_float16_max": 0, "above_half_float16_max": 0}],
                      cast={"tensors": 1, "elements": 4, "changed": 1, "underflow": 0, "non_finite": 0, "above_float16_max": 0,
                            "max_abs_error": 1e-8} if name == "anchor_w16" else None)
    record.update(changes)
    return record


def write_summary_run(folder, run="run", reused=(), changes=None):
    folder = Path(folder)
    stages = []

    def place(name, file, record, origin):
        artifacts = folder / name / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / file).write_text(json.dumps(record))
        (folder / name / "manifest.json").write_text(json.dumps({
            "version": 1, "key": "k" * 64, "status": "completed", "run_id": origin, "stage": name,
            "outputs": runner.outputs({"outputs": [file]}, artifacts)}))
        stages.append({"id": name, "role": "verify" if name == "layers" else "reference",
                       "status": "executed" if origin == run else "reused", "origin": origin, "artifacts": str(artifacts),
                       "execute_s": 1.0 if origin == run else 0})
        return sha256(artifacts / file)

    digest = place("reference", "reference.json", {
        "schema": reference.SCHEMA, "passed": True, "dtype": anchor_compare.REFERENCE_DTYPE, "device": "mps",
        "model": EXPECTED["model"], "cases": [{**case, **COUNTS, "files": copy.deepcopy(FILES)} for case in EXPECTED["cases"]],
        "fixtures_sha256": EXPECTED["fixtures_sha256"]}, "earlier")
    for name, file in anchor_summary.STAGES.items():
        place(name, file, stage_record(name, digest, **(changes or {}).get(name, {})), "earlier" if name in reused else run)
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
            path.write_text(data if isinstance(data, str) else json.dumps(data))
            paths.append(path)
        argv = ["summary", "--report", str(report), *(item for path in paths for item in ("--comparison", str(path)))]
        output, errors = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = anchor_summary.main()
        self.errors = errors.getvalue()
        return code, output.getvalue()

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

    def test_records_that_are_not_the_recorded_outputs_of_their_run_are_rejected(self):
        def manifest(folder, name, **changes):
            path = folder / name / "manifest.json"
            path.write_text(json.dumps({**json.loads(path.read_text()), **changes}))

        def substitute(folder):
            other = write_summary_run(folder / "other", run="other", changes={"half": {"timing": {"total_s": 3.0}}})
            shutil.copy(other.parent / "half/artifacts/reference.json", folder / "half/artifacts/reference.json")

        def point(folder):
            other = write_summary_run(folder / "other", run="other")
            data = json.loads((folder / "report.json").read_text())
            next(stage for stage in data["stages"] if stage["id"] == "anchor")["artifacts"] = str(other.parent / "anchor/artifacts")
            (folder / "report.json").write_text(json.dumps(data))

        variants = {"another run's record of the same step": substitute, "a step that names another run's output": point,
                    "no manifest": lambda folder: (folder / "anchor/manifest.json").unlink(),
                    "a manifest of another form": lambda folder: (folder / "anchor/manifest.json").write_text("[]"),
                    "a manifest without outputs": lambda folder: manifest(folder, "anchor", outputs={}),
                    "a manifest of another version": lambda folder: manifest(folder, "anchor", version=2),
                    "a manifest of another run": lambda folder: manifest(folder, "anchor", run_id="other"),
                    "a manifest of another step": lambda folder: manifest(folder, "anchor", stage="anchor_w16"),
                    "a reused reference with a manifest of the given run": lambda folder: manifest(folder, "reference", run_id="run")}
        for label, change in variants.items():
            with tempfile.TemporaryDirectory() as folder, self.subTest(label=label):
                report = write_summary_run(Path(folder) / "run")
                change(report.parent)
                code, output = self.summarize(report, [comparison_record(self.hashes(report, "anchor", "reference"))])
                self.assertEqual((code, output), (1, ""))

    def test_records_computed_from_another_reference_record_are_rejected(self):
        for name in ("anchor", "half", "layers"):
            with tempfile.TemporaryDirectory() as folder, self.subTest(name=name):
                report = write_summary_run(folder, changes={name: {"source": {"reference_sha256": "e" * 64}}})
                code, output = self.summarize(report, [comparison_record(self.hashes(report, "anchor", "reference"))])
                self.assertEqual((code, output), (1, ""))

    def test_a_comparison_given_twice_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            report = write_summary_run(folder)
            comparison = comparison_record(self.hashes(report, "anchor", "reference"))
            self.assertEqual(self.summarize(report, [comparison, comparison]), (1, ""))

    def test_two_comparisons_of_the_same_pair_of_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            report = write_summary_run(folder)
            first, second = self.hashes(report, "anchor", "reference"), self.hashes(report, "anchor", "half")
            reformatted = json.dumps(comparison_record(first), indent=4, sort_keys=True)
            variants = {"the same comparison written again": ([comparison_record(first), reformatted], 1),
                        "the baseline and the candidate swapped": ([comparison_record(first), comparison_record(first[::-1])], 1),
                        "two different pairs": ([comparison_record(first), comparison_record(second)], 0)}
            for label, (comparisons, expected_code) in variants.items():
                with self.subTest(label=label):
                    code, output = self.summarize(report, comparisons)
                    self.assertEqual(code, expected_code)
                    if expected_code == 0:
                        self.assertEqual([(item["baseline"], item["candidate"]) for item in json.loads(output)["comparisons"]],
                                         [("anchor", "reference"), ("anchor", "half")])
                    else:
                        self.assertEqual((output, self.errors), ("", "Cannot summarize local records (PipelineError).\n"))

    def test_values_that_the_step_or_the_group_requires_are_not_dropped(self):
        for name, change in (("anchor", lambda record: record.pop("ranges")), ("anchor_w16", lambda record: record.update(cast=None))):
            record = stage_record(name, "a" * 64)
            change(record)
            with self.subTest(name=name), self.assertRaises(PipelineError):
                anchor_summary.logits_record({"record": record, "sha256": "c" * 64}, name)
        data = comparison_record(["a" * 64, "b" * 64])
        data["groups"][0].update(scope="window")
        with self.assertRaises(PipelineError):
            anchor_summary.comparison(data, {"a" * 64: "anchor", "b" * 64: "reference"})

    def test_inputs_of_another_shape_stop_with_a_pipeline_error(self):
        def report_change(folder, change):
            path = folder / "report.json"
            path.write_text(json.dumps(change(json.loads(path.read_text()))))

        def stage(folder, name, **changes):
            report_change(folder, lambda data: {**data, "stages": [{**item, **changes} if item["id"] == name else item
                                                                   for item in data["stages"]]})

        def manifest(folder, name, change):
            path = folder / name / "manifest.json"
            path.write_text(json.dumps(change(json.loads(path.read_text()))))

        files = {"a report that is a list": lambda folder: (folder / "report.json").write_text("[]"),
                 "stages as an object": lambda folder: report_change(folder, lambda data: {**data, "stages": {}}),
                 "a stage that is text": lambda folder: report_change(folder, lambda data: {**data, "stages": ["anchor"]}),
                 "artifacts as a number": lambda folder: stage(folder, "anchor", artifacts=1),
                 "an origin as a list": lambda folder: stage(folder, "reference", origin=["earlier"]),
                 "a manifest that is a list": lambda folder: manifest(folder, "anchor", lambda data: []),
                 "manifest outputs as a list": lambda folder: manifest(folder, "anchor", lambda data: {**data, "outputs": []}),
                 "a manifest entry as text": lambda folder: manifest(folder, "anchor", lambda data: {
                     **data, "outputs": {"artifacts": {"tree": {"reference.json": "x"}}}})}
        records = {"a record schema as a list": {"anchor": {"schema": [anchor.SCHEMA]}},
                   "compute as a list": {"anchor": {"compute": []}}, "cases as an object": {"half": {"cases": {}}},
                   "the model as a list": {"layers": {"model": []}}, "the source as text": {"anchor": {"source": "text"}},
                   "checks as a list": {"anchor": {"checks": []}}, "timing as a list": {"half": {"timing": [1.0]}},
                   "layer types as text": {"layers": {"layer_types": "sliding"}},
                   "ranges as an object": {"anchor": {"ranges": {}}}, "a cast as a list": {"anchor_w16": {"cast": []}}}
        comparisons = {"a comparison that is a list": lambda hashes: [],
                       "comparison cases as an object": lambda hashes: {**comparison_record(hashes), "cases": {}},
                       "a group as text": lambda hashes: {**comparison_record(hashes), "groups": ["chat"]},
                       "a comparison of three records": lambda hashes: comparison_record([*hashes, hashes[0]]),
                       "a comparison of a record outside the report": lambda hashes: comparison_record(["a" * 64, hashes[1]])}
        variants = {**{label: (change, None, None) for label, change in files.items()},
                    **{label: (None, changes, None) for label, changes in records.items()},
                    **{label: (None, None, change) for label, change in comparisons.items()}}
        for label, (change, changes, data) in variants.items():
            with tempfile.TemporaryDirectory() as folder, self.subTest(label=label):
                report = write_summary_run(folder, changes=changes)
                hashes = self.hashes(report, "anchor", "reference")
                if change:
                    change(report.parent)
                code, output = self.summarize(report, [data(hashes) if data else comparison_record(hashes)])
                self.assertEqual((code, output, self.errors), (1, "", "Cannot summarize local records (PipelineError).\n"))

    def test_free_text_in_a_shareable_field_is_rejected(self):
        for change in (lambda r: r["cases"][0].update(kind="free text"), lambda r: r["compute"].update(weights="a/b"),
                       lambda r: r.update(passed="yes")):
            record = stage_record("anchor", "a" * 64)
            change(record)
            with self.subTest(change=change), self.assertRaises(PipelineError):
                anchor_summary.logits_record({"record": record, "sha256": "c" * 64}, "anchor")


if __name__ == "__main__":
    unittest.main()
