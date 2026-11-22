import io
import json
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
    from transformers.models.gemma4 import modeling_gemma4
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
except ImportError:
    torch = None

from Tools.Conversion.adapters.gemma4_31b import layers
from Tools.Conversion.adapters.gemma4_31b.prepare import TOKENIZER_FILES, load_contract
from Tools.Conversion.adapters.pinned import sha256
from Tools.Conversion.storage import digest, read_json, write_json

RECIPE = Path(__file__).resolve().parents[1] / "recipes/gemma4_31b"
SLIDING, FULL = "sliding_attention", "full_attention"
SHARD = "model-00001-of-00001.safetensors"
TEXT = {"model_type": "gemma4_text", "hidden_size": 48, "intermediate_size": 96, "vocab_size": 301,
        "num_hidden_layers": 12, "layer_types": ([SLIDING] * 5 + [FULL]) * 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "num_global_key_value_heads": 1, "head_dim": 16, "global_head_dim": 32,
        "sliding_window": 8, "max_position_embeddings": 4096, "rms_norm_eps": 1e-06,
        "rope_parameters": {SLIDING: {"rope_type": "default", "rope_theta": 10000.0},
                            FULL: {"rope_type": "proportional", "partial_rotary_factor": 0.25,
                                   "rope_theta": 1000000.0}},
        "final_logit_softcapping": 30.0, "attention_k_eq_v": True, "tie_word_embeddings": True,
        "hidden_activation": "gelu_pytorch_tanh", "hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0,
        "enable_moe_block": False, "attention_bias": False, "use_bidirectional_attention": "vision",
        "bos_token_id": 2, "pad_token_id": 0, "eos_token_id": 1}


def write_contract(folder, files):
    contract = read_json(RECIPE / "model.json")
    config = {key: TEXT.get(key) for key in contract["config"]}
    config.update(layer_pattern=TEXT["layer_types"][:6], eos_token_id=[TEXT["eos_token_id"]])
    tokenizer = {"policy": "keep-distributed", "files": {name: files[name]["sha256"] for name in TOKENIZER_FILES}}
    contract.update(files=files, config=config, tokenizer=tokenizer)
    path = Path(folder) / "model.json"
    write_json(path, contract)
    return path


def original(seed=0):
    torch.manual_seed(seed)
    config = Gemma4TextConfig(**TEXT)
    config._attn_implementation = "eager"
    model = modeling_gemma4.Gemma4ForCausalLM(config).eval()
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            noise = torch.randn_like(tensor)
            tensor.copy_(1 + 0.3 * noise if name.endswith("norm.weight") else 0.2 * noise)
        for layer in model.model.layers:
            layer.layer_scalar.copy_(torch.rand(1) + 0.5)
        model.model.embed_tokens.embed_scale.fill_(layers.rounded_scale(config.hidden_size))
    return model


def write_snapshot(folder):
    from safetensors.torch import save_file

    snapshot = Path(folder) / "snapshot"
    snapshot.mkdir()
    state = {"model.language_model." + name.removeprefix("model."): tensor.to(torch.bfloat16).contiguous()
             for name, tensor in original().state_dict().items() if name != "lm_head.weight"}
    save_file(state, str(snapshot / SHARD))
    documents = {"config.json": {"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"],
                                 "text_config": TEXT},
                 "model.safetensors.index.json": {"metadata": {}, "weight_map": dict.fromkeys(state, SHARD)},
                 "generation_config.json": {"bos_token_id": TEXT["bos_token_id"]}, "tokenizer.json": {"model": {}},
                 "tokenizer_config.json": {"bos_token": "<bos>"}}
    for name, value in documents.items():
        write_json(snapshot / name, value)
    (snapshot / "chat_template.jinja").write_text("{{ bos_token }}")
    files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(snapshot.iterdir())}
    return snapshot, write_contract(folder, files)


def write_run(folder, implementation, contract, arrays):
    folder.mkdir(parents=True)
    pinned = load_contract(contract)
    inputs = {}
    for name, array in arrays.items():
        np.save(folder / f"{name}.npy", array)
        ids = layers.tokens(name, pinned["config"])
        inputs[name] = {"positions": len(ids), "tokens_sha256": digest(ids),
                        "logits": layers.file_record(folder / f"{name}.npy")}
    model = {"id": pinned["model_id"], "revision": pinned["revision"], "contract_sha256": sha256(contract),
             "verified_before_loading": pinned["files"]}
    write_json(folder / "record.json", {"implementation": implementation, "layers": 6, "inputs": inputs,
                                        "layer_types": TEXT["layer_types"][:6], "model": model})


def edit_record(folder, change):
    record = read_json(folder / "record.json")
    change(record)
    write_json(folder / "record.json", record)


def set_row(name, row, values):
    def change(arrays):
        arrays[name][row] = -10.0
        arrays[name][row, :len(values)] = values
    return change


def add(name, row, column, value):
    def change(arrays):
        arrays[name][row, column] += value
    return change


def state(path):
    return (os.readlink(path) if path.is_symlink() else None), (path.read_bytes() if path.exists() else None)


class InputTests(unittest.TestCase):
    def test_the_inputs_start_with_bos_and_the_class_reads_the_window_then_chunks_of_one_two_four_and_eight(self):
        config = {"bos_token_id": 2, "vocab_size": 262144}
        short, long = (layers.tokens(name, config) for name in ("short", "long"))
        self.assertEqual((len(short), len(long), short[0], long[0]), (26, 1400, 2, 2))
        self.assertTrue(all(2 < token < 262144 for token in short[1:] + long[1:]))
        self.assertEqual(layers.tokens("long", config), long)
        bounds = layers.calls(1400, 1024)
        self.assertEqual(bounds[:6], [[0, 1024], [1024, 1025], [1025, 1027], [1027, 1031], [1031, 1039], [1039, 1040]])
        self.assertEqual([start for start, _ in bounds[1:]], [end for _, end in bounds[:-1]])
        self.assertEqual(bounds[-1][1], 1400)
        self.assertEqual(layers.calls(26, 1024), [[0, 26]])

    def test_a_run_refuses_first_layers_that_do_not_end_with_a_full_attention_layer_before_writing_or_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            contract = write_contract(folder, dict.fromkeys(TOKENIZER_FILES, {"bytes": 1, "sha256": "0" * 64}))
            config = load_contract(contract)["config"]
            for count in (6, 12):
                layers.check_layers(config, count)
            with mock.patch.object(layers, "verify_inputs") as verify, \
                    mock.patch.object(layers, "original_model") as original_model, \
                    mock.patch.object(layers, "class_model") as class_model:
                for implementation in layers.IMPLEMENTATIONS:
                    for count in (0, 1, 5, 7, 13):
                        output = Path(folder) / f"{implementation}-{count}"
                        with self.subTest(implementation=implementation, layers=count):
                            with self.assertRaisesRegex(ValueError, "must end with a full-attention layer"):
                                layers.run(implementation, contract, Path(folder) / "snapshot", count, output)
                            self.assertFalse(output.exists())
            verify.assert_not_called()
            original_model.assert_not_called()
            class_model.assert_not_called()


@unittest.skipIf(np is None, "NumPy is required for the comparison")
class CompareTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        self.contract = write_contract(self.folder, dict.fromkeys(TOKENIZER_FILES, {"bytes": 1, "sha256": "0" * 64}))
        generator = np.random.default_rng(0)
        self.arrays = {name: generator.normal(size=(length, TEXT["vocab_size"])).astype(np.float32)
                       for name, (length, _) in layers.INPUTS.items()}

    def pair(self, label, both=None, model_class=None):
        arrays = {name: array.copy() for name, array in self.arrays.items()}
        if both:
            both(arrays)
        changed = {name: array.copy() for name, array in arrays.items()}
        if model_class:
            model_class(changed)
        write_run(self.folder / label / "original", "original", self.contract, arrays)
        write_run(self.folder / label / "class", "class", self.contract, changed)
        return self.folder / label

    def compare(self, run):
        return layers.compare(run / "original", run / "class", self.contract)

    def test_matching_runs_pass_and_every_criterion_is_applied(self):
        result = self.compare(self.pair("same"))
        self.assertTrue(result["passed"], result["problems"])
        self.assertEqual([result["inputs"][name]["max_abs_logit"] for name in ("short", "long")], [0.0, 0.0])
        above = np.nextafter(np.float32(2), np.float32(3))
        cases = {"short logit": (None, add("short", 3, 4, 0.002), "short", {"max_abs_logit"}),
                 "long logit": (set_row("long", 3, [30.0]), add("long", 3, 5, 0.006), "long", {"max_abs_logit"}),
                 "long probability": (set_row("long", 7, [10.0, 10.0]), add("long", 7, 0, 0.004), "long",
                                      {"max_probability_error"}),
                 "long top-1": (set_row("long", 9, [2.0, above]), set_row("long", 9, [above, 2.0]), "long",
                                {"top1_mismatches"})}
        for label, (both, model_class, name, failing) in cases.items():
            with self.subTest(case=label):
                result = self.compare(self.pair(label.replace(" ", "-"), both, model_class))
                self.assertFalse(result["passed"])
                self.assertEqual(result["problems"], [])
                checks = result["inputs"][name]["checks"]
                self.assertEqual({key for key, passed in checks.items() if not passed}, failing)

    def test_runs_that_must_not_pass(self):
        def nan(arrays):
            arrays["long"][5, 6] = np.nan

        def infinity(arrays):
            arrays["short"][1, 2] = np.inf

        cases = {"the same NaN in both runs": (nan, None), "the same infinity in both runs": (infinity, None),
                 "a missing row": (None, lambda arrays: arrays.update(long=arrays["long"][:-1])),
                 "another vocabulary": (None, lambda arrays: arrays.update(short=np.zeros((26, 302), np.float32)))}
        for label, (both, model_class) in cases.items():
            with self.subTest(case=label):
                result = self.compare(self.pair(label.replace(" ", "-"), both, model_class))
                self.assertFalse(result["passed"])
                self.assertTrue(result["problems"])
        edits = {"another input": lambda record: record["inputs"]["long"].update(tokens_sha256="0" * 64),
                 "another layer count": lambda record: record.update(layers=5),
                 "another contract": lambda record: record["model"].update(contract_sha256="0" * 64),
                 "the original implementation twice": lambda record: record.update(implementation="original")}
        for label, edit in edits.items():
            with self.subTest(case=label):
                run = self.pair(label.replace(" ", "-"))
                edit_record(run / "class", edit)
                result = self.compare(run)
                self.assertFalse(result["passed"])
                self.assertTrue(result["problems"])
        run = self.pair("files")
        with self.subTest(case="a run compared with itself"):
            self.assertFalse(layers.compare(run / "original", run / "original", self.contract)["passed"])
        with self.subTest(case="logits changed after the record"):
            np.save(run / "class/long.npy", self.arrays["long"] + 1)
            self.assertFalse(self.compare(run)["passed"])
        with self.subTest(case="missing logits"):
            (run / "class/long.npy").unlink()
            self.assertFalse(self.compare(run)["passed"])

    def test_records_that_both_lack_or_share_a_layer_count_layer_types_or_model_off_the_contract_never_pass(self):
        kinds = TEXT["layer_types"]
        edits = {"no layer count": lambda record: record.pop("layers"),
                 "no layer types": lambda record: record.pop("layer_types"),
                 "a count that ends with a sliding-window layer": lambda record: record.update(layers=5,
                                                                                               layer_types=kinds[:5]),
                 "a count beyond the layers": lambda record: record.update(layers=13, layer_types=kinds + [FULL]),
                 "a boolean count": lambda record: record.update(layers=True, layer_types=kinds[:1]),
                 "other layer types": lambda record: record.update(layer_types=[FULL] * 6),
                 "another model": lambda record: record["model"].update(id="other/model"),
                 "another revision": lambda record: record["model"].update(revision="0" * 40),
                 "fewer verified files": lambda record: record["model"]["verified_before_loading"].popitem()}
        for label, edit in edits.items():
            with self.subTest(case=label):
                run = self.pair(label.replace(" ", "-"))
                for kind in layers.IMPLEMENTATIONS:
                    edit_record(run / kind, edit)
                result = self.compare(run)
                self.assertFalse(result["passed"])
                self.assertTrue(result["problems"])
        full_first = {"layer_pattern": [FULL], "num_hidden_layers": 1}
        self.assertEqual(layers.check_layers(full_first, 1), [FULL])
        for value in (True, 1.0, None):
            with self.subTest(layers=value), self.assertRaisesRegex(ValueError, "must end with a full-attention layer"):
                layers.check_layers(full_first, value)

    def test_a_run_refuses_an_existing_file_or_symbolic_link_as_its_output_before_reading_and_leaves_it(self):
        target = self.folder / "target"
        target.write_text("target")
        outputs = {"file": lambda path: path.write_text("earlier"),
                   "dangling link": lambda path: path.symlink_to(self.folder / "missing"),
                   "link": lambda path: path.symlink_to(target)}
        for label, make in outputs.items():
            output = self.folder / label.replace(" ", "-")
            make(output)
            before = state(output)
            with self.subTest(output=label):
                with mock.patch.object(layers, "verify_inputs") as verify, self.assertRaises(FileExistsError):
                    layers.run("original", self.contract, self.folder / "snapshot", 6, output)
                verify.assert_not_called()
                self.assertEqual(state(output), before)

    def test_compare_writes_only_its_result_as_json_to_standard_output_and_exits_with_one_unless_it_passes(self):
        for label, model_class in {"passing": None, "failing": add("short", 3, 4, 0.002)}.items():
            run = self.pair(label, model_class=model_class)
            result = self.compare(run)
            arguments = ["layers", "compare", "--original", str(run / "original"), "--model-class", str(run / "class")]
            with self.subTest(run=label):
                with mock.patch("sys.argv", arguments), mock.patch.object(layers, "compare", return_value=result), \
                        mock.patch("sys.stdout", new_callable=io.StringIO) as printed:
                    code = layers.main()
                self.assertEqual(result["passed"], label == "passing")
                self.assertEqual(json.loads(printed.getvalue()), result)
                self.assertEqual(printed.getvalue(), json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
                self.assertEqual(code, 0 if result["passed"] else 1)

    def test_runs_that_read_the_same_file_never_pass_and_copies_are_not_stopped_by_that_check(self):
        for label, link in {"symbolic-link": os.symlink, "hard-link": os.link}.items():
            with self.subTest(link=label):
                run = self.pair(label)
                (run / "class/long.npy").unlink()
                link(run / "original/long.npy", run / "class/long.npy")
                result = self.compare(run)
                self.assertFalse(result["passed"])
                self.assertIn("long: both runs read the same logits file", result["problems"])
        with self.subTest(link="record"):
            run = self.pair("record")
            (run / "class/record.json").unlink()
            os.link(run / "original/record.json", run / "class/record.json")
            self.assertIn("both runs read the same record.json", self.compare(run)["problems"])
        with self.subTest(link="copy"):
            run = self.pair("copy")
            shutil.copyfile(run / "original/long.npy", run / "class/long.npy")
            result = self.compare(run)
            self.assertTrue(result["passed"], result["problems"])


CONDITIONS = [("short-c1", "short", 26, 0, 1), ("short-c26", "short", 26, 0, 26),
              ("window-c1", "context", 1300, 0, 1), ("window-c2", "context", 1300, 0, 2),
              ("window-c4", "context", 1300, 0, 4), ("window-c8", "context", 1300, 0, 8),
              ("prefix-c1", "context", 1300, 1000, 1), ("capacity-2048", "context", 2056, 2040, 4),
              ("capacity-4096", "context", 4104, 4088, 8), ("capacity-8192", "context", 8200, 8184, 2),
              ("capacity-16384", "context", 16392, 16376, 4), ("tail", "context", 16896, 16832, 1)]


class ConditionTests(unittest.TestCase):
    def test_the_twelve_conditions_read_prefixes_of_the_short_and_the_context_input_and_keep_their_rows(self):
        self.assertEqual([tuple(condition) for condition in layers.CONDITIONS], CONDITIONS)
        config = {"bos_token_id": 2, "vocab_size": 262144}
        context = layers.tokens("context", config)
        self.assertEqual((len(context), context[0]), (16896, 2))
        self.assertTrue(all(2 < token < 262144 for token in context[1:]))
        self.assertEqual(layers.tokens("context", config), context)
        self.assertEqual(layers.kept_rows("short"), list(range(26)))
        self.assertEqual(layers.kept_rows("context"), [*range(1300), *range(2040, 2056), *range(4088, 4104),
                                                       *range(8184, 8200), *range(16376, 16392), *range(16832, 16896)])

    def test_a_context_run_refuses_a_layer_count_or_an_existing_output_before_writing_or_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            contract = write_contract(folder, dict.fromkeys(TOKENIZER_FILES, {"bytes": 1, "sha256": "0" * 64}))
            (Path(folder) / "used").mkdir()
            (Path(folder) / "dangling").symlink_to(Path(folder) / "missing")
            with mock.patch.object(layers, "verify_inputs") as verify, \
                    mock.patch.object(layers, "original_model") as original_model:
                for count in (0, 5, 7, 13):
                    with self.subTest(layers=count), self.assertRaisesRegex(ValueError, "full-attention"):
                        layers.context(contract, Path(folder) / "snapshot", count, Path(folder) / f"layers-{count}")
                    self.assertFalse((Path(folder) / f"layers-{count}").exists())
                for name in ("used", "dangling"):
                    with self.subTest(output=name), self.assertRaises(FileExistsError):
                        layers.context(contract, Path(folder) / "snapshot", 6, Path(folder) / name)
            verify.assert_not_called()
            original_model.assert_not_called()

    def test_a_context_run_refuses_an_existing_output_before_it_reads_the_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            broken = Path(folder) / "broken.json"
            broken.write_text("{")
            outputs = {"file": lambda path: path.write_text("earlier"), "folder": lambda path: path.mkdir(),
                       "dangling link": lambda path: path.symlink_to(Path(folder) / "missing")}
            for label, make in outputs.items():
                output = Path(folder) / label.replace(" ", "-")
                make(output)
                seen = lambda: (os.readlink(output) if output.is_symlink() else None,
                                output.read_bytes() if output.is_file() else None)
                before = seen()
                with self.subTest(output=label), mock.patch.object(layers, "load_contract", wraps=layers.load_contract) as load, \
                        self.assertRaises(FileExistsError):
                    layers.context(broken, Path(folder) / "snapshot", 6, output)
                load.assert_not_called()
                self.assertEqual(seen(), before)

    @unittest.skipIf(os.geteuid() == 0, "root searches a folder without permission")
    def test_a_context_run_refuses_a_link_into_a_folder_without_search_permission_before_it_reads_the_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            closed, output, broken = Path(folder) / "closed", Path(folder) / "closed-link", Path(folder) / "broken.json"
            closed.mkdir()
            output.symlink_to(closed / "inner")
            broken.write_text("{")
            closed.chmod(0)
            try:
                with mock.patch.object(layers, "load_contract", wraps=layers.load_contract) as load, \
                        self.assertRaises(FileExistsError):
                    layers.context(broken, Path(folder) / "snapshot", 6, output)
                load.assert_not_called()
            finally:
                closed.chmod(0o700)
            self.assertEqual(os.readlink(output), str(closed / "inner"))


@unittest.skipIf(np is None or torch is None, "NumPy, PyTorch and Transformers are required for the run")
class ContextReferenceTests(unittest.TestCase):
    def test_reading_in_chunks_with_the_cache_keeps_the_logits_of_one_call_on_the_chosen_rows(self):
        model = original()
        ids = [2, *torch.randint(3, TEXT["vocab_size"], (60,), generator=torch.Generator().manual_seed(3)).tolist()]
        with torch.no_grad():
            expected = model(torch.tensor([ids]), use_cache=False).logits[0].numpy()
        keep = [0, 6, 7, 8, 9, 16, 23, 24, 25, 40, 60]
        for size in (1, 3, 8, 13, 61):
            logits, calls = layers.kept_logits(model, ids, keep, size)
            with self.subTest(size=size):
                self.assertEqual(logits.shape, (len(keep), TEXT["vocab_size"]))
                self.assertEqual(calls, [[start, min(start + size, len(ids))] for start in range(0, len(ids), size)])
                self.assertLess(float(np.abs(logits - expected[keep]).max()), 1e-3)

    def test_the_context_run_keeps_the_rows_of_the_conditions_for_both_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot, contract = write_snapshot(folder)
            config = load_contract(contract)["config"]
            record = layers.context(contract, snapshot, 6, Path(folder) / "context")
            self.assertEqual(read_json(Path(folder) / "context/record.json"), record)
            self.assertEqual((record["schema"], record["implementation"], record["layers"], record["layer_types"],
                              record["chunk"]), (layers.CONTEXT_SCHEMA, "original", 6, TEXT["layer_types"][:6], 1024))
            arrays = {}
            for name in ("short", "context"):
                ids, rows, entry = layers.tokens(name, config), layers.kept_rows(name), record["inputs"][name]
                arrays[name] = np.load(Path(folder) / f"context/{name}.npy")
                with self.subTest(input=name):
                    self.assertEqual((entry["positions"], entry["tokens_sha256"], entry["rows"]), (len(ids), digest(ids), rows))
                    self.assertEqual(arrays[name].shape, (len(rows), TEXT["vocab_size"]))
                    self.assertEqual(entry["logits"]["shape"], [len(rows), TEXT["vocab_size"]])
                    self.assertEqual(entry["logits"]["sha256"], sha256(Path(folder) / f"context/{name}.npy"))
            self.assertEqual(record["inputs"]["context"]["calls"],
                             [[start, min(start + 1024, 16896)] for start in range(0, 16896, 1024)])
            model, _, _ = layers.original_model(snapshot, 6)
            ids = layers.tokens("context", config)
            with torch.no_grad():
                window = model(torch.tensor([ids[:1300]]), use_cache=False).logits[0].numpy()
                short = model(torch.tensor([layers.tokens("short", config)]), use_cache=False).logits[0].numpy()
            self.assertLess(float(np.abs(arrays["context"][:1300] - window).max()), 1e-3)
            self.assertLess(float(np.abs(arrays["short"] - short).max()), 1e-3)


@unittest.skipIf(np is None or torch is None, "NumPy, PyTorch and Transformers are required for the run")
class OriginalRunTests(unittest.TestCase):
    def test_the_original_run_reads_the_first_layers_with_rounded_weights_and_the_rounded_scale(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot, contract = write_snapshot(folder)
            config = load_contract(contract)["config"]
            reference = original()
            reference.load_state_dict({name: tensor.to(torch.bfloat16).to(torch.float16).float()
                                       for name, tensor in reference.state_dict().items()})
            record = layers.run("original", contract, snapshot, 12, Path(folder) / "all")
            for name in layers.INPUTS:
                ids = layers.tokens(name, config)
                with torch.no_grad():
                    expected = reference(torch.tensor([ids]), use_cache=False).logits[0].numpy()
                with self.subTest(input=name):
                    np.testing.assert_array_equal(np.load(Path(folder) / "all" / f"{name}.npy"), expected)
                    self.assertEqual(record["inputs"][name]["logits"]["shape"], [len(ids), TEXT["vocab_size"]])
                    self.assertEqual(record["inputs"][name]["tokens_sha256"], digest(ids))
            self.assertEqual(record["embed_scale"], 6.9375)
            first = layers.run("original", contract, snapshot, 6, Path(folder) / "first")
            self.assertEqual(first["layer_types"], TEXT["layer_types"][:6])
            self.assertEqual(read_json(Path(folder) / "first/record.json"), first)

    def test_a_run_refuses_an_existing_output_or_a_changed_checkpoint_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot, contract = write_snapshot(folder)
            (Path(folder) / "used").mkdir()
            shard = snapshot / SHARD
            data = bytearray(shard.read_bytes())
            data[-1] ^= 1
            with mock.patch.object(layers, "original_model") as load:
                with self.assertRaises(FileExistsError):
                    layers.run("original", contract, snapshot, 6, Path(folder) / "used")
                shard.write_bytes(data)
                with self.assertRaises(ValueError):
                    layers.run("original", contract, snapshot, 6, Path(folder) / "changed")
            load.assert_not_called()
            self.assertEqual(list((Path(folder) / "changed").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
