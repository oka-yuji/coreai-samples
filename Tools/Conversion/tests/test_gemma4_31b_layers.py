import copy
import hashlib
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

from Tools.Conversion.adapters.gemma4_31b import export, layers, native
from Tools.Conversion.adapters.gemma4_31b.prepare import TOKENIZER_FILES, load_contract
from Tools.Conversion.adapters.pinned import sha256
from Tools.Conversion.storage import digest, file_hash, read_json, write_json

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


def write_contract(folder, files, text=TEXT):
    contract = read_json(RECIPE / "model.json")
    config = {key: text.get(key) for key in contract["config"]}
    config.update(layer_pattern=text["layer_types"][:6], eos_token_id=[text["eos_token_id"]])
    tokenizer = {"policy": "keep-distributed", "files": {name: files[name]["sha256"] for name in TOKENIZER_FILES}}
    contract.update(files=files, config=config, tokenizer=tokenizer)
    path = Path(folder) / "model.json"
    write_json(path, contract)
    return path


def original(seed=0, text=TEXT):
    torch.manual_seed(seed)
    config = Gemma4TextConfig(**text)
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


def write_snapshot(folder, **changes):
    from safetensors.torch import save_file

    text = {**TEXT, **changes}
    snapshot = Path(folder) / "snapshot"
    snapshot.mkdir()
    state = {"model.language_model." + name.removeprefix("model."): tensor.to(torch.bfloat16).contiguous()
             for name, tensor in original(text=text).state_dict().items() if name != "lm_head.weight"}
    save_file(state, str(snapshot / SHARD))
    documents = {"config.json": {"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"],
                                 "text_config": text},
                 "model.safetensors.index.json": {"metadata": {}, "weight_map": dict.fromkeys(state, SHARD)},
                 "generation_config.json": {"bos_token_id": TEXT["bos_token_id"]}, "tokenizer.json": {"model": {}},
                 "tokenizer_config.json": {"bos_token": "<bos>"}}
    for name, value in documents.items():
        write_json(snapshot / name, value)
    (snapshot / "chat_template.jinja").write_text("{{ bos_token }}")
    files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(snapshot.iterdir())}
    return snapshot, write_contract(folder, files, text)


TAPS = (1, 2, 3, 4, 5)


def write_run(folder, implementation, contract, arrays, taps=TAPS):
    folder.mkdir(parents=True)
    pinned = load_contract(contract)
    inputs = {}
    for name, outputs in arrays.items():
        ids = layers.tokens(name, pinned["config"])
        inputs[name] = {"positions": len(ids), "tokens_sha256": digest(ids)}
        for output, array in outputs.items():
            np.save(folder / f"{name}.{output}.npy", array)
            inputs[name][output] = layers.file_record(folder / f"{name}.{output}.npy")
    model = {"id": pinned["model_id"], "revision": pinned["revision"], "contract_sha256": sha256(contract),
             "verified_before_loading": pinned["files"]}
    write_json(folder / "record.json", {"implementation": implementation, "layers": 6, "tap_layers": list(taps),
                                        "inputs": inputs, "layer_types": TEXT["layer_types"][:6], "model": model})


def edit_record(folder, change):
    record = read_json(folder / "record.json")
    change(record)
    write_json(folder / "record.json", record)


def set_row(name, row, values):
    def change(arrays):
        arrays[name]["logits"][row] = -10.0
        arrays[name]["logits"][row, :len(values)] = values
    return change


def add(name, row, column, value, output="logits"):
    def change(arrays):
        arrays[name][output][row, column] += value
    return change


def copied(arrays):
    return {name: {output: array.copy() for output, array in outputs.items()} for name, outputs in arrays.items()}


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
                                layers.run(implementation, contract, Path(folder) / "snapshot", count, "1", output)
                            self.assertFalse(output.exists())
            verify.assert_not_called()
            original_model.assert_not_called()
            class_model.assert_not_called()

    def test_tap_layers_are_ascending_distinct_counts_from_one_to_the_layer_count(self):
        self.assertEqual(layers.check_taps("1,2,3,4,5", 6), (1, 2, 3, 4, 5))
        self.assertEqual(layers.check_taps("6", 6), (6,))
        self.assertEqual(layers.check_taps("1,17,29,47,59", 60), (1, 17, 29, 47, 59))
        for text in ("0,1", "1,7", "7", "2,1", "1,1", "", ",", "1,,2", "1,", "01", " 1", "1 ", "+1", "-1", "1.5", "1_0",
                     "a", "1;2", "١"):
            with self.subTest(taps=text), self.assertRaisesRegex(ValueError, "tap layers"):
                layers.check_taps(text, 6)
        for value in (None, 1, [1, 2], (1, 2)):
            with self.subTest(taps=value), self.assertRaisesRegex(ValueError, "tap layers"):
                layers.check_taps(value, 6)

    def test_a_run_refuses_tap_layers_out_of_range_out_of_order_or_repeated_before_writing_or_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            contract = write_contract(folder, dict.fromkeys(TOKENIZER_FILES, {"bytes": 1, "sha256": "0" * 64}))
            with mock.patch.object(layers, "verify_inputs") as verify, \
                    mock.patch.object(layers, "original_model") as original_model, \
                    mock.patch.object(layers, "class_model") as class_model:
                for implementation in layers.IMPLEMENTATIONS:
                    for taps in ("0,1", "1,7", "3,2", "2,2", ""):
                        output = Path(folder) / f"{implementation}-{taps or 'none'}"
                        with self.subTest(implementation=implementation, taps=taps):
                            with self.assertRaisesRegex(ValueError, "tap layers"):
                                layers.run(implementation, contract, Path(folder) / "snapshot", 6, taps, output)
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
        widths = {"logits": TEXT["vocab_size"], "hidden_states": TEXT["hidden_size"],
                  "tap_hidden_states": TEXT["hidden_size"] * len(TAPS)}
        self.arrays = {name: {output: generator.normal(size=(length, width)).astype(np.float32)
                              for output, width in widths.items()} for name, (length, _) in layers.INPUTS.items()}

    def pair(self, label, both=None, model_class=None):
        arrays = copied(self.arrays)
        if both:
            both(arrays)
        changed = copied(arrays)
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
        self.assertEqual((result["layers"], result["tap_layers"], result["ratio_limit"], result["contract_sha256"]),
                         (6, list(TAPS), 1e-3, sha256(self.contract)))
        for name in layers.INPUTS:
            outputs = result["inputs"][name]["outputs"]
            self.assertEqual(outputs["hidden_states"], {"largest_ratio": 0.0, "passed": True})
            self.assertEqual(outputs["tap_hidden_states"], {str(tap): {"largest_ratio": 0.0, "passed": True} for tap in TAPS})
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
            arrays["long"]["logits"][5, 6] = np.nan

        def infinity(arrays):
            arrays["short"]["logits"][1, 2] = np.inf

        cases = {"the same NaN in both runs": (nan, None), "the same infinity in both runs": (infinity, None),
                 "a missing row": (None, lambda arrays: arrays["long"].update(logits=arrays["long"]["logits"][:-1])),
                 "another vocabulary": (None, lambda arrays: arrays["short"].update(logits=np.zeros((26, 302), np.float32)))}
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
        for output in layers.OUTPUTS:
            run = self.pair(f"files-{output}")
            with self.subTest(case=f"{output} changed after the record"):
                np.save(run / f"class/long.{output}.npy", self.arrays["long"][output] + 1)
                self.assertFalse(self.compare(run)["passed"])
            with self.subTest(case=f"missing {output}"):
                (run / f"class/long.{output}.npy").unlink()
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
                    layers.run("original", self.contract, self.folder / "snapshot", 6, "1", output)
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
        for output in layers.OUTPUTS:
            for label, link in {"symbolic-link": os.symlink, "hard-link": os.link}.items():
                with self.subTest(link=label, output=output):
                    run = self.pair(f"{label}-{output}")
                    (run / f"class/long.{output}.npy").unlink()
                    link(run / f"original/long.{output}.npy", run / f"class/long.{output}.npy")
                    result = self.compare(run)
                    self.assertFalse(result["passed"])
                    self.assertIn(f"long: both runs read the same {output} file", result["problems"])
        with self.subTest(link="record"):
            run = self.pair("record")
            (run / "class/record.json").unlink()
            os.link(run / "original/record.json", run / "class/record.json")
            self.assertIn("both runs read the same record.json", self.compare(run)["problems"])
        with self.subTest(link="copy"):
            run = self.pair("copy")
            for output in layers.OUTPUTS:
                shutil.copyfile(run / f"original/long.{output}.npy", run / f"class/long.{output}.npy")
            result = self.compare(run)
            self.assertTrue(result["passed"], result["problems"])

    def test_hidden_states_and_each_tap_layer_pass_only_below_a_thousandth_of_the_largest_reference_value_per_position(self):
        width = TEXT["hidden_size"]

        def peak(name, output, row, column, value, rest):
            def change(arrays):
                arrays[name][output][row] = rest
                arrays[name][output][row, column] = value
            return change

        hidden, tap3 = peak("long", "hidden_states", 7, 0, 10.0, 1.0), peak("long", "tap_hidden_states", 9, 2 * width, 10.0, 1.0)
        small = peak("short", "hidden_states", 4, 3, 0.1, 0.05)
        cases = {"hidden states just below": (hidden, add("long", 7, 5, 0.0099, "hidden_states"), "long", set()),
                 "hidden states just above": (hidden, add("long", 7, 5, 0.0101, "hidden_states"), "long",
                                              {"hidden_states"}),
                 "a tap layer just above": (tap3, add("long", 9, 2 * width + 1, 0.0101, "tap_hidden_states"), "long",
                                            {"tap_hidden_states"}),
                 "a position with small values": (small, add("short", 4, 0, 0.000101, "hidden_states"), "short",
                                                  {"hidden_states"}),
                 "a difference equal to the limit": (peak("long", "hidden_states", 11, 0, 1000.0, 0.0),
                                                     add("long", 11, 1, 1.0, "hidden_states"), "long", {"hidden_states"}),
                 "a reference position of zeros": (peak("short", "tap_hidden_states", 2, 0, 0.0, 0.0), None, "short",
                                                   {"tap_hidden_states"})}
        for label, (both, model_class, name, failing) in cases.items():
            with self.subTest(case=label):
                result = self.compare(self.pair(label.replace(" ", "-"), both, model_class))
                self.assertEqual(result["problems"], [])
                self.assertEqual(result["passed"], not failing)
                checks = result["inputs"][name]["checks"]
                self.assertEqual({key for key, passed in checks.items() if not passed}, failing)
        result = self.compare(self.pair("tap-3", tap3, add("long", 9, 2 * width + 1, 0.0101, "tap_hidden_states")))
        taps = result["inputs"]["long"]["outputs"]["tap_hidden_states"]
        self.assertEqual({tap for tap, entry in taps.items() if not entry["passed"]}, {"3"})
        self.assertGreaterEqual(taps["3"]["largest_ratio"], 1e-3)

    def test_the_result_names_the_contract_and_the_layers_only_when_both_runs_agree_with_the_contract_and_each_other(self):
        edits = {"another contract": (lambda record: record["model"].update(contract_sha256="0" * 64), "contract_sha256"),
                 "another model": (lambda record: record["model"].update(id="other/model"), "contract_sha256"),
                 "another layer count": (lambda record: record.update(layers=12, layer_types=TEXT["layer_types"]), "layers")}
        for label, (edit, key) in edits.items():
            run = self.pair(label.replace(" ", "-"))
            edit_record(run / "class", edit)
            result = self.compare(run)
            with self.subTest(case=label):
                self.assertIsNone(result[key])
                self.assertFalse(result["passed"])
                self.assertTrue(result["problems"])

    def test_non_finite_hidden_states_or_tap_layers_never_pass(self):
        for output in ("hidden_states", "tap_hidden_states"):
            def nan(arrays, output=output):
                arrays["short"][output][3, 1] = np.nan

            with self.subTest(output=output):
                result = self.compare(self.pair(f"nan-{output}", nan))
                self.assertFalse(result["passed"])
                self.assertFalse(result["inputs"]["short"]["checks"][output])

    def test_runs_without_the_same_valid_tap_layers_or_with_arrays_of_another_width_never_pass(self):
        edits = {"no tap layers": lambda record: record.pop("tap_layers"),
                 "four tap layers": lambda record: record.update(tap_layers=[1, 2, 3, 4]),
                 "a tap layer zero": lambda record: record.update(tap_layers=[0, 1, 2, 3, 4]),
                 "a tap layer beyond the layers": lambda record: record.update(tap_layers=[1, 2, 3, 4, 7]),
                 "descending tap layers": lambda record: record.update(tap_layers=[5, 4, 3, 2, 1]),
                 "a repeated tap layer": lambda record: record.update(tap_layers=[1, 1, 2, 3, 4]),
                 "a boolean tap layer": lambda record: record.update(tap_layers=[True, 2, 3, 4, 5]),
                 "tap layers as text": lambda record: record.update(tap_layers="1,2,3,4,5")}
        for label, edit in edits.items():
            for both in (False, True):
                with self.subTest(case=label, both=both):
                    run = self.pair(f"{label.replace(' ', '-')}-{both}")
                    for kind in layers.IMPLEMENTATIONS if both else ("class",):
                        edit_record(run / kind, edit)
                    result = self.compare(run)
                    self.assertFalse(result["passed"])
                    self.assertTrue(result["problems"])
        run = self.pair("narrow", None, lambda arrays: arrays["long"].update(
            tap_hidden_states=arrays["long"]["tap_hidden_states"][:, :-TEXT["hidden_size"]]))
        result = self.compare(run)
        self.assertFalse(result["passed"])
        self.assertTrue(result["problems"])


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
                        layers.context(contract, Path(folder) / "snapshot", count, Path(folder) / f"layers-{count}",
                                       "float16")
                    self.assertFalse((Path(folder) / f"layers-{count}").exists())
                for name in ("used", "dangling"):
                    with self.subTest(output=name), self.assertRaises(FileExistsError):
                        layers.context(contract, Path(folder) / "snapshot", 6, Path(folder) / name, "float16")
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
                    layers.context(broken, Path(folder) / "snapshot", 6, output, "float16")
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
                    layers.context(broken, Path(folder) / "snapshot", 6, output, "float16")
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
            record = layers.context(contract, snapshot, 6, Path(folder) / "context", "float16")
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
            taps = (1, 6, 7, 12)
            record = layers.run("original", contract, snapshot, 12, "1,6,7,12", Path(folder) / "all")
            self.assertEqual(record["tap_layers"], list(taps))
            for name in layers.INPUTS:
                ids = layers.tokens(name, config)
                last = []
                hook = reference.model.layers[-1].register_forward_hook(lambda module, args, output: last.append(output))
                with torch.no_grad():
                    output = reference(torch.tensor([ids]), use_cache=False, output_hidden_states=True)
                hook.remove()
                columns = [*output.hidden_states[:-1], last[0]]
                expected = {"logits": output.logits[0], "hidden_states": output.hidden_states[-1][0],
                            "tap_hidden_states": torch.cat([columns[tap][0] for tap in taps], dim=-1)}
                with self.subTest(input=name):
                    for key, value in expected.items():
                        np.testing.assert_array_equal(np.load(Path(folder) / "all" / f"{name}.{key}.npy"), value.numpy())
                        self.assertEqual(record["inputs"][name][key]["shape"], list(value.shape))
                    self.assertEqual(record["inputs"][name]["tap_hidden_states"]["shape"],
                                     [len(ids), TEXT["hidden_size"] * len(taps)])
                    self.assertEqual(record["inputs"][name]["tokens_sha256"], digest(ids))
                    self.assertFalse(np.array_equal(expected["hidden_states"].numpy(), last[0][0].numpy()))
            self.assertEqual(record["embed_scale"], 6.9375)
            first = layers.run("original", contract, snapshot, 6, "1,2,3,4,5", Path(folder) / "first")
            self.assertEqual((first["layer_types"], first["tap_layers"]), (TEXT["layer_types"][:6], [1, 2, 3, 4, 5]))
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
                    layers.run("original", contract, snapshot, 6, "1", Path(folder) / "used")
                shard.write_bytes(data)
                with self.assertRaises(ValueError):
                    layers.run("original", contract, snapshot, 6, "1", Path(folder) / "changed")
            load.assert_not_called()
            self.assertEqual(list((Path(folder) / "changed").iterdir()), [])


def quantized_tensors(text, seed=5):
    generator = torch.Generator().manual_seed(seed)
    parameters = dict(original(text=text).named_parameters())
    tensors = {}
    for name in sorted(export.quantized_names(text["layer_types"][:6])):
        rows, columns = parameters[name].shape
        tensors[f"{name}.quantized_data"] = torch.randint(0, 256, (rows, columns), dtype=torch.uint8, generator=generator)
        tensors[f"{name}.scale"] = (torch.rand(rows, columns // 32, generator=generator) * 0.01 + 1e-3).to(torch.float16)
        tensors[f"{name}.minval"] = (-0.5 * torch.rand(rows, columns // 32, generator=generator)).to(torch.float16)
    return tensors


MAIN_ASSET = b"main asset"


def resync(target):
    record, bundle = read_json(target / "export.json"), target / export.BUNDLE
    files = {path.relative_to(bundle).as_posix(): file_hash(path) for path in sorted(bundle.rglob("*")) if path.is_file()}
    record["bundle"].update(files=files, bytes=sum(entry["bytes"] for entry in files.values()))
    write_json(target / "export.json", record)


def complete_export(folder, contract, precision="fp16", functions=("main", "prefill"), quantized=None, **changes):
    pinned, name, eight_bit = load_contract(contract), export.name_of(6, functions, precision), precision == "8bit"
    config, kinds, compression = pinned["config"], layers.check_layers(pinned["config"], 6), "8bit" if eight_bit else None
    bundle = folder / export.BUNDLE
    (bundle / f"{name}.aimodel").mkdir(parents=True)
    (bundle / f"{name}.aimodel/main.mlirb").write_bytes(MAIN_ASSET)
    (bundle / f"{name}.aimodel/main.hash").write_bytes(hashlib.sha256(MAIN_ASSET).digest())
    write_json(bundle / f"{name}.aimodel/metadata.json", {"assetVersion": "2.0"})
    write_json(bundle / "metadata.json", export.metadata(pinned, name, 16896, compression, TAPS))
    (bundle / "tokenizer").mkdir()
    for item in TOKENIZER_FILES:
        shutil.copyfile(Path(contract).parent / "snapshot" / item, bundle / "tokenizer" / item)
    names, checkpoint = sorted(export.quantized_names(kinds)), 2000
    elements = {"torch.float16": 1000, **({"torch.uint8": 4000} if eight_bit else {})}
    required = export.required_bytes(checkpoint, precision)
    record = {"schema": export.SCHEMA, "passed": True, "problems": [], "checks": {"asset": True, "free_disk": True},
              "model": native.pinned_model(pinned, contract), "layers": 6, "layer_types": kinds,
              "functions": list(functions), "context": 16896, "precision": export.PRECISIONS[precision],
              "compression_config": copy.deepcopy(export.QUANTIZATION) if eight_bit else None,
              "upstream_revision": pinned["upstream_revision"],
              "weights": {"tensors": 85, "elements": sum(elements.values()), "elements_by_dtype": elements,
                          "quantized_layouts": len(names) if eight_bit else 0},
              **({"quantized": quantized} if eight_bit else {}),
              "asset": {"functions": export.expected_functions(config, kinds, functions, TAPS),
                        "output_order": {item: list(export.RETURNS[item]) for item in functions},
                        "storage_types": {"Float16": 1004, "Float32": export.float32_elements(config), "Int32": 1,
                                          "UInt32": 1, "UInt64": 1, **({"UInt8": 4128} if eight_bit else {})},
                        "dequantizations": export.dequantizations(names, kinds, functions, TAPS) if eight_bit else 0,
                        "float16_minus_weights": 4, **({"uint8_minus_weights": 128} if eight_bit else {})},
              "tokenizer": dict(pinned["tokenizer"]["files"]),
              "metadata": export.metadata(pinned, name, 16896, compression, TAPS),
              "bundle": {"folder": export.BUNDLE, "name": name, "main_hash": hashlib.sha256(MAIN_ASSET).hexdigest()},
              "disk": {"weight_bytes": 2 * checkpoint, "estimated_write_bytes": export.WRITES[precision] * checkpoint,
                       "margin_bytes": export.MARGIN_BYTES, "required_free_bytes": required,
                       "free_before_bytes": required, "lowest_free_bytes": export.MARGIN_BYTES},
              "resources": {"swap": {"increase_bytes": 0}}, **changes}
    write_json(folder / "export.json", record)
    resync(folder)
    return folder


def write_export(folder, contract, tensors, **changes):
    folder.mkdir()
    digests = export.save_quantized(folder / layers.QUANTIZED, list(tensors.items()))
    weights = sorted({name.rsplit(".", 1)[0] for name in tensors})
    quantized = {"layers": [{"name": name, "class": "Dequantize", "shape": None, **export.LAYOUT} for name in weights],
                 "skipped": 0, "warnings_enabled": True, "file": file_hash(folder / layers.QUANTIZED), "tensors": digests}
    return complete_export(folder, contract, "8bit", quantized=quantized, **changes)


def named(relative, target):
    return relative.format(name=read_json(target / "export.json")["bundle"]["name"])


def removed(relative):
    def change(target):
        path = target / export.BUNDLE / named(relative, target)
        shutil.rmtree(path) if path.is_dir() else path.unlink()
        resync(target)
    return change


def main_hash_only(target):
    keep = target / export.BUNDLE / named("{name}.aimodel/main.hash", target)
    for path in (target / export.BUNDLE).rglob("*"):
        if path.is_file() and path != keep:
            path.unlink()
    resync(target)


def rewritten(relative, content):
    def change(target):
        (target / export.BUNDLE / named(relative, target)).write_bytes(content)
        resync(target)
    return change


def linked(relative):
    def change(target):
        path = target / named(relative, target)
        moved = target.parent / f"{target.name}-moved-{path.name}"
        path.rename(moved)
        path.symlink_to(moved, target_is_directory=moved.is_dir())
    return change


def renamed(target):
    record, bundle = read_json(target / "export.json"), target / export.BUNDLE
    old = record["bundle"]["name"]
    (bundle / f"{old}.aimodel").rename(bundle / f"{old}_copy.aimodel")
    record["metadata"].update(name=f"{old}_copy", assets={"main": f"{old}_copy.aimodel"})
    record["bundle"]["name"] = f"{old}_copy"
    write_json(bundle / "metadata.json", record["metadata"])
    write_json(target / "export.json", record)
    resync(target)


def changed(change):
    def apply(target):
        record = read_json(target / "export.json")
        change(record)
        write_json(target / "export.json", record)
    return apply


BUNDLE_DEFECTS = {
    "a bundle of main.hash only": main_hash_only,
    "a bundle without main.mlirb": removed("{name}.aimodel/main.mlirb"),
    "a bundle without the tokenizer": removed("tokenizer"),
    "a bundle without its metadata.json": removed("metadata.json"),
    "a bundle with an extra file": rewritten("notes.txt", b"x"),
    "a bundle under a name that its layers, functions and precision do not give": renamed,
    "a tokenizer file of other content under a matching record": rewritten("tokenizer/tokenizer.json", b"{}"),
    "a metadata.json of the bundle of other content under a matching record": rewritten("metadata.json", b'{"name": "x"}'),
    "a main.mlirb that main.hash does not hash under a matching record": rewritten("{name}.aimodel/main.mlirb", b"other"),
    "a bundle size other than the sum of its files": changed(lambda record: record["bundle"].update(
        bytes=record["bundle"]["bytes"] + 1)),
    "an asset of other outputs": changed(lambda record: record["asset"]["output_order"].update(main=["logits"]))}


def interrupted_rename(target):
    (target / export.BUNDLE).rename(target / export.INCOMPLETE)
    write_json(target / "failure.json", {"schema": export.SCHEMA, "passed": False, "error": "OSError"})


def in_blocks_of(tensors, width):
    changed = dict(tensors)
    for key, data in tensors.items():
        if key.endswith(".quantized_data"):
            weight, (rows, columns) = key.removesuffix(".quantized_data"), data.shape
            changed[f"{weight}.scale"] = torch.full((rows, columns // width), 1e-3, dtype=torch.float16)
            changed[f"{weight}.minval"] = torch.zeros(rows, columns // width, dtype=torch.float16)
    return changed


def float16_restore(data, scale, minval):
    blocks = data.shape[1] // scale.shape[1]
    product = data.numpy().astype(np.float16) * np.repeat(scale.numpy(), blocks, axis=1)
    return torch.from_numpy(product + np.repeat(minval.numpy(), blocks, axis=1))


@unittest.skipIf(np is None or torch is None, "NumPy, PyTorch and Transformers are required for the run")
class RestoredReferenceTests(unittest.TestCase):
    TEXT = {**TEXT, "hidden_size": 64}

    def test_restoring_rounds_each_operation_to_float16_like_the_converter(self):
        tensors = quantized_tensors(self.TEXT)
        name = "model.layers.0.mlp.down_proj.weight"
        data, scale, minval = (tensors[f"{name}.{part}"] for part in ("quantized_data", "scale", "minval"))
        restored = layers.restore(data, scale, minval)
        self.assertEqual((restored.dtype, tuple(restored.shape)), (torch.float16, tuple(data.shape)))
        self.assertTrue(torch.equal(restored, float16_restore(data, scale, minval)))
        blocks = data.shape[1] // scale.shape[1]
        once = (data.float() * scale.float().repeat_interleave(blocks, 1) + minval.float().repeat_interleave(blocks, 1))
        self.assertFalse(torch.equal(restored, once.to(torch.float16)))

    def test_the_restored_reference_reads_the_quantized_weights_and_records_the_tensors_that_it_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot, contract = write_snapshot(folder, hidden_size=64)
            tensors = quantized_tensors(self.TEXT)
            source = write_export(Path(folder) / "export", contract, tensors)
            record = layers.context(contract, snapshot, 6, Path(folder) / "restored", "8bit", source)
            self.assertEqual(read_json(Path(folder) / "restored/record.json"), record)
            self.assertEqual(record["weights"], layers.RESTORED)
            self.assertEqual(record["quantized"], {"export_sha256": sha256(source / "export.json"),
                                                   "file": file_hash(source / layers.QUANTIZED), "tensors": 123})
            rounded = layers.context(contract, snapshot, 6, Path(folder) / "rounded", "float16")
            self.assertEqual(rounded["weights"], layers.WEIGHTS)
            self.assertNotIn("quantized", rounded)
            model, _, _ = layers.original_model(snapshot, 6)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if f"{name}.quantized_data" in tensors:
                        parameter.copy_(float16_restore(*(tensors[f"{name}.{part}"] for part in
                                                          ("quantized_data", "scale", "minval"))).float())
                ids = layers.tokens("short", load_contract(contract)["config"])
                expected = model(torch.tensor([ids]), use_cache=False).logits[0].numpy()
            short = np.load(Path(folder) / "restored/short.npy")
            self.assertLess(float(np.abs(short - expected).max()), 1e-3)
            self.assertGreater(float(np.abs(short - np.load(Path(folder) / "rounded/short.npy")).max()), 0.1)

    def test_a_restored_reference_refuses_tensors_or_records_that_differ_from_the_export(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            snapshot, contract = write_snapshot(folder, hidden_size=64)
            tensors = quantized_tensors(self.TEXT)
            name = "model.layers.1.self_attn.k_proj.weight"

            def record_of(target, change):
                record = read_json(target / "export.json")
                change(record)
                write_json(target / "export.json", record)

            def flipped(target, recorded):
                data = bytearray((target / layers.QUANTIZED).read_bytes())
                data[-1] ^= 1
                (target / layers.QUANTIZED).write_bytes(bytes(data))
                if recorded:
                    record_of(target, lambda record: record["quantized"].update(file=file_hash(target / layers.QUANTIZED)))

            cases = {"a tensor hash that differs from the export record": (tensors, lambda target: record_of(
                         target, lambda record: record["quantized"]["tensors"].update({f"{name}.scale": "0" * 64}))),
                     "a file changed after the export": (tensors, lambda target: flipped(target, False)),
                     "a changed tensor under an updated file record": (tensors, lambda target: flipped(target, True)),
                     "the same tensors in another file": (tensors, lambda target: (
                         (target / layers.QUANTIZED).unlink(),
                         export.save_quantized(target / layers.QUANTIZED, list(reversed(tensors.items()))))),
                     "a file with a tensor that the record does not list": (
                         {**tensors, f"{name}.zero_point": torch.zeros(32, 2, dtype=torch.uint8)},
                         lambda target: record_of(target, lambda record: (
                             record["quantized"]["tensors"].pop(f"{name}.zero_point"),
                             record["quantized"].update(file=file_hash(target / layers.QUANTIZED))))),
                     "an extra zero point": ({**tensors, f"{name}.zero_point": torch.zeros(4, 2, dtype=torch.uint8)}, None),
                     "a missing minimum": ({key: value for key, value in tensors.items() if key != f"{name}.minval"},
                                           None),
                     "a weight that the model does not have": (
                         {key.replace("layers.1.", "layers.7.") if key.startswith(f"{name}.") else key: value
                          for key, value in tensors.items()}, None),
                     "a weight of another shape": ({**tensors, f"{name}.quantized_data": tensors[
                         f"{name}.quantized_data"][:, :32].contiguous(), f"{name}.scale": tensors[f"{name}.scale"][:, :1]
                         .contiguous(), f"{name}.minval": tensors[f"{name}.minval"][:, :1].contiguous()}, None),
                     "data of another shape with the scales of the weight": ({**tensors, f"{name}.quantized_data": tensors[
                         f"{name}.quantized_data"][:, :32].contiguous()}, None),
                     "scales of another type": ({**tensors, f"{name}.scale": tensors[f"{name}.scale"].float()}, None),
                     "another precision": (tensors, lambda target: record_of(target, lambda record: record.update(
                         precision="float16"))),
                     "an export that did not pass": (tensors, lambda target: record_of(target, lambda record: record.update(
                         passed=False))),
                     "an export without a recorded weight": (tensors, lambda target: record_of(
                         target, lambda record: record["quantized"]["layers"].pop())),
                     "an export with another recorded weight": (tensors, lambda target: record_of(
                         target, lambda record: record["quantized"]["layers"].append({"name": "embed_tokens.weight"}))),
                     "an export whose weights are not a list": (tensors, lambda target: record_of(
                         target, lambda record: record["quantized"].update(layers={"name": "a"}))),
                     "an export whose quantized record is a list": (tensors, lambda target: record_of(
                         target, lambda record: record.update(quantized=["file"]))),
                     "another layer count": (tensors, lambda target: record_of(target, lambda record: record.update(
                         layers=12))),
                     "another contract": (tensors, lambda target: record_of(target, lambda record: record["model"].update(
                         contract_sha256="0" * 64)))}
            for index, (label, (chosen, change)) in enumerate(cases.items()):
                target = write_export(folder / f"export-{index}", contract, chosen)
                if change:
                    change(target)
                output = folder / f"restored-{index}"
                with self.subTest(case=label), self.assertRaises(ValueError), \
                        mock.patch.object(layers, "original_model", wraps=layers.original_model) as load:
                    layers.context(contract, snapshot, 6, output, "8bit", target)
                load.assert_not_called()
                self.assertFalse(os.path.lexists(output))
            index = snapshot / "model.safetensors.index.json"
            saved = index.read_bytes()
            for label, content in {"a list": [], "a weight map that is a list": {"weight_map": []},
                                   "a shard outside the snapshot": {"weight_map": {
                                       key: "../shard.safetensors" for key in read_json(index)["weight_map"]}}}.items():
                write_json(index, content)
                output = folder / f"index-{label.replace(' ', '-')}"
                with self.subTest(index=label), self.assertRaises(ValueError):
                    layers.context(contract, snapshot, 6, output, "8bit", write_export(
                        folder / f"export-index-{label.replace(' ', '-')}", contract, tensors))
                self.assertFalse(os.path.lexists(output))
            index.write_bytes(saved)
            float16 = complete_export(folder / "float16-export", contract)
            with self.assertRaisesRegex(ValueError, "not an 8-bit export"), \
                    mock.patch.object(layers, "original_model", wraps=layers.original_model) as load:
                layers.context(contract, snapshot, 6, folder / "restored-float16", "8bit", float16)
            load.assert_not_called()
            self.assertFalse(os.path.lexists(folder / "restored-float16"))
            source = write_export(folder / "export", contract, tensors)
            for label, (weights, given) in {"8bit without an export": ("8bit", None),
                                            "float16 with an export": ("float16", source),
                                            "another kind of weights": ("bfloat16", None)}.items():
                with self.subTest(case=label), mock.patch.object(layers, "load_contract") as load, \
                        self.assertRaises(ValueError):
                    layers.context(contract, snapshot, 6, folder / label.replace(" ", "-"), weights, given)
                load.assert_not_called()
                self.assertFalse((folder / label.replace(" ", "-")).exists())

    def test_a_failure_after_the_reference_folder_is_made_leaves_a_record_of_its_type_only(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            snapshot, contract = write_snapshot(folder, hidden_size=64)
            source = write_export(folder / "export", contract, quantized_tensors(self.TEXT))
            cases = {"8bit": ("original_model", source), "float16": ("verify_inputs", None)}
            for error in (RuntimeError("private-example-sentinel"), KeyboardInterrupt()):
                for weights, (target, export_folder) in cases.items():
                    output = folder / f"failed-{weights}-{type(error).__name__}"
                    with self.subTest(weights=weights, error=type(error).__name__), mock.patch.object(
                            layers, target, side_effect=error), self.assertRaises(type(error)):
                        layers.context(contract, snapshot, 6, output, weights, export_folder)
                    self.assertEqual(read_json(output / "failure.json"), {
                        "schema": layers.CONTEXT_SCHEMA, "passed": False, "weights": weights,
                        "error": type(error).__name__})
                    self.assertEqual(sorted(path.name for path in output.iterdir()), ["failure.json"])

    def test_a_restored_reference_refuses_an_export_that_did_not_finish_did_not_pass_or_has_other_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            snapshot, contract = write_snapshot(folder, hidden_size=64)
            tensors = quantized_tensors(self.TEXT)
            name = "model.layers.0.self_attn.q_proj.weight"

            def entry(target, **changes):
                record = read_json(target / "export.json")
                for item in record["quantized"]["layers"]:
                    if item["name"] == name:
                        item.update(changes)
                write_json(target / "export.json", record)

            def edited(change):
                return lambda target: write_json(target / "export.json", change(read_json(target / "export.json")))

            cases = {"the last rename failed": (tensors, interrupted_rename),
                     "a failure record next to the bundle": (tensors, lambda target: write_json(
                         target / "failure.json", {"schema": export.SCHEMA, "passed": False, "error": "OSError"})),
                     "an unfinished bundle next to the bundle": (tensors, lambda target: (
                         target / export.INCOMPLETE).mkdir()),
                     "a bundle file changed after the export": (tensors, lambda target: (
                         target / export.BUNDLE / "metadata.json").write_text("{ }")),
                     "passed as the number one": (tensors, edited(lambda record: {**record, "passed": 1})),
                     "an export with a problem": (tensors, edited(lambda record: {**record, "problems": ["asset"]})),
                     "an export without tap layers": (tensors, edited(lambda record: {**record, "metadata": {}})),
                     "an export of another function layout": (tensors, edited(
                         lambda record: {**record, "functions": ["main"]})),
                     "a weight recorded in blocks of 16": (tensors, lambda target: entry(target, block=[1, 16])),
                     "a weight recorded with a zero point": (tensors, lambda target: entry(
                         target, zero_point="torch.uint8")),
                     "a weight recorded with scales of another type": (tensors, lambda target: entry(
                         target, scale="torch.float32")),
                     "a weight listed twice": (tensors, edited(lambda record: {**record, "quantized": {
                         **record["quantized"],
                         "layers": record["quantized"]["layers"] + record["quantized"]["layers"][:1]}}))}
            for index, (label, (chosen, change)) in enumerate(cases.items()):
                target = write_export(folder / f"export-{index}", contract, chosen)
                if change:
                    change(target)
                output = folder / f"restored-{index}"
                with self.subTest(case=label):
                    with self.assertRaises(ValueError), \
                            mock.patch.object(layers, "original_model", wraps=layers.original_model) as load:
                        layers.context(contract, snapshot, 6, output, "8bit", target)
                    load.assert_not_called()
                    self.assertFalse(os.path.lexists(output))
            record = layers.context(contract, snapshot, 6, folder / "accepted", "8bit",
                                    write_export(folder / "export-accepted", contract, tensors))
            self.assertEqual(record["quantized"]["tensors"], 123)

    def test_a_restored_reference_refuses_an_export_whose_bundle_or_record_is_not_complete_or_not_of_the_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            snapshot, contract = write_snapshot(folder, hidden_size=64)
            tensors = quantized_tensors(self.TEXT)
            for index, (label, change) in enumerate(BUNDLE_DEFECTS.items()):
                target = write_export(folder / f"export-{index}", contract, tensors)
                change(target)
                output = folder / f"restored-{index}"
                with self.subTest(case=label):
                    with self.assertRaises(ValueError), \
                            mock.patch.object(layers, "original_model", wraps=layers.original_model) as load:
                        layers.context(contract, snapshot, 6, output, "8bit", target)
                    load.assert_not_called()
                    self.assertFalse(os.path.lexists(output))

    def test_a_restored_reference_refuses_scales_and_minimums_in_blocks_other_than_32(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            snapshot, contract = write_snapshot(folder, hidden_size=64)
            tensors = quantized_tensors(self.TEXT)
            for width in (16, 64):
                source = write_export(folder / f"export-{width}", contract, in_blocks_of(tensors, width))
                output = folder / f"restored-{width}"
                with self.subTest(width=width):
                    with self.assertRaisesRegex(ValueError, r"blocks of \[1, 32\]"), \
                            mock.patch.object(layers, "original_model", wraps=layers.original_model) as load:
                        layers.context(contract, snapshot, 6, output, "8bit", source)
                    load.assert_not_called()
                    self.assertFalse(os.path.lexists(output))


if __name__ == "__main__":
    unittest.main()
