import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from coreai_models._constants import MAIN_GRAPH_NAME
from coreai_models.models.base import TraceSpec
from coreai_models.primitives.macos.sdpa import SDPA
import torch
from torch.export.graph_signature import OutputKind
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

from Tools.Conversion import local
from Tools.Conversion.adapters.gemma4_31b import layers, model
from Tools.Conversion.storage import PipelineError
from Tools.Conversion.tests.test_gemma4_31b_layers import FULL, SLIDING, TEXT, original, write_snapshot

TOLERANCE = 1e-3
LENGTH, PREFILL = 61, 20
IDS = torch.randint(1, TEXT["vocab_size"], (LENGTH,), generator=torch.Generator().manual_seed(7)).tolist()
SHARED = ("model.embed_tokens.weight", "model.norm.weight")
TAPS, SHIFTABLE = (1, 6, 7, 12), (2, 6, 7, 11)
WIDTH = TEXT["hidden_size"]


def chunks(size):
    return [size] * (LENGTH // size) + ([LENGTH % size] if LENGTH % size else [])


READINGS = {"one call": ([LENGTH], 0), "one token at a time": (chunks(1), 0), "chunks of 2": (chunks(2), 0),
            "chunks of 4": (chunks(4), 0), "chunks of 8": (chunks(8), 0),
            "prefill, then one token at a time": ([PREFILL] + [1] * (LENGTH - PREFILL), PREFILL)}


def model_class(reference, taps=TAPS, **changes):
    built = model.Gemma4ForCausalLM(model.text_config(Gemma4TextConfig(**{**TEXT, **changes}))).eval()
    built.load_state_dict({name.removeprefix("model.") if name in SHARED else name: tensor
                           for name, tensor in reference.state_dict().items() if name != "lm_head.weight"})
    built.tap_layers = taps
    return built


def read(built, sizes, prefill):
    states, rows, done = built.states(LENGTH), {name: [] for name in layers.OUTPUTS}, 0
    with torch.no_grad():
        for index, size in enumerate(sizes):
            built.set_prefill_mode(index == 0 and prefill > 0)
            output = built(torch.tensor([IDS[done:done + size]], dtype=torch.int32),
                           torch.arange(done + size, dtype=torch.int32).unsqueeze(0), *states)
            done += size
            names = layers.OUTPUTS[2:] if built.prefill_mode else layers.OUTPUTS
            if len(output) != len(names):
                raise AssertionError(f"The function returned {len(output)} outputs, not {names}")
            for name, tensor in zip(names, output):
                rows[name].append(tensor[0])
    built.set_prefill_mode(False)
    return {name: torch.cat(parts) for name, parts in rows.items()}


def largest_differences(built, expected):
    return {name: float((read(built, sizes, prefill)["logits"] - expected[prefill:]).abs().max())
            for name, (sizes, prefill) in READINGS.items()}


def tapped(columns, taps):
    return torch.cat([columns[tap] for tap in taps], dim=-1)


def matches(found, expected):
    return found.shape == expected.shape and all(
        layers.ratios(found[:, start:start + WIDTH], expected[:, start:start + WIDTH])["passed"]
        for start in range(0, expected.shape[-1], WIDTH))


def replica(future=True, shorter=0, shift=0):
    def window(position_ids, query_len, size):
        seq_len = position_ids.shape[-1]
        positions = position_ids.narrow(-1, seq_len - query_len, query_len) + shift
        length = torch.sym_min(seq_len, size + query_len - 1 - shorter)
        keys = position_ids.narrow(-1, seq_len - length, length)
        distance = positions[0].unsqueeze(-1) - keys[0].unsqueeze(0)
        allowed = distance < size
        return positions, length, allowed & (distance >= 0) if future else allowed
    return window


def unmask_full_attention(built):
    for layer in built.model.layers:
        if not layer.self_attn.sliding:
            layer.self_attn.sdpa = SDPA(scale=1.0, is_causal=False)


MUTATIONS = {"a window one position wider": ({"sliding_window": TEXT["sliding_window"] + 1}, model.window, None),
             "a window one position narrower": ({"sliding_window": TEXT["sliding_window"] - 1}, model.window, None),
             "later positions not masked in the window": ({}, replica(future=False), None),
             "later positions not masked in the full-attention layers": ({}, model.window, unmask_full_attention),
             "a window slice one position shorter": ({}, replica(shorter=1), None),
             "positions shifted by one": ({}, replica(shift=1), None)}


class ModelClassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = original()
        last = []
        hook = cls.reference.model.layers[-1].register_forward_hook(lambda module, args, output: last.append(output))
        with torch.no_grad():
            output = cls.reference(torch.tensor([IDS]), use_cache=False, output_hidden_states=True)
        hook.remove()
        cls.expected, cls.normed = output.logits[0], output.hidden_states[-1][0]
        cls.columns = [state[0] for state in output.hidden_states[:-1]] + [last[0][0]]

    def test_every_reading_matches_the_original_logits(self):
        for name, difference in largest_differences(model_class(self.reference), self.expected).items():
            with self.subTest(reading=name):
                self.assertLess(difference, TOLERANCE)

    def test_every_reading_gives_the_final_norm_output_and_the_outputs_of_the_tap_layers_of_the_original(self):
        built, expected = model_class(self.reference), tapped(self.columns, TAPS)
        self.assertEqual(len(self.columns), TEXT["num_hidden_layers"] + 1)
        for name, (sizes, prefill) in READINGS.items():
            found = read(built, sizes, prefill)
            with self.subTest(reading=name):
                self.assertEqual(found["tap_hidden_states"].shape, (LENGTH, WIDTH * len(TAPS)))
                self.assertEqual(found["hidden_states"].shape, (LENGTH - prefill, WIDTH))
                self.assertTrue(matches(found["hidden_states"], self.normed[prefill:]))
                self.assertTrue(matches(found["tap_hidden_states"], expected))

    def test_the_comparison_detects_shifted_tap_layers_and_the_final_norm_on_the_wrong_side(self):
        expected = tapped(self.columns, SHIFTABLE)
        self.assertTrue(matches(read(model_class(self.reference, taps=SHIFTABLE), [LENGTH], 0)["tap_hidden_states"],
                                expected))
        shifted = {"every tap layer one earlier": tuple(tap - 1 for tap in SHIFTABLE),
                   "every tap layer one later": tuple(tap + 1 for tap in SHIFTABLE),
                   "the last tap layer one later": SHIFTABLE[:-1] + (SHIFTABLE[-1] + 1,)}
        for label, taps in shifted.items():
            with self.subTest(mutation=label):
                found = read(model_class(self.reference, taps=taps), [LENGTH], 0)["tap_hidden_states"]
                self.assertFalse(matches(found, expected))
        built = model_class(self.reference)
        built.norm = torch.nn.Identity()
        before = read(built, [LENGTH], 0)["hidden_states"]
        with self.subTest(mutation="the hidden states before the final norm"):
            self.assertTrue(matches(before, self.columns[-1]))
            self.assertFalse(matches(before, self.normed))
        found = read(model_class(self.reference), [LENGTH], 0)["tap_hidden_states"]
        scalars = [layer.layer_scalar for layer in self.reference.model.layers]
        candidates = {"the last tap layer after the final norm": tapped(self.columns[:-1] + [self.normed], TAPS),
                      "the tap layers before layer_scalar": torch.cat([self.columns[tap] / scalars[tap - 1]
                                                                       for tap in TAPS], dim=-1),
                      "the tap layers in the reverse order": tapped(self.columns, TAPS[::-1])}
        self.assertTrue(matches(found, tapped(self.columns, TAPS)))
        for label, candidate in candidates.items():
            with self.subTest(mutation=label):
                self.assertFalse(matches(found, candidate))

    def test_the_main_graph_returns_the_logits_the_final_norm_output_and_the_taps_and_prefill_only_the_taps(self):
        self.assertEqual(model.Gemma4ForCausalLM.export_output_names(),
                         {MAIN_GRAPH_NAME: ("logits", "hidden_states", "tap_hidden_states")})
        built = model_class(self.reference)
        spec = TraceSpec(max_context_length=64, cache_seq_len=32, query_len=4, offset=2)
        inputs = built.build_reference_inputs(built.config, torch.float32, spec)[MAIN_GRAPH_NAME]
        shapes = built.build_dynamic_shapes(built.config, spec)[MAIN_GRAPH_NAME]
        widths = {}
        for prefill in (False, True):
            built.set_prefill_mode(prefill)
            with torch.no_grad():
                program = torch.export.export(built, args=(), kwargs=inputs, dynamic_shapes=shapes)
            values = next(node for node in program.graph.nodes if node.op == "output").args[0]
            widths[prefill] = [value.meta["val"].shape[-1] for value, output in zip(values, program.graph_signature.output_specs)
                               if output.kind == OutputKind.USER_OUTPUT]
        built.set_prefill_mode(False)
        self.assertEqual(widths, {False: [TEXT["vocab_size"], WIDTH, WIDTH * len(TAPS)], True: [WIDTH * len(TAPS)]})

    def test_the_comparison_detects_each_mutation_of_the_window_and_the_positions(self):
        exact = largest_differences(model_class(self.reference), self.expected)
        with mock.patch.object(model, "window", replica()):
            self.assertEqual(largest_differences(model_class(self.reference), self.expected), exact)
        for name, (changes, window, change) in MUTATIONS.items():
            built = model_class(self.reference, **changes)
            if change:
                change(built)
            with self.subTest(mutation=name), mock.patch.object(model, "window", window):
                self.assertGreaterEqual(max(largest_differences(built, self.expected).values()), TOLERANCE)

    def test_the_states_are_four_key_value_caches_that_follow_the_layer_pattern(self):
        self.assertEqual(model.Gemma4ForCausalLM.export_state_names(),
                         {MAIN_GRAPH_NAME: ("keyCache", "valueCache", "slidingKeyCache", "slidingValueCache")})
        full_shape = (1, TEXT["num_global_key_value_heads"], 5, TEXT["global_head_dim"])
        sliding_shape = (1, TEXT["num_key_value_heads"], 5, TEXT["head_dim"])
        for count in range(1, TEXT["num_hidden_layers"] + 1):
            kinds = TEXT["layer_types"][:count]
            built = model.Gemma4ForCausalLM(model.text_config(Gemma4TextConfig(**TEXT), num_layers=count),
                                            model_device="meta")
            with self.subTest(layers=count):
                self.assertEqual([tuple(state.shape) for state in built.states(5)],
                                 [(kinds.count(FULL), *full_shape)] * 2 + [(kinds.count(SLIDING), *sliding_shape)] * 2)
                self.assertEqual([(layer.self_attn.sliding, layer.self_attn.slot) for layer in built.model.layers],
                                 [(kind == SLIDING, kinds[:index].count(kind)) for index, kind in enumerate(kinds)])

    def test_each_state_keeps_one_dynamic_dimension_in_the_exported_graph(self):
        built = model_class(self.reference)
        spec = TraceSpec(max_context_length=64, cache_seq_len=32, query_len=4, offset=2)
        inputs = built.build_reference_inputs(built.config, torch.float32, spec)
        shapes = built.build_dynamic_shapes(built.config, spec)
        built.validate_export_contract(inputs, shapes)
        with torch.no_grad():
            program = torch.export.export(built, args=(), kwargs=inputs[MAIN_GRAPH_NAME],
                                          dynamic_shapes=shapes[MAIN_GRAPH_NAME]).run_decompositions()
        values = {node.name: node.meta["val"] for node in program.graph.nodes if node.op == "placeholder"}
        mutated = {output.target for output in program.graph_signature.output_specs
                   if output.kind == OutputKind.USER_INPUT_MUTATION}
        self.assertEqual(mutated, set(model.STATE_ARGUMENTS))
        for name in model.STATE_ARGUMENTS:
            with self.subTest(state=name):
                self.assertEqual([index for index, size in enumerate(values[name].shape)
                                  if isinstance(size, torch.SymInt)], [3])

    def test_unsupported_configurations_fail_before_any_weight_is_read(self):
        rope = TEXT["rope_parameters"]
        changes = {"per-layer inputs": {"hidden_size_per_layer_input": 8},
                   "shared key and value layers": {"num_kv_shared_layers": 2},
                   "mixture of experts": {"enable_moe_block": True},
                   "separate values in the full-attention layers": {"attention_k_eq_v": False},
                   "attention bias": {"attention_bias": True},
                   "bidirectional attention": {"use_bidirectional_attention": "all"},
                   "untied embeddings": {"tie_word_embeddings": False},
                   "another activation": {"hidden_activation": "gelu"},
                   "another sliding-window RoPE": {"rope_parameters": {**rope, SLIDING: {**rope[SLIDING],
                                                                                        "rope_type": "linear"}}},
                   "another full-attention RoPE": {"rope_parameters": {**rope, FULL: {**rope[FULL],
                                                                                      "rope_type": "default"}}},
                   "a scaled full-attention RoPE": {"rope_parameters": {**rope, FULL: {**rope[FULL], "factor": 2.0}}},
                   "another attention": {"layer_types": ["chunked_attention"] * TEXT["num_hidden_layers"]},
                   "head dimensions stored per layer": {"global_head_dim": None}}
        for name, change in changes.items():
            with self.subTest(configuration=name), self.assertRaises(ValueError):
                model.text_config(SimpleNamespace(**{**TEXT, **change}))
        with tempfile.TemporaryDirectory() as folder:
            config = {"model_type": "gemma4", "text_config": {**TEXT, "hidden_size_per_layer_input": 8}}
            (Path(folder) / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "Unsupported Gemma 4 configuration"):
                model.Gemma4ForCausalLM.from_hf_memory_efficient(folder, hf_config_attr="text_config",
                                                                 hf_state_dict_prefix=layers.PREFIX)


class RunTests(unittest.TestCase):
    def test_the_model_class_run_matches_the_original_run_and_a_shorter_window_slice_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            snapshot, contract = write_snapshot(folder)
            count, taps = TEXT["num_hidden_layers"], ",".join(map(str, TAPS))
            layers.run("original", contract, snapshot, count, taps, folder / "original")
            record = layers.run("class", contract, snapshot, count, taps, folder / "class")
            self.assertEqual(record["inputs"]["long"]["calls"][:3], [[0, 8], [8, 9], [9, 11]])
            self.assertEqual(record["upstream_revision"], json.loads(contract.read_text())["upstream_revision"])
            result = layers.compare(folder / "original", folder / "class", contract)
            self.assertEqual(result["problems"], [])
            for name in layers.INPUTS:
                with self.subTest(input=name):
                    self.assertLess(result["inputs"][name]["max_abs_logit"], TOLERANCE)
                    self.assertEqual(result["inputs"][name]["top1_mismatches"], 0)
                    checks = result["inputs"][name]["checks"]
                    self.assertTrue(checks["hidden_states"] and checks["tap_hidden_states"])
                    outputs = result["inputs"][name]["outputs"]
                    self.assertLess(outputs["hidden_states"]["largest_ratio"], layers.RATIO)
                    self.assertEqual(set(outputs["tap_hidden_states"]), {str(tap) for tap in TAPS})
                    self.assertLess(max(entry["largest_ratio"] for entry in outputs["tap_hidden_states"].values()),
                                    layers.RATIO)
            with mock.patch.object(model, "window", replica(shorter=1)):
                layers.run("class", contract, snapshot, count, taps, folder / "shorter")
            result = layers.compare(folder / "original", folder / "shorter", contract)
            self.assertEqual(result["problems"], [])
            self.assertGreaterEqual(result["inputs"]["long"]["max_abs_logit"], TOLERANCE)
            self.assertFalse(result["inputs"]["long"]["checks"]["tap_hidden_states"])
            layers.run("class", contract, snapshot, count, "1,6,7,11", folder / "shifted")
            result = layers.compare(folder / "original", folder / "shifted", contract)
            self.assertFalse(result["passed"])
            self.assertIn("the runs differ in tap_layers", result["problems"])

    def test_the_model_class_run_refuses_a_converter_checkout_that_fails_the_check_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            snapshot, contract = write_snapshot(folder)
            with mock.patch.object(local, "check_checkout", side_effect=PipelineError("another revision")), \
                    mock.patch.object(model.Gemma4ForCausalLM, "from_hf_memory_efficient") as load, \
                    self.assertRaises(PipelineError):
                layers.run("class", contract, snapshot, TEXT["num_hidden_layers"], "1", folder / "refused")
            load.assert_not_called()
            self.assertFalse((folder / "refused/record.json").exists())


if __name__ == "__main__":
    unittest.main()
