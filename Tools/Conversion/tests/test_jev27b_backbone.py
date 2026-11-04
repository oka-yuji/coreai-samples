import dataclasses
import hashlib
import importlib.util
from types import SimpleNamespace
import unittest

from Tools.Conversion.adapters.jev27b.decision import INPUTS, OUTPUT, STATES

CONVERTER = importlib.util.find_spec("coreai_models") is not None


def composite_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    text = {"vocab_size": 512, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"], "full_attention_interval": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32, "linear_num_key_heads": 2,
            "linear_key_head_dim": 32, "linear_num_value_heads": 4, "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4, "max_position_embeddings": 4096, "tie_word_embeddings": False}
    return Qwen3_5Config(text_config=text)


@unittest.skipUnless(CONVERTER, "The JEV-27B converter environment is required")
class BackboneContractTests(unittest.TestCase):
    def test_the_backbone_exports_main_with_hidden_states_and_four_states(self):
        from Tools.Conversion.adapters.jev27b.backbone import TypedDecisionBackbone

        self.assertIs(TypedDecisionBackbone.exports_prefill_graph, False)
        self.assertEqual(TypedDecisionBackbone._extra_hf_shared_keys, [])
        self.assertEqual(TypedDecisionBackbone.export_output_names(), {"main": (OUTPUT,)})
        self.assertEqual(TypedDecisionBackbone.export_state_names(), {"main": STATES})

    def test_a_composite_config_builds_the_text_model_without_an_lm_head(self):
        from Tools.Conversion.adapters.jev27b.backbone import TypedDecisionBackbone

        config = composite_config()
        model = TypedDecisionBackbone(config=config, model_device="meta")
        self.assertFalse(hasattr(model, "lm_head"))
        self.assertIs(model.config, config.text_config)
        self.assertEqual(model.export_input_names(), {"main": INPUTS})
        self.assertEqual(len(model.model.layers), 4)

    def test_the_load_check_counts_fusions_and_reports_unloaded_parameters(self):
        import torch

        from Tools.Conversion.adapters.jev27b.backbone import TypedDecisionBackbone, load_check

        config = composite_config()
        model = TypedDecisionBackbone(config=config, model_device="meta")
        result = load_check(model, model.config, torch.bfloat16)
        self.assertEqual(result["fused"], {"in_proj_all": 3, "qkv_gate_proj": 1})
        self.assertEqual(result["fused_expected"], result["fused"])
        self.assertIsNone(result["contract_error"])
        self.assertFalse(result["checks"]["all_parameters_loaded"])
        self.assertFalse(result["passed"])
        self.assertTrue(all(value for name, value in result["checks"].items() if name != "all_parameters_loaded"))

    def test_the_flat_entry_reads_unprefixed_keys_and_other_types_pass_through(self):
        from Tools.Conversion.adapters.jev27b.backbone import COMPOSITE, FLAT, TypedDecisionBackbone, entry_for, install

        @dataclasses.dataclass(frozen=True)
        class Entry:
            macos_class: object
            hf_config_attr: object = "text_config"
            hf_state_dict_prefix: str = "model.language_model."

        original = Entry(macos_class=object)
        pipeline = SimpleNamespace(get_model_entry=lambda name: original if name == COMPOSITE else name)
        flat = entry_for(pipeline, FLAT)
        self.assertEqual((flat.macos_class, flat.hf_config_attr, flat.hf_state_dict_prefix), (TypedDecisionBackbone, None, ""))
        self.assertEqual(entry_for(pipeline, COMPOSITE),
                         Entry(TypedDecisionBackbone, "text_config", "model.language_model."))
        install(pipeline, FLAT)
        self.assertEqual(pipeline.get_model_entry(FLAT), flat)
        self.assertIs(pipeline.get_model_entry(COMPOSITE), original)
        self.assertEqual(pipeline.get_model_entry("qwen3"), "qwen3")
        with self.assertRaises(ValueError):
            entry_for(pipeline, "qwen3")

    def test_tensor_digests_cover_raw_bytes_and_skip_meta_tensors(self):
        import torch

        from Tools.Conversion.adapters.jev27b.backbone import digest, tensor_hashes

        values = torch.tensor([1.5, -0.0], dtype=torch.bfloat16)
        self.assertEqual(digest(values), hashlib.sha256(values.view(torch.int16).numpy().tobytes()).hexdigest())
        self.assertNotEqual(digest(values), digest(torch.tensor([1.5, 0.0], dtype=torch.bfloat16)))
        self.assertEqual(digest(torch.zeros(0)), hashlib.sha256(b"").hexdigest())
        self.assertIsNone(digest(torch.zeros(2, device="meta")))
        module = torch.nn.Linear(2, 3)
        module.register_buffer("flag", torch.tensor(True))
        names = [(entry["kind"], entry["name"]) for entry in tensor_hashes(module)["tensors"]]
        self.assertEqual(names, [("parameter", "weight"), ("parameter", "bias"), ("buffer", "flag")])

    def test_zero_states_hold_at_least_the_traced_cache_length(self):
        import torch

        from Tools.Conversion.adapters.jev27b.backbone import zero_states

        text = composite_config().text_config
        shapes = [tuple(state.shape) for state in zero_states(text, torch.float32, 3000, device="meta")]
        self.assertEqual(shapes, [(1, 1, 2, 3000, 32), (1, 1, 2, 3000, 32), (3, 1, 4, 256), (3, 1, 4, 32, 32)])
        self.assertEqual(zero_states(text, torch.float32, 5, device="meta")[0].shape[3], 2048)


if __name__ == "__main__":
    unittest.main()
