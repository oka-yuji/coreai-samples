import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from Tools.Conversion.adapters.qwen3_tts.runtime import validate_inputs
from Tools.Conversion.adapters.qwen3_tts.source import canonical_positions, compare_tensor, validate_source
from Tools.Conversion.storage import file_hash
from Tools.Conversion.tests.test_qwen3_tts_profiles import configuration


class QwenTTSContractTests(unittest.TestCase):
    def test_changed_source_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ("a" * 40)
            root.mkdir()
            config = root / "config.json"
            config.write_text(json.dumps(configuration("0b6", 1024)))
            contract = {"model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "revision": root.name, "files": {"config.json": file_hash(config)}}
            self.assertEqual(validate_source(root, contract), root.resolve())
            config.write_text("{}")
            with self.assertRaises(ValueError):
                validate_source(root, contract)

    def test_unmanaged_tokenizer_is_rejected_but_download_metadata_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ("a" * 40)
            root.mkdir()
            config = root / "config.json"
            config.write_text(json.dumps(configuration("0b6", 1024)))
            contract = {"model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "revision": root.name, "files": {"config.json": file_hash(config)}}
            metadata = root / ".cache/huggingface/download/model.metadata"
            metadata.parent.mkdir(parents=True)
            metadata.write_text("download metadata")
            self.assertEqual(validate_source(root, contract), root.resolve())
            (root / "tokenizer.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Unmanaged source file"):
                validate_source(root, contract)

    def test_predictor_head_must_follow_its_prefix(self):
        for length, head in [(2, 1), (17, 15), (3, -1)]:
            with self.subTest(length=length, head=head), self.assertRaises(ValueError):
                validate_inputs("predictor", (np.zeros((1, length, 1024), np.float32), np.array([head], np.int32)))

    def test_codec_rejects_special_tokens_and_wrong_dtype(self):
        for value, dtype in [(2048, np.int32), (-1, np.int32), (0, np.int64)]:
            with self.subTest(value=value, dtype=dtype), self.assertRaises(ValueError):
                validate_inputs("codec", (np.full((1, 16, 4), value, dtype=dtype),))

    def test_talker_rejects_mismatched_position_axes(self):
        with self.assertRaises(ValueError):
            validate_inputs("talker", (np.zeros((1, 5, 1024), np.float32), np.zeros((1, 1, 5), np.int32)))

    def test_non_finite_candidates_cannot_pass(self):
        self.assertFalse(compare_tensor([1.0], [np.nan])["passed"])
        with self.assertRaises(ValueError):
            compare_tensor([np.nan], [1.0])

    def test_float_positions_are_normalized_without_rounding(self):
        positions = np.array([0.0, 255.0, 256.0, 511.0], dtype=np.float32)
        result = canonical_positions(positions)
        self.assertEqual(result.dtype, np.int32)
        np.testing.assert_array_equal(result, [0, 255, 256, 511])
        for invalid in ([0.5], [-1], [32768], [np.nan]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                canonical_positions(invalid)


@unittest.skipUnless(importlib.util.find_spec("qwen_tts"), "The pinned Qwen TTS environment is required")
class QwenTTSWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration

        cls.torch = torch
        torch.set_num_threads(2)
        torch.manual_seed(17)
        predictor = {"hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 2,
                     "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
                     "vocab_size": 16, "max_position_embeddings": 512, "num_code_groups": 16,
                     "layer_types": ["full_attention", "full_attention"]}
        config = Qwen3TTSTalkerConfig(
            hidden_size=8, intermediate_size=16, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=4,
            vocab_size=32, text_hidden_size=8, text_vocab_size=64, max_position_embeddings=512,
            num_code_groups=16, codec_eos_token_id=31, code_predictor_config=predictor,
            rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 0], "interleaved": True}
        )
        config._attn_implementation = "eager"
        config.code_predictor_config._attn_implementation = "eager"
        cls.talker = Qwen3TTSTalkerForConditionalGeneration(config).eval().requires_grad_(False)

    def test_exported_talker_preserves_length_and_position_boundaries(self):
        from Tools.Conversion.adapters.qwen3_tts.inputs import prepare_inputs
        from Tools.Conversion.adapters.qwen3_tts.wrappers import Talker, dynamic_shapes

        torch = self.torch
        module = Talker(SimpleNamespace(talker=self.talker)).eval()
        def padded(inputs):
            return tuple(torch.from_numpy(value) for value in prepare_inputs("talker", tuple(value.numpy() for value in inputs)))

        example = padded((torch.randn(1, 4, 8), torch.arange(4, dtype=torch.int32).view(1, 1, -1).expand(3, 1, -1)))
        with torch.no_grad():
            exported = torch.export.export(module, example, dynamic_shapes=dynamic_shapes("talker"), strict=False)
            for length, offset in [(1, 511), (4, 255), (256, 0), (512, 0), (1, 0)]:
                with self.subTest(length=length, offset=offset):
                    inputs = (torch.randn(1, length, 8), (torch.arange(length, dtype=torch.int32) + offset).view(1, 1, -1).expand(3, 1, -1))
                    torch.testing.assert_close(exported.module()(*padded(inputs)), module(*inputs))

    def test_exported_predictor_preserves_first_and_last_heads(self):
        from Tools.Conversion.adapters.qwen3_tts.inputs import prepare_inputs
        from Tools.Conversion.adapters.qwen3_tts.wrappers import CodePredictor, dynamic_shapes

        torch = self.torch
        module = CodePredictor(SimpleNamespace(talker=self.talker)).eval()
        def padded(inputs):
            return tuple(torch.from_numpy(value) for value in prepare_inputs("predictor", tuple(value.numpy() for value in inputs)))

        with torch.no_grad():
            exported = torch.export.export(module, padded((torch.randn(1, 4, 8), torch.tensor([2], dtype=torch.int32))),
                                            dynamic_shapes=dynamic_shapes("predictor"), strict=False)
            for length, head in [(2, 0), (9, 7), (16, 14), (2, 0)]:
                inputs = (torch.randn(1, length, 8), torch.tensor([head], dtype=torch.int32))
                torch.testing.assert_close(exported.module()(*padded(inputs)), module(*inputs))

    def test_causal_codec_padding_preserves_variable_lengths(self):
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2CausalConvNet
        from Tools.Conversion.adapters.qwen3_tts.codec import CausalConvolution

        torch = self.torch
        original = Qwen3TTSTokenizerV2CausalConvNet(3, 4, 7, dilation=3).eval().requires_grad_(False)
        module = CausalConvolution(original).eval()
        with torch.no_grad():
            exported = torch.export.export(module, (torch.randn(1, 3, 4),), strict=False,
                                            dynamic_shapes=({2: torch.export.Dim("frames", min=1, max=256)},))
            for length in (1, 3, 25):
                inputs = torch.randn(1, 3, length)
                torch.testing.assert_close(module(inputs), original(inputs), rtol=0, atol=0)
                torch.testing.assert_close(exported.module()(inputs), original(inputs), rtol=0, atol=0)

    def test_full_codec_export_preserves_the_sliding_attention_boundary(self):
        from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2DecoderConfig
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Decoder
        from Tools.Conversion.adapters.qwen3_tts.wrappers import CodecDecoder, dynamic_shapes

        torch = self.torch
        torch.manual_seed(23)
        config = Qwen3TTSTokenizerV2DecoderConfig(
            codebook_size=16, codebook_dim=8, hidden_size=8, latent_dim=8,
            num_attention_heads=2, num_key_value_heads=1, head_dim=4, intermediate_size=16,
            num_hidden_layers=2, num_quantizers=16, decoder_dim=32,
            upsample_rates=[8, 5, 4, 3], upsampling_ratios=[2, 2], sliding_window=72
        )
        config._attn_implementation = "eager"
        original = Qwen3TTSTokenizerV2Decoder(config).eval().requires_grad_(False)
        for name, parameter in original.named_parameters():
            if name.endswith("embedding_sum"):
                parameter.copy_(torch.randn_like(parameter))
        wrapper = CodecDecoder(SimpleNamespace(speech_tokenizer=SimpleNamespace(model=SimpleNamespace(decoder=original)))).eval()
        with torch.no_grad():
            example = (torch.randint(0, 16, (1, 16, 4), dtype=torch.int32),)
            exported = torch.export.export(wrapper, example, dynamic_shapes=dynamic_shapes("codec"), strict=False)
            for frames in (1, 3, 73):
                with self.subTest(frames=frames):
                    codes = torch.randint(0, 16, (1, 16, frames), dtype=torch.int32)
                    expected = original(codes)
                    torch.testing.assert_close(wrapper(codes), expected, rtol=0, atol=0)
                    torch.testing.assert_close(exported.module()(codes), expected, rtol=0, atol=0)

    def test_host_generation_matches_native_sampling_order_and_eos(self):
        self.assert_host_generation(self.talker)

    def test_distinct_predictor_width_preserves_native_generation(self):
        import copy
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration

        config = copy.deepcopy(self.talker.config)
        config.code_predictor_config.hidden_size = 12
        model = Qwen3TTSTalkerForConditionalGeneration(config).eval().requires_grad_(False)
        model.code_predictor.small_to_mtp_projection.bias.fill_(0.4)
        self.assert_host_generation(model)

    def assert_host_generation(self, source_talker):
        import copy
        from Tools.Conversion.adapters.qwen3_tts.inputs import prepare_inputs
        from Tools.Conversion.adapters.qwen3_tts.synthesis import CoreAIBridge
        from Tools.Conversion.adapters.qwen3_tts.wrappers import CodePredictor, Talker

        torch = self.torch
        talker = copy.deepcopy(source_talker)
        model = SimpleNamespace(talker=talker)
        modules = {"talker": Talker(model).eval(), "predictor": CodePredictor(model).eval()}

        class Components:
            def call(self, kind, values):
                with torch.no_grad():
                    result = modules[kind](*(torch.from_numpy(np.array(value)) for value in prepare_inputs(kind, values)))
                if not isinstance(result, tuple):
                    result = (result,)
                return tuple(value.detach().numpy() for value in result)

        bridge = CoreAIBridge(SimpleNamespace(model=model), Components())
        hidden, trailing, pad = torch.randn(1, 3, 8), torch.randn(1, 2, 8), torch.randn(1, 1, 8)
        for sample, eos_bias in [(False, None), (True, None), (True, 100.0), (False, -100.0)]:
            with self.subTest(sample=sample, eos_bias=eos_bias), torch.no_grad():
                if eos_bias is not None:
                    head = torch.nn.Linear(8, 32, bias=True)
                    head.weight.zero_()
                    head.bias.zero_()
                    head.bias[31] = eos_bias
                    talker.codec_head = head
                    modules["talker"] = Talker(model).eval()
                settings = {"max_new_tokens": 6, "min_new_tokens": 2, "do_sample": sample,
                            "top_k": 5, "top_p": 0.9, "temperature": 0.9, "repetition_penalty": 1.05,
                            "subtalker_dosample": sample, "subtalker_top_k": 5,
                            "subtalker_top_p": 0.9, "subtalker_temperature": 0.9,
                            "eos_token_id": 31, "suppress_tokens": list(range(16, 31)),
                            "output_hidden_states": True, "return_dict_in_generate": True}
                inputs = {"inputs_embeds": hidden, "attention_mask": torch.ones(1, 3, dtype=torch.long),
                          "trailing_text_hidden": trailing, "tts_pad_embed": pad}
                torch.manual_seed(42)
                expected = talker.generate(**inputs, **settings)
                torch.manual_seed(42)
                actual = bridge.generate(**inputs, **settings)
                torch.testing.assert_close(actual.sequences, expected.sequences, rtol=0, atol=0)
                wanted_frames = torch.stack([item[1] for item in expected.hidden_states if item[1] is not None], dim=1)
                actual_frames = torch.stack([item[1] for item in actual.hidden_states if item[1] is not None], dim=1)
                torch.testing.assert_close(actual_frames, wanted_frames, rtol=0, atol=0)
                self.assertEqual(actual_frames.shape[1], actual.sequences.shape[1] - 1)
                if eos_bias == 100.0:
                    self.assertEqual(actual.sequences.shape[1], 3)
                elif eos_bias == -100.0:
                    self.assertEqual(actual.sequences.shape[1], 6)


if __name__ == "__main__":
    unittest.main()
