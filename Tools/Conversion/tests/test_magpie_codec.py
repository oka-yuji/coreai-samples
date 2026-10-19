import importlib.util
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec("nemo"), "The pinned Magpie environment is required")
class MagpieCodecTests(unittest.TestCase):
    def test_exportable_codec_preserves_audio_and_dynamic_lengths(self):
        import torch
        from nemo.collections.tts.modules.audio_codec_modules import CausalHiFiGANDecoder
        from nemo.core.classes.common import typecheck
        from Tools.Conversion.adapters.magpie.wrappers import CodecDecoder, dynamic_shapes

        torch.set_num_threads(2)
        torch.manual_seed(57)
        typecheck.set_typecheck_enabled(False)
        original = CausalHiFiGANDecoder(
            input_dim=32, up_sample_rates=[8, 8, 4, 2, 2], base_channels=64,
            activation="half_snake", output_activation="clamp", pad_mode="zeros",
            n_groups_equal_to_out_channels=True
        ).eval().requires_grad_(False)
        wrapper = CodecDecoder(SimpleNamespace(_codec_model=SimpleNamespace(audio_decoder=original)))
        self.assertTrue(hasattr(original.pre_conv.conv, "parametrizations"))
        with torch.no_grad():
            example = (torch.randn(1, 32, 4), torch.tensor([4], dtype=torch.int32))
            exported = torch.export.export(wrapper, example, dynamic_shapes=dynamic_shapes("codec"), strict=False)
            for frames, valid in [(2, 2), (7, 5), (32, 32)]:
                with self.subTest(frames=frames, valid=valid):
                    inputs = (torch.randn(1, 32, frames), torch.tensor([valid], dtype=torch.int32))
                    expected = original(*inputs)[0]
                    torch.testing.assert_close(wrapper(*inputs), expected)
                    torch.testing.assert_close(exported.module()(*inputs), expected)

    def test_missing_codec_inference_weights_are_rejected(self):
        import torch
        from Tools.Conversion.adapters.magpie.source import validate_codec_weights

        model = torch.nn.Module()
        for name in ("audio_encoder", "audio_decoder", "vector_quantizer"):
            setattr(model, name, torch.nn.Linear(2, 3))
        state = model.state_dict()
        self.assertTrue(validate_codec_weights(model, state)["strict_inference_weights"])
        state.pop("audio_decoder.weight")
        with self.assertRaises(RuntimeError):
            validate_codec_weights(model, state)


if __name__ == "__main__":
    unittest.main()
