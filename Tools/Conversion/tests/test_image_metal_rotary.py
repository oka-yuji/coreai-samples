import unittest

try:
    import torch
    from coreai_torch import TorchConverter, get_decomp_table
except ImportError as error:
    raise unittest.SkipTest(f"Core AI authoring unavailable: {error}") from error

from Tools.Conversion.adapters.qwen_image21.metal_rotary import Float32Rotary


class ImageMetalRotaryTests(unittest.TestCase):
    def inputs(self, length=3):
        value = torch.arange(length * 32 * 128, dtype=torch.float32).reshape(1, length, 32, 128) / 128
        cosine = torch.zeros(1, length, 1, 64)
        sine = torch.ones_like(cosine)
        return value, cosine, sine

    def test_quarter_turn_preserves_every_head_and_adjacent_pair(self):
        value, cosine, sine = self.inputs()
        expected = torch.empty_like(value)
        expected[..., 0::2] = -value[..., 1::2]
        expected[..., 1::2] = value[..., 0::2]
        actual = Float32Rotary()(value, cosine, sine)
        self.assertEqual(actual.shape, value.shape)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, expected))

    def test_incompatible_geometry_and_precision_are_rejected_before_kernel_dispatch(self):
        rotary = Float32Rotary()
        value, cosine, sine = self.inputs()
        cases = [(value.half(), cosine, sine), (value, cosine.half(), sine), (value, cosine, sine.half()),
                 (value.repeat(2, 1, 1, 1), cosine, sine), (value[:, :, :16], cosine, sine),
                 (value[..., :64], cosine, sine), (value[:, :0], cosine[:, :0], sine[:, :0]),
                 (value, cosine[:, :1], sine), (value, cosine, sine[..., :32])]
        for inputs in cases:
            with self.subTest(shapes=[tuple(item.shape) for item in inputs]), self.assertRaises(ValueError):
                rotary(*inputs)

    def test_public_export_keeps_the_metal_fma_and_supports_multiple_instances(self):
        for length in (2, 5):
            with self.subTest(length=length):
                rotary = Float32Rotary()
                inputs = self.inputs(length)
                program = torch.export.export(rotary, inputs, strict=False).run_decompositions(get_decomp_table())
                converter = TorchConverter()
                converter.register_custom_kernels([rotary.kernel])
                converter.add_exported_program(program, input_names=["value", "cosine", "sine"],
                                               output_names=["rotated"], state_names=[], entrypoint_name="main")
                intermediate = str(converter.to_coreai())
                self.assertIn("metal4_kernel", intermediate)
                self.assertIn("fma(real, co, -(imag * si))", intermediate)
                self.assertIn("fma(real, si, imag * co)", intermediate)
                self.assertEqual(tuple(program.module()(*inputs).shape), (1, length, 32, 128))


if __name__ == "__main__":
    unittest.main()
