import unittest

try:
    import torch
    import coreai_torch
except ImportError as error:
    raise unittest.SkipTest(f"Qwen image conversion dependencies unavailable: {error}") from error

from Tools.Conversion.adapters.qwen_image21.metal_rotary import Float32Rotary, reference_rotary
from Tools.Conversion.adapters.qwen_image21.mobile_profile import NATIVE_COMPLEX_CANDIDATE, make_profile, validate_mobile_profile


class ImageNativeComplexTests(unittest.TestCase):
    def test_candidate_preserves_every_compression_and_workload_setting(self):
        old = make_profile("w4-dit-g16-blocks16-23-f32", "a" * 64, {"fixture.py": "b" * 64})
        new = make_profile(NATIVE_COMPLEX_CANDIDATE, "a" * 64, {"fixture.py": "b" * 64})
        self.assertNotEqual(old["id"], new["id"])
        self.assertEqual(old["components"], new["components"])
        self.assertEqual(old["workload"], new["workload"])
        self.assertEqual(old["source"], new["source"])
        validate_mobile_profile(new)
        self.assertIsNotNone(Float32Rotary().kernel)
        self.assertIsNone(Float32Rotary(native_complex=True).kernel)

    def test_native_boundary_matches_the_source_and_exports_complex_operations(self):
        generator = torch.Generator().manual_seed(379)
        for length in (1, 17, 2304):
            with self.subTest(length=length):
                value = torch.randn(1, length, 32, 128, generator=generator)
                cosine = torch.randn(1, length, 1, 64, generator=generator)
                sine = torch.randn(1, length, 1, 64, generator=generator)
                model = Float32Rotary(native_complex=True)
                self.assertTrue(torch.equal(model(value, cosine, sine), reference_rotary(value, cosine, sine).reshape(value.shape)))
        sample = (torch.zeros(1, 2, 32, 128), torch.ones(1, 2, 1, 64), torch.zeros(1, 2, 1, 64))
        program = torch.export.export(Float32Rotary(native_complex=True), sample, strict=False)
        operations = {str(node.target) for node in program.graph.nodes if node.op == "call_function"}
        self.assertIn("aten.view_as_complex.default", operations)
        self.assertIn("aten.view_as_real.default", operations)
        self.assertFalse(any("metal" in name for name in operations))

    def test_native_boundary_rejects_dtype_and_shape_changes(self):
        value = torch.zeros(1, 2, 32, 128)
        cosine = torch.ones(1, 2, 1, 64)
        model = Float32Rotary(native_complex=True)
        for inputs in [(value.half(), cosine, cosine), (value[:, :, :31], cosine, cosine),
                       (value, cosine[:, :1], cosine[:, :1])]:
            with self.subTest(shapes=[list(x.shape) for x in inputs]), self.assertRaises(ValueError):
                model(*inputs)


if __name__ == "__main__":
    unittest.main()
