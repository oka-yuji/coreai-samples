import copy
import unittest

import torch

from Tools.Conversion.adapters.qwen_image21.mobile_models import compress_original, configure_original
from Tools.Conversion.adapters.qwen_image21.mobile_profile import make_profile
from Tools.Conversion.adapters.qwen_image21.mobile_reference import component_paths, reference_preflight, selected_runs
from Tools.Conversion.adapters.qwen_image21.mobile_source import independent_component


class ImageMobileSourceTests(unittest.TestCase):
    def profile(self, candidate="w4"):
        return make_profile(candidate, "a" * 64, {"source.py": "b" * 64})

    def test_original_component_reproduces_exported_weights_without_the_export_forward(self):
        torch.manual_seed(216)
        original = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.GELU(), torch.nn.Linear(64, 8))
        export, compression = compress_original(configure_original(copy.deepcopy(original), "denoiser", self.profile()),
                                                "denoiser", self.profile())
        source, observed = independent_component(original, "denoiser", self.profile(), compression)
        self.assertEqual(observed, compression)
        self.assertFalse(any(torch.nn.utils.parametrize.is_parametrized(child) for child in source.modules()))
        values = torch.randn(3, 128).half()
        torch.testing.assert_close(source(values), export(values), atol=0, rtol=0)

    def test_missing_and_different_exported_quantized_weights_cannot_become_reference_b(self):
        for compression in (None, {"kind": "linear-integer", "layers": []}):
            with self.subTest(compression=compression), self.assertRaises(ValueError):
                independent_component(torch.nn.Sequential(torch.nn.Linear(128, 8)), "denoiser", self.profile(), compression)
        with self.assertRaises(ValueError):
            reference_preflight(self.profile(), {})
        self.assertEqual(reference_preflight(self.profile("mixed"), {}), ({}, {}))

    def test_partial_or_duplicate_component_bindings_and_run_selections_are_rejected(self):
        for values in (["text=one", "text=two"], ["unknown=one"], ["text="], ["text"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                component_paths(values)
        with self.assertRaises(ValueError):
            reference_preflight(self.profile("mixed"), component_paths(["text=one"]))
        manifest = {"runs": [{"run_id": "a"}, {"run_id": "b"}]}
        self.assertEqual(selected_runs(manifest, None)[1], ["a", "b"])
        for selection in ("", "a,a", "a,unknown"):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                selected_runs(manifest, selection)


if __name__ == "__main__":
    unittest.main()
