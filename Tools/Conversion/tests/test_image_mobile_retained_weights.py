import copy
import unittest

try:
    import diffusers
    import transformers.vision_utils
except ImportError as error:
    raise unittest.SkipTest(f"Qwen image dependencies unavailable: {error}") from error

import torch

from Tools.Conversion.adapters.qwen_image21.mobile_models import compress_original, configure_original
from Tools.Conversion.adapters.qwen_image21.mobile_profile import make_profile
from Tools.Conversion.adapters.qwen_image21.mobile_source import independent_component
from Tools.Conversion.adapters.qwen_image21.quantization import plain_reference, quantize, store_linear_weights_in_half


class RetainedWeightModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.common = torch.nn.Linear(128, 128)
        self.transformer_blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(32)])
        for index in (20, 21):
            block = self.transformer_blocks[index]
            block.attn = torch.nn.Module()
            for name in ("to_q", "to_k", "to_v"):
                setattr(block.attn, name, torch.nn.Linear(128, 128))
            block.attn.to_out = torch.nn.ModuleList([torch.nn.Linear(128, 128)])
            block.img_mlp = torch.nn.Module()
            for name in ("proj", "out", "gate_layer"):
                setattr(block.img_mlp, name, torch.nn.Linear(128, 128))

    def forward(self, value):
        return self.transformer_blocks[20].attn.to_q(self.common(value))


class ImageMobileRetainedWeightTests(unittest.TestCase):
    def test_retained_modules_keep_source_half_weights_and_reference_matches_export(self):
        torch.manual_seed(216)
        original = RetainedWeightModel()
        profile = make_profile("w4-blocks20-21-f32", "a" * 64, {"source.py": "b" * 64})
        exported, records = compress_original(configure_original(copy.deepcopy(original), "denoiser", profile), "denoiser", profile)
        self.assertEqual([row["name"] for row in records["layers"]], ["common.linear"])
        retained = profile["components"]["denoiser"]["weight_encoding"]["dense_modules"]
        self.assertEqual([row["name"] for row in records["dense_layers"]], retained)
        for name in retained:
            layer = exported.get_submodule(name)
            self.assertFalse(torch.nn.utils.parametrize.is_parametrized(layer))
            torch.testing.assert_close(layer.weight, original.get_submodule(name.removesuffix(".linear")).weight.half(), rtol=0, atol=0)
        source, observed = independent_component(copy.deepcopy(original), "denoiser", profile, records)
        self.assertEqual(observed, records)
        self.assertFalse(any(torch.nn.utils.parametrize.is_parametrized(layer) for layer in source.modules()))
        values = torch.randn(7, 128)
        torch.testing.assert_close(source(values), exported(values), rtol=0, atol=0)

    def test_missing_duplicate_changed_or_unrecorded_retained_weights_are_rejected(self):
        model = store_linear_weights_in_half(torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.Linear(64, 8)))
        exported, records = quantize(model, 4, dense_modules=["0.linear"])
        for change in (
            lambda value: value.pop("dense_layers"),
            lambda value: value["dense_layers"].append(copy.deepcopy(value["dense_layers"][0])),
            lambda value: value["dense_layers"][0]["weight"].update(sha256="0" * 64),
            lambda value: value["dense_layers"][0].update(bias=None),
        ):
            mutated = copy.deepcopy(records)
            change(mutated)
            with self.assertRaises(ValueError):
                plain_reference(exported, mutated)
        with torch.no_grad():
            exported[0].linear.weight[0, 0] += 1
        with self.assertRaises(ValueError):
            plain_reference(exported, records)

    def test_invalid_retained_module_sets_fail_before_compression(self):
        for names in (["missing.linear"], ["0.linear", "0.linear"], ["0.linear", "1.linear"], [True], "0.linear"):
            model = store_linear_weights_in_half(torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.Linear(64, 8)))
            with self.subTest(names=names), self.assertRaises(ValueError):
                quantize(model, 4, dense_modules=names)


if __name__ == "__main__":
    unittest.main()
