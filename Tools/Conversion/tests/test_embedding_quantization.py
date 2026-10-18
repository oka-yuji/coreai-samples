import json
from pathlib import Path
import unittest

import numpy as np

from Tools.Conversion.adapters.embeddinggemma2.quantization import dequantize_table, quantize_table
from Tools.Conversion.local import recipe_path
from Tools.Conversion.storage import PipelineError


class EmbeddingQuantizationTests(unittest.TestCase):
    def test_signed_nibble_order_and_row_scales_are_preserved(self):
        original = np.array([[7, -7, 0, 1], [-2, 2, 14, -14], [0, 0, 0, 0]], dtype=np.float32)
        packed, scales = quantize_table(original)
        np.testing.assert_array_equal(packed[0], [0x97, 0x10])
        np.testing.assert_array_equal(scales, [1, 2, 1])
        np.testing.assert_array_equal(dequantize_table(packed, scales), original)

    def test_nonfinite_and_malformed_tables_are_rejected(self):
        for value in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                quantize_table(np.array([[0, value]], dtype=np.float32))
        with self.assertRaises(ValueError):
            quantize_table(np.zeros((1, 3)))
        for scales in (np.array([0]), np.array([-1]), np.array([np.nan]), np.ones(2)):
            with self.assertRaises(ValueError):
                dequantize_table(np.zeros((1, 1), dtype=np.uint8), scales)
        with self.assertRaises(ValueError):
            dequantize_table(np.array([[0x18]], dtype=np.uint8), np.ones(1))

    def test_quantized_recipe_uses_matched_references_and_separate_profile(self):
        path = recipe_path('embeddinggemma2', 'int4_int8')
        recipe = json.loads(path.read_text())
        stages = {s['id']: s for s in recipe['steps']}
        for name in ('text_export', 'vision_export', 'bundle', 'verify'):
            self.assertIn('quantized_reference', stages[name]['depends'])
            self.assertIn('$quantized_reference/reference', stages[name]['command'])
        self.assertIn('--original-reference', stages['verify']['command'])
        self.assertEqual(recipe_path('embeddinggemma2').name, 'recipe.json')
        with self.assertRaises(PipelineError):
            recipe_path('qwen3_06b', 'int4_int8')


class QuantizedModuleTests(unittest.TestCase):
    def test_position_table_configuration_keeps_child_projection_quantized(self):
        import torch
        from Tools.Conversion.adapters.embeddinggemma2.quantization import prepare_quantization

        class Patch(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = torch.nn.Linear(8, 8)
                self.position_embedding_table = torch.nn.Parameter(torch.zeros(2, 4, 8))

            def forward(self, values):
                return self.input_proj(values) + self.position_embedding_table[0, 0]

        class Vision(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vision = torch.nn.Module()
                self.vision.patch_embedder = Patch()

            def forward(self, values):
                return self.vision.patch_embedder(values)

        module = Vision().eval().requires_grad_(False)
        quantizer, prepared, layers = prepare_quantization(module, 'vision', (torch.ones(1, 8),))
        self.assertEqual(len(layers), 2)
        patch = prepared.vision.patch_embedder
        self.assertTrue(torch.nn.utils.parametrize.is_parametrized(patch, 'position_embedding_table'))
        self.assertTrue(torch.nn.utils.parametrize.is_parametrized(patch.input_proj, 'weight'))
        reference = prepared(torch.ones(1, 8))
        finalized = quantizer.finalize()
        exported = torch.export.export(finalized, (torch.ones(1, 8),), strict=False)
        torch.testing.assert_close(exported.module()(torch.ones(1, 8)), reference)
