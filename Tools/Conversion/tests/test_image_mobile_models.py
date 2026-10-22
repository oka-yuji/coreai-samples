import json
from pathlib import Path
import tempfile
import unittest

import torch
from diffusers import QwenImage21Transformer2DModel

from Tools.Conversion.adapters.qwen_image21.mobile_models import VisionBoundary, load_denoiser, parameter_precision


class ImageMobileModelTests(unittest.TestCase):
    def test_vision_rotation_keeps_full_precision_trigonometry_for_low_precision_queries(self):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel, apply_rotary_pos_emb_vision

        config = Qwen3VLVisionConfig(depth=3, hidden_size=32, intermediate_size=64, out_hidden_size=32,
                                    num_heads=4, deepstack_visual_indexes=[0, 1, 2])
        for dtype in (torch.float16, torch.bfloat16):
            model = parameter_precision(Qwen3VLVisionModel(config), dtype)
            boundary = VisionBoundary(model, dtype)
            cosine, sine = boundary.encoder.cosine, boundary.encoder.sine
            self.assertEqual(boundary.encoder.position.dtype, dtype)
            self.assertEqual(cosine.dtype, torch.float32)
            self.assertEqual(sine.dtype, torch.float32)
            torch.manual_seed(216)
            query = torch.randn(1024, 4, 8).to(dtype)
            full, _ = apply_rotary_pos_emb_vision(query, query, cosine, sine)
            rounded, _ = apply_rotary_pos_emb_vision(query, query, cosine.to(dtype), sine.to(dtype))
            self.assertGreater(torch.count_nonzero(full != rounded).item(), 0)

    def test_low_precision_parameters_preserve_the_source_temporal_frequency_buffer(self):
        model = QwenImage21Transformer2DModel(
            num_layers=1, num_attention_heads=1, attention_head_dim=32, context_in_dim=64, axes_dims_rope=(8, 12, 12))
        expected = model.time_text_embed.time_proj.freqs.clone()
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            for dtype in (torch.float16, torch.bfloat16):
                source = QwenImage21Transformer2DModel.from_pretrained(folder, torch_dtype=dtype, local_files_only=True)
                converted = parameter_precision(model, dtype)
                self.assertEqual(next(converted.parameters()).dtype, dtype)
                self.assertEqual(converted.time_text_embed.time_proj.freqs.dtype, torch.float32)
                torch.testing.assert_close(converted.time_text_embed.time_proj.freqs, expected, atol=0, rtol=0)
                torch.testing.assert_close(converted.time_text_embed.time_proj.freqs, source.time_text_embed.time_proj.freqs,
                                           atol=0, rtol=0)

    def test_meta_loading_restores_temporal_buffers_and_unregistered_rotary_tables(self):
        model = QwenImage21Transformer2DModel(
            num_layers=1, num_attention_heads=1, attention_head_dim=32, context_in_dim=64, axes_dims_rope=(8, 12, 12))
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder)
            target = snapshot / "transformer"
            model.save_pretrained(target)
            index = {"weight_map": {name: "diffusion_pytorch_model.safetensors" for name in model.state_dict()}}
            (target / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index))
            restored = load_denoiser(snapshot)
        self.assertTrue(all(not value.is_meta for value in (*restored.parameters(), *restored.buffers(), *restored.pos_embed.freqs)))
        for actual, expected in zip(restored.pos_embed.freqs, model.pos_embed.freqs, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(restored.time_text_embed.time_proj.freqs, model.time_text_embed.time_proj.freqs,
                                   atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
