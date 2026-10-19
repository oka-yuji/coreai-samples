import importlib.util
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec("nemo"), "The pinned Magpie environment is required")
class MagpieWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from nemo.collections.tts.modules.transformer_2501 import Transformer
        from Tools.Conversion.adapters.magpie.wrappers import Decoder, LocalTransformer, TextEncoder

        cls.torch = torch
        torch.set_num_threads(2)
        torch.manual_seed(12)
        config = {"n_layers": 2, "d_model": 8, "d_ffn": 16, "sa_n_heads": 2,
                  "kernel_size": 1, "p_dropout": 0.0, "p_dropout_out": 0.0,
                  "is_causal": True, "max_length_causal_mask": 32, "use_learnable_pos_emb": True}
        model = SimpleNamespace(
            text_embedding=torch.nn.Embedding(24, 8),
            encoder=Transformer(**config),
            decoder=Transformer(**config, has_xattn=True, xa_d_memory=8, xa_n_heads=1, xa_d_head=4),
            final_proj=torch.nn.Linear(8, 30),
            local_transformer=Transformer(**config),
            local_transformer_out_projections=torch.nn.ModuleList([torch.nn.Linear(8, 7) for _ in range(16)]),
            inference_parameters=SimpleNamespace(apply_prior_to_layers=[0])
        )
        cls.model = model
        cls.encoder = TextEncoder(model).eval().requires_grad_(False)
        cls.decoder = Decoder(model).eval().requires_grad_(False)
        cls.local = LocalTransformer(model).eval().requires_grad_(False)

    def test_encoder_matches_the_original_embedding_and_transformer(self):
        torch = self.torch
        tokens = torch.tensor([[2, 3, 4, 5]], dtype=torch.int32)
        mask = torch.ones(1, 4)
        expected = self.model.encoder(self.model.text_embedding(tokens) * mask.unsqueeze(-1), mask)["output"]
        torch.testing.assert_close(self.encoder(tokens, mask), expected)

    def test_decoder_preserves_last_step_logits_and_each_attention_layer(self):
        torch = self.torch
        hidden, memory = torch.randn(2, 5, 8), torch.randn(2, 4, 8)
        mask, memory_mask = torch.ones(2, 5), torch.ones(2, 4)
        prior = torch.tensor([[[0.1, 0.3, 1.0, 0.2]], [[1.0, 1.0, 1.0, 1.0]]])
        original = self.model.decoder(hidden, mask, memory, memory_mask, attn_prior=[prior, None])
        logits, last, attention = self.decoder(hidden, mask, memory, memory_mask, prior)
        torch.testing.assert_close(last, original["output"][:, -1])
        torch.testing.assert_close(logits, self.model.final_proj(original["output"][:, -1]))
        for index in range(2):
            torch.testing.assert_close(attention[:, index], original["attn_probabilities"][index]["cross_attn_probabilities"][0][:, 0, -1])

    def test_local_transformer_selects_the_requested_codebook(self):
        torch = self.torch
        for length, codebook in [(1, 0), (4, 3), (16, 15)]:
            with self.subTest(length=length, codebook=codebook):
                hidden = torch.randn(2, length, 8)
                original = self.model.local_transformer(hidden, torch.ones(2, length))["output"][:, -1]
                expected = self.model.local_transformer_out_projections[codebook](original)
                torch.testing.assert_close(self.local(hidden, torch.tensor([codebook], dtype=torch.int32)), expected)

    def test_exported_local_graph_accepts_shorter_and_longer_sequences(self):
        torch = self.torch
        exported = torch.export.export(
            self.local, (torch.randn(2, 4, 8), torch.tensor([3], dtype=torch.int32)), strict=False,
            dynamic_shapes=({1: torch.export.Dim("sequence", min=1, max=16)}, {})
        )
        for length, codebook in [(1, 0), (16, 15)]:
            inputs = (torch.randn(2, length, 8), torch.tensor([codebook], dtype=torch.int32))
            torch.testing.assert_close(exported.module()(*inputs), self.local(*inputs))

    def test_core_host_sampling_matches_the_upstream_helper(self):
        import numpy as np
        from nemo.collections.tts.modules.magpietts_modules import LocalTransformerHelper
        from Tools.Conversion.adapters.magpie.synthesize import CoreAIBridge
        from Tools.Conversion.adapters.magpie.wrappers import LocalTransformer

        torch = self.torch
        model = SimpleNamespace(
            local_transformer=self.model.local_transformer,
            local_transformer_in_projection=torch.nn.Identity(),
            local_transformer_audio_out_projection=torch.nn.Identity(),
            audio_in_projection=torch.nn.Identity(),
            audio_embeddings=torch.nn.ModuleList([torch.nn.Embedding(12, 8) for _ in range(16)]),
            local_transformer_out_projections=torch.nn.ModuleList([torch.nn.Linear(8, 12) for _ in range(16)]),
            audio_eos_id=5, codebook_size=4
        )
        helper = LocalTransformerHelper(
            local_transformer=model.local_transformer, audio_embeddings=model.audio_embeddings,
            audio_in_projection=model.audio_in_projection,
            local_transformer_in_projection=model.local_transformer_in_projection,
            local_transformer_audio_out_projection=model.local_transformer_audio_out_projection,
            local_transformer_out_projections=model.local_transformer_out_projections,
            num_audio_codebooks=8, frame_stacking_factor=2, audio_eos_id=5, mask_token_id=8, codebook_size=4
        )
        wrapper = LocalTransformer(model).eval()

        class Components:
            def call(self, kind, values):
                inputs = tuple(torch.from_numpy(np.array(value)) for value in values)
                with torch.no_grad():
                    return (wrapper(*inputs).numpy(),)

        bridge = CoreAIBridge(model, Components())
        hidden = torch.randn(2, 8)
        for extra in ({}, {"unfinished_items": {0: True}}, {"finished_items": {0: True}}):
            with self.subTest(extra=extra), torch.no_grad():
                settings = {"temperature": 0.6, "topk": 4, "use_cfg": True, "cfg_scale": 2.5, "use_kv_cache": False, **extra}
                torch.manual_seed(42)
                expected = helper.sample_autoregressive(hidden, **settings)
                torch.manual_seed(42)
                actual = bridge.local_sample(hidden, **settings)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
