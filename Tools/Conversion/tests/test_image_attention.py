import unittest

try:
    import torch
    from torch.nn import functional as F
except ImportError as error:
    raise unittest.SkipTest(f"Torch unavailable: {error}") from error

from Tools.Conversion.adapters.qwen_image21.denoiser import RealAttention


class AttentionFixture(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.inner_dim = 8
        self.heads = 2
        self.to_q = torch.nn.Linear(8, 8, bias=False, dtype=dtype)
        self.to_k = torch.nn.Linear(8, 8, bias=False, dtype=dtype)
        self.to_v = torch.nn.Linear(8, 8, bias=False, dtype=dtype)
        self.norm_q = self.norm_k = torch.nn.Identity()
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(8, 8, bias=False, dtype=dtype), torch.nn.Identity()])


class ImageAttentionTests(unittest.TestCase):
    def compare(self, length, dtype, mask, tolerance):
        torch.manual_seed(220)
        attention = AttentionFixture(dtype).eval().requires_grad_(False)
        hidden = torch.randn(2, length, 8).to(dtype)
        cosine = torch.ones(1, length, 1, 2)
        sine = torch.zeros_like(cosine)
        with torch.no_grad():
            projected = [layer(hidden).reshape(2, length, 2, 4).transpose(1, 2)
                         for layer in (attention.to_q, attention.to_k, attention.to_v)]
            expected = F.scaled_dot_product_attention(*projected, attn_mask=mask).transpose(1, 2).flatten(2)
            expected = attention.to_out[0](expected)
            actual = RealAttention()(attention, hidden, rotary_emb=(cosine, sine), attention_mask=mask)
        self.assertEqual(actual.shape, hidden.shape)
        self.assertEqual(actual.dtype, dtype)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        return actual

    def test_float32_masks_and_fully_masked_rows_match_the_reference_across_chunk_boundaries(self):
        for length in (1, 257, 513):
            with self.subTest(length=length):
                mask = torch.ones(length, length, dtype=torch.bool).tril()
                mask[-1] = False
                actual = self.compare(length, torch.float32, mask, 1e-6)
                self.assertEqual(torch.count_nonzero(actual[:, -1]).item(), 0)

    def test_broadcast_masks_and_a_partial_final_chunk_match_the_reference(self):
        mask = torch.ones(1, 1, 513, 513, dtype=torch.bool)
        mask[..., 256, :] = False
        actual = self.compare(513, torch.float32, mask, 1e-6)
        self.assertEqual(torch.count_nonzero(actual[:, 256]).item(), 0)

    def test_unmasked_float32_attention_matches_the_reference(self):
        self.compare(513, torch.float32, None, 1e-6)

    def test_single_query_masks_broadcast_over_every_chunk(self):
        for shape in ((1, 513), (1, 1, 1, 513)):
            with self.subTest(shape=shape):
                mask = torch.ones(shape, dtype=torch.bool)
                mask[..., -1] = False
                self.compare(513, torch.float32, mask, 1e-6)
                self.compare(513, torch.float32, torch.zeros_like(mask), 1e-6)

    def test_floating_attention_bias_preserves_the_existing_semantics(self):
        mask = torch.zeros(33, 33)
        mask.masked_fill_(torch.ones(33, 33, dtype=torch.bool).triu(1), -torch.inf)
        self.compare(33, torch.float32, mask, 0)

    def test_low_precision_attention_preserves_the_existing_semantics(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                mask = torch.ones(1, 1, 257, 257, dtype=torch.bool).tril()
                self.compare(257, dtype, mask, 0)


if __name__ == "__main__":
    unittest.main()
