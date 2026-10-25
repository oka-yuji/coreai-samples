import torch
from torch import nn
import torch.nn.functional as F

from .contracts import WINDOW_PATCHES


def rotate_half(values):
    half = values.shape[-1] // 2
    return torch.cat((-values[..., half:], values[..., :half]), dim=-1)


def rms_norm(norm, values):
    inverse = values.abs().amax(-1, keepdim=True).clamp_min(0.001).reciprocal()
    scaled = values * inverse
    variance = (scaled * scaled).mean(-1, keepdim=True) + (norm.variance_epsilon * inverse) * inverse
    return norm.weight * (scaled * torch.rsqrt(variance))


class Vision(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.patch = visual.patch_embed.proj
        self.blocks, self.merger = visual.blocks, visual.merger
        self.full = set(visual.fullatt_block_indexes)
        self.heads = visual.blocks[0].attn.num_heads

    def forward(self, pixel_values, key_valid, cos, sin):
        count = pixel_values.shape[0]
        hidden = F.linear(pixel_values, self.patch.weight.reshape(self.patch.weight.shape[0], -1))
        bias = torch.where(key_valid > 0.5, 0.0, float("-inf")).to(hidden.dtype)
        window_bias, full_bias = bias.reshape(-1, 1, 1, WINDOW_PATCHES), bias.reshape(1, 1, 1, -1)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        for index, block in enumerate(self.blocks):
            attention = block.attn
            values = attention.qkv(rms_norm(block.norm1, hidden)).reshape(count, 3, self.heads, -1)
            query, key, value = values.permute(1, 0, 2, 3).unbind(0)
            query = query * cos + rotate_half(query) * sin
            key = key * cos + rotate_half(key) * sin
            if index in self.full:
                query, key, value = (part.transpose(0, 1).unsqueeze(0) for part in (query, key, value))
                output = F.scaled_dot_product_attention(query, key, value, attn_mask=full_bias,
                                                        scale=attention.scaling)[0].transpose(0, 1)
            else:
                query, key, value = (part.reshape(-1, WINDOW_PATCHES, self.heads, part.shape[-1]).transpose(1, 2)
                                     for part in (query, key, value))
                output = F.scaled_dot_product_attention(query, key, value, attn_mask=window_bias,
                                                        scale=attention.scaling).transpose(1, 2)
            hidden = hidden + attention.proj(output.reshape(count, -1))
            hidden = hidden + block.mlp(rms_norm(block.norm2, hidden))
        return self.merger.mlp(rms_norm(self.merger.ln_q, hidden).reshape(-1, self.merger.hidden_size))


class Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        language = model.model.language_model
        config = model.config.text_config
        self.layers, self.norm, self.embed, self.head = language.layers, language.norm, language.embed_tokens, model.lm_head
        self.head_dim, self.kv_heads, self.hidden = config.head_dim, config.num_key_value_heads, config.hidden_size
        self.repeats = config.num_attention_heads // config.num_key_value_heads
        self.vocab = config.vocab_size

    def decode(self, hidden, cos, sin, mask, key_cache=None, value_cache=None):
        batch, count, _ = hidden.shape
        total = mask.shape[-1]
        past = total - count
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        for index, layer in enumerate(self.layers):
            attention = layer.self_attn
            normalized = rms_norm(layer.input_layernorm, hidden)
            query = rms_norm(attention.q_norm, attention.q_proj(normalized).view(batch, count, -1, self.head_dim))
            key = rms_norm(attention.k_norm, attention.k_proj(normalized).view(batch, count, -1, self.head_dim))
            value = attention.v_proj(normalized).view(batch, count, -1, self.head_dim).transpose(1, 2)
            query, key = query.transpose(1, 2), key.transpose(1, 2)
            query = query * cos + rotate_half(query) * sin
            key = key * cos + rotate_half(key) * sin
            if key_cache is not None:
                key_cache[index:index + 1, :, :, past:total, :] = key.unsqueeze(0)
                value_cache[index:index + 1, :, :, past:total, :] = value.unsqueeze(0)
                key, value = key_cache[index, :, :, :total, :], value_cache[index, :, :, :total, :]
            output = F.scaled_dot_product_attention(query, key.repeat_interleave(self.repeats, dim=1),
                                                    value.repeat_interleave(self.repeats, dim=1),
                                                    attn_mask=mask, scale=attention.scaling)
            hidden = hidden + attention.o_proj(output.transpose(1, 2).reshape(batch, count, -1))
            hidden = hidden + layer.mlp(rms_norm(layer.post_attention_layernorm, hidden))
        return self.head(rms_norm(self.norm, hidden[:, -1, :])).reshape(1, self.vocab)


class Prefill(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, inputs_embeds, cos, sin, mask, keyCache, valueCache):
        return self.decoder.decode(inputs_embeds, cos, sin, mask, keyCache, valueCache)


class Main(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, input_ids, cos, sin, mask, keyCache, valueCache):
        return self.decoder.decode(self.decoder.embed(input_ids), cos, sin, mask, keyCache, valueCache)


class StatelessPrefill(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, inputs_embeds, cos, sin, mask):
        return self.decoder.decode(inputs_embeds, cos, sin, mask)


class Embed(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("table", weight.to(torch.bfloat16))

    def forward(self, input_ids):
        return F.embedding(input_ids, self.table).to(torch.float32)
