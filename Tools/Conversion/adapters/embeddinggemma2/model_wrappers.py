import types

import torch
from torch import nn


def materialized_repeat(values, repetitions):
    if repetitions == 1:
        return values
    indices = torch.tensor(
        [index for index in range(values.shape[1]) for _ in range(repetitions)],
        dtype=torch.int64, device=values.device,
    )
    return torch.index_select(values, 1, indices)


def use_materialized_key_values():
    import transformers.models.embedding_gemma2.modeling_embedding_gemma2 as implementation
    implementation.repeat_kv = materialized_repeat


def scaled_norm(self, values):
    inverse = values.abs().amax(-1, keepdim=True).clamp_min(0.001).reciprocal()
    scaled = values * inverse
    variance = (scaled * scaled).mean(-1, keepdim=True) + (self.eps * inverse) * inverse
    return scaled * torch.pow(variance, -0.5)


def use_scaled_norms(module):
    names = []
    for name, child in module.named_modules():
        if type(child).__name__ in ("EmbeddingGemma2RMSNorm", "Gemma4RMSNorm"):
            child._norm = types.MethodType(scaled_norm, child)
            names.append(name)
    return names


class TextTower(nn.Module):
    def __init__(self, model, length=128, safe_normalize=False):
        super().__init__()
        self.text = model.language_model
        self.safe_normalize = safe_normalize
        if length >= self.text.config.sliding_window:
            raise ValueError("This fixed short-sequence wrapper does not cover sliding-window boundaries")
        positions = torch.arange(length).unsqueeze(0)
        dummy = torch.zeros(1, length, self.text.config.hidden_size)
        for kind in set(self.text.config.layer_types):
            cosine, sine = self.text.rotary_emb(dummy, positions, kind)
            self.register_buffer(f"cos_{kind}", cosine.detach())
            self.register_buffer(f"sin_{kind}", sine.detach())
        self.text.embed_tokens = nn.Identity()
        self.text.rotary_emb = nn.Identity()

    def forward(self, inputs_embeds, attention_mask):
        bias = (attention_mask.to(inputs_embeds.dtype) - 1)[:, None, None, :] * 10000
        per_layer = self.text.ple(inputs_embeds)
        hidden = inputs_embeds
        for index, layer in enumerate(self.text.layers):
            kind = self.text.config.layer_types[index]
            position = (getattr(self, f"cos_{kind}"), getattr(self, f"sin_{kind}"))
            hidden = layer(hidden, per_layer[:, :, index, :], attention_mask=bias, position_embeddings=position)
        hidden = self.text.embedding_projection(self.text.norm(hidden)).float()
        mask = attention_mask.float().unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        if self.safe_normalize:
            pooled = pooled * pooled.abs().amax(-1, keepdim=True).clamp_min(0.001).reciprocal()
        return torch.nn.functional.normalize(pooled, dim=-1)


class VisionTower(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision = model.vision_tower
        self.bridge = model.embed_vision
        if self.vision.config.standardize:
            raise ValueError("This wrapper requires the pinned non-standardizing vision configuration")

    def forward(self, pixel_values, image_position_ids, pooling_weights):
        positions = image_position_ids.to(torch.int64)
        padding = (positions == -1).all(-1)
        hidden = self.vision.patch_embedder(pixel_values, positions, padding)
        position_embeddings = self.vision.encoder.rotary_emb(hidden, positions)
        bias = padding.to(hidden.dtype)[:, None, None, :] * -10000
        for layer in self.vision.encoder.layers:
            hidden = layer(hidden, attention_mask=bias, position_embeddings=position_embeddings, position_ids=positions)
        pooled = torch.matmul(pooling_weights.float(), hidden.float()) * self.vision.pooler.root_hidden_size
        return self.bridge(pooled)
