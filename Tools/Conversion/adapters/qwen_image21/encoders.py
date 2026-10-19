from itertools import chain
from pathlib import Path
from typing import Literal

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextModel,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionModel,
    Qwen3VLVisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
)
from transformers.vision_utils import (
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)


def load_encoder(snapshot: Path, kind: Literal["text", "vision"]):
    if kind not in ("text", "vision"):
        raise ValueError(f"Unknown encoder kind: {kind}")
    directory = snapshot / "text_encoder"
    config = Qwen3VLConfig.from_pretrained(directory, local_files_only=True)
    encoder_config = config.text_config if kind == "text" else config.vision_config
    encoder_config._attn_implementation = "sdpa"
    model_class = Qwen3VLTextModel if kind == "text" else Qwen3VLVisionModel
    prefix = "model.language_model." if kind == "text" else "model.visual."
    with torch.device("meta"):
        model = model_class(encoder_config)
    with safe_open(directory / "model.safetensors", framework="pt", device="cpu") as source:
        state = {
            key.removeprefix(prefix): source.get_tensor(key).to(dtype=torch.float32)
            for key in source.keys()
            if key.startswith(prefix)
        }
    model.load_state_dict(state, strict=True, assign=True)
    with torch.device("cpu"):
        if kind == "text":
            model.rotary_emb = Qwen3VLTextRotaryEmbedding(encoder_config)
        else:
            model.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(encoder_config)
    remaining = [
        name for name, value in chain(model.named_parameters(), model.named_buffers())
        if value.is_meta
    ]
    if remaining:
        raise RuntimeError(f"Encoder contains unmaterialized tensors: {remaining}")
    return model.float().cpu().eval().requires_grad_(False)


class TextEncoder(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.embed_tokens = original.embed_tokens
        self.layers = original.layers

    def forward(self, input_ids, vision, deepstack, cosine, sine, mask, visual_mask):
        hidden = torch.where(visual_mask.unsqueeze(-1), vision, self.embed_tokens(input_ids))
        for index, layer in enumerate(self.layers):
            hidden = layer(
                hidden,
                position_embeddings=(cosine, sine),
                attention_mask=mask,
                use_cache=False,
            )
            if index < 3:
                hidden = hidden + deepstack[index]
        return hidden


class VisionBlock(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.block = original

    def forward(self, hidden, cosine, sine):
        block = self.block
        value = block.norm1(hidden)
        query, key, value = (
            block.attn.qkv(value)
            .reshape(hidden.shape[0], 3, block.attn.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        query, key = apply_rotary_pos_emb_vision(query, key, cosine, sine)
        attended = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            dropout_p=0.0,
            is_causal=False,
            scale=block.attn.scaling,
        )
        attended = attended.squeeze(0).transpose(0, 1).reshape(hidden.shape)
        hidden = hidden + block.attn.proj(attended)
        return hidden + block.mlp(block.norm2(hidden))


class VisionEncoder(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.patch_weight = nn.Parameter(
            original.patch_embed.proj.weight.detach().flatten(1), requires_grad=False
        )
        self.patch_bias = original.patch_embed.proj.bias
        self.blocks = nn.ModuleList(VisionBlock(block) for block in original.blocks)
        self.merger = original.merger
        self.deepstack_mergers = original.deepstack_merger_list
        self.deepstack_indexes = tuple(original.deepstack_visual_indexes)
        if len(self.deepstack_indexes) != 3:
            raise ValueError("The vision encoder requires three deepstack outputs")
        with torch.no_grad():
            grid = torch.tensor([[1, 32, 32]], dtype=torch.int64, device="cpu")
            indices, weights = get_vision_interpolation_indices_and_weights(
                grid,
                num_grid_per_side=original.num_grid_per_side,
                mode=original.interpolation_mode,
                align_corners=original.interpolation_align_corners,
                spatial_merge_size=original.spatial_merge_size,
            )
            position = (original.pos_embed(indices) * weights[:, :, None]).sum(1)
            positions = get_vision_position_ids(grid, original.spatial_merge_size)
            cosine, sine = original.rotary_pos_emb(position, positions)
        self.register_buffer("position", position.detach())
        self.register_buffer("cosine", cosine.detach())
        self.register_buffer("sine", sine.detach())

    def forward(self, pixel_values):
        hidden = F.linear(pixel_values, self.patch_weight, self.patch_bias) + self.position
        deepstack = []
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, self.cosine, self.sine)
            if index in self.deepstack_indexes:
                merger = self.deepstack_mergers[self.deepstack_indexes.index(index)]
                deepstack.append(merger(hidden))
        return self.merger(hidden), deepstack[0], deepstack[1], deepstack[2]
