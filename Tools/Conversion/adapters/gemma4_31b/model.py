from types import SimpleNamespace

import coreai_torch.composite_ops
from coreai_models._constants import (
    KEY_CACHE_NAME, MAIN_GRAPH_NAME, SLIDING_KEY_CACHE_NAME, SLIDING_VALUE_CACHE_NAME, VALUE_CACHE_NAME,
)
from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import DecomposedRoPE, RoPE
from coreai_models.primitives.macos.sdpa import SDPA
import torch
from torch import nn
import torch.nn.functional as F

SLIDING = "sliding_attention"
FULL = "full_attention"
STATES = (KEY_CACHE_NAME, VALUE_CACHE_NAME, SLIDING_KEY_CACHE_NAME, SLIDING_VALUE_CACHE_NAME)
STATE_ARGUMENTS = ("k_cache", "v_cache", "sliding_k_cache", "sliding_v_cache")


def bfloat16(value):
    return float(torch.tensor(value, dtype=torch.float32, device="cpu").to(torch.bfloat16))


def text_config(hf_config, num_layers=None, max_context_length=None):
    def read(name):
        value = getattr(hf_config, name, None)
        if value is None:
            raise ValueError(f"The Gemma 4 text configuration has no {name}")
        return value

    rope = read("rope_parameters")
    sliding_rope, full_rope = rope.get(SLIDING) or {}, rope.get(FULL) or {}
    unsupported = {
        "per-layer inputs": bool(getattr(hf_config, "hidden_size_per_layer_input", 0)),
        "shared key and value layers": bool(getattr(hf_config, "num_kv_shared_layers", 0)),
        "mixture of experts": bool(getattr(hf_config, "enable_moe_block", False)),
        "separate values in the full-attention layers": not read("attention_k_eq_v"),
        "attention bias": bool(getattr(hf_config, "attention_bias", False)),
        "bidirectional attention": getattr(hf_config, "use_bidirectional_attention", None) == "all",
        "untied embeddings": not read("tie_word_embeddings"),
        "another activation": read("hidden_activation") != "gelu_pytorch_tanh",
        "another sliding-window RoPE": sliding_rope.get("rope_type") != "default",
        "another full-attention RoPE": full_rope.get("rope_type") != "proportional"
        or full_rope.get("factor", 1.0) != 1.0,
    }
    if any(unsupported.values()):
        raise ValueError(f"Unsupported Gemma 4 configuration: {[name for name, found in unsupported.items() if found]}")
    layers = read("num_hidden_layers") if num_layers is None else num_layers
    layer_types = list(read("layer_types"))[:layers]
    if len(layer_types) != layers or not set(layer_types) <= {SLIDING, FULL}:
        raise ValueError("The layer types do not cover the layers or name another attention")
    return SimpleNamespace(
        hidden_size=read("hidden_size"), intermediate_size=read("intermediate_size"), vocab_size=read("vocab_size"),
        num_hidden_layers=layers, layer_types=layer_types, num_attention_heads=read("num_attention_heads"),
        num_key_value_heads=read("num_key_value_heads"),
        num_global_key_value_heads=read("num_global_key_value_heads"), head_dim=read("head_dim"),
        global_head_dim=read("global_head_dim"), sliding_window=read("sliding_window"),
        rms_norm_eps=read("rms_norm_eps"), final_logit_softcapping=read("final_logit_softcapping"),
        sliding_rope_theta=float(sliding_rope["rope_theta"]), full_rope_theta=float(full_rope["rope_theta"]),
        partial_rotary_factor=float(full_rope.get("partial_rotary_factor", 1.0)),
        max_position_embeddings=max_context_length or read("max_position_embeddings"))


def proportional_frequencies(head_dim, theta, partial):
    angles = int(partial * head_dim // 2)
    rotated = 1.0 / (theta ** (torch.arange(0, 2 * angles, 2, dtype=torch.int64).to(torch.float32) / head_dim))
    return torch.cat((rotated, torch.zeros(head_dim // 2 - angles, dtype=torch.float32)))


def window(position_ids, query_len, size):
    seq_len = position_ids.shape[-1]
    positions = position_ids.narrow(-1, seq_len - query_len, query_len)
    length = torch.sym_min(seq_len, size + query_len - 1)
    keys = position_ids.narrow(-1, seq_len - length, length)
    distance = positions[0].unsqueeze(-1) - keys[0].unsqueeze(0)
    return positions, length, (distance >= 0) & (distance < size)


class Attention(nn.Module):
    def __init__(self, config, sliding, slot):
        super().__init__()
        self.sliding, self.slot = sliding, slot
        self.heads = config.num_attention_heads
        self.head_dim = config.head_dim if sliding else config.global_head_dim
        self.kv_heads = config.num_key_value_heads if sliding else config.num_global_key_value_heads
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.q_proj = nn.Linear(hidden, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False) if sliding else None
        self.o_proj = nn.Linear(self.heads * self.head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)
        self.v_norm = coreai_torch.composite_ops.RMSNormImpl(eps=eps)
        with torch.device("cpu"):
            self.unit_scale = torch.ones(self.head_dim, dtype=torch.float32)
            self.rope = RoPE(base=config.sliding_rope_theta) if sliding else DecomposedRoPE(
                dims=self.head_dim,
                inv_freq=proportional_frequencies(self.head_dim, config.full_rope_theta, config.partial_rotary_factor))
        self.sdpa = SDPA(scale=1.0, is_causal=not sliding)

    def forward(self, x, positions, offset, query_len, seq_len, cache, length, mask):
        batch = x.shape[0]
        query = self.q_norm(self.q_proj(x).reshape(batch, query_len, self.heads, self.head_dim)).transpose(1, 2)
        projected = self.k_proj(x).reshape(batch, query_len, self.kv_heads, self.head_dim)
        key = self.k_norm(projected).transpose(1, 2)
        source = projected if self.v_proj is None else self.v_proj(x).reshape(batch, query_len, self.kv_heads,
                                                                               self.head_dim)
        value = self.v_norm(source, self.unit_scale.to(device=x.device, dtype=x.dtype)).transpose(1, 2)
        query = self.rope(query, position_ids=positions)
        key = self.rope(key, position_ids=positions)
        key, value = cache.update_and_fetch(self.slot, offset, key, value, seq_len=seq_len, query_len=query_len)
        if self.sliding:
            key, value = key.narrow(-2, seq_len - length, length), value.narrow(-2, seq_len - length, length)
        output = self.sdpa(query, key, value, attn_mask=mask if self.sliding else None)
        return self.o_proj(output.transpose(1, 2).reshape(batch, query_len, self.heads * self.head_dim))


class MLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config, sliding, slot):
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.self_attn = Attention(config, sliding, slot)
        self.mlp = MLP(hidden, config.intermediate_size)
        self.input_layernorm = RMSNorm(hidden, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(hidden, eps=eps)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, x, *attention):
        h = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), *attention))
        h = h + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(h)))
        return h * self.layer_scalar


class Gemma4ForCausalLM(BaseForCausalLM):
    exports_prefill_graph = True

    @classmethod
    def _get_reauthored_config(cls, hf_config, max_context_length=None, num_layers=None):
        return text_config(hf_config, num_layers=num_layers, max_context_length=max_context_length)

    def _init_model(self, config):
        slots, layers = {SLIDING: 0, FULL: 0}, []
        for kind in config.layer_types:
            layers.append(DecoderLayer(config, kind == SLIDING, slots[kind]))
            slots[kind] += 1
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.model = nn.ModuleDict({"layers": nn.ModuleList(layers)})
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_scale = bfloat16(config.hidden_size ** 0.5)

    def forward(self, input_ids, position_ids, k_cache, v_cache, sliding_k_cache, sliding_v_cache):
        query_len, seq_len = input_ids.shape[-1], position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)
        positions, length, mask = window(position_ids, query_len, self.config.sliding_window)
        full, sliding = KVCache(k_cache, v_cache), KVCache(sliding_k_cache, sliding_v_cache)
        weight = self.embed_tokens.weight
        h = self.embed_tokens(input_ids) * torch.tensor(self.embed_scale, dtype=weight.dtype, device=weight.device)
        for layer in self.model.layers:
            cache = sliding if layer.self_attn.sliding else full
            h = layer(h, positions, offset, query_len, seq_len, cache, length, mask)
        if self.prefill_mode:
            return ()
        cap = self.config.final_logit_softcapping
        return torch.tanh(F.linear(self.norm(h), weight) / cap) * cap

    @classmethod
    def export_state_names(cls):
        return {MAIN_GRAPH_NAME: STATES}

    def states(self, seq_len, dtype=torch.float32):
        config, kinds = self.config, self.config.layer_types
        full = (kinds.count(FULL), 1, config.num_global_key_value_heads, seq_len, config.global_head_dim)
        sliding = (kinds.count(SLIDING), 1, config.num_key_value_heads, seq_len, config.head_dim)
        return tuple(torch.zeros(shape, dtype=dtype, device="cpu") for shape in (full, full, sliding, sliding))

    def build_reference_inputs(self, config, target_dtype, spec):
        input_ids = torch.randint(1, config.vocab_size, (1, spec.query_len), dtype=torch.int32)
        position_ids = torch.arange(spec.query_len + spec.offset, dtype=torch.int32).unsqueeze(0)
        states = self.states(spec.cache_seq_len, target_dtype)
        return {MAIN_GRAPH_NAME: {"input_ids": input_ids, "position_ids": position_ids,
                                  **dict(zip(STATE_ARGUMENTS, states))}}

    def build_dynamic_shapes(self, config, spec):
        maximum = spec.max_context_length
        shapes = {"input_ids": {1: torch.export.Dim("seq_ids", max=maximum - 2)},
                  "position_ids": {1: torch.export.Dim("seq_pos", min=spec.query_len, max=maximum - 1)}}
        for name in STATE_ARGUMENTS:
            shapes[name] = None if spec.caches_are_static else {
                KVCache.seq_len_dim(): torch.export.Dim(f"{name}_len", min=spec.cache_seq_len, max=maximum)}
        return {MAIN_GRAPH_NAME: shapes}

    def _mutate_state_dict(self, state_dict):
        for key in [key for key in state_dict if key.startswith("layers.")]:
            state_dict["model." + key] = state_dict.pop(key)
