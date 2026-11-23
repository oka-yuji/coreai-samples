from types import SimpleNamespace

from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import DecomposedRoPE, RoPE
from coreai_models.primitives.macos.sdpa import SDPA
import torch
from torch import nn
import torch.nn.functional as F

from .model import FULL, MLP, SLIDING, bfloat16, proportional_frequencies

INPUTS = ("input_ids", "hidden_states", "position_ids", "k_sliding", "v_sliding", "k_full", "v_full")
OUTPUT = "draft_tokens"
TARGET = ("hidden_size", "vocab_size", "num_key_value_heads", "num_global_key_value_heads", "head_dim",
          "global_head_dim", "sliding_window")
ROPE = {SLIDING: {"rope_theta": 10000.0, "rope_type": "default"},
        FULL: {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"}}
SYNTHETIC = {
    "target": {"hidden_size": 48, "vocab_size": 301, "num_key_value_heads": 2, "num_global_key_value_heads": 1,
               "head_dim": 16, "global_head_dim": 32, "sliding_window": 1024},
    "assistant": {"backbone_hidden_size": 48, "use_ordered_embeddings": False, "tie_word_embeddings": True,
                  "text_config": {"vocab_size": 301, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 4,
                                  "layer_types": [SLIDING] * 3 + [FULL], "num_attention_heads": 4,
                                  "num_key_value_heads": 2, "num_global_key_value_heads": 1, "head_dim": 16,
                                  "global_head_dim": 32, "sliding_window": 1024, "rms_norm_eps": 1e-6,
                                  "rope_parameters": ROPE, "num_kv_shared_layers": 4, "hidden_size_per_layer_input": 0,
                                  "vocab_size_per_layer_input": 0, "enable_moe_block": False,
                                  "use_double_wide_mlp": False, "attention_bias": False,
                                  "hidden_activation": "gelu_pytorch_tanh", "attention_k_eq_v": True,
                                  "final_logit_softcapping": None, "tie_word_embeddings": True,
                                  "max_position_embeddings": 32768, "pad_token_id": 0, "bos_token_id": 2,
                                  "eos_token_id": 1}}}


def assistant_config(document, target):
    text = document.get("text_config") if isinstance(document, dict) else None
    if not isinstance(text, dict) or not isinstance(target, dict) or set(target) != set(TARGET):
        raise ValueError("The assistant configuration needs a text_config and the target needs " + ", ".join(TARGET))

    def read(source, name):
        if name not in source or source[name] is None:
            raise ValueError(f"The Gemma 4 assistant configuration has no {name}")
        return source[name]

    layers, kinds = read(text, "num_hidden_layers"), read(text, "layer_types")
    heads, kv, global_kv = (read(text, name) for name in ("num_attention_heads", "num_key_value_heads",
                                                          "num_global_key_value_heads"))
    shared = {"backbone_hidden_size": "hidden_size", **{name: name for name in TARGET if name != "hidden_size"}}
    found = {name: read(document if name == "backbone_hidden_size" else text, name) for name in shared}
    unsupported = {
        "ordered embeddings with centroids": read(document, "use_ordered_embeddings") is not False,
        "untied embeddings": read(document, "tie_word_embeddings") is not True,
        "layers that compute their own keys and values": read(text, "num_kv_shared_layers") != layers,
        "per-layer inputs": read(text, "hidden_size_per_layer_input") != 0,
        "mixture of experts": read(text, "enable_moe_block") is not False,
        "double-wide MLP": read(text, "use_double_wide_mlp") is not False,
        "attention bias": read(text, "attention_bias") is not False,
        "another activation": read(text, "hidden_activation") != "gelu_pytorch_tanh",
        "another RoPE": read(text, "rope_parameters") != ROPE,
        "layer types other than one or more sliding-window layers and full-attention layers":
            type(layers) is not int or not isinstance(kinds, list) or len(kinds) != layers
            or any(kind not in (SLIDING, FULL) for kind in kinds) or SLIDING not in kinds or FULL not in kinds,
        "query heads that are not a multiple of the key and value heads":
            any(type(value) is not int or value <= 0 for value in (heads, kv, global_kv)) or heads % kv or heads % global_kv,
        **{f"another {name} than the target {target_name}": found[name] != target[target_name]
           for name, target_name in shared.items()}}
    if any(unsupported.values()):
        raise ValueError(f"Unsupported Gemma 4 assistant: {[name for name, wrong in unsupported.items() if wrong]}")
    return SimpleNamespace(**found, hidden_size=read(text, "hidden_size"),
                           intermediate_size=read(text, "intermediate_size"), layer_types=list(kinds),
                           num_attention_heads=heads, rms_norm_eps=read(text, "rms_norm_eps"),
                           sliding_rope_theta=ROPE[SLIDING]["rope_theta"], full_rope_theta=ROPE[FULL]["rope_theta"],
                           partial_rotary_factor=ROPE[FULL]["partial_rotary_factor"])


class SharedAttention(nn.Module):
    def __init__(self, config, sliding):
        super().__init__()
        self.sliding, self.heads = sliding, config.num_attention_heads
        self.head_dim = config.head_dim if sliding else config.global_head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.heads * self.head_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(self.heads * self.head_dim, config.hidden_size, bias=False)
        with torch.device("cpu"):
            self.rope = RoPE(base=config.sliding_rope_theta) if sliding else DecomposedRoPE(
                dims=self.head_dim,
                inv_freq=proportional_frequencies(self.head_dim, config.full_rope_theta, config.partial_rotary_factor))
        self.sdpa = SDPA(scale=1.0)

    def forward(self, x, position_ids, key, value):
        query = self.q_norm(self.q_proj(x).reshape(1, 1, self.heads, self.head_dim)).transpose(1, 2)
        output = self.sdpa(self.rope(query, position_ids=position_ids), key, value)
        return self.o_proj(output.transpose(1, 2).reshape(1, 1, self.heads * self.head_dim))


class DraftLayer(nn.Module):
    def __init__(self, config, sliding):
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.self_attn = SharedAttention(config, sliding)
        self.mlp = MLP(hidden, config.intermediate_size)
        self.input_layernorm = RMSNorm(hidden, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(hidden, eps=eps)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, x, position_ids, key, value):
        h = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), position_ids, key, value))
        h = h + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(h)))
        return h * self.layer_scalar


class Gemma4AssistantDrafter(nn.Module):
    def __init__(self, config, steps):
        super().__init__()
        if type(steps) is not int or steps < 1:
            raise ValueError(f"The drafter needs a whole number of steps from 1: {steps!r}")
        self.config, self.steps = config, steps
        backbone, hidden = config.backbone_hidden_size, config.hidden_size
        self.target_embed = nn.Embedding(config.vocab_size, backbone)
        self.pre_projection = nn.Linear(2 * backbone, hidden, bias=False)
        self.post_projection = nn.Linear(hidden, backbone, bias=False)
        self.model = nn.ModuleDict({"embed_tokens": nn.Embedding(config.vocab_size, hidden),
                                    "layers": nn.ModuleList(DraftLayer(config, kind == SLIDING)
                                                            for kind in config.layer_types),
                                    "norm": RMSNorm(hidden, eps=config.rms_norm_eps)})
        self.embed_scale = bfloat16(backbone ** 0.5)

    def check(self, input_ids, hidden_states, position_ids, k_sliding, v_sliding, k_full, v_full):
        config, dtype = self.config, self.pre_projection.weight.dtype
        wrong = [name for name, tensor, shape, kind in (
            ("input_ids", input_ids, (1, 1), torch.int32), ("position_ids", position_ids, (1, 1), torch.int32),
            ("hidden_states", hidden_states, (1, 1, config.backbone_hidden_size), dtype))
            if tuple(tensor.shape) != shape or tensor.dtype != kind]
        for name, key, value, heads, width, largest in (
                ("sliding", k_sliding, v_sliding, config.num_key_value_heads, config.head_dim, config.sliding_window + 1),
                ("full", k_full, v_full, config.num_global_key_value_heads, config.global_head_dim, None)):
            if key.dim() != 5 or tuple(key.shape[:3]) != (1, 1, heads) or key.shape[4] != width \
                    or key.dtype != dtype or value.shape != key.shape or value.dtype != dtype \
                    or key.shape[3] < 1 or largest is not None and key.shape[3] > largest:
                wrong.append(f"{name} keys and values")
        if wrong:
            raise ValueError(f"The drafter inputs differ from the contract: {wrong}")

    def embed(self, input_ids):
        weight = self.target_embed.weight
        return self.target_embed(input_ids) * torch.tensor(self.embed_scale, dtype=weight.dtype, device=weight.device)

    def step(self, embedded, hidden_states, position_ids, k_sliding, v_sliding, k_full, v_full):
        x = self.pre_projection(torch.cat([embedded, hidden_states], dim=-1))
        caches = {True: (k_sliding[0], v_sliding[0]), False: (k_full[0], v_full[0])}
        for layer in self.model.layers:
            x = layer(x, position_ids, *caches[layer.self_attn.sliding])
        x = self.model.norm(x)
        return F.linear(x, self.model.embed_tokens.weight), self.post_projection(x)

    def outputs(self, input_ids, hidden_states, position_ids, k_sliding, v_sliding, k_full, v_full):
        self.check(input_ids, hidden_states, position_ids, k_sliding, v_sliding, k_full, v_full)
        return self.step(self.embed(input_ids), hidden_states, position_ids, k_sliding, v_sliding, k_full, v_full)

    def forward(self, input_ids, hidden_states, position_ids, k_sliding, v_sliding, k_full, v_full):
        self.check(input_ids, hidden_states, position_ids, k_sliding, v_sliding, k_full, v_full)
        token, hidden, drafted = input_ids, hidden_states, []
        for _ in range(self.steps):
            logits, hidden = self.step(self.embed(token), hidden, position_ids, k_sliding, v_sliding, k_full, v_full)
            token = logits.argmax(dim=-1).to(torch.int32)
            drafted.append(token)
        return torch.cat(drafted, dim=-1)


def traced(model, context):
    config, dtype = model.config, model.pre_projection.weight.dtype
    sliding = (1, 1, config.num_key_value_heads, 5, config.head_dim)
    full = (1, 1, config.num_global_key_value_heads, 9, config.global_head_dim)
    inputs = {"input_ids": torch.zeros(1, 1, dtype=torch.int32),
              "hidden_states": torch.zeros(1, 1, config.backbone_hidden_size, dtype=dtype),
              "position_ids": torch.full((1, 1), 9, dtype=torch.int32),
              "k_sliding": torch.zeros(sliding, dtype=dtype), "v_sliding": torch.zeros(sliding, dtype=dtype),
              "k_full": torch.zeros(full, dtype=dtype), "v_full": torch.zeros(full, dtype=dtype)}
    window = torch.export.Dim("sliding", min=1, max=config.sliding_window + 1)
    confirmed = torch.export.Dim("confirmed", min=1, max=context)
    shapes = {"input_ids": None, "hidden_states": None, "position_ids": None, "k_sliding": {3: window},
              "v_sliding": {3: window}, "k_full": {3: confirmed}, "v_full": {3: confirmed}}
    return inputs, shapes


def export(drafters, context):
    import coreai_torch
    from coreai_models.export.externalize import EXTERNALIZE_SPECS
    from coreai_models.export.mlir_ops import register_custom_torch_lowering, remove_functionalization

    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    for model in drafters:
        inputs, shapes = traced(model, context)

        def export_fn(module, inputs=inputs, shapes=shapes):
            with torch.no_grad():
                program = torch.export.export(module, args=(), kwargs=inputs, dynamic_shapes=shapes)
            program = program.run_decompositions(coreai_torch.get_decomp_table())
            remove_functionalization(program)
            return program

        converter.add_pytorch_module(model, export_fn=export_fn, externalize_modules=EXTERNALIZE_SPECS,
                                     input_names=INPUTS, output_names=(OUTPUT,), entrypoint_name=f"draft_{model.steps}")
    register_custom_torch_lowering(converter)
    return converter.to_coreai()


def synthetic(seed, steps):
    model = Gemma4AssistantDrafter(assistant_config(SYNTHETIC["assistant"], SYNTHETIC["target"]), steps).eval()
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            noise = torch.randn(tensor.shape, generator=generator)
            tensor.copy_(1 + 0.3 * noise if name.endswith("norm.weight") else 0.2 * noise)
        for layer in model.model.layers:
            layer.layer_scalar.copy_(torch.rand(1, generator=generator) + 0.5)
    return model
