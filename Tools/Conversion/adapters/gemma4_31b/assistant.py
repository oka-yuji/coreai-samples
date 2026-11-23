from types import SimpleNamespace

SLIDING, FULL = "sliding_attention", "full_attention"
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


def checkpoint_shapes(config):
    hidden, inner = config.hidden_size, config.intermediate_size
    shapes = {"model.embed_tokens.weight": [config.vocab_size, hidden], "model.norm.weight": [hidden],
              "pre_projection.weight": [hidden, 2 * config.backbone_hidden_size],
              "post_projection.weight": [config.backbone_hidden_size, hidden]}
    for index, kind in enumerate(config.layer_types):
        width = config.head_dim if kind == SLIDING else config.global_head_dim
        prefix = f"model.layers.{index}."
        shapes.update({f"{prefix}{name}.weight": [hidden] for name in (
            "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm")})
        shapes.update({f"{prefix}layer_scalar": [1], f"{prefix}mlp.gate_proj.weight": [inner, hidden],
                       f"{prefix}mlp.up_proj.weight": [inner, hidden], f"{prefix}mlp.down_proj.weight": [hidden, inner],
                       f"{prefix}self_attn.q_proj.weight": [config.num_attention_heads * width, hidden],
                       f"{prefix}self_attn.q_norm.weight": [width],
                       f"{prefix}self_attn.o_proj.weight": [hidden, config.num_attention_heads * width]})
    return shapes
