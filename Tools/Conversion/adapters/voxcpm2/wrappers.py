import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.weight_norm import WeightNorm

KEPT_FP32 = ("MiniCPMRMSNorm", "SinusoidalPosEmb")
ASSETS = ("tslm", "ralm", "proj", "locenc", "dit", "vae")


def scaled(layer, value):
    return value * (layer.scale_depth / math.sqrt(layer.num_hidden_layers)) if layer.use_mup else value


class Cache(nn.Module):
    def __init__(self, lm, capacity):
        super().__init__()
        config = lm.config
        self.layers, self.norm, self.capacity = lm.layers, lm.norm, capacity
        self.heads, self.kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.dim = config.kv_channels or config.hidden_size // config.num_attention_heads
        self.rope = lm.rope_emb is not None
        if self.rope:
            self.register_buffer("cos", lm.rope_emb.cos_cached[:capacity].clone())
            self.register_buffer("sin", lm.rope_emb.sin_cached[:capacity].clone())
        shape = (len(self.layers), self.kv_heads, capacity, self.dim)
        self.register_buffer("k_cache", torch.zeros(shape))
        self.register_buffer("v_cache", torch.zeros(shape))

    def write(self, index, keys, values):
        index = index.expand(len(self.layers), -1, -1, -1)
        self.k_cache.scatter_(2, index, torch.cat(keys))
        self.v_cache.scatter_(2, index, torch.cat(values))


class Prefill(Cache):
    def __init__(self, lm, capacity, length):
        super().__init__(lm, capacity)
        self.length = length

    def forward(self, embeds, length):
        count = self.length
        position = (self.cos[:count], self.sin[:count]) if self.rope else None
        keep = (torch.arange(count) < length).view(1, 1, count, 1).to(embeds.dtype)
        hidden, keys, values = embeds, [], []
        for layer in self.layers:
            hidden, (key, value) = layer(hidden, position, True)
            keys.append(key * keep)
            values.append(value * keep)
        self.write(torch.arange(count).view(1, 1, count, 1).expand(1, self.kv_heads, count, self.dim), keys, values)
        return self.norm(hidden)


class Step(Cache):
    def forward(self, embed, position):
        from voxcpm.modules.minicpm4.model import apply_rotary_pos_emb

        index = position.long()
        slots = torch.arange(self.capacity)
        visible = (slots <= index).view(1, 1, 1, -1)
        current = (slots == index).view(1, 1, -1, 1)
        if self.rope:
            cos, sin = self.cos.index_select(0, index), self.sin.index_select(0, index)
        hidden, keys, values = embed, [], []
        for number, layer in enumerate(self.layers):
            attention = layer.self_attn
            x = layer.input_layernorm(hidden)
            query = attention.q_proj(x).view(1, 1, self.heads, self.dim).transpose(1, 2)
            key = attention.k_proj(x).view(1, 1, self.kv_heads, self.dim).transpose(1, 2)
            value = attention.v_proj(x).view(1, 1, self.kv_heads, self.dim).transpose(1, 2)
            if self.rope:
                query, key = apply_rotary_pos_emb(query, key, cos, sin)
            out = F.scaled_dot_product_attention(query.contiguous(),
                                                 torch.where(current, key, self.k_cache[number:number + 1]),
                                                 torch.where(current, value, self.v_cache[number:number + 1]),
                                                 attn_mask=visible, enable_gqa=True)
            hidden = hidden + scaled(layer, attention.o_proj(out.transpose(1, 2).reshape(1, self.heads * self.dim)))
            hidden = hidden + scaled(layer, layer.mlp(layer.post_attention_layernorm(hidden)))
            keys.append(key)
            values.append(value)
        self.write(index.view(1, 1, 1, 1).expand(1, self.kv_heads, 1, self.dim), keys, values)
        return self.norm(hidden)


class Embed(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.table = lm.embed_tokens

    def forward(self, ids):
        return self.table(ids)


class Heads(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.lm_to_dit, self.residual_to_dit = model.lm_to_dit_proj, model.res_to_dit_proj
        self.stop_proj, self.stop_activation, self.stop_head = model.stop_proj, model.stop_actn, model.stop_head

    def forward(self, lm_hidden, residual_hidden):
        dit_hidden = torch.cat((self.lm_to_dit(lm_hidden), self.residual_to_dit(residual_hidden)), dim=-1)
        return dit_hidden, self.stop_head(self.stop_activation(self.stop_proj(lm_hidden)))


class Fuse(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fsq, self.fusion = model.fsq_layer, model.fusion_concat_proj

    def forward(self, lm_raw, curr_embed):
        lm_hidden = self.fsq(lm_raw)
        return lm_hidden, self.fusion(torch.cat((lm_hidden, curr_embed), dim=-1))


class PrefillFuse(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fusion = model.fusion_concat_proj

    def forward(self, hidden):
        return self.fusion(torch.cat((hidden, torch.zeros_like(hidden)), dim=-1))


class LocEnc(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder, self.projection = model.feat_encoder, model.enc_to_lm_proj

    def forward(self, patch):
        return self.projection(self.encoder(patch.transpose(1, 2).unsqueeze(1)))[:, 0]


class DiT(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.estimator = model.feat_decoder.estimator

    def forward(self, x, mu, t, cond, dt):
        return self.estimator(x, mu, t, cond, dt)


def effective_weight(module):
    hooks = [hook for hook in module._forward_pre_hooks.values() if isinstance(hook, WeightNorm)]
    return (hooks[0].compute_weight(module) if hooks else module.weight).detach().clone()


class VAEStream(nn.Module):
    def __init__(self, vae):
        super().__init__()
        decoder = vae.decoder
        bucket = int(decoder.get_sr_idx(torch.tensor([vae.out_sample_rate], dtype=torch.int32))[0])
        self.constants, self.state_names, self.plan = 0, [], []
        for layer, condition in zip(decoder.model, decoder.sr_cond_model, strict=True):
            if condition is not None:
                if condition.cond_type != "scale_bias" or not isinstance(condition.out_layer, nn.Identity):
                    raise ValueError("Unsupported sample-rate condition")
                self.plan.append(("affine", self.constant(condition.scale_embed.weight[bucket].view(1, -1, 1)),
                                  self.constant(condition.bias_embed.weight[bucket].view(1, -1, 1))))
            self.plan.extend(self.flatten(layer))

    def constant(self, tensor):
        name = f"c{self.constants:03d}"
        self.constants += 1
        self.register_buffer(name, tensor.detach().clone())
        return name

    def state(self, channels, length):
        name = f"state_{len(self.state_names):02d}"
        self.register_buffer(name, torch.zeros(1, channels, length))
        self.state_names.append(name)
        return name

    def bias(self, module):
        return None if module.bias is None else self.constant(module.bias)

    def flatten(self, module):
        from voxcpm.modules.audiovae.audio_vae_v2 import (CausalConv1d, CausalDecoderBlock, CausalResidualUnit,
                                                          CausalTransposeConv1d, Snake1d)

        if isinstance(module, CausalConv1d):
            pad = module._CausalConv1d__padding * 2 - module._CausalConv1d__output_padding
            return [("conv", self.constant(effective_weight(module)), self.bias(module), module.stride[0],
                     module.dilation[0], module.groups, self.state(module.in_channels, pad) if pad > 0 else None)]
        if isinstance(module, CausalTransposeConv1d):
            trim = module._CausalTransposeConv1d__padding * 2 - module._CausalTransposeConv1d__output_padding
            context = (module.kernel_size[0] - 1) // module.stride[0]
            if context < 1 or module.dilation[0] != 1:
                raise ValueError("Unsupported transposed convolution")
            return [("transpose", self.constant(effective_weight(module)), self.bias(module), module.stride[0],
                     module.groups, context, trim, self.state(module.in_channels, context))]
        if isinstance(module, Snake1d):
            return [("snake", self.constant(module.alpha))]
        if isinstance(module, nn.Tanh):
            return [("tanh",)]
        if isinstance(module, CausalResidualUnit):
            return [("residual", self.flatten(module.block))]
        if isinstance(module, CausalDecoderBlock):
            return self.flatten(module.block)
        if isinstance(module, nn.Sequential):
            return [step for child in module for step in self.flatten(child)]
        raise TypeError(f"Unsupported VAE module: {type(module).__name__}")

    def run(self, plan, x):
        for step in plan:
            kind = step[0]
            if kind == "conv":
                _, weight, bias, stride, dilation, groups, state = step
                if state is not None:
                    cache = getattr(self, state)
                    x = torch.cat((cache, x), dim=-1)
                    cache.copy_(x[..., -cache.shape[-1]:])
                x = F.conv1d(x, getattr(self, weight), bias and getattr(self, bias), stride, 0, dilation, groups)
            elif kind == "transpose":
                _, weight, bias, stride, groups, context, trim, state = step
                cache = getattr(self, state)
                full = torch.cat((cache, x), dim=-1)
                cache.copy_(full[..., -context:])
                y = F.conv_transpose1d(full, getattr(self, weight), bias and getattr(self, bias), stride, 0, 0, groups)
                x = y[..., context * stride:y.shape[-1] - trim]
            elif kind == "snake":
                alpha = getattr(self, step[1])
                x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
            elif kind == "affine":
                x = x * getattr(self, step[1]) + getattr(self, step[2])
            elif kind == "tanh":
                x = torch.tanh(x)
            else:
                x = x + self.run(step[1], x)
        return x

    def forward(self, latent):
        return self.run(self.plan, latent)


def keep_fp32(node):
    stack = node.meta.get("nn_module_stack") or {}
    names = " ".join(str(value[1]) for value in stack.values())
    return any(name in names for name in KEPT_FP32)


def functions(model, asset, contract):
    execution = contract["execution"]
    capacity, length = execution["capacity"], execution["prefill_length"]
    hidden = contract["architecture"]["hidden_size"]
    patch = (1, contract["architecture"]["feat_dim"], contract["architecture"]["patch_size"])
    kv = ["k_cache", "v_cache"]
    first = torch.tensor([length], dtype=torch.int32)
    if asset in ("tslm", "ralm"):
        lm = model.base_lm if asset == "tslm" else model.residual_lm
        rows = [("prefill", Prefill(lm, capacity, length), (torch.zeros(1, length, hidden), first), ["embeds", "length"], ["hidden"], kv),
                ("step", Step(lm, capacity), (torch.zeros(1, hidden), first), ["embed", "position"], ["hidden"], kv)]
        if asset == "tslm":
            rows.insert(0, ("embed", Embed(lm), (torch.zeros(1, length, dtype=torch.int32),), ["ids"], ["embeds"], []))
        return rows
    if asset == "proj":
        return [("heads", Heads(model), (torch.zeros(1, hidden), torch.zeros(1, hidden)), ["lm_hidden", "residual_hidden"],
                 ["dit_hidden", "stop_logits"], []),
                ("fuse", Fuse(model), (torch.zeros(1, hidden), torch.zeros(1, hidden)), ["lm_raw", "curr_embed"],
                 ["lm_hidden", "residual_input"], []),
                ("prefill_fuse", PrefillFuse(model), (torch.zeros(1, length, hidden),), ["hidden"], ["residual_inputs"], [])]
    if asset == "locenc":
        return [("main", LocEnc(model), (torch.zeros(patch),), ["patch"], ["curr_embed"], [])]
    if asset == "dit":
        pair = (2, *patch[1:])
        return [("main", DiT(model), (torch.zeros(pair), torch.zeros(2, hidden), torch.zeros(2), torch.zeros(pair), torch.zeros(2)),
                 ["x", "mu", "t", "cond", "dt"], ["velocity"], [])]
    if asset == "vae":
        module = VAEStream(model.audio_vae)
        return [("main", module, (torch.zeros(patch),), ["latent"], ["audio"], module.state_names)]
    raise ValueError(f"Unknown asset: {asset}")
