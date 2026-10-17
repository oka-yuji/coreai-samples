import torch
from torch import nn
from torch.nn import functional as F


def causal_mask(hidden):
    positions = torch.arange(hidden.shape[1], device=hidden.device)
    allowed = positions[:, None] >= positions[None, :]
    return torch.where(allowed, 0.0, torch.finfo(hidden.dtype).min).to(hidden.dtype)[None, None, :, :]


class TextProjection(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.projection = model.talker.text_projection

    def forward(self, hidden):
        return self.projection(hidden)


class Talker(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.talker.model
        self.head = model.talker.codec_head

    def forward(self, hidden, positions, last_index=None):
        result = self.model(inputs_embeds=hidden, attention_mask=causal_mask(hidden), position_ids=positions,
                            past_key_values=None, use_cache=False, output_hidden_states=False)
        last = (result.last_hidden_state[:, -1, :] if last_index is None else
                torch.index_select(result.last_hidden_state, 1, last_index).squeeze(1))
        return self.head(last), last


class CodePredictor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.talker.code_predictor.model
        self.projection = model.talker.code_predictor.small_to_mtp_projection
        self.register_buffer("weights", torch.stack([head.weight.detach() for head in model.talker.code_predictor.lm_head]))

    def forward(self, hidden, codebook):
        projected = self.projection(hidden)
        result = self.model(inputs_embeds=projected, attention_mask={"full_attention": causal_mask(projected)},
                            past_key_values=None, use_cache=False, output_hidden_states=False)
        last = torch.index_select(result.last_hidden_state, 1, codebook + 1).squeeze(1)
        weight = torch.index_select(self.weights, 0, codebook).squeeze(0)
        return F.linear(last, weight)


class CodecDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        from .codec import exportable_codec

        self.decoder = exportable_codec(model.speech_tokenizer.model.decoder)

    def forward(self, codes):
        return self.decoder(codes)


def make_wrapper(model, kind):
    constructor = {"projection": TextProjection, "talker": Talker, "predictor": CodePredictor, "codec": CodecDecoder}[kind]
    return constructor(model).eval().requires_grad_(False)


def names(kind, *, padded=False):
    return {"projection": (["hidden"], ["projected"]),
            "talker": (["hidden", "positions", "last_index"] if padded else ["hidden", "positions"], ["logits", "hidden_out"]),
            "predictor": (["hidden", "codebook"], ["logits"]),
            "codec": (["codes"], ["audio"])}[kind]


def dynamic_shapes(kind):
    if kind == "projection":
        return ({1: torch.export.Dim("text_length", min=1, max=512)},)
    if kind == "talker":
        return ({}, {}, {})
    if kind == "predictor":
        return ({}, {})
    return ({2: torch.export.Dim("codec_frames", min=1, max=256)},)
