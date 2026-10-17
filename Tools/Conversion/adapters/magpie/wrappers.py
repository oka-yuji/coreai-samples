import torch
from torch import nn
from torch.nn import functional as F


class TextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embedding = model.text_embedding
        self.encoder = model.encoder
        self.encoder.reset_cache(False)

    def forward(self, tokens, mask):
        hidden = self.embedding(tokens) * mask.unsqueeze(-1)
        return self.encoder(hidden, mask)["output"]


class Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder
        self.projection = model.final_proj
        self.prior_layers = tuple(model.inference_parameters.apply_prior_to_layers)
        self.decoder.reset_cache(False)

    def forward(self, hidden, mask, memory, memory_mask, prior):
        priors = [prior if index in self.prior_layers else None for index in range(self.decoder.n_layers)]
        result = self.decoder(hidden, mask, memory, memory_mask, attn_prior=priors)
        last = result["output"][:, -1, :]
        attention = torch.stack([
            layer["cross_attn_probabilities"][0][:, 0, -1, :]
            for layer in result["attn_probabilities"]
        ], dim=1)
        return self.projection(last), last, attention


class LocalTransformer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.transformer = model.local_transformer
        self.transformer.reset_cache(False)
        self.register_buffer("weights", torch.stack([layer.weight.detach() for layer in model.local_transformer_out_projections]))
        self.register_buffer("biases", torch.stack([layer.bias.detach() for layer in model.local_transformer_out_projections]))

    def forward(self, hidden, codebook):
        mask = torch.ones_like(hidden[:, :, 0])
        last = self.transformer(hidden, mask)["output"][:, -1, :]
        weight = torch.index_select(self.weights, 0, codebook).squeeze(0)
        bias = torch.index_select(self.biases, 0, codebook).squeeze(0)
        return F.linear(last, weight, bias)


class CodecDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        from .codec import exportable_codec

        self.decoder = exportable_codec(model._codec_model.audio_decoder)

    def forward(self, latent, lengths):
        return self.decoder(inputs=latent, input_len=lengths)[0]


def make_wrapper(model, kind):
    constructors = {"encoder": TextEncoder, "decoder": Decoder,
                    "local": LocalTransformer, "codec": CodecDecoder}
    return constructors[kind](model).eval().requires_grad_(False)


def names(kind):
    return {
        "encoder": (["tokens", "mask"], ["encoded"]),
        "decoder": (["hidden", "mask", "memory", "memory_mask", "prior"], ["logits", "hidden_out", "attention"]),
        "local": (["hidden", "codebook"], ["logits"]),
        "codec": (["latent", "lengths"], ["audio"])
    }[kind]


def dynamic_shapes(kind):
    if kind == "encoder":
        text = torch.export.Dim("text_length", min=2, max=512)
        return ({1: text}, {1: text})
    if kind == "decoder":
        sequence = torch.export.Dim("sequence_length", min=2, max=512)
        text = torch.export.Dim("text_length", min=2, max=512)
        return ({1: sequence}, {1: sequence}, {1: text}, {1: text}, {2: text})
    if kind == "local":
        sequence = torch.export.Dim("local_length", min=1, max=16)
        return ({1: sequence}, {})
    frames = torch.export.Dim("codec_frames", min=2, max=512)
    return ({2: frames}, {})
