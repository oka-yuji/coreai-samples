import copy

import torch
from torch import nn
from torch.nn import functional as F


class CausalConvolution(nn.Module):
    def __init__(self, original):
        super().__init__()
        if original.stride != 1 or original.conv.stride != (1,):
            raise ValueError("The pinned audio decoder requires stride-one causal convolutions")
        self.conv = original.conv
        self.padding = int(original.padding)

    def forward(self, hidden):
        return self.conv(F.pad(hidden, (self.padding, 0), mode="constant", value=0)).contiguous()


class DecoderTransformer(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.transformer = original
        self.window = original.config.sliding_window

    def forward(self, inputs_embeds):
        positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        distance = positions[:, None] - positions[None, :]
        allowed = distance >= 0
        minimum = torch.finfo(inputs_embeds.dtype).min
        masks = {"full_attention": torch.where(allowed, 0.0, minimum)[None, None]}
        if self.transformer.has_sliding_layers:
            masks["sliding_attention"] = torch.where(allowed & (distance < self.window), 0.0, minimum)[None, None]
        return self.transformer(inputs_embeds=inputs_embeds, attention_mask=masks,
                                position_ids=positions.unsqueeze(0), past_key_values=None, use_cache=False)


def exportable_codec(original):
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2CausalConvNet

    decoder = copy.deepcopy(original)

    def replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, Qwen3TTSTokenizerV2CausalConvNet):
                setattr(parent, name, CausalConvolution(child))
            else:
                replace(child)

    replace(decoder)
    decoder.pre_transformer = DecoderTransformer(decoder.pre_transformer)
    return decoder
