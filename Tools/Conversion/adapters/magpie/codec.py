import copy

import torch
from torch import nn
from torch.nn import functional as F


def apply_length_mask(value, lengths):
    positions = torch.arange(value.shape[-1], device=value.device).view(1, 1, -1)
    return value * (positions < lengths.view(-1, 1, 1)).to(value.dtype)


def materialized_convolution(original):
    arguments = {"in_channels": original.in_channels, "out_channels": original.out_channels,
                 "kernel_size": original.kernel_size, "stride": original.stride,
                 "padding": original.padding, "dilation": original.dilation,
                 "groups": original.groups, "bias": original.bias is not None,
                 "padding_mode": original.padding_mode}
    if isinstance(original, nn.ConvTranspose1d):
        arguments["output_padding"] = original.output_padding
        result = nn.ConvTranspose1d(**arguments)
    elif isinstance(original, nn.Conv1d):
        result = nn.Conv1d(**arguments)
    else:
        raise ValueError("Unsupported codec convolution")
    result.weight = nn.Parameter(original.weight.detach().clone(), requires_grad=False)
    if original.bias is not None:
        result.bias = nn.Parameter(original.bias.detach().clone(), requires_grad=False)
    return result


class CausalConvolution(nn.Module):
    def __init__(self, original):
        super().__init__()
        if original.conv.stride != (1,) or original.extra_pad_mode != "constant":
            raise ValueError("The pinned decoder requires stride-one, constant-padded convolutions")
        self.conv = materialized_convolution(original.conv)
        self.activation = original.activation
        self.padding = (self.conv.kernel_size[0] - 1) * self.conv.dilation[0]

    def forward(self, inputs, input_len):
        result = self.activation(self.conv(F.pad(inputs, (self.padding, 0))))
        return apply_length_mask(result, input_len)


class CausalTransposedConvolution(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.conv = materialized_convolution(original.conv)
        self.activation = original.activation
        self.left = int(original.padding_left)
        self.right = int(original.padding_right)

    def forward(self, inputs, input_len):
        result = self.conv(inputs)
        result = result[..., self.left:result.shape[-1] - self.right]
        return apply_length_mask(self.activation(result), input_len)


def exportable_codec(original):
    from nemo.collections.tts.modules.audio_codec_modules import CausalConv1dNorm, CausalConvTranspose1dNorm

    decoder = copy.deepcopy(original)
    def replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, CausalConv1dNorm):
                setattr(parent, name, CausalConvolution(child))
            elif isinstance(child, CausalConvTranspose1dNorm):
                setattr(parent, name, CausalTransposedConvolution(child))
            else:
                replace(child)

    replace(decoder)
    if any(hasattr(module, "parametrizations") for module in decoder.modules()):
        raise ValueError("An unexpected parametrized codec component remains")
    return decoder
