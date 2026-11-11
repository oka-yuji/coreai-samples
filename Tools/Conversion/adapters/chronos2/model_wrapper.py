import torch
from chronos.chronos_bolt import InstanceNorm

from .artifacts import QUANTILES, validate_config


class ExportNorm(InstanceNorm):
    def forward(self, x, loc_scale=None):
        original_dtype = x.dtype
        x = x.to(torch.float32)
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num((x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
            scale = torch.where(scale == 0, torch.full_like(scale, self.eps), scale)
        else:
            loc, scale = loc_scale
        scaled = (x - loc) / scale
        if self.use_arcsinh:
            scaled = torch.arcsinh(scaled)
        return scaled.to(original_dtype), (loc, scale)


class Chronos2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        validate_config(model.config.to_dict())
        if (type(model.instance_norm) is not InstanceNorm or model.instance_norm.eps != 1e-5
                or model.instance_norm.use_arcsinh is not True
                or model.patch.patch_size != 16 or model.patch.patch_stride != 16 or model.num_quantiles != 21):
            raise ValueError("Unexpected normalization, patch or quantile contract")
        if model.training or any(value.is_floating_point() and value.dtype != torch.float32
                                 for value in (*model.parameters(), *model.buffers())):
            raise ValueError("The export model must be evaluated in FP32")
        if not torch.equal(model.quantiles.cpu(), torch.tensor(QUANTILES, dtype=torch.float32)):
            raise ValueError("Unexpected quantile order")
        model.instance_norm = ExportNorm(eps=model.instance_norm.eps, use_arcsinh=model.instance_norm.use_arcsinh)
        self.model = model

    def forward(self, context):
        if context.dtype != torch.float32 or context.ndim != 2 or context.shape[0] != 1:
            raise ValueError("Only Float32 context[1, length] is supported")
        length = context.shape[1]
        if length < 16 or length > 8192 or length % 16:
            raise ValueError("Context length must be 16 to 8192 in complete patches")
        if not torch.compiler.is_compiling():
            if torch.isinf(context).any() or not torch.isfinite(context).any():
                raise ValueError("Context must contain an observation and no infinity")
        return self.model(context=context, num_output_patches=1).quantile_preds


def lower_isnan(value):
    return value != value


def lower_unfold(value, dimension, size, step):
    if (dimension not in (-1, value.ndim - 1) or size != 16 or step != 16
            or value.shape[-1] < 16 or value.shape[-1] % 16):
        raise ValueError("Only complete nonoverlapping last-axis patches of size and stride 16 are supported")
    return value.reshape(*value.shape[:-1], value.shape[-1] // 16, 16)


def decomposition_table(base):
    table = dict(base)
    table[torch.ops.aten.isnan.default] = lower_isnan
    table[torch.ops.aten.unfold.default] = lower_unfold
    return table


def export_program(module, base):
    context = torch.linspace(10.0, 50.0, 1008, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        program = torch.export.export(module, (context,), strict=False, dynamic_shapes={
            "context": {1: 16 * torch.export.Dim("patches", min=1, max=512)},
        })
        return program.run_decompositions(decomposition_table(base))
