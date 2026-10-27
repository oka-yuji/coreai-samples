import hashlib
import re

import torch
from torch import nn


DECODER_LINEAR = re.compile(r"layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))")


def tensor_record(value):
    value = value.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    return {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch."),
            "sha256": hashlib.sha256(raw).hexdigest()}


def targets(module, kind, bits):
    if kind not in ("decoder", "vision") or type(bits) is not int or bits not in (4, 8):
        raise ValueError("Select decoder INT4/INT8 or vision INT8")
    if kind == "vision" and bits != 8:
        raise ValueError("Only the declared vision INT8 comparison is supported")
    names = []
    for name, child in module.named_modules():
        if torch.nn.utils.parametrize.is_parametrized(child):
            raise ValueError("Quantization requires uncompressed source weights")
        if type(child) is nn.Linear and (kind == "vision" or DECODER_LINEAR.fullmatch(name)):
            if not name or not torch.isfinite(child.weight).all():
                raise ValueError("Quantization requires named, finite Linear weights")
            if kind == "decoder" and child.in_features % 64:
                raise ValueError("Decoder weights require complete blocks of 64 input channels")
            names.append(name)
    if not names:
        raise ValueError("No declared Linear weights were found")
    if kind == "decoder":
        observed = {name for name, child in module.named_modules() if type(child) is nn.Linear and name.startswith("layers.")}
        if set(names) != observed or len(names) != 7 * len(module.layers):
            raise ValueError("Every decoder layer must contain exactly the seven declared Linear weights")
    return names


class LinearInputs(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, value):
        for layer in self.layers:
            value = layer(torch.zeros((1, layer.in_features), dtype=layer.weight.dtype, device=layer.weight.device))
        return value


def quantize(module, kind, bits):
    from coreai_opt.quantization import ExecutionMode, ModuleQuantizerConfig, QuantizationSpec, Quantizer, QuantizerConfig
    from coreai_opt.quantization.spec import PerBlockGranularity, PerChannelGranularity, QuantizationScheme

    names = targets(module, kind, bits)
    selected = {name + ".weight" for name in names}
    retained = {name: tensor_record(value) for name, value in module.named_parameters(remove_duplicate=False)
                if name not in selected}
    granularity = PerBlockGranularity(axis=1, block_size=64) if kind == "decoder" else PerChannelGranularity(axis=0)
    spec = QuantizationSpec(dtype=torch.int4 if bits == 4 else torch.int8,
                            qscheme=QuantizationScheme.SYMMETRIC_WITH_CLIPPING, granularity=granularity)
    config = QuantizerConfig(global_config=None, execution_mode=ExecutionMode.EAGER)
    config.set_module_type(nn.Linear, ModuleQuantizerConfig(
        op_input_spec=None, op_output_spec=None, op_state_spec={"weight": spec}))
    layers = LinearInputs([module.get_submodule(name) for name in names]).eval().requires_grad_(False)
    quantizer = Quantizer(layers, config)
    prepared = quantizer.prepare((torch.zeros(1),))
    if not all(torch.nn.utils.parametrize.is_parametrized(layer, "weight") for layer in prepared.layers):
        raise ValueError("A required Linear weight was skipped")
    finalized = quantizer.finalize()
    records, reference = [], {}
    for name, linear in zip(names, finalized.layers, strict=True):
        params = linear.parametrizations.weight[0]
        integers, scales, zero = params.quantized_data, params.scale, params.zero_point
        expected_scale = (linear.out_features, linear.in_features // 64) if kind == "decoder" else (linear.out_features, 1)
        if (integers.dtype != torch.int8 or tuple(scales.shape) != expected_scale or zero is None
                or zero.shape != scales.shape or params.minval is not None or torch.any(zero != 0)
                or not torch.isfinite(scales).all() or torch.any(scales <= 0)
                or integers.min().item() < -(2 ** (bits - 1)) or integers.max().item() >= 2 ** (bits - 1)):
            raise ValueError("Finalized weights differ from the declared integer contract")
        scale = scales.repeat_interleave(64, dim=1) if kind == "decoder" else scales
        effective = (integers.float() * scale.float()).to(linear.weight.dtype)
        if not torch.equal(effective, linear.weight):
            raise ValueError("Independent dequantization differs from the exported effective weight")
        reference[name + ".weight"] = effective.detach().cpu().contiguous()
        records.append({"name": name, "integer": tensor_record(integers), "scale": tensor_record(scales),
                        "export_scale": tensor_record(scales.half()),
                        "zero_point": tensor_record(zero), "effective": tensor_record(effective)})
        module.set_submodule(name, linear)
    for name, expected in retained.items():
        if tensor_record(module.get_parameter(name)) != expected:
            raise ValueError("An excluded weight or bias changed: " + name)
    observed = {name for name, child in module.named_modules() if torch.nn.utils.parametrize.is_parametrized(child)}
    if observed != set(names):
        raise ValueError("The compressed modules differ from the declared target set")
    return module, {"kind": kind, "bits": bits, "scheme": "symmetric_with_clipping",
                    "granularity": "block64_axis1" if kind == "decoder" else "channel_axis0",
                    "layers": records, "retained": retained}, reference


def audit_program(program, record, prefix=""):
    suffix = ".parametrizations.weight.0.quantized_data"
    observed = {name.removesuffix(suffix) for name in program.state_dict if name.endswith(suffix)}
    expected = {prefix + row["name"] for row in record["layers"]}
    if observed != expected:
        raise ValueError("The exported function does not contain exactly the declared compressed modules")
    inputs = {spec.target for spec in program.graph_signature.input_specs}
    for row in record["layers"]:
        name = prefix + row["name"] + ".parametrizations.weight.0."
        for parameter, key in (("quantized_data", "integer"), ("scale", "export_scale"), ("zero_point", "zero_point")):
            target = name + parameter
            if target not in inputs or tensor_record(program.state_dict[target]) != row[key]:
                raise ValueError("The exported compression parameters differ from the declared identity: " + target)
    return {"modules": sorted(observed), "parameters_match": True}


def save_reference(path, tensors, record):
    from safetensors.torch import save_file

    if path.exists():
        raise FileExistsError(path)
    save_file(tensors, str(path), metadata={"kind": record["kind"], "bits": str(record["bits"]),
                                          "scheme": record["scheme"], "granularity": record["granularity"]})
