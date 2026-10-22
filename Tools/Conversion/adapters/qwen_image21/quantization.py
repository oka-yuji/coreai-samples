import copy
import hashlib

import torch


class StoredLinear(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear.to(dtype=torch.float16)

    def forward(self, value):
        bias = self.linear.bias.float() if self.linear.bias is not None else None
        return torch.nn.functional.linear(value.float(), self.linear.weight.float(), bias)


class StoredEmbedding(torch.nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding.to(dtype=torch.float16)

    def forward(self, indices):
        return self.embedding(indices).float()


def store_linear_weights_in_half(module):
    for name, child in list(module.named_modules()):
        if type(child) is torch.nn.Linear:
            if not name:
                raise ValueError("Wrap a root Linear before preparing mixed-precision storage")
            module.set_submodule(name, StoredLinear(child))
    return module


def tensor_record(value):
    value = value.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    return {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch."),
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class LinearCalibration(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, value):
        for layer in self.layers:
            value = layer(torch.zeros((1, layer.in_features), dtype=layer.weight.dtype, device=layer.weight.device))
        return value


def dense_weight(linear, name):
    if (type(linear) is not torch.nn.Linear or torch.nn.utils.parametrize.is_parametrized(linear)
            or linear.weight.dtype != torch.float16 or not torch.isfinite(linear.weight).all()
            or (linear.bias is not None and (linear.bias.dtype != torch.float16 or not torch.isfinite(linear.bias).all()))):
        raise ValueError("Retained Linear weights require finite Float16 storage without parametrization")
    return {"name": name, "weight": tensor_record(linear.weight),
            "bias": tensor_record(linear.bias) if linear.bias is not None else None,
            "payload_bits": linear.weight.numel() * 16}


def quantize(module, bits, group_size=64, *, dense_modules=()):
    if bits not in (4, 8) or type(bits) is not int or type(group_size) is not int or group_size not in (16, 64):
        raise ValueError("Mobile compression requires W4 or W8 with a declared group16 or group64")
    from coreai_opt.quantization import ExecutionMode, ModuleQuantizerConfig, QuantizationSpec, Quantizer, QuantizerConfig
    from coreai_opt.quantization.spec import PerBlockGranularity, QuantizationScheme

    names = [name for name, child in module.named_modules() if type(child) is torch.nn.Linear]
    if not names or "" in names or any(torch.nn.utils.parametrize.is_parametrized(child) for child in module.modules()):
        raise ValueError("Compression requires a nonempty, unparametrized Linear set")
    if (not isinstance(dense_modules, (tuple, list)) or any(type(name) is not str for name in dense_modules)
            or len(dense_modules) != len(set(dense_modules)) or not set(dense_modules) < set(names)):
        raise ValueError("Retained modules must be a unique proper subset of the declared Linear set")
    retained = [dense_weight(module.get_submodule(name), name) for name in dense_modules]
    names = [name for name in names if name not in dense_modules]
    for name in names:
        weight = module.get_submodule(name).weight
        if weight.dtype != torch.float16 or weight.shape[1] % group_size:
            raise ValueError("Each quantized Linear must have Float16 storage and complete declared blocks")
    spec = QuantizationSpec(dtype=torch.int4 if bits == 4 else torch.int8,
                            qscheme=QuantizationScheme.SYMMETRIC_WITH_CLIPPING,
                            granularity=PerBlockGranularity(axis=1, block_size=group_size))
    config = QuantizerConfig(global_config=None, execution_mode=ExecutionMode.EAGER)
    config.set_module_type(torch.nn.Linear, ModuleQuantizerConfig(
        op_input_spec=None, op_output_spec=None, op_state_spec={"weight": spec}))
    calibration = LinearCalibration([module.get_submodule(name) for name in names]).eval().requires_grad_(False)
    quantizer = Quantizer(calibration, config)
    prepared = quantizer.prepare((torch.zeros(1),))
    if not all(torch.nn.utils.parametrize.is_parametrized(layer, "weight") for layer in prepared.layers):
        raise ValueError("Required Linear weights were not quantized")
    finalized = quantizer.finalize()
    for name, layer in zip(names, finalized.layers, strict=True):
        module.set_submodule(name, layer)
    records = [quantized_weight(module.get_submodule(name), name, bits, group_size)[1] for name in names]
    result = {"kind": "linear-integer", "bits": bits, "group_size": group_size, "axis": 1,
              "scheme": "symmetric_with_clipping", "storage_dtype": "float16",
              "effective_weight": "round16((q-zp)*scale16)", "layers": records}
    if retained:
        result["dense_layers"] = retained
    return module, result


def quantized_weight(linear, name, bits, group_size=64):
    if not torch.nn.utils.parametrize.is_parametrized(linear, "weight") or len(linear.parametrizations.weight) != 1:
        raise ValueError("A quantized weight must have exactly one finalized parametrization")
    params = linear.parametrizations.weight[0]
    required = ("quantized_data", "scale", "zero_point", "minval", "input_dtype")
    if any(not hasattr(params, key) for key in required):
        raise ValueError("The weight is not a finalized integer representation")
    q, scale, zero = params.quantized_data, params.scale, params.zero_point
    if (q.dtype != torch.int8 or scale.dtype != torch.float16 or zero is None or zero.dtype != torch.int8
            or params.minval is not None or params.input_dtype != (torch.int4 if bits == 4 else torch.int8)
            or q.ndim != 2 or list(q.shape) != [linear.out_features, linear.in_features]
            or list(scale.shape) != [q.shape[0], q.shape[1] // group_size] or zero.shape != scale.shape
            or q.shape[1] % group_size or not torch.isfinite(scale).all() or not (scale > 0).all()
            or torch.any(zero != 0) or q.min().item() < -(2 ** (bits - 1)) or q.max().item() >= 2 ** (bits - 1)):
        raise ValueError("Finalized quantization parameters violate the declared storage contract")
    with torch.no_grad():
        independent = ((q.float() - zero.float().repeat_interleave(group_size, dim=1))
                       * scale.float().repeat_interleave(group_size, dim=1)).half()
        if not torch.equal(independent, linear.weight):
            raise ValueError("Independent dequantization differs from the finalized effective weight")
    record = {"name": name, "shape": list(q.shape), "integer": tensor_record(q),
              "scale": tensor_record(scale), "zero_point": tensor_record(zero),
              "effective_float16": tensor_record(independent),
              "payload_bits_scenario": q.numel() * bits + scale.numel() * 16 + zero.numel() * bits}
    return independent, record


def plain_reference(module, compression):
    reference = copy.deepcopy(module)
    names = [row["name"] for row in compression["layers"]]
    retained = compression.get("dense_layers", [])
    retained_names = [row["name"] for row in retained]
    ordinary = [name for name, child in reference.named_modules() if type(child) is torch.nn.Linear]
    if (set(ordinary) != set(retained_names) or len(retained_names) != len(set(retained_names))
            or set(names) & set(retained_names)):
        raise ValueError("Independent reference must cover every and only the retained dense Linear weight")
    for row in retained:
        if dense_weight(reference.get_submodule(row["name"]), row["name"]) != row:
            raise ValueError("A retained dense weight or bias differs from the exported identity")
    observed = [name for name, child in reference.named_modules()
                if torch.nn.utils.parametrize.is_parametrized(child, "weight")]
    if set(names) != set(observed) or len(names) != len(set(names)):
        raise ValueError("Independent reference must cover every and only the declared compressed weight")
    with torch.no_grad():
        for row in compression["layers"]:
            name = row["name"]
            linear = reference.get_submodule(name)
            weight, identity = quantized_weight(linear, name, compression["bits"], compression["group_size"])
            if identity != row:
                raise ValueError("The quantized weight identity changed before reference construction")
            parent_name = name.rpartition(".")[0]
            parent = reference.get_submodule(parent_name)
            dtype = torch.float32 if isinstance(parent, StoredLinear) else torch.float16
            dense = torch.nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None, dtype=dtype)
            dense.weight.copy_(weight.to(dtype))
            if linear.bias is not None:
                dense.bias.copy_(linear.bias.to(dtype))
            reference.set_submodule(name, dense.requires_grad_(False))
    if any(torch.nn.utils.parametrize.is_parametrized(child) for child in reference.modules()):
        raise ValueError("The independent reference must not retain quantization operators")
    return reference.eval().requires_grad_(False)
