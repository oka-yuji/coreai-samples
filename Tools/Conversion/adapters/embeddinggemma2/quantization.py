import numpy as np


PROFILE = "int4-int8-ptq-s128-v70"
TABLE_SHAPE = (262144, 512)


def quantize_table(table):
    values = np.asarray(table, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] % 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite rows with an even width")
    scales = np.max(np.abs(values), axis=1) / np.float32(7)
    scales = np.where(scales == 0, np.float32(1), scales).astype("<f4")
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Invalid INT4 scales")
    integers = np.clip(np.rint(values / scales[:, None]), -7, 7).astype(np.int8)
    unsigned = integers.view(np.uint8) & 15
    packed = unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
    return packed, scales


def dequantize_table(packed, scales):
    data = np.asarray(packed)
    scale = np.asarray(scales)
    if data.dtype != np.uint8 or data.ndim != 2 or scale.shape != (data.shape[0],):
        raise ValueError("Invalid INT4 table layout")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid INT4 scales")
    integers = np.empty((data.shape[0], data.shape[1] * 2), dtype=np.int8)
    integers[:, 0::2] = (data & 15).astype(np.int8)
    integers[:, 1::2] = (data >> 4).astype(np.int8)
    integers[integers >= 8] -= 16
    if np.any(integers == -8):
        raise ValueError("The symmetric INT4 table excludes -8")
    return integers.astype(np.float32) * scale[:, None]


def prepare_quantization(module, kind, example):
    import torch
    from torch import nn
    from coreai_opt.quantization import ExecutionMode, ModuleQuantizerConfig, QuantizationSpec, Quantizer, QuantizerConfig
    from coreai_opt.quantization.spec import PerChannelGranularity, QuantizationScheme
    if kind not in ("text", "vision"):
        raise ValueError("Unknown encoder kind")
    bits = 4 if kind == "text" else 8
    spec = QuantizationSpec(dtype=torch.int4 if bits == 4 else torch.int8,
                            qscheme=QuantizationScheme.SYMMETRIC_WITH_CLIPPING,
                            granularity=PerChannelGranularity(axis=0))
    config = QuantizerConfig(global_config=None, execution_mode=ExecutionMode.EAGER)
    for target in (nn.Linear, nn.Conv2d):
        config.set_module_type(target, ModuleQuantizerConfig(
            op_input_spec=None, op_output_spec=None, op_state_spec={"weight": spec}))
    layers = [{"name": name, "shape": list(child.weight.shape), "bits": bits}
              for name, child in module.named_modules() if type(child) in (nn.Linear, nn.Conv2d)]
    if kind == "vision":
        position = module.vision.patch_embedder.position_embedding_table
        position_spec = QuantizationSpec(
            dtype=torch.int8, qscheme=QuantizationScheme.SYMMETRIC_WITH_CLIPPING,
            granularity=PerChannelGranularity(axis=2))
        config.set_module_name(r"vision\.patch_embedder", ModuleQuantizerConfig(
            op_input_spec=None, op_output_spec=None, op_state_spec={"weight": spec},
            module_state_spec={"position_embedding_table": position_spec}))
        layers.append({"name": "vision.patch_embedder", "parameter": "position_embedding_table",
                       "shape": list(position.shape), "bits": 8, "axis": 2})
    if not layers:
        raise ValueError("No quantizable encoder weights")
    quantizer = Quantizer(module, config)
    prepared = quantizer.prepare(example)
    for layer in layers:
        if not torch.nn.utils.parametrize.is_parametrized(
            prepared.get_submodule(layer["name"]), layer.get("parameter", "weight")
        ):
            raise ValueError("A required weight was not quantized: " + layer["name"])
    return quantizer, prepared, layers
