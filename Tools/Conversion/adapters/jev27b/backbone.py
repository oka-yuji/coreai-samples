import dataclasses
import hashlib
import importlib.metadata
from pathlib import Path

import torch
from torch.nn.utils import parametrize
import coreai_models.export.pipeline as pipeline
from coreai_models._constants import MAIN_GRAPH_NAME, TRACE_KV_CACHE_SEQ_LEN
from coreai_models.models.base import BaseForCausalLM, TraceSpec
from coreai_models.models.macos.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5Model
from coreai_models.primitives.macos.cache import DeltaNetCache, KVCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

from .decision import HASH_SCHEMA, INPUTS, OUTPUT, STATES

COMPOSITE = "qwen3_5"
FLAT = "qwen3_5_text"
FUSED = {"in_proj_all": "linear_attention", "qkv_gate_proj": "full_attention"}
PACKAGES = ("torch", "coreai-opt", "coreai-torch", "transformers")


class TypedDecisionBackbone(Qwen3_5ForCausalLM):
    exports_prefill_graph = False
    _extra_hf_shared_keys = []

    def _init_model(self, config):
        if isinstance(config, Qwen3_5Config):
            config = config.text_config
            self.config = config
        self.model = Qwen3_5Model(config)

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(self, input_ids, position_ids, k_cache, v_cache, conv_states, recurrent_states):
        return self.model(input_ids, position_ids, KVCache(k_cache, v_cache), DeltaNetCache(conv_states, recurrent_states))

    @classmethod
    def export_output_names(cls):
        return {MAIN_GRAPH_NAME: (OUTPUT,)}


def entry_for(pipeline, model_type):
    entry = dataclasses.replace(pipeline.get_model_entry(COMPOSITE), macos_class=TypedDecisionBackbone)
    if model_type == FLAT:
        return dataclasses.replace(entry, hf_config_attr=None, hf_state_dict_prefix="")
    if model_type != COMPOSITE:
        raise ValueError(f"No typed-decision entry for {model_type}")
    return entry


def install(pipeline, model_type):
    entry, lookup = entry_for(pipeline, model_type), pipeline.get_model_entry
    pipeline.get_model_entry = lambda name: entry if name == model_type else lookup(name)
    return entry


def zero_states(config, dtype, tokens, device="cpu"):
    with torch.device(device):
        key, value = TypedDecisionBackbone.create_kv_cache_tensors(
            config, dtype=dtype, seq_len=max(tokens, TRACE_KV_CACHE_SEQ_LEN))
        conv, recurrent = TypedDecisionBackbone.create_delta_cache_tensors(config, dtype=dtype)
    return key, value, conv, recurrent


def versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def digest(tensor):
    data = tensor.detach()
    if data.device.type == "meta":
        return None
    data = data.cpu().contiguous().reshape(-1)
    return hashlib.sha256(memoryview(data.view(torch.uint8).numpy())).hexdigest()


def quantized_layout(model):
    layout = []
    for name, module in model.named_modules():
        if not parametrize.is_parametrized(module):
            continue
        for tensor, chain in module.parametrizations.items():
            for index, step in enumerate(chain):
                data, scale = getattr(step, "quantized_data", None), getattr(step, "scale", None)
                if data is None or scale is None:
                    continue
                layout.append({"name": f"{name}.parametrizations.{tensor}.{index}", "class": type(step).__name__,
                               "input_dtype": str(step.input_dtype), "output_dtype": str(step.output_dtype),
                               "zero_point": step.zero_point is not None, "minval": step.minval is not None,
                               "block": [size // part for size, part in zip(data.shape, scale.shape)]})
    return layout


def tensor_hashes(model):
    tensors = []
    for kind, items in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in items:
            tensors.append({"name": name, "kind": kind, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                            "device": tensor.device.type, "bytes": tensor.numel() * tensor.element_size(),
                            "sha256": digest(tensor)})
    return {"tensors": tensors, "layout": quantized_layout(model), "count": len(tensors),
            "bytes": sum(entry["bytes"] for entry in tensors)}


def layer_count(config, layer_type):
    return (config.layer_types or [])[: config.num_hidden_layers].count(layer_type)


def load_check(model, config, dtype):
    spec = TraceSpec(max_context_length=TRACE_KV_CACHE_SEQ_LEN)
    references = model.build_reference_inputs(config, dtype, spec)
    try:
        model.validate_export_contract(references, model.build_dynamic_shapes(config, spec))
        contract = None
    except ValueError as error:
        contract = str(error)
    fused = {name: sum(isinstance(module, torch.nn.Linear) and path.endswith("." + name)
                       for path, module in model.named_modules()) for name in FUSED}
    expected = {name: layer_count(config, layer_type) for name, layer_type in FUSED.items()}
    parameters = list(model.named_parameters())
    meta = sorted(name for name, tensor in parameters if tensor.is_meta)
    result = {"model_class": type(model).__name__, "config_class": type(config).__name__,
              "exports_prefill_graph": model.exports_prefill_graph, "extra_hf_shared_keys": list(model._extra_hf_shared_keys),
              "input_names": model.export_input_names(), "state_names": model.export_state_names(),
              "output_names": model.export_output_names(), "lm_head": hasattr(model, "lm_head"),
              "fused": fused, "fused_expected": expected, "meta_parameters": meta, "parameters": len(parameters),
              "elements": sum(tensor.numel() for _, tensor in parameters),
              "parameter_dtypes": sorted({str(tensor.dtype) for _, tensor in parameters}),
              "contract_inputs": list(references[MAIN_GRAPH_NAME]), "contract_error": contract}
    result["checks"] = {
        "no_prefill_graph": result["exports_prefill_graph"] is False,
        "no_extra_shared_keys": result["extra_hf_shared_keys"] == [],
        "inputs": result["input_names"] == {MAIN_GRAPH_NAME: INPUTS},
        "hidden_state_output": result["output_names"] == {MAIN_GRAPH_NAME: (OUTPUT,)},
        "states": result["state_names"] == {MAIN_GRAPH_NAME: STATES},
        "no_lm_head": not result["lm_head"],
        "fused": fused == expected and all(expected.values()),
        "all_parameters_loaded": not meta,
        "export_contract": contract is None,
    }
    result["passed"] = all(result["checks"].values())
    return result


def check_location(upstream, recipe):
    import coreai_models

    upstream = Path(upstream).resolve()
    if not Path(coreai_models.__file__).resolve().is_relative_to(upstream / "python/src"):
        raise ValueError("Exporter imported from an unexpected checkout")
    if not Path(recipe).resolve().is_relative_to(upstream):
        raise ValueError("The quantization recipe must come from the exporter checkout")


def export_config(contract, recipe, model_id, output_dir, name):
    from coreai_models.llm.export import _load_compression_config_object

    if recipe.stem != contract["compression"]:
        raise ValueError("The recipe file differs from the contract compression")
    return pipeline.ExportConfig(hf_model_id=model_id, variant="macOS", compression=recipe.stem,
                                 compression_config_object=_load_compression_config_object(recipe, "macOS"),
                                 compute_precision=contract["precision"], max_context_length=contract["context_length"],
                                 output_dir=str(output_dir), output_name=name, overwrite=False)


def hash_record(model, identity):
    return {"schema": HASH_SCHEMA, **identity, "versions": versions(), **tensor_hashes(model)}
