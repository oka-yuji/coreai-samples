import dataclasses

import torch
from coreai_torch.composite_ops import GatedDeltaUpdate
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

from . import backbone

VARIANT = "gdn-fp32"


class Fp32GatedDeltaUpdate(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, query, key, value, g, beta, initial_state):
        output, state = self.inner(*(tensor.to(torch.float32) for tensor in (query, key, value, g, beta, initial_state)))
        return output.to(query.dtype), state.to(initial_state.dtype)


def wrap_gated_delta_updates(model):
    owners = [module for module in model.modules()
              if isinstance(getattr(module, "_gated_delta_update", None), GatedDeltaUpdate)]
    for module in owners:
        module._gated_delta_update = Fp32GatedDeltaUpdate(module._gated_delta_update)
    return len(owners)


def wrapped_count(model):
    return sum(isinstance(module, Fp32GatedDeltaUpdate) for module in model.modules())


class TypedDecisionBackboneGdnFp32(backbone.TypedDecisionBackbone):
    def _init_model(self, config):
        super()._init_model(config)
        text = config.text_config if isinstance(config, Qwen3_5Config) else config
        expected = backbone.layer_count(text, "linear_attention")
        wrapped = wrap_gated_delta_updates(self.model)
        if not expected or wrapped != expected:
            raise ValueError(f"Wrapped {wrapped} gated delta updates for {expected} linear-attention layers")


def entry_for(pipeline):
    return dataclasses.replace(backbone.entry_for(pipeline, backbone.COMPOSITE),
                               macos_class=TypedDecisionBackboneGdnFp32)


def install(pipeline):
    entry, lookup = entry_for(pipeline), pipeline.get_model_entry
    pipeline.get_model_entry = lambda name: entry if name == backbone.COMPOSITE else lookup(name)
    return entry
