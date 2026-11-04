import dataclasses

from . import backbone, backbone_gdnfp32
from .decision import LAST_OUTPUT


class TypedDecisionBackboneLastToken(backbone_gdnfp32.TypedDecisionBackboneGdnFp32):
    @backbone.BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(self, input_ids, position_ids, k_cache, v_cache, conv_states, recurrent_states):
        hidden = self.model(input_ids, position_ids, backbone.KVCache(k_cache, v_cache),
                            backbone.DeltaNetCache(conv_states, recurrent_states))
        return hidden.select(1, -1).reshape(1, 1, self.config.hidden_size)

    @classmethod
    def export_output_names(cls):
        return {backbone.MAIN_GRAPH_NAME: (LAST_OUTPUT,)}


def install(pipeline):
    entry = dataclasses.replace(backbone_gdnfp32.entry_for(pipeline), macos_class=TypedDecisionBackboneLastToken)
    lookup = pipeline.get_model_entry
    pipeline.get_model_entry = lambda name: entry if name == backbone.COMPOSITE else lookup(name)
    return entry
