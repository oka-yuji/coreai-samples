import copy
import importlib.util
from pathlib import Path
import unittest

from Tools.Conversion.adapters.jev27b.decision import HASH_SCHEMA, compare_hashes
from Tools.Conversion.adapters.jev27b.export_variant import ir_counts, verdict
from Tools.Conversion.recipe import load_recipe

ROOT = Path(__file__).resolve().parents[3]
BINDINGS = {name: "/unused" for name in ("base", "corpus", "hf_cache", "jglue", "python", "runner", "setup_record",
                                         "snapshot", "upstream", "verify")}

CONVERTER = importlib.util.find_spec("coreai_models") is not None
F32, BF16 = "tensor<1x4x?x8xf32>", "tensor<1x4x?x8xbf16>"
IR = "\n".join([
    '  coreai.graph private noinline @layer.gdu_1(%arg0: tensor<1x4x?x8xf32> {coreai.name = "query"}) -> '
    '(tensor<1x?x4x8xf32>) attributes {composite_decl = #coreai.composite_declaration<"gated_delta_update" = {}>} {',
    '  coreai.graph private noinline @layer.norm_2(%arg0: tensor<1x?x8xbf16>) -> (tensor<1x?x8xbf16>) '
    'attributes {composite_decl = #coreai.composite_declaration<"rms_norm" = {}>} {',
    "    %1:2 = coreai.invoke @layer.gdu_1(%a, %b, %c, %d, %e, %f)  : "
    f"({F32}, {F32}, {F32}, tensor<1x4x?xf32>, tensor<1x4x?xf32>, tensor<1x4x8x8xf32>) -> (tensor<1x?x4x8xf32>)",
    "    %2:2 = coreai.invoke @layer.gdu_1(%a, %b, %c, %d, %e, %f)  : "
    f"({BF16}, {BF16}, {BF16}, tensor<1x4x?xbf16>, tensor<1x4x?xbf16>, tensor<1x4x8x8xbf16>) -> (tensor<1x?x4x8xbf16>)",
    "    %3 = coreai.invoke @layer.norm_2(%g)  : (tensor<1x?x8xbf16>) -> tensor<1x?x8xbf16>",
    "    %4 = coreai.cast %3 : tensor<1x?x8xbf16> to tensor<1x?x8xf32>",
    "    %5 = coreai.cast_like %4 : tensor<1x?x8xf32>",
])


def record(digests):
    return {"schema": HASH_SCHEMA, "model_id": "owner/model", "revision": "0" * 40, "model_type": "qwen3_5",
            "device": "cpu", "versions": {"torch": "2.9.0"}, "recipe_sha256": "1" * 64, "compute_precision": "bfloat16",
            "compression": "qwen3_5_cfg", "context_length": 262144,
            "tensors": [{"name": name, "kind": "parameter", "shape": [2], "dtype": "torch.uint8", "sha256": digest}
                        for name, digest in digests.items()],
            "layout": [{"name": "a", "block": [1, 32]}]}


class VariantCheckTests(unittest.TestCase):
    def test_the_ir_counts_float32_gated_delta_update_calls_and_casts(self):
        counts = ir_counts(IR)
        self.assertEqual(counts["composite_declarations"], {"gated_delta_update": 1, "rms_norm": 1})
        self.assertEqual(counts["invoke_sites_by_kind"], {"gated_delta_update": 2, "rms_norm": 1})
        self.assertEqual((counts["gated_delta_update_sites"], counts["gated_delta_update_float32_sites"]), (2, 1))
        self.assertEqual(counts["cast_operations"], 1)

    def test_the_variant_passes_only_with_equal_hashes_every_layer_wrapped_and_float32_updates(self):
        baseline = record({"a": "a" * 64, "b": "b" * 64})
        hashes = compare_hashes(baseline, copy.deepcopy(baseline))
        ir = {"gated_delta_update_sites": 48, "gated_delta_update_float32_sites": 48}
        self.assertTrue(verdict(hashes, 48, 48, ir, True, 0)["passed"])
        changed = record({"a": "a" * 64, "b": "c" * 64})
        result = verdict(compare_hashes(baseline, changed), 48, 48, ir, True, 0)
        self.assertEqual((result["passed"], result["checks"]["hashes_match"]), (False, False))
        self.assertEqual(compare_hashes(baseline, changed)["differing"], ["b"])
        renamed = record({"a": "a" * 64, "b.inner": "b" * 64})
        self.assertFalse(compare_hashes(baseline, renamed)["hashes_match"])
        for wrapped, layers, sites, float32, inspected, aborts in ((47, 48, 48, 48, True, 0), (0, 0, 0, 0, True, 0),
                                                                   (48, 48, 48, 47, True, 0), (48, 48, 48, 48, False, 0),
                                                                   (48, 48, 48, 48, True, 1)):
            ir = {"gated_delta_update_sites": sites, "gated_delta_update_float32_sites": float32}
            self.assertFalse(verdict(hashes, wrapped, layers, ir, inspected, aborts)["passed"])

    def test_the_recipe_exports_the_variant_after_the_export_and_runs_typed_decide_on_it(self):
        recipe = load_recipe(ROOT / "Tools/Conversion/recipes/jev27b/recipe.json", BINDINGS, ROOT)
        steps = {step["id"]: step for step in recipe["steps"]}
        ids = list(steps)
        variant, decide = steps["export_gdnfp32"], steps["typed_decide_f"]
        self.assertEqual(ids[ids.index("export") + 1], "export_gdnfp32")
        self.assertEqual(variant["depends"], ["export", "merge", "prepare"])
        self.assertIn("$export", variant["command"])
        self.assertTrue(set(steps["export"]["inputs"]) < set(variant["inputs"]))
        self.assertEqual(recipe["variables"]["bundle_gdnfp32"], variant["outputs"][0])
        self.assertEqual(decide["depends"], ["export_gdnfp32", "typed_input"])
        self.assertIn("$export_gdnfp32/$bundle_gdnfp32", decide["command"])


@unittest.skipUnless(CONVERTER, "The JEV-27B converter environment is required")
class WrapperTests(unittest.TestCase):
    def fake(self):
        import torch
        from coreai_torch.composite_ops import GatedDeltaUpdate

        class Fake(GatedDeltaUpdate):
            def __init__(self):
                super().__init__()
                self.seen = []

            def forward(self, query, key, value, g, beta, initial_state):
                self.seen.append([tensor.dtype for tensor in (query, key, value, g, beta, initial_state)])
                return query.transpose(1, 2) * 2, initial_state + 1

        return Fake, torch

    def test_the_wrapper_runs_the_update_in_float32_and_returns_the_input_types(self):
        from Tools.Conversion.adapters.jev27b.backbone_gdnfp32 import Fp32GatedDeltaUpdate

        Fake, torch = self.fake()
        inner = Fake()
        wrapper = Fp32GatedDeltaUpdate(inner)
        sequence = torch.full((1, 2, 3, 4), 0.5, dtype=torch.bfloat16)
        gates = torch.full((1, 2, 3), 0.25, dtype=torch.bfloat16)
        output, state = wrapper(sequence, sequence, sequence, gates, gates, torch.zeros(1, 2, 4, 4, dtype=torch.bfloat16))
        self.assertEqual(inner.seen, [[torch.float32] * 6])
        self.assertEqual((output.dtype, state.dtype), (torch.bfloat16, torch.bfloat16))
        self.assertEqual((tuple(output.shape), tuple(state.shape)), ((1, 3, 2, 4), (1, 2, 4, 4)))
        self.assertTrue(torch.equal(output.float(), torch.ones(1, 3, 2, 4)))
        self.assertIs(wrapper.inner, inner)
        self.assertIsInstance(wrapper.inner, Fake)

    def test_every_update_is_wrapped_once_without_renaming_any_tensor(self):
        from Tools.Conversion.adapters.jev27b.backbone_gdnfp32 import (Fp32GatedDeltaUpdate, wrap_gated_delta_updates,
                                                                      wrapped_count)

        Fake, torch = self.fake()

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(2, 2)
                self._gated_delta_update = Fake()

        model = torch.nn.Sequential(Layer(), torch.nn.Linear(2, 2), Layer(), Layer())
        names = [name for name, _ in model.named_parameters()] + [name for name, _ in model.named_buffers()]
        self.assertEqual(wrap_gated_delta_updates(model), 3)
        self.assertEqual(wrapped_count(model), 3)
        self.assertEqual(wrap_gated_delta_updates(model), 0)
        self.assertTrue(all(isinstance(layer._gated_delta_update, Fp32GatedDeltaUpdate)
                            and isinstance(layer._gated_delta_update.inner, Fake) for layer in (model[0], model[2], model[3])))
        self.assertEqual([name for name, _ in model.named_parameters()] + [name for name, _ in model.named_buffers()],
                         names)

    def test_the_backbone_variant_wraps_every_linear_attention_layer_and_is_installed(self):
        from Tools.Conversion.adapters.jev27b import backbone
        from Tools.Conversion.adapters.jev27b.backbone_gdnfp32 import TypedDecisionBackboneGdnFp32, entry_for, wrapped_count
        from Tools.Conversion.tests.test_jev27b_backbone import composite_config

        model = TypedDecisionBackboneGdnFp32(config=composite_config(), model_device="meta")
        self.assertEqual(wrapped_count(model), 3)
        original = backbone.TypedDecisionBackbone(config=composite_config(), model_device="meta")
        self.assertEqual([name for name, _ in model.named_parameters()],
                         [name for name, _ in original.named_parameters()])
        self.assertIs(entry_for(backbone.pipeline).macos_class, TypedDecisionBackboneGdnFp32)


if __name__ == "__main__":
    unittest.main()
