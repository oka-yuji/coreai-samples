from pathlib import Path
import tempfile
import unittest

import torch
import numpy as np
from torch import nn

from Tools.Conversion.adapters.teleocr.quantization import audit_program, quantize, save_reference, targets, tensor_record
from Tools.Conversion.local import recipe_path
from Tools.Conversion.recipe import load_recipe
from Tools.Conversion.adapters.teleocr.verify import compressed_storage_matches
from Tools.Conversion.adapters.teleocr.contracts import logit_metrics
from Tools.Conversion.adapters.teleocr.quantized import conversion_passes, plain_reference
from Tools.Conversion.adapters.teleocr.convert import export


def decoder(width=64):
    result = nn.Module()
    result.layers = nn.ModuleList()
    for _ in range(2):
        layer = nn.Module()
        layer.self_attn, layer.mlp = nn.Module(), nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            layer.self_attn.add_module(name, nn.Linear(width, width))
        for name in ("gate_proj", "up_proj", "down_proj"):
            layer.mlp.add_module(name, nn.Linear(width, width, bias=False))
        layer.norm = nn.LayerNorm(width)
        result.layers.append(layer)
    result.embed, result.head = nn.Embedding(128, width), nn.Linear(width, 128, bias=False)
    result.head.weight = result.embed.weight
    return result.eval().requires_grad_(False)


class TeleOCRQuantizationTests(unittest.TestCase):
    def test_export_audit_rejects_missing_extra_and_changed_compression_parameters(self):
        model, record, _ = quantize(nn.Sequential(nn.Linear(64, 32)).eval().requires_grad_(False), "vision", 8)
        program = export(model, (torch.ones(1, 64),), half=True)
        self.assertEqual(audit_program(program, record), {"modules": ["0"], "parameters_match": True})
        key = "0.parametrizations.weight.0.quantized_data"
        original = program.state_dict[key]
        program.state_dict[key] = torch.zeros_like(original)
        with self.assertRaisesRegex(ValueError, "declared identity"):
            audit_program(program, record)
        program.state_dict[key] = original
        program.state_dict["extra.parametrizations.weight.0.quantized_data"] = original
        with self.assertRaisesRegex(ValueError, "exactly the declared"):
            audit_program(program, record)
        del program.state_dict["extra.parametrizations.weight.0.quantized_data"]
        del program.state_dict[key]
        with self.assertRaisesRegex(ValueError, "exactly the declared"):
            audit_program(program, record)

    def test_plain_reference_preserves_the_original_compressed_model(self):
        for bits in (4, 8):
            with self.subTest(bits=bits):
                model, _, weights = quantize(decoder(), "decoder", bits)
                inputs = torch.randn(1, 64)
                before = model.layers[0].self_attn.q_proj(inputs)
                reference = plain_reference(model, weights)
                torch.testing.assert_close(model.layers[0].self_attn.q_proj(inputs), before, rtol=0, atol=0)
                torch.testing.assert_close(reference.layers[0].self_attn.q_proj(inputs), before, rtol=0, atol=0)
                self.assertTrue(torch.nn.utils.parametrize.is_parametrized(model.layers[0].self_attn.q_proj))
                self.assertFalse(any(torch.nn.utils.parametrize.is_parametrized(child) for child in reference.modules()))
                self.assertIs(reference.embed.weight, reference.head.weight)

    def test_conversion_requires_fixed_error_bounds_independent_of_the_argmax_margin(self):
        reference = np.array([0.0, 10.0, 1.0])
        for output in ([1e6, -1e6, -1e6], [0.0, 0.0, 0.0], [0.0, 20.0, 2.0], [0.0, float("nan"), 1.0]):
            with self.subTest(output=output):
                self.assertFalse(conversion_passes(logit_metrics(reference, np.array(output))))
        self.assertTrue(conversion_passes(logit_metrics(reference, reference + 0.001)))
        tie = np.array([10.0, 10.001, 1.0])
        near = np.array([10.001, 10.0, 1.0])
        self.assertNotEqual(tie.argmax(), near.argmax())
        self.assertTrue(conversion_passes(logit_metrics(tie, near)))

    def test_inspection_rejects_missing_wrong_or_incomplete_integer_weight_storage(self):
        audit = {"modules": ["decoder.layers.0.mlp.up_proj"], "parameters_match": True}
        record = {"bits": 4, "kind": "decoder", "functions": {"prefill": audit, "main": audit},
                  "layers": [{"name": "layers.0.mlp.up_proj", "integer": {"shape": [128, 64]}}]}
        functions = [{"name": "prefill"}, {"name": "main"}]
        for storage in ([], [{"name": "Float16", "count": 8192}], [{"name": "Int8", "count": 8192}],
                        [{"name": "Int4", "count": 8191}]):
            with self.subTest(storage=storage):
                self.assertFalse(compressed_storage_matches(record, {"summary": {"storageTypes": storage, "functions": functions}}))
        report = {"summary": {"storageTypes": [{"name": "Int4", "count": 8192}], "functions": functions}}
        self.assertTrue(compressed_storage_matches(record, report))
        del record["functions"]["main"]
        self.assertFalse(compressed_storage_matches(record, report))

    def test_recipe_keeps_three_candidates_separate_and_requires_matching_components(self):
        path = recipe_path("teleocr", "int4_int8")
        recipe = load_recipe(path, {"python": "/usr/bin/python3", "snapshot": "/unused"}, Path(__file__).resolve().parents[3])
        stages = {step["id"]: step for step in recipe["steps"]}
        for name, vision, decoder_name in (("int8", "vision", "decoder_int8"), ("int4", "vision", "decoder_int4"),
                                            ("vision_int8", "vision_int8", "decoder")):
            self.assertEqual(stages["bundle_" + name]["depends"], ["reference", vision, "embed", decoder_name])
            self.assertEqual(stages["verify_" + name]["depends"], [vision, "embed", decoder_name, "bundle_" + name])
            self.assertEqual(stages["verify_" + name]["checks"], [{"file": "comparison.json", "key": ["passed"], "equals": True}])
        for name in ("decoder_int8", "decoder_int4", "vision_int8"):
            self.assertTrue(stages[name]["heavy"])
            self.assertIn("reference.safetensors", stages[name]["outputs"])
            self.assertIn("$repo/Tools/Conversion/adapters/teleocr/quantization.py", stages[name]["inputs"])
        self.assertEqual(recipe_path("teleocr").name, "recipe.json")

    def test_decoder_compression_retains_tied_head_embedding_norms_and_biases(self):
        for bits in (4, 8):
            with self.subTest(bits=bits):
                torch.manual_seed(314)
                model = decoder()
                head = model.head.weight
                retained = {name: tensor_record(value) for name, value in model.named_parameters(remove_duplicate=False)
                            if not any(name == item + ".weight" for item in targets(model, "decoder", bits))}
                model, record, reference = quantize(model, "decoder", bits)
                self.assertIs(model.head.weight, head)
                self.assertIs(model.embed.weight, head)
                self.assertEqual(len(record["layers"]), 14)
                self.assertEqual(record["retained"], retained)
                self.assertEqual(record["granularity"], "block64_axis1")
                for name, weight in reference.items():
                    linear = model.get_submodule(name.removesuffix(".weight"))
                    inputs = torch.randn(3, linear.in_features)
                    torch.testing.assert_close(linear(inputs), nn.functional.linear(inputs, weight, linear.bias),
                                               rtol=0, atol=0)
                with tempfile.TemporaryDirectory() as directory:
                    from safetensors.torch import load_file
                    path = Path(directory) / "reference.safetensors"
                    save_reference(path, reference, record)
                    loaded = load_file(str(path))
                    self.assertEqual(set(loaded), set(reference))
                    for name in reference:
                        torch.testing.assert_close(loaded[name], reference[name], rtol=0, atol=0)
                    with self.assertRaises(FileExistsError):
                        save_reference(path, reference, record)

    def test_vision_int8_supports_non_block_width_and_retains_the_patch_projection(self):
        torch.manual_seed(314)
        model = nn.Module()
        model.patch = nn.Conv3d(3, 8, kernel_size=(2, 2, 2))
        model.blocks = nn.ModuleList([nn.Linear(76, 32), nn.Linear(32, 76)])
        original = tensor_record(model.patch.weight)
        model, record, reference = quantize(model, "vision", 8)
        self.assertEqual(record["granularity"], "channel_axis0")
        self.assertEqual({row["name"] for row in record["layers"]}, {"blocks.0", "blocks.1"})
        self.assertEqual(tensor_record(model.patch.weight), original)
        self.assertEqual(set(reference), {"blocks.0.weight", "blocks.1.weight"})

    def test_unsupported_precision_incomplete_blocks_and_unknown_layers_are_rejected(self):
        for kind, bits in (("decoder", 3), ("decoder", True), ("vision", 4), ("embed", 8)):
            with self.subTest(kind=kind, bits=bits), self.assertRaises(ValueError):
                targets(decoder(), kind, bits)
        with self.assertRaisesRegex(ValueError, "complete blocks"):
            targets(decoder(63), "decoder", 4)
        model = decoder()
        model.layers[0].extra = nn.Linear(64, 64)
        with self.assertRaisesRegex(ValueError, "seven declared"):
            targets(model, "decoder", 4)
        del model.layers[0].extra
        with torch.no_grad():
            model.layers[0].self_attn.q_proj.weight[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            targets(model, "decoder", 4)

    def test_a_second_compression_cannot_requantize_the_same_model(self):
        model, _, _ = quantize(decoder(), "decoder", 4)
        with self.assertRaisesRegex(ValueError, "uncompressed"):
            quantize(model, "decoder", 8)


if __name__ == "__main__":
    unittest.main()
