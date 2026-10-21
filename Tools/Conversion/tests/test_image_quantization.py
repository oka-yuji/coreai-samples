import copy
import unittest

import torch

from Tools.Conversion.adapters.qwen_image21.quantization import (
    plain_reference, quantize, store_linear_weights_in_half,
)


class ImageQuantizationTests(unittest.TestCase):
    def make(self, mixed):
        torch.manual_seed(216)
        model = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.GELU(), torch.nn.Linear(64, 8))
        model = store_linear_weights_in_half(model) if mixed else model.half()
        values = torch.randn(5, 128, dtype=torch.float32 if mixed else torch.float16)
        return model.eval().requires_grad_(False), (values,)

    def test_finalized_w4_and_w8_references_preserve_weights_biases_and_compute_precision(self):
        for bits in (4, 8):
            for mixed in (False, True):
                with self.subTest(bits=bits, mixed=mixed):
                    model, example = self.make(mixed)
                    finalized, record = quantize(model, example, bits)
                    reference = plain_reference(finalized, record)
                    self.assertEqual(len(record["layers"]), 2)
                    self.assertFalse(any(torch.nn.utils.parametrize.is_parametrized(m) for m in reference.modules()))
                    self.assertTrue(all(row["scale"]["dtype"] == "float16" for row in record["layers"]))
                    self.assertEqual(reference(*example).dtype, torch.float32 if mixed else torch.float16)
                    torch.testing.assert_close(reference(*example), finalized(*example), atol=0, rtol=0)
                    for row in record["layers"]:
                        expected = finalized.get_submodule(row["name"])
                        actual = reference.get_submodule(row["name"])
                        torch.testing.assert_close(actual.weight, expected.weight.to(actual.weight.dtype), atol=0, rtol=0)
                        torch.testing.assert_close(actual.bias, expected.bias.to(actual.bias.dtype), atol=0, rtol=0)

    def test_missing_duplicate_and_changed_weight_identities_are_rejected(self):
        model, example = self.make(True)
        finalized, record = quantize(model, example, 4)
        for kind in ("missing", "duplicate", "changed"):
            altered = copy.deepcopy(record)
            if kind == "missing":
                altered["layers"].pop()
            elif kind == "duplicate":
                altered["layers"].append(copy.deepcopy(altered["layers"][0]))
            else:
                altered["layers"][0]["integer"]["sha256"] = "0" * 64
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                plain_reference(finalized, altered)
        changed = copy.deepcopy(finalized)
        params = changed.get_submodule(record["layers"][0]["name"]).parametrizations.weight[0]
        params.scale.mul_(2)
        with self.assertRaises(ValueError):
            plain_reference(changed, record)

    def test_unsupported_storage_groups_and_repeated_quantization_fail_closed(self):
        model, example = self.make(False)
        for bits, group in ((2, 64), (True, 64), (4, 32)):
            with self.subTest(bits=bits, group=group), self.assertRaises(ValueError):
                quantize(model, example, bits, group)
        with self.assertRaises(ValueError):
            quantize(torch.nn.Sequential(torch.nn.Linear(128, 8)).eval(), (torch.ones(1, 128),), 4)
        with self.assertRaises(ValueError):
            quantize(torch.nn.Sequential(torch.nn.Linear(96, 8)).half().eval(), (torch.ones(1, 96).half(),), 4)
        finalized, _ = quantize(model, example, 4)
        with self.assertRaises(ValueError):
            quantize(finalized, example, 4)

    def test_nonfinite_and_invalid_export_scale_never_produce_a_reference(self):
        model, example = self.make(True)
        finalized, record = quantize(model, example, 4)
        for value in (0.0, -1.0, float("nan"), float("inf")):
            candidate = copy.deepcopy(finalized)
            params = candidate.get_submodule(record["layers"][0]["name"]).parametrizations.weight[0]
            params.scale.fill_(value)
            with self.subTest(scale=value), self.assertRaises(ValueError):
                plain_reference(candidate, record)


if __name__ == "__main__":
    unittest.main()
