import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from chronos.chronos_bolt import InstanceNorm

from Tools.Conversion.adapters.chronos2 import convert, model_wrapper, reference
from Tools.Conversion.adapters.chronos2.artifacts import CORE_CONFIG, FORECAST_CONFIG, QUANTILES
from Tools.Conversion.adapters.chronos2.artifacts import validate_config, vector_document
from Tools.Conversion.storage import read_json, write_json
from Tools.Conversion.tests.test_chronos2_support import ChronosFiles


class SyntheticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = {**copy.deepcopy(CORE_CONFIG), "chronos_config": copy.deepcopy(FORECAST_CONFIG),
                  "torch_dtype": "float32"}
        self.config = SimpleNamespace(to_dict=lambda: copy.deepcopy(config))
        self.instance_norm = InstanceNorm(eps=1e-5, use_arcsinh=True)
        self.patch = SimpleNamespace(patch_size=16, patch_stride=16)
        self.num_quantiles = 21
        self.register_buffer("quantiles", torch.tensor(QUANTILES, dtype=torch.float32))

    def forward(self, context, num_output_patches):
        if num_output_patches != 1:
            raise ValueError("Expected one output patch")
        normalized, scale = self.instance_norm(context)
        patches = torch.nan_to_num(normalized).unfold(-1, 16, 16)
        output = patches.mean(dim=-2).unsqueeze(1).expand(1, 21, 16)
        return SimpleNamespace(quantile_preds=self.instance_norm.inverse(output, scale))


class WrapperTests(unittest.TestCase):
    def test_norm_matches_original_formula_and_inherits_inverse(self):
        original = InstanceNorm(eps=1e-5, use_arcsinh=True)
        adapted = model_wrapper.ExportNorm(eps=1e-5, use_arcsinh=True)
        self.assertIs(model_wrapper.ExportNorm.inverse, InstanceNorm.inverse)
        for values in ([0.0] * 16, [3.0] * 16, [float("nan")] * 15 + [-4.0],
                       [float("nan"), 1.0, 3.0, 5.0], [float("nan")] * 16):
            with self.subTest(values=values):
                inputs = torch.tensor([values], dtype=torch.float32)
                expected, expected_scale = original(inputs)
                actual, scale = adapted(inputs)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
                for actual_part, expected_part in zip(scale, expected_scale):
                    torch.testing.assert_close(actual_part, expected_part, atol=0, rtol=0, equal_nan=True)
                torch.testing.assert_close(adapted.inverse(actual, scale), original.inverse(expected, expected_scale),
                                           atol=0, rtol=0, equal_nan=True)
                torch.testing.assert_close(adapted(inputs, scale)[0], original(inputs, scale)[0],
                                           atol=0, rtol=0, equal_nan=True)
                self.assertEqual(scale[1].dtype, torch.float32)

    def test_decomposition_copy_keeps_process_table_unchanged(self):
        sentinel = object()
        base = {torch.ops.aten.isnan.default: sentinel}
        copied = model_wrapper.decomposition_table(base)
        self.assertEqual(base, {torch.ops.aten.isnan.default: sentinel})
        self.assertIsNot(copied, base)
        value = torch.tensor([1.0, float("nan"), float("inf"), -float("inf")])
        torch.testing.assert_close(copied[torch.ops.aten.isnan.default](value), torch.isnan(value))
        for length in (16, 32, 1008):
            tensor = torch.arange(length, dtype=torch.float32).reshape(1, -1)
            for dimension in (-1, 1):
                torch.testing.assert_close(model_wrapper.lower_unfold(tensor, dimension, 16, 16),
                                           tensor.unfold(dimension, 16, 16))

    def test_unsupported_unfold_contracts_are_rejected(self):
        for dimension, size, step, length in ((0, 16, 16, 32), (-2, 16, 16, 32), (-1, 8, 16, 32),
                                               (-1, 16, 8, 32), (-1, 16, 16, 17), (-1, 16, 16, 0)):
            with self.subTest(dimension=dimension, size=size, step=step, length=length), self.assertRaises(ValueError):
                model_wrapper.lower_unfold(torch.ones(1, length), dimension, size, step)

    def test_wrapper_rejects_shape_mask_precision_and_model_contract_changes(self):
        wrapper = model_wrapper.Chronos2Wrapper(SyntheticModel().eval()).eval()
        invalid = [torch.ones(2, 16), torch.ones(16), torch.ones(1, 17), torch.ones(1, 8208),
                   torch.ones(1, 16, dtype=torch.float64), torch.full((1, 16), float("nan")),
                   torch.full((1, 16), float("inf"))]
        for context in invalid:
            with self.subTest(shape=context.shape, dtype=context.dtype), self.assertRaises(ValueError):
                wrapper(context)
        for mutate in (lambda m: setattr(m.instance_norm, "eps", 1e-4),
                       lambda m: setattr(m.instance_norm, "use_arcsinh", False),
                       lambda m: setattr(m.patch, "patch_stride", 8),
                       lambda m: setattr(m, "quantiles", m.quantiles.flip(0)), lambda m: m.half(), lambda m: m.train()):
            model = SyntheticModel().eval()
            mutate(model)
            with self.assertRaises(ValueError):
                model_wrapper.Chronos2Wrapper(model)

    def test_small_synthetic_export_supports_boundary_patch_counts(self):
        wrapper = model_wrapper.Chronos2Wrapper(SyntheticModel().eval()).eval()
        exported = model_wrapper.export_program(wrapper, {})
        for length in (16, 32, 1008, 8192):
            with self.subTest(length=length):
                context = torch.linspace(-3, 9, length).reshape(1, -1)
                context[0, 0] = float("nan")
                graph = exported.module(check_guards=length != 16)
                torch.testing.assert_close(graph(context), wrapper(context))
        ranges = [(value.lower, value.upper) for value in exported.range_constraints.values()]
        self.assertIn((16, 8192), ranges)
        for length in (17, 8208):
            with self.subTest(invalid_length=length), self.assertRaises(RuntimeError):
                exported.module(check_guards=False)(torch.ones(1, length))
        targets = {node.target for node in exported.graph.nodes}
        self.assertNotIn(torch.ops.aten.isnan.default, targets)
        self.assertNotIn(torch.ops.aten.unfold.default, targets)


class ReferenceTests(ChronosFiles, unittest.TestCase):
    def test_installed_official_configuration_preserves_the_pinned_contract(self):
        from chronos.chronos2.config import Chronos2CoreConfig

        config = Chronos2CoreConfig(**self.config)
        validate_config(config.to_dict())

    def test_reference_calls_unmodified_official_model_without_gradients(self):
        model = SyntheticModel().eval()
        normalization = model.instance_norm
        calls = []
        def observe(module, args, kwargs):
            calls.append((torch.is_grad_enabled(), kwargs["context"].dtype, kwargs["context"].shape,
                          kwargs["num_output_patches"]))
        hook = model.register_forward_pre_hook(observe, with_kwargs=True)
        with patch.object(reference, "load_official", return_value=model):
            document = reference.reference(self.model_path, self.fixture_path, self.snapshot,
                                           self.ready, self.root / "reference-out")
        hook.remove()
        self.assertIs(model.instance_norm, normalization)
        self.assertIs(type(normalization), InstanceNorm)
        self.assertEqual(len(calls), 18)
        self.assertTrue(all(not grad and dtype == torch.float32 and patches == 1
                            for grad, dtype, _, patches in calls))
        self.assertEqual(document["cases"][12]["context"], self.fixtures["cases"][12]["context"])
        wrapper = model_wrapper.Chronos2Wrapper(SyntheticModel().eval()).eval()
        _, comparison = convert.compare_wrapper(wrapper, document, self.identity, self.fixtures)
        self.assertTrue(comparison["passed"])
        self.assertEqual(comparison["elements"], 6048)
        document["cases"][-1]["quantiles"][0][-1][-1] += 1.0
        _, comparison = convert.compare_wrapper(wrapper, document, self.identity, self.fixtures)
        self.assertFalse(comparison["passed"])

    def test_reference_and_export_reject_existing_output_before_loading(self):
        out = self.root / "existing"
        out.mkdir()
        (out / "keep").write_text("keep")
        with patch.object(reference, "load_official") as loader, self.assertRaises(ValueError):
            reference.reference(self.model_path, self.fixture_path, self.snapshot, self.ready, out)
        loader.assert_not_called()
        with patch.object(convert, "load_official") as loader, self.assertRaises(ValueError):
            convert.convert(self.model_path, self.fixture_path, self.snapshot, self.ready, self.reference_path, out)
        loader.assert_not_called()
        self.assertEqual((out / "keep").read_text(), "keep")

    def test_conversion_uses_release_explicit_names_and_all_case_comparison(self):
        values = reference.run_reference(SyntheticModel().eval(), self.fixtures)
        expected = vector_document("reference", self.identity, self.fixtures, values)
        write_json(self.reference_path, expected)
        program = Mock()
        def save_asset(path, metadata):
            self.assertEqual(path.suffix, ".aimodel")
            self.assertEqual(metadata.author, "Amazon")
            self.assertEqual(metadata.license, "Apache-2.0")
            path.mkdir()
            (path / "main.mlirb").write_bytes(b"synthetic conversion")
        program.save_asset.side_effect = save_asset
        converter = Mock()
        converter.to_coreai.return_value = program
        factory = Mock(return_value=converter)
        factory.Mode.RELEASE = "release"
        table = {}
        fake_torch = SimpleNamespace(TorchConverter=factory, get_decomp_table=lambda: table)
        fake_runtime = SimpleNamespace(AIModelAssetMetadata=Mock)
        exported = object()
        import sys
        with patch.dict(sys.modules, {"coreai_torch": fake_torch, "coreai.runtime": fake_runtime}), \
                patch.object(model_wrapper, "export_program", return_value=exported) as exporter, \
                patch.object(convert, "load_official", return_value=SyntheticModel().eval()):
            result = convert.convert(self.model_path, self.fixture_path, self.snapshot, self.ready,
                                     self.reference_path, self.root / "converted")
        factory.assert_called_once_with(mode="release")
        converter.add_exported_program.assert_called_once_with(
            exported, input_names=["context"], output_names=["quantiles"], state_names=[], entrypoint_name="main")
        self.assertIs(exporter.call_args.args[1], table)
        self.assertEqual(result["identity"], self.identity)
        comparison = read_json(self.root / "converted/wrapper-comparison.json")
        self.assertTrue(comparison["passed"])
        self.assertEqual((comparison["count"], comparison["elements"]), (18, 6048))
        expected["cases"][-1]["quantiles"][0][-1][-1] += 1
        write_json(self.reference_path, expected)
        with patch.dict(sys.modules, {"coreai_torch": fake_torch, "coreai.runtime": fake_runtime}), \
                patch.object(model_wrapper, "export_program") as exporter, \
                patch.object(convert, "load_official", return_value=SyntheticModel().eval()), \
                self.assertRaises(ValueError):
            convert.convert(self.model_path, self.fixture_path, self.snapshot, self.ready,
                            self.reference_path, self.root / "rejected")
        exporter.assert_not_called()
        self.assertFalse((self.root / "rejected/Model.aimodel").exists())


if __name__ == "__main__":
    unittest.main()
