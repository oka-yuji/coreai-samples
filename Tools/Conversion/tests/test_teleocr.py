import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from Tools.Conversion import local
from Tools.Conversion.adapters.teleocr.contracts import (
    FUNCTIONS, IOS_CONTEXT, IOS_FUNCTIONS, IOS_MASKED, REVISION, WINDOW_LIMIT, digest, inspected_functions, logit_metrics,
    metrics, restore_order, window_inputs, window_order,
)
from Tools.Conversion.adapters.teleocr.prepare import validate
from Tools.Conversion.recipe import load_recipe

ROOT = Path(__file__).resolve().parents[3]


def original_window_index(grid_h, grid_w):
    units_h, units_w = grid_h // 2, grid_w // 2
    index = np.arange(units_h * units_w).reshape(units_h, units_w)
    padded = np.pad(index, ((0, 4 - units_h % 4), (0, 4 - units_w % 4)), constant_values=-100)
    rows, columns = padded.shape[0] // 4, padded.shape[1] // 4
    windows = padded.reshape(rows, 4, columns, 4).transpose(0, 2, 1, 3).reshape(-1)
    return windows[windows != -100]


def inspect_type(tensor):
    dims = " × ".join("?" if dim is None else str(dim) for dim in tensor["shape"])
    return f"NDArray ({tensor['dtype']}, {dims})"


class TeleOCRContractTests(unittest.TestCase):
    def test_padded_window_order_keeps_the_original_unit_order(self):
        for grid in ((4, 4), (32, 32), (64, 64), (74, 74), (86, 26), (26, 86), (2, 30)):
            order = window_order(*grid)
            windows = order.reshape(-1, 16)
            self.assertTrue(np.all((windows >= 0).any(axis=1)))
            np.testing.assert_array_equal(order[order >= 0], original_window_index(*grid))

    def test_window_inputs_mark_padding_and_restore_the_source_order(self):
        rng = np.random.default_rng(0)
        grid = (6, 10)
        units = grid[0] * grid[1] // 4
        pixels = rng.normal(size=(units * 4, 1176)).astype(np.float32)
        rotary = rng.normal(size=(units * 4, 40)).astype(np.float32)
        inputs, order = window_inputs(pixels, rotary, *grid)
        padded = np.repeat(order < 0, 4)
        self.assertEqual(inputs["pixel_values"].shape, (len(order) * 4, 1176))
        self.assertEqual(len(inputs["key_valid"]) % 64, 0)
        np.testing.assert_array_equal(inputs["key_valid"], (~padded).astype(np.float32))
        self.assertTrue(np.all(inputs["pixel_values"][padded] == 0))
        self.assertTrue(np.all(inputs["cos"][padded] == 1) and np.all(inputs["sin"][padded] == 0))
        first = int(np.flatnonzero(order >= 0)[1])
        np.testing.assert_array_equal(inputs["pixel_values"][first * 4:first * 4 + 4],
                                      pixels[order[first] * 4:order[first] * 4 + 4])
        np.testing.assert_allclose(inputs["cos"][first * 4], np.cos(np.concatenate([rotary, rotary], -1)[order[first] * 4]))
        merged = np.where(order >= 0, order, -1).astype(np.float32)[:, None]
        np.testing.assert_array_equal(restore_order(merged, order)[:, 0], np.arange(units))

    def test_window_order_rejects_incomplete_units_and_oversized_images(self):
        with self.assertRaises(ValueError):
            window_order(5, 8)
        with self.assertRaises(ValueError):
            window_order(8 * (WINDOW_LIMIT + 1), 8)

    def test_window_inputs_reject_pixel_rows_outside_the_patch_grid(self):
        with self.assertRaisesRegex(ValueError, "Pixel rows do not match the patch grid"):
            window_inputs(np.zeros((32, 1176)), np.zeros((32, 40)), 6, 10)

    def test_metrics_reject_mismatched_empty_or_nonfinite_reference_arrays(self):
        pairs = [(np.ones(3), np.ones(4)), (np.array([np.nan]), np.ones(1)), (np.array([]), np.array([]))]
        for measure in (metrics, logit_metrics):
            for reference, actual in pairs:
                with self.subTest(measure=measure.__name__, reference=reference), self.assertRaises(ValueError):
                    measure(reference, actual)

    def test_nonfinite_candidates_are_reported_and_never_match(self):
        self.assertFalse(metrics(np.ones(3), np.array([1.0, np.nan, 1.0]))["finite"])
        reference = np.array([0.0, 1.0, 3.0, 2.0])
        for candidate, expected in (([np.nan, 1.0, 3.0, 2.0], False), ([0.0, 1.0, np.nan, 2.0], False),
                                    ([0.0, 1.0, 3.1, 2.0], True)):
            with self.subTest(candidate=candidate):
                result = logit_metrics(reference, np.array(candidate))
                self.assertIs(result["finite"], expected)
                self.assertIs(result["argmax_match"], expected)

    def test_inspect_report_parsing_matches_the_declared_functions(self):
        for functions in (FUNCTIONS["main"], IOS_FUNCTIONS["main"], IOS_FUNCTIONS["vision"]):
            report = {"summary": {"functions": [
                {"name": name, **{group: [{"name": tensor["name"], "type": inspect_type(tensor)}
                                          for tensor in function[group]] for group in ("inputs", "outputs", "states")}}
                for name, function in functions.items()]}}
            self.assertEqual(inspected_functions(report), functions)
        changed = copy.deepcopy(report)
        changed["summary"]["functions"][0]["inputs"][0]["type"] = "NDArray (Float16, ? × 1176)"
        self.assertNotEqual(inspected_functions(changed), IOS_FUNCTIONS["vision"])

    def test_ios_contract_fixes_every_shape_and_shares_one_kv_layout(self):
        self.assertEqual(sorted(IOS_FUNCTIONS["vision"]), ["windows_100", "windows_16", "windows_32", "windows_64"])
        self.assertEqual(sorted(IOS_FUNCTIONS["main"]),
                         ["main", "prefill_1024", "prefill_2048", "prefill_256", "prefill_512"])
        for asset in IOS_FUNCTIONS.values():
            for function in asset.values():
                for tensor in (*function["inputs"], *function["outputs"], *function["states"]):
                    self.assertNotIn(None, tensor["shape"])
        states = {str(function["states"]) for function in IOS_FUNCTIONS["main"].values()}
        self.assertEqual(len(states), 1)
        self.assertEqual(IOS_FUNCTIONS["main"]["main"]["states"][0]["shape"], [28, 1, 8, IOS_CONTEXT, 128])

    def test_prepare_rejects_a_changed_pinned_file(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / REVISION
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}")
            specification = {"model_id": "XingChen-AGI/TeleOCR", "revision": REVISION,
                             "files": {"config.json": {"bytes": 2, "sha256": digest(snapshot / "config.json")}}}
            validate(snapshot, specification)
            (snapshot / "config.json").write_text("[]")
            with self.assertRaises(ValueError):
                validate(snapshot, specification)

    def test_recipe_gates_every_component_and_the_bundle(self):
        recipe = load_recipe(ROOT / "Tools/Conversion/recipes/teleocr/recipe.json",
                             {"python": sys.executable, "snapshot": "/unused"}, ROOT)
        steps = {step["id"]: step for step in recipe["steps"]}
        self.assertEqual(list(steps), ["prepare", "reference", "vision", "embed", "decoder", "bundle", "verify"])
        for name in ("vision", "embed", "decoder", "verify"):
            self.assertEqual(steps[name]["checks"], [{"file": "comparison.json", "key": ["passed"], "equals": True}])
        self.assertTrue(all(steps[name]["heavy"] for name in ("reference", "vision", "embed", "decoder")))

    def test_ios_recipe_keeps_the_dynamic_baseline_and_gates_every_ios_bundle(self):
        path = local.recipe_path("teleocr", "ios")
        self.assertEqual(path.name, "recipe-ios.json")
        recipe = load_recipe(path, {"python": sys.executable, "snapshot": "/unused"}, ROOT)
        steps = {step["id"]: step for step in recipe["steps"]}
        self.assertEqual(list(steps)[:7], ["prepare", "reference", "vision", "embed", "decoder", "bundle", "verify"])
        gated = ["vision_ios", "embed_ios", "decoder_ios", "decoder_ios_pal4", "verify_ios", "verify_ios_pal4"]
        for name in gated:
            self.assertEqual(steps[name]["checks"], [{"file": "comparison.json", "key": ["passed"], "equals": True}])
        self.assertIn("vision", steps["vision_ios"]["depends"])
        for name, precision in (("bundle_ios", "fp16"), ("bundle_ios_pal4", "pal4")):
            command = " ".join(steps[name]["command"])
            self.assertIn(f"--platform ios --precision {precision}", command)
        self.assertIn("decoder_ios_pal4", steps["verify_ios_pal4"]["depends"])

    def test_dynamic_quantized_and_ios_recipes_keep_separate_work_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "teleocr.paths.json").write_text("{}")
            for precision, name in ((None, "teleocr"), ("int4_int8", "teleocr-quantized"), ("ios", "teleocr-ios")):
                with self.subTest(precision=precision):
                    recipe = local.recipe_path("teleocr", precision)
                    command = local.pipeline_command("run", recipe, workspace, workspace / "heavy-job", False)
                    self.assertEqual(command[command.index("--work-dir") + 1], str(workspace / "work" / name))
                    self.assertEqual(command[command.index("--lock-dir") + 1], str(workspace / "heavy-job"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "The pinned TeleOCR environment is required")
class TeleOCRStaticWrapperTests(unittest.TestCase):
    def test_static_decoder_rejects_unbounded_errors_and_accepts_only_close_top_two_divergence(self):
        from Tools.Conversion.adapters.teleocr.ios import decoder_comparison_passes
        reference = np.array([0.0, 10.0, 1.0])
        for values in ([0.0, 0.0, 0.0], [60000.0, -60000.0, -60000.0], [0.0, 20.0, 2.0], [0.0, np.nan, 1.0]):
            check = logit_metrics(reference, np.array(values))
            with self.subTest(values=values):
                self.assertFalse(decoder_comparison_passes(check))
                self.assertFalse(check["numeric_divergence"])
        self.assertTrue(decoder_comparison_passes(logit_metrics(reference, reference + 0.001)))
        near = logit_metrics(np.array([10.0, 10.001, 1.0]), np.array([10.001, 10.0, 1.0]))
        self.assertTrue(decoder_comparison_passes(near))
        self.assertTrue(near["numeric_divergence"])
        unrelated = logit_metrics(np.array([10.0, 10.001, 9.999]), np.array([10.0, 9.998, 10.002]))
        self.assertFalse(decoder_comparison_passes(unrelated))

    def test_static_inputs_pad_hide_and_mark_the_decode_position(self):
        from Tools.Conversion.adapters.teleocr.ios import padded_window_rows, prefill_inputs, step_inputs
        rows = {"pixel_values": np.ones((64, 1176), np.float32), "key_valid": np.ones(64, np.float32),
                "cos": np.full((64, 80), 0.5, np.float32), "sin": np.full((64, 80), 0.5, np.float32)}
        padded = padded_window_rows(rows, 2)
        self.assertEqual(padded["pixel_values"].shape, (128, 1176))
        self.assertTrue(np.all(padded["pixel_values"][64:] == 0) and np.all(padded["key_valid"][64:] == 0))
        self.assertTrue(np.all(padded["cos"][64:] == 1) and np.all(padded["sin"][64:] == 0))
        feeds = prefill_inputs(np.ones((1, 3, 4), np.float32), np.zeros((1, 3, 2), np.float32),
                               np.zeros((1, 3, 2), np.float32), 5)
        expected = np.full((5, 5), IOS_MASKED, np.float32)
        for row in range(3):
            expected[row, :row + 1] = 0
        np.testing.assert_array_equal(feeds["mask"][0, 0], expected)
        self.assertEqual(feeds["logit_row"].tolist(), [2])
        self.assertTrue(np.all(feeds["inputs_embeds"][0, 3:] == 0) and np.all(feeds["cos"][0, 3:] == 1))
        step = step_inputs(7, 4, 10, 1e6, 8)
        self.assertEqual(int(step["write_mask"].sum()), 1)
        self.assertEqual(int(step["write_mask"].argmax()), 4)
        self.assertTrue(np.all(step["mask"][..., :5] == 0) and np.all(step["mask"][..., 5:] == IOS_MASKED))
        self.assertEqual(step["mask"].shape[-1], IOS_CONTEXT)

    def test_static_prefill_and_one_hot_decode_match_the_dynamic_decoder(self):
        import torch
        from torch import nn
        from Tools.Conversion.adapters.teleocr.wrappers import Decoder, StaticMain, StaticPrefill
        torch.manual_seed(3)
        layers, heads, kv_heads, head_dim, hidden, vocab, capacity = 2, 4, 2, 8, 32, 40, 24

        def norm(size):
            return SimpleNamespace(weight=torch.rand(size) + 0.5, variance_epsilon=1e-6)

        def layer():
            attention = nn.Module()
            attention.q_proj, attention.k_proj = nn.Linear(hidden, heads * head_dim), nn.Linear(hidden, kv_heads * head_dim)
            attention.v_proj, attention.o_proj = nn.Linear(hidden, kv_heads * head_dim), nn.Linear(heads * head_dim, hidden)
            attention.q_norm, attention.k_norm, attention.scaling = norm(head_dim), norm(head_dim), head_dim ** -0.5
            return SimpleNamespace(self_attn=attention, input_layernorm=norm(hidden),
                                   post_attention_layernorm=norm(hidden), mlp=nn.Linear(hidden, hidden))

        embed = nn.Embedding(vocab, hidden)
        head = nn.Linear(hidden, vocab, bias=False)
        head.weight = embed.weight
        model = SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(layers=[layer() for _ in range(layers)], norm=norm(hidden),
                                                                 embed_tokens=embed)),
            lm_head=head, config=SimpleNamespace(text_config=SimpleNamespace(
                head_dim=head_dim, num_key_value_heads=kv_heads, hidden_size=hidden, num_attention_heads=heads,
                vocab_size=vocab)))
        core = Decoder(model).eval()
        count, bucket, steps = 5, 8, 3
        ids = torch.randint(0, vocab, (1, count + steps))
        cos, sin = torch.rand(1, count + steps, head_dim), torch.rand(1, count + steps, head_dim)
        with torch.no_grad():
            shape = (layers, 1, kv_heads, capacity, head_dim)
            keys, values = torch.zeros(shape), torch.zeros(shape)
            causal = torch.full((count, count), float("-inf")).triu(1)[None, None]
            expected = [core.decode(core.embed(ids[:, :count]), cos[:, :count], sin[:, :count], causal, keys, values)]
            for step in range(steps):
                position = count + step
                expected.append(core.decode(core.embed(ids[:, position:position + 1]), cos[:, position:position + 1],
                                            sin[:, position:position + 1], torch.zeros(1, 1, 1, position + 1), keys, values))
            keys, values = torch.zeros(shape), torch.zeros(shape)
            mask = torch.full((1, 1, bucket, bucket), IOS_MASKED)
            mask[0, 0, :count, :count] = torch.full((count, count), IOS_MASKED).triu(1)
            embeddings = torch.cat([core.embed(ids[:, :count]), torch.zeros(1, bucket - count, hidden)], 1)
            padding = (torch.ones(1, bucket - count, head_dim), torch.zeros(1, bucket - count, head_dim))
            actual = [StaticPrefill(core)(embeddings, torch.cat([cos[:, :count], padding[0]], 1),
                                          torch.cat([sin[:, :count], padding[1]], 1), mask,
                                          torch.tensor([count - 1], dtype=torch.int32), keys, values)]
            for step in range(steps):
                position = count + step
                decode_mask = torch.full((1, 1, 1, capacity), IOS_MASKED)
                decode_mask[..., :position + 1] = 0
                write = torch.zeros(capacity)
                write[position] = 1
                actual.append(StaticMain(core)(ids[:, position:position + 1].to(torch.int32), cos[:, position:position + 1],
                                               sin[:, position:position + 1], decode_mask, write, keys, values))
        for reference, candidate in zip(expected, actual):
            torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
