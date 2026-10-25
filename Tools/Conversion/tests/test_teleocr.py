import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from Tools.Conversion.adapters.teleocr.contracts import (
    FUNCTIONS, REVISION, WINDOW_LIMIT, digest, inspected_functions, restore_order, window_inputs, window_order,
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

    def test_inspect_report_parsing_matches_the_declared_functions(self):
        report = {"summary": {"functions": [
            {"name": name, **{group: [{"name": tensor["name"], "type": inspect_type(tensor)} for tensor in function[group]]
                              for group in ("inputs", "outputs", "states")}}
            for name, function in FUNCTIONS["main"].items()]}}
        self.assertEqual(inspected_functions(report), FUNCTIONS["main"])
        changed = copy.deepcopy(report)
        changed["summary"]["functions"][0]["states"][0]["type"] = "NDArray (Float32, 28 × 1 × 8 × ? × 128)"
        self.assertNotEqual(inspected_functions(changed), FUNCTIONS["main"])

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


if __name__ == "__main__":
    unittest.main()
