import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Tools.Conversion.adapters.embeddinggemma2.preprocessing import KEYS, compare


def write_dump(folder, identifiers=("photo", "gradient"), revision="abc", png="same"):
    folder.mkdir()
    fixtures = []
    for index, name in enumerate(identifiers):
        (folder / name).mkdir()
        arrays = {"pixels": np.full((4, 3, 3), index, dtype=np.uint8),
                  "input_ids": np.arange(8, dtype=np.int64)[None],
                  "attention_mask": np.ones((1, 8), dtype=np.int64),
                  "pixel_values": np.linspace(-1, 1, 12, dtype=np.float32).reshape(1, 3, 4),
                  "image_position_ids": np.arange(6, dtype=np.int64).reshape(1, 3, 2)}
        for key, array in arrays.items():
            np.save(folder / name / f"{key}.npy", array, allow_pickle=False)
        fixtures.append({"id": name, "png_sha256": f"{png}-{name}"})
    (folder / "summary.json").write_text(json.dumps(
        {"model_revision": revision, "versions": {"pillow": folder.name}, "fixtures": fixtures}))
    return folder


def bump(path, index, amount):
    array = np.load(path, allow_pickle=False)
    array.flat[index] += amount
    np.save(path, array, allow_pickle=False)


class EmbeddingPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.left = write_dump(self.root / "left")
        self.right = write_dump(self.root / "right")

    def tearDown(self):
        self.temp.cleanup()

    def equal_arrays(self, result):
        return {(row["id"], key): item["equal"] for row in result["fixtures"] for key, item in row["arrays"].items()}

    def test_identical_dumps_are_equal_with_zero_difference(self):
        result = compare(self.left, self.right)
        self.assertTrue(result["equal"])
        self.assertEqual([row["id"] for row in result["fixtures"]], ["photo", "gradient"])
        self.assertEqual((result["left"]["versions"], result["right"]["versions"]), ({"pillow": "left"}, {"pillow": "right"}))
        for row in result["fixtures"]:
            self.assertTrue(row["pixels_equal"] and row["png_bytes_equal"])
            self.assertEqual(list(row["arrays"]), KEYS)
            for item in row["arrays"].values():
                self.assertTrue(item["equal"])
                self.assertEqual(item["max_abs_difference"], 0)

    def test_png_byte_differences_are_reported_without_failing_equality(self):
        result = compare(self.left, write_dump(self.root / "encoded", png="encoded"))
        self.assertTrue(result["equal"])
        self.assertEqual([row["png_bytes_equal"] for row in result["fixtures"]], [False, False])

    def test_one_changed_element_fails_only_that_fixture_and_key(self):
        bump(self.right / "gradient" / "pixel_values.npy", 5, 0.25)
        result = compare(self.left, self.right)
        self.assertFalse(result["equal"])
        self.assertEqual([key for key, equal in self.equal_arrays(result).items() if not equal], [("gradient", "pixel_values")])
        self.assertEqual(result["fixtures"][1]["arrays"]["pixel_values"]["max_abs_difference"], 0.25)

    def test_changed_pixels_fail_equality(self):
        bump(self.right / "photo" / "pixels.npy", 0, 7)
        result = compare(self.left, self.right)
        self.assertFalse(result["equal"])
        self.assertEqual([row["pixels_equal"] for row in result["fixtures"]], [False, True])
        self.assertTrue(all(self.equal_arrays(result).values()))

    def test_shape_or_dtype_differences_fail_equality(self):
        np.save(self.right / "photo" / "attention_mask.npy", np.ones((1, 9), dtype=np.int64), allow_pickle=False)
        np.save(self.right / "gradient" / "image_position_ids.npy", np.arange(6, dtype=np.int32).reshape(1, 3, 2), allow_pickle=False)
        np.save(self.right / "photo" / "pixels.npy", np.zeros((4, 3, 3), dtype=np.int64), allow_pickle=False)
        result = compare(self.left, self.right)
        self.assertFalse(result["equal"])
        self.assertEqual([row["pixels_equal"] for row in result["fixtures"]], [False, True])
        shape = result["fixtures"][0]["arrays"]["attention_mask"]
        self.assertEqual((shape["equal"], shape["shape"], shape["max_abs_difference"]), (False, {"left": [1, 8], "right": [1, 9]}, None))
        dtype = result["fixtures"][1]["arrays"]["image_position_ids"]
        self.assertEqual((dtype["equal"], dtype["dtype"], dtype["max_abs_difference"]), (False, {"left": "int64", "right": "int32"}, 0))

    def test_unsigned_differences_do_not_wrap(self):
        for folder, value in ((self.left, 3), (self.right, 7)):
            np.save(folder / "photo" / "input_ids.npy", np.full((1, 8), value, dtype=np.uint8), allow_pickle=False)
        self.assertEqual(compare(self.left, self.right)["fixtures"][0]["arrays"]["input_ids"]["max_abs_difference"], 4.0)

    def test_fixture_identity_mismatch_is_rejected(self):
        cases = [(self.left, {"identifiers": ("gradient", "photo")}), (self.left, {"identifiers": ("photo",)}),
                 (self.left, {"revision": "other"}), (write_dump(self.root / "empty", identifiers=()), {"identifiers": ()})]
        for index, (left, options) in enumerate(cases):
            with self.subTest(options=options), self.assertRaises(ValueError):
                compare(left, write_dump(self.root / f"mismatch-{index}", **options))


if __name__ == "__main__":
    unittest.main()
