import unittest

import numpy as np

from Tools.Conversion.adapters.qwen_image21.contracts import layout, pad, tensor_metrics, validate_arrays


class ImageLayoutTests(unittest.TestCase):
    def test_text_is_causal_and_target_sees_the_entire_prefix_and_target_image(self):
        values = layout(np.array([False, False, False]))
        attention = values["mask"][0, 0]
        self.assertEqual(values["length"], 1027)
        self.assertEqual(values["reference_count"], 0)
        np.testing.assert_array_equal(attention[:3, :4], [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
        ])
        self.assertTrue(attention[3:1027, :1027].all())
        np.testing.assert_array_equal(values["text_index"][:3], [0, 1, 2])
        np.testing.assert_array_equal(values["image_index"][3:1027], np.arange(1024))
        np.testing.assert_array_equal(values["target_indices"], np.arange(3, 1027))
        np.testing.assert_array_equal(np.flatnonzero(values["target"]), np.arange(3, 1027))

    def test_reference_is_bidirectional_without_seeing_later_text_or_target(self):
        slots = np.array([False, False] + [True] * 256 + [False, False, False])
        values = layout(slots)
        attention = values["mask"][0, 0]
        self.assertEqual(values["length"], 2053)
        self.assertEqual(values["reference_count"], 1)
        self.assertFalse(attention[:2, 2:].any())
        self.assertTrue(attention[2:1026, :1026].all())
        self.assertFalse(attention[2:1026, 1026:].any())
        self.assertTrue(attention[1026, :1027].all())
        self.assertFalse(attention[1026, 1027:].any())
        self.assertTrue(attention[1029:2053, :2053].all())
        np.testing.assert_array_equal(values["text_index"][[0, 1, 1026, 1027, 1028]], [0, 1, 258, 259, 260])
        np.testing.assert_array_equal(values["image_index"][2:1026], np.arange(1024))
        np.testing.assert_array_equal(values["image_index"][1029:2053], np.arange(1024, 2048))
        np.testing.assert_array_equal(values["target_indices"], np.arange(1029, 2053))
        np.testing.assert_array_equal(np.flatnonzero(values["target"]), np.arange(1029, 2053))
        self.assertFalse(values["image_mask"][[0, 1, 1026, 1027, 1028]].any())
        self.assertTrue(values["image_mask"][2:1026].all())
        self.assertTrue(values["image_mask"][1029:2053].all())

    def test_adjacent_reference_and_target_remain_separate_attention_blocks(self):
        values = layout(np.ones(256, dtype=np.bool_))
        attention = values["mask"][0, 0]
        self.assertEqual(values["length"], 2048)
        self.assertTrue(values["image_mask"][:2048].all())
        self.assertTrue(attention[:1024, :1024].all())
        self.assertFalse(attention[:1024, 1024:2048].any())
        self.assertTrue(attention[1024:2048, :2048].all())
        np.testing.assert_array_equal(values["target_indices"], np.arange(1024, 2048))
        np.testing.assert_array_equal(values["image_index"][:2048], np.arange(2048))

    def test_padding_never_becomes_an_attention_key_or_output_target(self):
        values = layout(np.array([False, False, False]))
        attention = values["mask"][0, 0]
        self.assertEqual(attention.shape, (2304, 2304))
        self.assertEqual(attention.dtype, np.bool_)
        self.assertFalse(attention[:, 1027:].any())
        self.assertTrue(attention[1027:, :1027].all())
        self.assertFalse(values["image_mask"][1027:].any())
        self.assertFalse(values["target"][1027:].any())
        self.assertTrue((values["text_index"][1027:] == 0).all())
        self.assertTrue((values["image_index"][1027:] == 0).all())

    def test_maximum_prompt_with_one_reference_fills_the_joint_budget_exactly(self):
        values = layout(np.array([False] * 128 + [True] * 256 + [False] * 128))
        self.assertEqual(values["length"], 2304)
        self.assertEqual(values["text_index"][1152], 384)
        self.assertEqual(values["text_index"][1279], 511)
        np.testing.assert_array_equal(values["target_indices"], np.arange(1280, 2304))
        self.assertEqual(values["image_index"][2303], 2047)
        self.assertEqual(layout(np.zeros(512, dtype=np.bool_))["length"], 1536)

    def test_image_slot_count_must_be_zero_or_exactly_one_merged_image(self):
        for count in (1, 255, 257, 512):
            with self.subTest(count=count), self.assertRaises(ValueError):
                layout(np.ones(count, dtype=np.bool_))

    def test_layout_rejects_empty_oversized_nonboolean_and_batched_prompts(self):
        for slots in (
            np.array([], dtype=np.bool_),
            np.zeros(513, dtype=np.bool_),
            np.array([0, 1], dtype=np.int32),
            np.array([[False, False]]),
            np.array(False),
        ):
            with self.subTest(shape=slots.shape, dtype=slots.dtype), self.assertRaises(ValueError):
                layout(slots)


class ImageTensorContractTests(unittest.TestCase):
    def test_padding_preserves_values_and_dtype_and_zeros_only_the_unused_region(self):
        source = np.array([[1, 2], [3, 4]], dtype=np.int32)
        actual = pad(source, (3, 4))
        self.assertEqual(actual.dtype, np.int32)
        np.testing.assert_array_equal(actual, [[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 0, 0]])
        np.testing.assert_array_equal(pad(source, (2, 2)), source)

    def test_padding_rejects_truncation_and_rank_changes(self):
        for source, shape in ((np.ones((2, 4)), (3, 3)), (np.ones((2, 2)), (4,))):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                pad(source, shape)

    def test_metrics_use_both_error_limits(self):
        reference = np.array([2.0, 2.0])
        actual = np.array([2.25, 2.0])
        passing = tensor_metrics(reference, actual, max_limit=0.125, rmse_limit=0.1)
        self.assertTrue(passing["passed"])
        self.assertAlmostEqual(passing["normalized_max"], 0.125)
        self.assertAlmostEqual(passing["normalized_rmse"], 0.08838834764831845)
        self.assertFalse(tensor_metrics(reference, actual, max_limit=0.1, rmse_limit=0.1)["passed"])
        self.assertFalse(tensor_metrics(reference, actual, max_limit=0.2, rmse_limit=0.05)["passed"])
        self.assertTrue(tensor_metrics(np.zeros(2), np.zeros(2))["passed"])

    def test_metrics_reject_mismatched_empty_or_nonfinite_reference_arrays(self):
        pairs = [(np.ones((2, 2)), np.ones(4)), (np.array([]), np.array([]))]
        pairs.extend((np.array([value]), np.array([0.0])) for value in (np.nan, np.inf, -np.inf))
        for reference, actual in pairs:
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                tensor_metrics(reference, actual)

    def test_nonfinite_candidate_metrics_cannot_pass(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                result = tensor_metrics(np.ones(2), np.array([1.0, value]))
                self.assertFalse(result["finite"])
                self.assertFalse(result["passed"])

    def test_array_validation_requires_exact_names_shape_dtype_and_finite_values(self):
        specification = [{"name": "features", "shape": [2, 2], "dtype": "float32"}]
        valid = np.ones((2, 2), dtype=np.float32)
        validate_arrays({"features": valid}, specification)
        invalid = [
            {},
            {"features": valid, "extra": valid},
            {"features": valid.reshape(4)},
            {"features": valid.astype(np.float64)},
        ]
        for value in (np.nan, np.inf, -np.inf):
            candidate = valid.copy()
            candidate[1, 0] = value
            invalid.append({"features": candidate})
        for index, values in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                validate_arrays(values, specification)

    def test_gather_indices_accept_endpoints_and_reject_negative_or_one_past_the_end(self):
        for name, limit in (("input_ids", 151936), ("text_index", 512), ("image_index", 2048)):
            specification = [{"name": name, "shape": [2], "dtype": "int32"}]
            with self.subTest(name=name, case="endpoints"):
                validate_arrays({name: np.array([0, limit - 1], dtype=np.int32)}, specification)
            for invalid in (-1, limit):
                with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                    validate_arrays({name: np.array([0, invalid], dtype=np.int32)}, specification)


if __name__ == "__main__":
    unittest.main()
