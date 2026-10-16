import unittest

import numpy as np

from Tools.Conversion.adapters.embeddinggemma2.contracts import compare_vectors, spatial_pooling_weights, comparison_gate


class ContractTests(unittest.TestCase):
    def test_fp32_comparison_cannot_use_the_low_precision_gate(self):
        specification = {"fp32_gate": {"cosine_min": 0.99999, "max_abs_max": 0.0001},
                         "low_precision_gate": {"cosine_min": 0.999, "max_abs_max": 0.01}}
        expected = np.eye(1, 768)
        candidate = expected.copy()
        candidate[0, 1] = 0.005
        candidate /= np.linalg.norm(candidate)
        strict = comparison_gate(specification, ["fp32", "fp32"])
        low = comparison_gate(specification, ["fp16_safe", "fp16_safe"])
        self.assertFalse(compare_vectors(expected, candidate, **strict)["pass"])
        self.assertTrue(compare_vectors(expected, candidate, **low)["pass"])

    def test_unknown_precision_cannot_silently_select_a_weaker_gate(self):
        with self.assertRaises(ValueError):
            comparison_gate({}, ["unknown"])

    def test_spatial_neighbors_are_pooled_instead_of_contiguous_nine_rows(self):
        positions = np.array([(x, y) for y in range(3) for x in range(6)], dtype=np.int32)
        weights, mask = spatial_pooling_weights(positions, 2)
        result = weights @ np.arange(18, dtype=np.float32)
        np.testing.assert_allclose(result, [7, 10], atol=1e-6)
        self.assertEqual(mask.tolist(), [True, True])

    def test_padding_does_not_contribute_to_any_spatial_pool(self):
        positions = np.full((18, 2), -1, dtype=np.int32)
        positions[:9] = [(x, y) for y in range(3) for x in range(3)]
        weights, mask = spatial_pooling_weights(positions, 2)
        hidden = np.concatenate([np.ones(9), np.full(9, 9999)])
        np.testing.assert_allclose(weights @ hidden, [1, 0], atol=1e-6)
        self.assertEqual(mask.tolist(), [True, False])

    def test_patch_order_can_change_without_changing_spatial_pooling(self):
        positions = np.array([(x, y) for y in range(3) for x in range(6)], dtype=np.int32)
        permutation = np.arange(18)[::-1]
        weights, _ = spatial_pooling_weights(positions[permutation], 2)
        np.testing.assert_allclose(weights @ permutation, [7, 10], atol=1e-6)

    def test_duplicate_positions_are_rejected(self):
        positions = np.array([(x, y) for y in range(3) for x in range(6)], dtype=np.int32)
        positions[1] = positions[0]
        with self.assertRaises(ValueError):
            spatial_pooling_weights(positions, 2)

    def test_incomplete_patch_grid_is_rejected(self):
        positions = np.array([(x, y) for y in range(3) for x in range(6)], dtype=np.int32)
        positions[-1] = [-1, -1]
        with self.assertRaises(ValueError):
            spatial_pooling_weights(positions, 2)

    def test_identical_normalized_vectors_pass(self):
        expected = np.eye(1, 768)
        self.assertTrue(compare_vectors(expected, expected, 0.99999, 0.0001)["pass"])

    def test_high_cosine_does_not_hide_wrong_vector_norm(self):
        expected = np.eye(1, 768)
        self.assertFalse(compare_vectors(expected, expected * 2, 0.999, 2)["pass"])

    def test_nonfinite_output_is_an_explicit_failure(self):
        expected = np.eye(1, 768)
        for value in [np.nan, np.inf, -np.inf]:
            actual = expected.copy()
            actual[0, 2] = value
            result = compare_vectors(expected, actual, 0.999, 0.01)
            self.assertFalse(result["pass"])
            self.assertFalse(result["rows"][0]["finite"])

    def test_zero_vector_cannot_pass(self):
        expected = np.eye(1, 768)
        self.assertFalse(compare_vectors(expected, np.zeros_like(expected), 0.999, 1)["pass"])

    def test_corrupted_reference_is_rejected(self):
        expected = np.eye(1, 768)
        expected[0, 1] = np.nan
        with self.assertRaises(ValueError):
            compare_vectors(expected, np.zeros_like(expected), 0.999, 1)


if __name__ == "__main__":
    unittest.main()
