import copy
import math
import sys
import unittest
from fractions import Fraction

from Tools.Conversion.adapters.chronos2.contracts import (
    MODEL_ID,
    QUANTILES,
    REVISION,
    compare_predictions,
    prepare_context,
    validate_predictions,
)


def predictions(value=0.0):
    return [[[value for _ in range(16)] for _ in range(21)]]


class ContextTests(unittest.TestCase):
    def test_model_identity_and_quantiles_are_pinned(self):
        self.assertEqual(MODEL_ID, "amazon/chronos-2")
        self.assertEqual(REVISION, "29ec3766d36d6f73f0696f85560a422f50e8498c")
        self.assertEqual(tuple(QUANTILES), (
            0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
            0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
            0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
        ))

    def test_length_boundaries_are_left_padded_to_complete_patches(self):
        for length in (1, 15, 16, 17, 31, 32, 33, 1008, 8191, 8192):
            with self.subTest(length=length):
                values = list(range(length))
                original = values.copy()
                actual = prepare_context(values)
                padded_length = ((length + 15) // 16) * 16
                self.assertIsInstance(actual, list)
                self.assertIsNot(actual, values)
                self.assertEqual(len(actual), padded_length)
                self.assertTrue(all(math.isnan(value) for value in actual[:-length]))
                self.assertEqual(actual[-length:], values)
                self.assertTrue(all(type(value) is float for value in actual))
                self.assertEqual(values, original)
                actual[-1] = -999.0
                self.assertEqual(values, original)

    def test_missing_values_and_raw_scale_are_preserved(self):
        values = [math.nan, -123.5, 0, math.nan, 9999]
        original = values.copy()
        actual = prepare_context(values)
        self.assertTrue(all(math.isnan(value) for value in actual[:12]))
        self.assertEqual(actual[12:14], [-123.5, 0.0])
        self.assertTrue(math.isnan(actual[14]))
        self.assertEqual(actual[15], 9999.0)
        self.assertEqual(len(values), len(original))
        self.assertTrue(all(value is previous for value, previous in zip(values, original)))

    def test_tuple_range_and_fraction_inputs_produce_floats(self):
        self.assertEqual(prepare_context((Fraction(1, 2), -2))[-2:], [0.5, -2.0])
        self.assertEqual(prepare_context(range(16)), [float(value) for value in range(16)])

    def test_float32_boundary_values_preserve_raw_context_and_missing_values(self):
        maximum = float.fromhex("0x1.fffffep+127")
        magnitudes = (
            math.nextafter(maximum, 0.0),
            maximum,
            math.nextafter(maximum, math.inf),
            math.nextafter(float.fromhex("0x1.ffffffp+127"), 0.0),
        )
        for magnitude in magnitudes:
            for sign in (-1, 1):
                with self.subTest(magnitude=magnitude, sign=sign):
                    values = [math.nan, sign * magnitude, math.nan]
                    original = values.copy()
                    actual = prepare_context(values)
                    self.assertEqual(len(actual), 16)
                    self.assertTrue(all(math.isnan(value) for value in actual[:-2]))
                    self.assertEqual(actual[-2], values[1])
                    self.assertTrue(math.isnan(actual[-1]))
                    self.assertEqual(len(values), len(original))
                    self.assertTrue(all(value is previous for value, previous in zip(values, original)))

    def test_float32_overflow_is_rejected_without_changing_input(self):
        overflow_boundary = float.fromhex("0x1.ffffffp+127")
        magnitudes = (
            overflow_boundary,
            math.nextafter(overflow_boundary, math.inf),
            2.0**128,
            1e39,
            sys.float_info.max,
        )
        for magnitude in magnitudes:
            for sign in (-1, 1):
                with self.subTest(magnitude=magnitude, sign=sign):
                    values = [1.0, sign * magnitude, math.nan]
                    original = values.copy()
                    with self.assertRaises(ValueError):
                        prepare_context(values)
                    self.assertEqual(len(values), len(original))
                    self.assertTrue(all(value is previous for value, previous in zip(values, original)))

    def test_empty_and_oversized_contexts_are_rejected(self):
        for values in ([], (), [1.0] * 8193):
            with self.subTest(length=len(values)), self.assertRaises(ValueError):
                prepare_context(values)

    def test_all_missing_contexts_are_rejected(self):
        for length in (1, 16, 8192):
            with self.subTest(length=length), self.assertRaises(ValueError):
                prepare_context([math.nan] * length)

    def test_invalid_elements_are_rejected_without_changing_input(self):
        for invalid in (True, False, "1", None, 1j, [1.0], (1.0,), math.inf, -math.inf, 10**400):
            with self.subTest(invalid=invalid):
                values = [1.0, invalid]
                original = values.copy()
                with self.assertRaises(ValueError):
                    prepare_context(values)
                self.assertEqual(values, original)

    def test_nonsequence_contexts_are_value_errors(self):
        for values in (None, 1.0, True, "123", b"123", {0: 1.0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                prepare_context(values)


class PredictionTests(unittest.TestCase):
    def test_nested_lists_and_tuples_accept_finite_reals_without_changes(self):
        values = [[[Fraction(21 - quantile, 7) - step for step in range(16)]
                   for quantile in range(21)]]
        original = copy.deepcopy(values)
        self.assertIsNone(validate_predictions(values))
        self.assertEqual(values, original)
        tuples = (tuple(tuple(row) for row in values[0]),)
        self.assertIsNone(validate_predictions(tuples))
        self.assertEqual(tuples, (tuple(tuple(row) for row in original[0]),))
        self.assertTrue(compare_predictions(tuples, values)["passed"])

    def test_invalid_shapes_and_containers_are_rejected(self):
        valid = predictions()
        invalid = (
            None, 1.0, "1", [], valid * 2, valid[0],
            [[]], [[0.0] * 21],
            [[[0.0] * 16 for _ in range(20)]],
            [[[0.0] * 16 for _ in range(22)]],
            [[[0.0] * 15 for _ in range(21)]],
            [[[0.0] * 17 for _ in range(21)]],
            [[[[0.0]] * 16 for _ in range(21)]],
            {0: valid[0]}, [{index: row for index, row in enumerate(valid[0])}],
            [[range(16) for _ in range(21)]],
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_predictions(values)

    def test_ragged_final_row_is_rejected(self):
        for length in (0, 15, 17):
            with self.subTest(length=length):
                values = predictions()
                values[0][-1] = [0.0] * length
                with self.assertRaises(ValueError):
                    validate_predictions(values)

    def test_invalid_elements_in_the_last_cell_are_rejected(self):
        for invalid in (True, False, "1", None, 1j, math.nan, math.inf, -math.inf, 10**400):
            with self.subTest(invalid=invalid):
                values = predictions()
                values[0][-1][-1] = invalid
                with self.assertRaises(ValueError):
                    validate_predictions(values)


class ComparisonTests(unittest.TestCase):
    def test_identical_predictions_pass_with_typed_metrics(self):
        values = predictions(-2)
        result = compare_predictions(values, values)
        self.assertEqual(result, {"passed": True, "count": 336, "max_absolute_error": 0.0})
        self.assertIs(type(result["passed"]), bool)
        self.assertIs(type(result["count"]), int)
        self.assertIs(type(result["max_absolute_error"]), float)

    def test_absolute_tolerance_includes_the_boundary(self):
        expected = predictions()
        for difference, passed in (
            (math.nextafter(1e-4, 0.0), True),
            (1e-4, True),
            (math.nextafter(1e-4, math.inf), False),
        ):
            for sign in (-1, 1):
                with self.subTest(difference=difference, sign=sign):
                    actual = predictions()
                    actual[0][-1][-1] = sign * difference
                    result = compare_predictions(actual, expected)
                    self.assertIs(result["passed"], passed)
                    self.assertEqual(result["count"], 336)
                    self.assertEqual(result["max_absolute_error"], difference)

    def test_relative_and_absolute_tolerances_are_added(self):
        for reference in (-1.0, 1.0, -1000.0, 1000.0):
            tolerance = 1e-4 + 1e-4 * abs(reference)
            for factor, passed in ((0.999, True), (1.001, False)):
                for sign in (-1, 1):
                    with self.subTest(reference=reference, factor=factor, sign=sign):
                        actual = reference + sign * factor * tolerance
                        result = compare_predictions(predictions(actual), predictions(reference))
                        self.assertIs(result["passed"], passed)
                        self.assertEqual(result["max_absolute_error"], abs(actual - reference))

    def test_relative_tolerance_uses_the_expected_value(self):
        actual = predictions(1.00020001)
        expected = predictions(1.0)
        self.assertIs(compare_predictions(actual, expected)["passed"], False)
        self.assertIs(compare_predictions(expected, actual)["passed"], True)

    def test_every_cell_is_checked(self):
        expected = predictions()
        for quantile in range(21):
            for step in range(16):
                with self.subTest(quantile=quantile, step=step):
                    actual = predictions()
                    actual[0][quantile][step] = 0.01
                    result = compare_predictions(actual, expected)
                    self.assertIs(result["passed"], False)
                    self.assertEqual(result["count"], 336)
                    self.assertEqual(result["max_absolute_error"], 0.01)

    def test_failure_still_reports_the_largest_error_without_mutation(self):
        actual = predictions(-1.0)
        expected = predictions(-1.0)
        actual[0][0][0] = -1.5
        actual[0][-1][-1] = -4.0
        original_actual = copy.deepcopy(actual)
        original_expected = copy.deepcopy(expected)
        self.assertEqual(compare_predictions(actual, expected), {
            "passed": False, "count": 336, "max_absolute_error": 3.0,
        })
        self.assertEqual(actual, original_actual)
        self.assertEqual(expected, original_expected)

    def test_both_sides_are_validated_before_comparison(self):
        invalid = [[], predictions() * 2]
        for value in (math.nan, math.inf, -math.inf, True, "0", None):
            values = predictions()
            values[0][-1][-1] = value
            invalid.append(values)
        for values in invalid:
            for actual, expected in ((values, predictions()), (predictions(), values)):
                with self.subTest(actual=actual, expected=expected), self.assertRaises(ValueError):
                    compare_predictions(actual, expected)

    def test_finite_extremes_pass_when_equal_and_fail_when_subtraction_overflows(self):
        for value in (sys.float_info.max, -sys.float_info.max):
            with self.subTest(value=value):
                self.assertIs(compare_predictions(predictions(value), predictions(value))["passed"], True)
                result = compare_predictions(predictions(value), predictions(-value))
                self.assertIs(result["passed"], False)
                self.assertEqual(result["count"], 336)
                self.assertEqual(result["max_absolute_error"], math.inf)


if __name__ == "__main__":
    unittest.main()
