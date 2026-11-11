import math
import struct
from collections.abc import Sequence
from numbers import Real


MODEL_ID = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
QUANTILES = (
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
    0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
    0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
)


def _as_float(value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Values must be real numbers, excluding booleans")
    try:
        return float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("Values must be representable as floats") from error


def prepare_context(values) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError("Context must be a one-dimensional numeric sequence")
    if not 1 <= len(values) <= 8192:
        raise ValueError("Context length must be between 1 and 8192")
    context = [_as_float(value) for value in values]
    if any(math.isinf(value) for value in context):
        raise ValueError("Context must not contain infinity")
    try:
        for value in context:
            struct.pack("!f", value)
    except OverflowError as error:
        raise ValueError("Context values must not overflow float32") from error
    if not any(math.isfinite(value) for value in context):
        raise ValueError("Context must contain at least one finite value")
    return [math.nan] * (-len(context) % 16) + context


def validate_predictions(values) -> None:
    if not isinstance(values, (list, tuple)) or len(values) != 1:
        raise ValueError("Predictions must have shape [1, 21, 16]")
    batch = values[0]
    if not isinstance(batch, (list, tuple)) or len(batch) != 21:
        raise ValueError("Predictions must have shape [1, 21, 16]")
    for row in batch:
        if not isinstance(row, (list, tuple)) or len(row) != 16:
            raise ValueError("Predictions must have shape [1, 21, 16]")
        if any(not math.isfinite(_as_float(value)) for value in row):
            raise ValueError("Predictions must contain only finite real numbers")


def compare_predictions(actual, expected) -> dict[str, bool | int | float]:
    validate_predictions(actual)
    validate_predictions(expected)
    passed = True
    count = 0
    max_absolute_error = 0.0
    for actual_row, expected_row in zip(actual[0], expected[0]):
        for actual_value, expected_value in zip(actual_row, expected_row):
            reference = float(expected_value)
            error = abs(float(actual_value) - reference)
            if not math.isfinite(error) or error > 1e-4 + 1e-4 * abs(reference):
                passed = False
            max_absolute_error = max(max_absolute_error, error)
            count += 1
    return {"passed": passed, "count": count, "max_absolute_error": max_absolute_error}
