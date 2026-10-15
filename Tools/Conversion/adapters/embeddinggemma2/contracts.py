import numpy as np


def comparison_gate(specification, profiles):
    allowed = {"fp32", "fp16_weights", "fp16_all", "fp16_safe"}
    if not profiles or any(profile not in allowed for profile in profiles):
        raise ValueError("All encoders must declare a known precision profile")
    return specification["fp32_gate" if all(profile == "fp32" for profile in profiles) else "low_precision_gate"]


def spatial_pooling_weights(positions, output_length, kernel_size=3):
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 2 or not np.issubdtype(positions.dtype, np.integer):
        raise ValueError("Patch positions must be integer pairs")
    if kernel_size <= 0 or output_length <= 0 or len(positions) != output_length * kernel_size**2:
        raise ValueError("Input patches do not match the fixed output shape")
    padding = np.all(positions == -1, axis=1)
    valid = positions[~padding]
    if not len(valid) or np.any(valid < 0):
        raise ValueError("The image has no valid rectangular patch grid")
    width, height = np.max(valid, axis=0) + 1
    if width % kernel_size or height % kernel_size or len(valid) != width * height:
        raise ValueError("The patch grid must contain complete spatial kernels")
    coordinates = valid[:, 0] + width * valid[:, 1]
    if len(np.unique(coordinates)) != len(valid):
        raise ValueError("Patch positions must be unique")
    groups = valid[:, 0] // kernel_size + (width // kernel_size) * (valid[:, 1] // kernel_size)
    if np.max(groups) >= output_length:
        raise ValueError("The patch grid exceeds the output token budget")
    weights = np.zeros((output_length, len(positions)), dtype=np.float32)
    weights[groups, np.flatnonzero(~padding)] = 1.0 / kernel_size**2
    return weights, np.any(weights != 0, axis=1)


def compare_vectors(reference, actual, cosine_min, max_abs_max, norm_tolerance=0.001):
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape or reference.ndim != 2 or reference.shape[1] != 768:
        raise ValueError("Both vector arrays must have the same nonempty [N, 768] shape")
    if not len(reference) or not np.all(np.isfinite(reference)):
        raise ValueError("Reference vectors must be finite and nonempty")
    reference_norms = np.linalg.norm(reference, axis=1)
    if np.any(np.abs(reference_norms - 1) > norm_tolerance):
        raise ValueError("Reference vectors must be normalized")
    records = []
    for expected, candidate in zip(reference, actual, strict=True):
        finite = bool(np.all(np.isfinite(candidate)))
        if not finite:
            records.append({"finite": False, "norm": None, "cosine": None, "max_abs": None, "pass": False})
            continue
        norm = float(np.linalg.norm(candidate))
        cosine = float(np.dot(expected, candidate) / (np.linalg.norm(expected) * norm)) if norm > 0 else None
        error = float(np.max(np.abs(expected - candidate)))
        passed = norm > 0 and abs(norm - 1) <= norm_tolerance and cosine >= cosine_min and error <= max_abs_max
        records.append({"finite": True, "norm": norm, "cosine": cosine, "max_abs": error, "pass": bool(passed)})
    return {"pass": all(row["pass"] for row in records), "rows": records}
