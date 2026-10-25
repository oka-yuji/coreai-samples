import hashlib
import json
from pathlib import Path

import numpy as np

MODEL_ID = "XingChen-AGI/TeleOCR"
REVISION = "70a19ec6a3b036311af19ae5e762e650f41a5980"
SYSTEM = "You are a helpful assistant."
PROMPTS = {
    "text": "Please output the text content from the image.",
    "table": "This is the image of a table. Please output the table in OTSL format.",
    "layout": "Analyze the image layout.",
}
IMAGE_TOKEN, VISION_START, VISION_END = 151655, 151652, 151653
STOP_TOKENS = [151645, 151643]
PATCH, TEMPORAL, MERGE, WINDOW_PIXELS = 14, 2, 2, 112
WINDOW_UNITS = WINDOW_PIXELS // PATCH // MERGE
UNIT_PATCHES = MERGE * MERGE
WINDOW_PATCHES = WINDOW_UNITS * WINDOW_UNITS * UNIT_PATCHES
WINDOW_LIMIT = 2048
TOKEN_LIMIT = 32768
VISION_NRMSE_LIMIT = 1e-4
VISION_CASES = {
    "min": {"size": [56, 56]},
    "square448": {"size": [448, 448]},
    "square896": {"size": [896, 896]},
    "square1036": {"size": [1036, 1036]},
    "portrait": {"size": [364, 1204], "crop": [60, 0, 420, 1190]},
    "landscape": {"size": [1204, 364], "crop": [0, 120, 1240, 495]},
}
REQUIRED_VISION_CASES = ["min", "square448", "square1036", "portrait", "landscape"]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def metrics(reference, actual):
    expected = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(actual, dtype=np.float64)
    if expected.shape != observed.shape or not expected.size or not np.isfinite(expected).all():
        raise ValueError("Reference and actual must have the same nonempty shape and a finite reference")
    if not np.isfinite(observed).all():
        return {"finite": False, "shape": list(expected.shape)}
    delta = observed - expected
    scale = float(np.sqrt(np.mean(expected * expected)))
    norm = float(np.linalg.norm(expected) * np.linalg.norm(observed))
    return {"finite": True, "shape": list(expected.shape), "max_abs": float(np.abs(delta).max()),
            "nrmse": float(np.sqrt(np.mean(delta * delta))) / max(scale, 1e-30),
            "cosine": float(np.sum(expected * observed) / norm) if norm else None,
            "reference_max_abs": float(np.abs(expected).max())}


def logit_metrics(reference, actual):
    result = metrics(reference, actual)
    expected = np.asarray(reference, dtype=np.float64).reshape(-1)
    observed = np.asarray(actual, dtype=np.float64).reshape(-1)
    top = np.sort(expected)[-2:]
    result.update(argmax=int(observed.argmax()), reference_argmax=int(expected.argmax()),
                  argmax_match=bool(observed.argmax() == expected.argmax()),
                  top5=[int(i) for i in np.argsort(-observed)[:5]],
                  reference_top5=[int(i) for i in np.argsort(-expected)[:5]],
                  reference_top2_gap=float(top[1] - top[0]))
    return result


def window_order(grid_h, grid_w):
    units_h, units_w = grid_h // MERGE, grid_w // MERGE
    if units_h * MERGE != grid_h or units_w * MERGE != grid_w or units_h < 1 or units_w < 1:
        raise ValueError("The patch grid must contain complete merge units")
    index = np.arange(units_h * units_w).reshape(units_h, units_w)
    padded = np.pad(index, ((0, -units_h % WINDOW_UNITS), (0, -units_w % WINDOW_UNITS)), constant_values=-1)
    rows, columns = padded.shape[0] // WINDOW_UNITS, padded.shape[1] // WINDOW_UNITS
    order = padded.reshape(rows, WINDOW_UNITS, columns, WINDOW_UNITS).transpose(0, 2, 1, 3).reshape(-1)
    if order.size // (WINDOW_UNITS * WINDOW_UNITS) > WINDOW_LIMIT:
        raise ValueError("The image exceeds the exported window limit")
    return order


def window_inputs(pixel_values, rotary, grid_h, grid_w):
    order = window_order(grid_h, grid_w)
    valid = order >= 0
    units = len(order)
    source = order.clip(min=0)
    patches = np.asarray(pixel_values, dtype=np.float32).reshape(-1, UNIT_PATCHES, pixel_values.shape[-1])
    angles = np.asarray(rotary, dtype=np.float32).reshape(-1, UNIT_PATCHES, rotary.shape[-1])
    if len(patches) != int(valid.sum()):
        raise ValueError("Pixel rows do not match the patch grid")
    mask = np.repeat(valid, UNIT_PATCHES).astype(np.float32)
    values = (patches[source] * valid[:, None, None]).reshape(units * UNIT_PATCHES, -1)
    emb = np.concatenate([angles, angles], axis=-1)[source].reshape(units * UNIT_PATCHES, -1)
    cos = np.where(mask[:, None] > 0, np.cos(emb), 1.0).astype(np.float32)
    sin = np.where(mask[:, None] > 0, np.sin(emb), 0.0).astype(np.float32)
    return {"pixel_values": values, "key_valid": mask, "cos": cos, "sin": sin}, order


def restore_order(features, order):
    order = np.asarray(order)
    valid = order >= 0
    result = np.zeros((int(valid.sum()), features.shape[-1]), dtype=features.dtype)
    result[order[valid]] = features[valid]
    return result


def tensor(name, dtype, *shape):
    return {"name": name, "dtype": dtype, "shape": list(shape)}


DECODER_CONTEXT = [tensor("cos", "Float16", 1, None, 128), tensor("sin", "Float16", 1, None, 128),
                   tensor("mask", "Float16", 1, 1, None, None)]
DECODER_STATES = [tensor("keyCache", "Float16", 28, 1, 8, None, 128),
                  tensor("valueCache", "Float16", 28, 1, 8, None, 128)]
DECODER_OUTPUTS = [tensor("logits", "Float16", 1, 151936)]
FUNCTIONS = {
    "vision": {"main": {"inputs": [tensor("pixel_values", "Float16", None, 1176), tensor("key_valid", "Float16", None),
                                   tensor("cos", "Float16", None, 80), tensor("sin", "Float16", None, 80)],
                        "outputs": [tensor("image_features", "Float16", None, 1024)], "states": []}},
    "embed": {"main": {"inputs": [tensor("input_ids", "Int32", 1, None)],
                       "outputs": [tensor("inputs_embeds", "Float32", 1, None, 1024)], "states": []}},
    "main": {"prefill": {"inputs": [tensor("inputs_embeds", "Float16", 1, None, 1024), *DECODER_CONTEXT],
                         "outputs": DECODER_OUTPUTS, "states": DECODER_STATES},
             "main": {"inputs": [tensor("input_ids", "Int32", 1, None), *DECODER_CONTEXT],
                      "outputs": DECODER_OUTPUTS, "states": DECODER_STATES}},
}


def parse_type(text):
    kind, _, rest = text.partition(" (")
    dtype, _, dims = rest.rstrip(")").partition(", ")
    shape = [None if dim == "?" else int(dim) for dim in dims.split(" × ")] if dims else []
    return {"kind": kind, "dtype": dtype, "shape": shape}


def inspected_functions(report):
    result = {}
    for function in report["summary"]["functions"]:
        result[function["name"]] = {
            group: [{"name": item["name"], **{key: value for key, value in parse_type(item["type"]).items() if key != "kind"}}
                    for item in function.get(group, [])]
            for group in ("inputs", "outputs", "states")}
    return result
