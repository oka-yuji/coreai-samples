import hashlib
import json
from pathlib import Path

import numpy as np


REVISION = "fc13ba1986fcb905854dc8de2f52eee395f49b5b"
DIFFUSERS_REVISION = "b800f828f3951c4d206e154411bcbfa693cd322e"
KINDS = ("vision", "text", "denoiser", "vae_encoder", "vae_decoder")
TEXT_LENGTH = 512
JOINT_LENGTH = 2304
SIGMAS = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)
PROFILE = {"version": 1, "source": "Qwen/Qwen-Image-2.1-Turbo", "revision": REVISION,
           "diffusers_revision": DIFFUSERS_REVISION, "precision": "fp32", "resolution": 512,
           "text_length": TEXT_LENGTH, "joint_length": JOINT_LENGTH, "batch": 1,
           "max_reference_images": 1, "cfg": 1, "prefix_cache": False, "sigmas": list(SIGMAS)}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tensor_metrics(reference, actual, max_limit=1e-3, rmse_limit=1e-4):
    expected, observed = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if expected.shape != observed.shape or not expected.size or not np.isfinite(expected).all():
        raise ValueError("Reference and actual must have the same nonempty shape and a finite reference")
    if not np.isfinite(observed).all():
        return {"finite": False, "passed": False}
    delta = observed - expected
    absolute_rmse = float(np.sqrt(np.mean(delta * delta)))
    normalized_max = float(np.max(np.abs(delta)) / max(float(np.max(np.abs(expected))), 1e-12))
    normalized_rmse = absolute_rmse / max(float(np.sqrt(np.mean(expected * expected))), 1e-12)
    return {"finite": True, "shape": list(expected.shape), "absolute_rmse": absolute_rmse,
            "normalized_max": normalized_max, "normalized_rmse": normalized_rmse,
            "passed": normalized_max <= max_limit and normalized_rmse <= rmse_limit}


def pad(value, shape):
    value = np.asarray(value)
    if value.ndim != len(shape) or any(actual > limit for actual, limit in zip(value.shape, shape, strict=True)):
        raise ValueError(f"Input shape {value.shape} exceeds the fixed profile {shape}")
    result = np.zeros(shape, dtype=value.dtype)
    result[tuple(slice(0, count) for count in value.shape)] = value
    return result


def layout(image_slots):
    image_slots = np.asarray(image_slots)
    if image_slots.dtype != np.bool_ or image_slots.ndim != 1 or not 0 < len(image_slots) <= TEXT_LENGTH:
        raise ValueError("Image slots must be a nonempty boolean prompt sequence within the profile")
    count = int(image_slots.sum())
    if count not in (0, 256):
        raise ValueError("The profile supports zero or one 512-square reference image")
    slots = np.concatenate((image_slots, np.ones(256, dtype=np.bool_)))
    repeats = np.where(slots, 4, 1)
    expanded = np.repeat(slots, repeats)
    length = len(expanded)
    if length > JOINT_LENGTH:
        raise ValueError("The expanded layout exceeds the joint token budget")
    text_index = np.repeat(np.arange(len(slots), dtype=np.int32), repeats)
    text_index[expanded] = 0
    image_index = np.zeros(length, dtype=np.int32)
    image_index[expanded] = np.arange(expanded.sum(), dtype=np.int32)
    positions = np.flatnonzero(expanded)
    ids = np.full(length, -1, dtype=np.int32)
    ids[positions] = np.repeat(np.arange(count // 256 + 1), 1024)
    target_indices = positions[-1024:]
    target = np.zeros(length, dtype=np.bool_)
    target[target_indices] = True
    sequence = np.arange(length)
    attention = (sequence[:, None] >= sequence[None, :]) | ((ids[:, None] >= 0) & (ids[:, None] == ids[None, :]))
    mask = np.zeros((1, 1, JOINT_LENGTH, JOINT_LENGTH), dtype=np.bool_)
    mask[0, 0, :length, :length] = attention
    mask[0, 0, length:, :length] = True
    return {"text_index": pad(text_index, (JOINT_LENGTH,)), "image_index": pad(image_index, (JOINT_LENGTH,)),
            "image_mask": pad(expanded, (JOINT_LENGTH,)), "mask": mask,
            "target": pad(target, (JOINT_LENGTH,)), "target_indices": target_indices,
            "compact_image_mask": expanded, "length": length, "reference_count": count // 256}


def validate_arrays(values, specification):
    if set(values) != {item["name"] for item in specification}:
        raise ValueError("The component input names do not match its contract")
    for item in specification:
        value = np.asarray(values[item["name"]])
        if list(value.shape) != item["shape"] or value.dtype.name != item["dtype"]:
            raise ValueError(f"Unexpected shape or dtype for {item['name']}")
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError("Component inputs must be finite")
        upper = {"input_ids": 151936, "text_index": TEXT_LENGTH, "image_index": 2048}.get(item["name"])
        if upper is not None and (np.any(value < 0) or np.any(value >= upper)):
            raise ValueError(f"Index outside the profile: {item['name']}")


def check_profile(profile):
    if profile != PROFILE:
        raise ValueError("The component has a different source or execution profile")


def require_heavy_job():
    import os
    import psutil

    lock = Path(os.environ.get("COREAI_HEAVY_JOB_LOCK", Path.home() / "Library/Caches/coreai-samples/heavy-job"))
    try:
        owner = read_json(lock / "owner.txt")
        owners = {os.getpid(), *(process.pid for process in psutil.Process().parents())}
        if type(owner.get("pid")) is not int or owner["pid"] not in owners:
            raise ValueError("The machine lock belongs to another job")
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError("Run this model operation through the conversion runner and its machine lock") from error


def validate_bundle(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("A bundle must be a real directory")
    manifest = read_json(root / "bundle.json")
    check_profile(manifest["profile"])
    if (manifest.get("component_verified") is not True or manifest.get("end_to_end_verified") is not True
            or manifest.get("components") != list(KINDS)):
        raise ValueError("The bundle must declare all five verified components and the complete pipeline")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Bundle entries must not be symbolic links")
        if path.is_file() and path != root / "bundle.json":
            actual.add(path.relative_to(root).as_posix())
        elif not path.is_file() and not path.is_dir():
            raise ValueError("The bundle contains a special file")
    inventory = manifest.get("assets")
    if not isinstance(inventory, dict) or set(inventory) != actual or not actual:
        raise ValueError("Bundle contents do not match the complete file inventory")
    for relative, expected in inventory.items():
        path = root / relative
        if (not isinstance(expected, dict) or type(expected.get("bytes")) is not int
                or path.stat().st_size != expected["bytes"] or digest(path) != expected.get("sha256")):
            raise ValueError(f"Bundle content fingerprint differs: {relative}")
    for kind in KINDS:
        component = read_json(root / kind / "component.json")
        check_profile(component["profile"])
        if component.get("kind") != kind or component.get("passed") is not True:
            raise ValueError("A component identity does not match the bundle")
    return manifest


def component_inventory(directory):
    directory = Path(directory)
    contract = read_json(directory / "component.json")
    kind = contract["kind"]
    if kind not in KINDS or contract["asset"] != f"{kind}.aimodel":
        raise ValueError("Unexpected component asset path")
    paths = [directory / "component.json", directory / "comparison.json"]
    asset = directory / contract["asset"]
    if asset.is_symlink() or not asset.is_dir():
        raise ValueError("The component asset must be a real directory")
    for path in asset.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("The component asset contains a link or special file")
        if path.is_file():
            paths.append(path)
    result = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("The component manifest must be a regular file")
        result[path.relative_to(directory).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    return result
