import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ...storage import file_hash, read_json, write_json

TYPES = {"float32": "<f4", "int32": "<i4"}
PATCH_ARRAYS = ("noise", "dit_hidden", "cond", "latent", "stop_logits")
SOURCES = ("ids.npy", "patches.npz", "calls.npz", "calls.json", "audio.npy")


def write_array(directory, key, value):
    value = np.asarray(value)
    kind = str(value.dtype)
    if kind not in TYPES:
        raise ValueError(f"{key} has the unsupported type {kind}")
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise ValueError(f"{key} is not finite")
    if value.size == 0:
        raise ValueError(f"{key} is empty")
    data = np.ascontiguousarray(value).astype(TYPES[kind]).tobytes()
    path = directory / f"{key}.bin"
    with path.open("xb") as handle:
        handle.write(data)
    return {"file": path.name, "type": kind, "shape": [int(size) for size in value.shape],
            "sha256": hashlib.sha256(data).hexdigest()}


def call_order(order, calls):
    counts = {}
    for row in order:
        if not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str) or row[1] != counts.get(row[0], 0):
            raise ValueError(f"The call order is not consecutive at {row}")
        counts[row[0]] = row[1] + 1
    for key in calls.files:
        name = ".".join(key.split(".")[:2])
        if key.count(".") != 3 or key.split(".")[2] not in ("in", "out") or counts.get(name) != calls[key].shape[0]:
            raise ValueError(f"{key} does not match the call order")
    recorded = {".".join(key.split(".")[:2]) for key in calls.files}
    if recorded != set(counts):
        raise ValueError("The call order and the recorded arrays name different calls")
    return counts


def convert_sentence(source, destination, name, entry, samples):
    if entry.get("finish") != "stop" or entry.get("patches", 0) < 1 or entry.get("stopped_patch") != entry["patches"] - 1:
        raise ValueError(f"{name} did not end with a stop")
    if sorted(entry.get("files", [])) != sorted(SOURCES):
        raise ValueError(f"{name} does not hold the expected files")
    ids = np.load(source / "ids.npy", allow_pickle=False)
    audio = np.load(source / "audio.npy", allow_pickle=False)
    order = read_json(source / "calls.json")
    destination.mkdir()
    arrays = {}
    with np.load(source / "patches.npz", allow_pickle=False) as patches, \
            np.load(source / "calls.npz", allow_pickle=False) as calls:
        counts = call_order(order, calls)
        if sorted(patches.files) != sorted(PATCH_ARRAYS):
            raise ValueError(f"{name} does not hold the patch arrays")
        if any(patches[key].shape[0] != entry["patches"] for key in PATCH_ARRAYS):
            raise ValueError(f"{name} holds another patch count")
        if counts.get("proj.heads") != entry["patches"] or audio.shape != (entry["patches"] * samples,):
            raise ValueError(f"{name} does not match its patch count")
        if ids.dtype != np.int32 or ids.ndim != 1 or not np.array_equal(calls["tslm.embed.in.ids"][0, 0, :ids.size], ids):
            raise ValueError(f"{name} holds other token IDs")
        for key in sorted(calls.files):
            arrays[key] = write_array(destination, key, calls[key])
        for key in PATCH_ARRAYS:
            arrays[f"patch.{key}"] = write_array(destination, f"patch.{key}", patches[key])
    arrays["audio"] = write_array(destination, "audio", audio)
    record = {"name": name, "seed": entry["seed"], "finish": entry["finish"], "patches": entry["patches"],
              "stoppedPatch": entry["stopped_patch"], "ids": [int(value) for value in ids], "calls": order,
              "arrays": arrays, "sources": {file: file_hash(source / file)["sha256"] for file in SOURCES}}
    write_json(destination / "case.json", record)
    return {"name": name, "patches": entry["patches"], "calls": len(order)}


def convert(check, output, samples):
    report = read_json(check / "check.json")
    if report.get("complete") is not True:
        raise ValueError("The check is not complete")
    entries = {name: fixture.get("swift_reference") for name, fixture in report["fixtures"].items()}
    if not entries or any(entry is None for entry in entries.values()):
        raise ValueError("Every fixture needs a recorded generation")
    seeds = {entry["seed"] for entry in entries.values()}
    if seeds != {report["seed"]}:
        raise ValueError("The recorded generations use another seed")
    output.mkdir(parents=True, exist_ok=False)
    cases = [convert_sentence(check / "swift_reference" / name, output / name, name, entries[name], samples)
             for name in sorted(entries)]
    write_json(output / "reference.json", {"format": 1, "seed": report["seed"], "samplesPerPatch": samples,
                                            "sentences": [case["name"] for case in cases],
                                            "check": file_hash(check / "check.json")["sha256"]})
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-patch", type=int, default=7680)
    args = parser.parse_args()
    print(json.dumps(convert(args.check, args.output, args.samples_per_patch)))


if __name__ == "__main__":
    main()
