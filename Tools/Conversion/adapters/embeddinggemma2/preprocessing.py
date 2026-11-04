import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np

from .prepare import validate_source

KEYS = ["input_ids", "attention_mask", "pixel_values", "image_position_ids"]
PACKAGES = ["pillow", "filelock", "setuptools", "transformers", "torch", "numpy"]


def describe(array):
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def dump(contract, source, images, output):
    from PIL import Image
    from transformers import AutoProcessor

    from .reference import image_fixtures

    source = validate_source(source, json.loads(contract.read_text()))
    if output.exists():
        raise FileExistsError(output)
    image_dir = output / "images"
    image_dir.mkdir(parents=True)
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    records = []
    for fixture in image_fixtures(images, image_dir):
        destination = output / fixture["id"]
        destination.mkdir()
        image = Image.open(output / fixture["image"]).convert("RGB")
        pixels = np.asarray(image)
        batch = processor(images=[image], max_soft_tokens=70, return_tensors="pt", padding="max_length", max_length=128, truncation=False)
        arrays = {key: batch[key].numpy() for key in KEYS}
        np.save(destination / "pixels.npy", pixels, allow_pickle=False)
        for key, array in arrays.items():
            np.save(destination / f"{key}.npy", array, allow_pickle=False)
        records.append({"id": fixture["id"], "png_sha256": fixture["sha256"], "pixels": describe(pixels),
                        "arrays": {key: describe(array) for key, array in arrays.items()}})
    versions = {"python": platform.python_version()} | {name: importlib.metadata.version(name) for name in PACKAGES}
    summary = {"model_revision": source.name, "versions": versions, "fixtures": records}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def compare(left, right):
    summaries = [json.loads((folder / "summary.json").read_text()) for folder in (left, right)]
    identifiers = [[fixture["id"] for fixture in summary["fixtures"]] for summary in summaries]
    if not identifiers[0] or identifiers[0] != identifiers[1] or summaries[0]["model_revision"] != summaries[1]["model_revision"]:
        raise ValueError("Both dumps must contain the same nonempty fixtures in the same order for the same model revision")
    fixtures = []
    for name, first, second in zip(identifiers[0], summaries[0]["fixtures"], summaries[1]["fixtures"]):
        pixels = [np.load(folder / name / "pixels.npy", allow_pickle=False) for folder in (left, right)]
        arrays = {}
        for key in KEYS:
            a, b = (np.load(folder / name / f"{key}.npy", allow_pickle=False) for folder in (left, right))
            same = a.shape == b.shape and a.dtype == b.dtype
            difference = None
            if a.shape == b.shape:
                difference = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max(initial=0))
            arrays[key] = {"shape": list(a.shape) if same else {"left": list(a.shape), "right": list(b.shape)},
                           "dtype": str(a.dtype) if same else {"left": str(a.dtype), "right": str(b.dtype)},
                           "equal": same and bool(np.array_equal(a, b)), "max_abs_difference": difference}
        fixtures.append({"id": name, "png_bytes_equal": first["png_sha256"] == second["png_sha256"],
                         "pixels_equal": pixels[0].dtype == pixels[1].dtype and bool(np.array_equal(*pixels)),
                         "arrays": arrays})
    return {"equal": all(row["pixels_equal"] and all(item["equal"] for item in row["arrays"].values()) for row in fixtures),
            "model_revision": summaries[0]["model_revision"],
            "left": {"versions": summaries[0]["versions"]}, "right": {"versions": summaries[1]["versions"]},
            "fixtures": fixtures}


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    dumping = commands.add_parser("dump")
    for name in ("--contract", "--source", "--images", "--output"):
        dumping.add_argument(name, type=Path, required=True)
    comparing = commands.add_parser("compare")
    for name in ("--left", "--right", "--output"):
        comparing.add_argument(name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "dump":
        dump(args.contract, args.source, args.images, args.output)
        return 0
    result = compare(args.left, args.right)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
