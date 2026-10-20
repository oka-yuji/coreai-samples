import argparse
import hashlib
import json
import math
from pathlib import Path

from .evaluation import REVISION, SIGMAS, canonical_hash, require, validate_input_pair, validate_inventory

DEFINITION_SHA256 = "fe0127bc661bebaa36f75aaeb7f1d21b5e4163c1381a8bcb1652d71efcd87303"


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def file_record(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def validate_cases(value):
    require(isinstance(value, dict), "Invalid fixture definition")
    require(value.get("format") == 1 and value.get("id") == "qmi-quality-30x2-v1", "Unsupported fixture definition")
    require(value.get("seeds") == [20261018, 20261021], "The two seeds are frozen")
    cases = value.get("cases")
    require(isinstance(cases, list) and len(cases) == 30, "Expected all thirty inputs")
    require(all(isinstance(case, dict) and isinstance(case.get("id"), str) for case in cases), "Invalid fixture entries")
    require(len({case["id"] for case in cases}) == 30, "Duplicate fixture id")
    for case in cases:
        require(case.get("language") in ("ja", "en") and isinstance(case.get("prompt"), str)
                and case["prompt"].strip(), "Invalid fixture prompt")
        require(case.get("axes") and set(case["axes"]) <= {"instruction", "text", "detail", "preservation", "alpha"},
                "Missing quality axes")
        require(case.get("reference") in (None, "circle", "objects", "person", "sign", "transparent", "gradient", "checker"),
                "Unknown authored reference")
    require(sum(case.get("reference") is not None for case in cases) == 15, "Expected fifteen edits and fifteen generations")
    digest = canonical_hash(value)
    require(digest == DEFINITION_SHA256, "The version-one fixture definition changed")
    return digest


def quality_axes():
    path = Path(__file__).resolve().parents[2] / "recipes/qwen_image21/evaluation-cases.json"
    value = json.loads(path.read_text())
    validate_cases(value)
    return {f"{case['id']}-{seed}": case["axes"] for case in value["cases"] for seed in value["seeds"]}


def validate_suite_runs(runs):
    require(isinstance(runs, list) and len(runs) == 60, "All sixty suite runs are required")
    identifiers = [run.get("run_id") for run in runs]
    require(len(set(identifiers)) == 60 and set(identifiers) == set(quality_axes()), "The frozen run identities differ")
    for run in runs:
        require(run["run_id"] == f"{run.get('fixture_id')}-{run.get('seed')}", "Run identity differs from its input")
        require(run.get("noise") == f"noise/{run['seed']}.npy", "Noise path differs from the frozen seed")


def validate_fixture_suite(root, manifest):
    import numpy as np
    root = Path(root)
    require(manifest.get("format") == 1 and manifest.get("id") == "qmi-quality-30x2-v1",
            "Unexpected fixture version")
    require(manifest.get("source_revision") == REVISION and manifest.get("definition_sha256") == DEFINITION_SHA256,
            "Frozen fixture source or definition changed")
    validate_suite_runs(manifest["runs"])
    inventory = validate_inventory(root, manifest["inventory"])
    definition_path = Path(__file__).resolve().parents[2] / "recipes/qwen_image21/evaluation-cases.json"
    definition = json.loads(definition_path.read_text())
    validate_cases(definition)
    definitions = {case["id"]: case for case in definition["cases"]}
    for run in manifest["runs"]:
        validate_input_pair(run, run)
        case = definitions[run["fixture_id"]]
        directory = root / run["fixture_id"]
        metadata = json.loads((directory / "input.json").read_text())
        reference = "images/" + case["reference"] + ".png" if case.get("reference") else None
        operation = "edit" if reference else "generate"
        require(metadata.get("prompt") == case["prompt"] and metadata.get("reference") == reference
                and metadata.get("operation") == run["operation"] == operation,
                "The run payload differs from its frozen case")
        require((directory / "prompt.txt").read_bytes() == case["prompt"].encode("utf-8"),
                "Saved prompt bytes differ from the executed prompt")
        bindings = {"prompt_sha256": directory / "prompt.txt", "noise_sha256": root / run["noise"],
                    "layout_sha256": directory / "layout.npz"}
        if reference:
            bindings["reference_sha256"] = root / reference
        for field, path in bindings.items():
            require(file_record(path)["sha256"] == run[field], f"Run input hash differs from its actual file: {field}")
        noise = np.load(root / run["noise"], allow_pickle=False)
        require(noise.shape == (1, 1024, 64) and noise.dtype == np.float32 and bool(np.isfinite(noise).all()),
                "Frozen noise shape, dtype or values changed")
        with np.load(directory / "encoded.npz", allow_pickle=False) as encoded:
            tokens = encoded["input_ids"]
            require(tokens.ndim == 2 and tokens.shape[0] == 1 and 0 < tokens.shape[1] <= 512
                    and np.issubdtype(tokens.dtype, np.integer) and bool(((tokens >= 0) & (tokens < 151936)).all()),
                    "Frozen tokenizer shape or dtype changed")
            require(hashlib.sha256(tokens.astype("<i4").tobytes()).hexdigest() == run["processed_tokens_sha256"],
                    "Run tokens differ from the declared hash")
    return {"runs": len(manifest["runs"]), "files": len(inventory), "passed": True}


def reference_images():
    from PIL import Image, ImageDraw, ImageFont
    images = {}
    for name in ("circle", "objects", "person", "sign", "transparent", "gradient", "checker"):
        image = Image.new("RGBA", (512, 512), (255, 255, 255, 0 if name == "transparent" else 255))
        draw = ImageDraw.Draw(image)
        if name == "circle":
            draw.ellipse((128, 128, 384, 384), fill=(220, 35, 35, 255))
        elif name == "objects":
            draw.rectangle((48, 176, 208, 336), fill=(220, 35, 35, 255))
            draw.ellipse((304, 176, 464, 336), fill=(30, 70, 220, 255))
        elif name == "person":
            draw.ellipse((152, 48, 360, 270), fill=(60, 40, 30, 255))
            draw.ellipse((170, 90, 342, 282), fill=(233, 184, 148, 255))
            draw.rectangle((230, 260, 282, 310), fill=(233, 184, 148, 255))
            draw.polygon([(160, 300), (352, 300), (390, 480), (122, 480)], fill=(30, 70, 220, 255))
            for x in (194, 268):
                draw.ellipse((x, 150, x + 50, 200), outline=(25, 25, 25, 255), width=5)
            draw.line((244, 172, 268, 172), fill=(25, 25, 25, 255), width=5)
            draw.arc((218, 204, 294, 244), 0, 180, fill=(120, 60, 50, 255), width=3)
        elif name == "sign":
            draw.rounded_rectangle((48, 144, 464, 368), radius=20, fill=(20, 65, 190, 255))
            font = ImageFont.load_default(size=80)
            draw.text((256, 256), "OPEN", font=font, anchor="mm", fill=(255, 255, 255, 255))
        elif name == "transparent":
            points = [(256 + (170 if i % 2 == 0 else 75) * math.sin(i * math.pi / 5),
                       256 - (170 if i % 2 == 0 else 75) * math.cos(i * math.pi / 5)) for i in range(10)]
            draw.polygon(points, fill=(220, 35, 35, 255))
        elif name == "gradient":
            for x in range(160, 352):
                value = int(90 + 100 * math.sin((x - 160) / 192 * math.pi))
                draw.line((x, 160, x, 444), fill=(20, 60, value, 255))
            draw.rectangle((198, 108, 314, 160), fill=(235, 235, 235, 255), outline=(120, 120, 120, 255), width=2)
        else:
            for y in range(8):
                for x in range(8):
                    if (x + y) % 2 == 0:
                        draw.rectangle((x * 64, y * 64, x * 64 + 63, y * 64 + 63), fill=(0, 0, 0, 255))
        images[name] = image
    return images


def freeze(definition, snapshot, output):
    import importlib.metadata
    import numpy as np
    from .preprocessing import denoiser_layout, initial_noise, prompt_inputs, text_inputs

    definition, snapshot, output = Path(definition), Path(snapshot), Path(output)
    require(snapshot.name == REVISION, "Unexpected source snapshot")
    cases = json.loads(definition.read_text())
    definition_hash = validate_cases(cases)
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    (output / "noise").mkdir()
    images = reference_images()
    for name, image in images.items():
        image.save(output / "images" / (name + ".png"))
    for seed in cases["seeds"]:
        np.save(output / "noise" / f"{seed}.npy", initial_noise(seed))
    entries = []
    for case in cases["cases"]:
        image = images.get(case.get("reference"))
        encoded, drop, image_id = prompt_inputs(snapshot, case["prompt"], image)
        length = int(encoded["input_ids"].shape[1])
        slots = (encoded["input_ids"][0] == image_id)[drop:]
        structure = denoiser_layout(slots)
        directory = output / case["id"]
        directory.mkdir()
        (directory / "prompt.txt").write_text(case["prompt"], encoding="utf-8")
        np.savez_compressed(directory / "encoded.npz", **encoded)
        features = [np.zeros((256, 4096), np.float32)] * 4 if image else None
        text = text_inputs(snapshot, encoded, features)
        np.savez_compressed(directory / "text-preprocessing.npz", **text)
        np.savez_compressed(directory / "layout.npz", **{key: value for key, value in structure.items()
                                                        if isinstance(value, np.ndarray)})
        reference = None if image is None else "images/" + case["reference"] + ".png"
        if image is not None:
            rgba = np.asarray(image, dtype=np.float32) / np.float32(255) * np.float32(2) - np.float32(1)
            np.save(directory / "vae-input.npy", np.ascontiguousarray(rgba.transpose(2, 0, 1)[None, :, None]))
        metadata = {"id": case["id"], "operation": "edit" if image else "generate",
                    "prompt": case["prompt"], "language": case["language"], "axes": case["axes"],
                    "reference": reference, "processed_tokens": length, "drop": drop,
                    "image_token_id": image_id, "expanded_tokens": structure["length"],
                    "source_revision": REVISION, "sigmas": SIGMAS, "reference_outputs": "not-run"}
        save_json(directory / "input.json", metadata)
        for seed in cases["seeds"]:
            entries.append({"run_id": f"{case['id']}-{seed}", "fixture_id": case["id"], "seed": seed,
                            "source_revision": REVISION, "sigmas": SIGMAS, "operation": metadata["operation"],
                            "prompt_sha256": file_record(directory / "prompt.txt")["sha256"],
                            "reference_sha256": file_record(output / reference)["sha256"] if reference else None,
                            "noise": f"noise/{seed}.npy", "noise_sha256": file_record(output / "noise" / f"{seed}.npy")["sha256"],
                            "processed_tokens_sha256": hashlib.sha256(encoded["input_ids"].astype("<i4").tobytes()).hexdigest(),
                            "layout_sha256": file_record(directory / "layout.npz")["sha256"]})
    inventory = {path.relative_to(output).as_posix(): file_record(path)
                 for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = {"format": 1, "id": cases["id"], "definition_sha256": definition_hash,
                "source_revision": REVISION, "inventory": inventory, "runs": entries,
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "diffusers", "Pillow", "numpy")},
                "baseline_status": "not-run", "quality_status": "unreviewed",
                "image_origin": cases["image_origin"], "rng_version": "torch-2.9.0-cpu-randn-packed-v1"}
    save_json(output / "fixtures.json", manifest)
    validate_fixture_suite(output, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.definition, args.snapshot, args.output)
    print(json.dumps({"inputs": 30, "runs": len(result["runs"]), "manifest": str(args.output / "fixtures.json")}))


if __name__ == "__main__":
    main()
