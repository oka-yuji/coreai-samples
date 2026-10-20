import argparse
import hashlib
import json
import math
from pathlib import Path

from .evaluation import REVISION, SIGMAS, canonical_hash, require, validate_inventory


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def file_record(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def validate_cases(value):
    require(value.get("format") == 1 and value.get("id") == "qmi-quality-30x2-v1", "Unsupported fixture definition")
    require(value.get("seeds") == [20261018, 20261021], "The two seeds are frozen")
    cases = value.get("cases")
    require(isinstance(cases, list) and len(cases) == 30, "Expected all thirty inputs")
    require(len({case["id"] for case in cases}) == 30, "Duplicate fixture id")
    for case in cases:
        require(case.get("language") in ("ja", "en") and isinstance(case.get("prompt"), str)
                and case["prompt"].strip(), "Invalid fixture prompt")
        require(case.get("axes") and set(case["axes"]) <= {"instruction", "text", "detail", "preservation", "alpha"},
                "Missing quality axes")
        require(case.get("reference") in (None, "circle", "objects", "person", "sign", "transparent", "gradient", "checker"),
                "Unknown authored reference")
    require(sum("reference" in case for case in cases) == 15, "Expected fifteen edits and fifteen generations")
    return canonical_hash(value)


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
    validate_inventory(output, inventory)
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
