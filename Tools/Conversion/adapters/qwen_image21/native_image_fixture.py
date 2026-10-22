import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

from .fixtures import save_json
from .preprocessing import prompt_inputs


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run(args):
    output = args.output
    if (output / "image-golden.json").exists():
        raise FileExistsError("Image goldens are immutable")
    pixels = np.arange(32 * 24 * 4, dtype=np.uint32).reshape(24, 32, 4)
    pixels = ((pixels * 37 + 17) % 256).astype(np.uint8)
    pixels[:, :, 3] = np.array([0, 1, 8, 63, 127, 128, 254, 255], np.uint8)[np.arange(32) % 8]
    source = Image.fromarray(pixels, "RGBA")
    source.save(output / "rgba-source.png")
    source.save(output / "rgba-srgb.png", icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    exif = Image.Exif()
    exif[274] = 6
    source.save(output / "rgba-oriented.png", exif=exif)
    oriented = ImageOps.exif_transpose(Image.open(output / "rgba-oriented.png")).convert("RGBA")
    variants = [(1, source), (2, source.transpose(Image.Transpose.FLIP_LEFT_RIGHT)),
                (3, source.transpose(Image.Transpose.ROTATE_180)), (4, source.transpose(Image.Transpose.FLIP_TOP_BOTTOM)),
                (5, source.transpose(Image.Transpose.TRANSPOSE)), (6, source.transpose(Image.Transpose.ROTATE_270)),
                (7, source.transpose(Image.Transpose.TRANSVERSE)), (8, source.transpose(Image.Transpose.ROTATE_90))]
    resizes = []
    for width, height in [(32, 24), (7, 5), (31, 17), (64, 48)]:
        resized = source.resize((width, height), Image.Resampling.BICUBIC)
        resizes.append({"width": width, "height": height, "rgbaSHA256": digest(np.asarray(resized))})
    canonical = source.crop((4, 0, 28, 24)).resize((512, 512), Image.Resampling.BICUBIC)
    canonical.save(output / "canonical-reference.png")
    encoded, _, _ = prompt_inputs(args.snapshot, "reference", canonical)
    white = Image.new("RGB", canonical.size, "white")
    white.paste(canonical, mask=canonical.getchannel("A"))
    vae = np.asarray(canonical, np.float32) / np.float32(255) * np.float32(2) - np.float32(1)
    vae = np.ascontiguousarray(vae.transpose(2, 0, 1)[None, :, None])
    pattern = np.array([-2, -1, -0.5, 0, 0.5, 1, 2, 0.0039215689,
                        0.1, -0.1, 0.9, -0.9, 0.25, -0.25, 0.75, -0.75], np.float32)
    decoded = np.resize(pattern, (1, 4, 1, 512, 512))
    pixels = np.clip(decoded[0, :, 0].transpose(1, 2, 0) / 2 + 0.5, 0, 1)
    final = (pixels * 255).round().astype(np.uint8)
    save_json(output / "image-golden.json", {
        "format": 1, "pillowVersion": "13.0.0", "referenceVersion": "center-crop-bicubic-straight-rgba8-v1",
        "sourceSHA256": digest(np.asarray(source)), "orientedSHA256": digest(np.asarray(oriented)),
        "orientations": [{"orientation": number, "width": value.width, "height": value.height,
                          "rgbaSHA256": digest(np.asarray(value))} for number, value in variants],
        "resizes": resizes, "canonicalSHA256": digest(np.asarray(canonical)),
        "whiteSHA256": digest(np.asarray(white)), "vaeSHA256": digest(vae),
        "visionSHA256": digest(encoded["pixel_values"].astype(np.float32)),
        "decoderPattern": pattern.tolist(), "decoderSHA256": digest(final),
        "colorPolicy": "Preserve encoded RGB channel values as Pillow convert(RGBA); do not apply an ICC display transform",
        "scope": "Original authored PNGs, all EXIF orientations, Pillow bicubic and Qwen3VL processor; no learned weights"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
