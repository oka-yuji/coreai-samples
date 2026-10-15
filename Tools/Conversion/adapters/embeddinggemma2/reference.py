import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from transformers import AutoConfig, AutoProcessor, EmbeddingGemma2Model

from .contracts import spatial_pooling_weights


def embedding(hidden, mask):
    weights = mask.to(torch.float32).unsqueeze(-1)
    pooled = (hidden.float() * weights).sum(1) / weights.sum(1)
    return torch.nn.functional.normalize(pooled, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    source = args.source
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    prompts = json.loads((source / "config_sentence_transformers.json").read_text())["prompts"]
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    config.audio_config = None
    config._attn_implementation = "eager"
    print("Loading original FP32 text and vision", flush=True)
    model = EmbeddingGemma2Model.from_pretrained(
        source, config=config, dtype=torch.float32, attn_implementation="eager", local_files_only=True
    ).eval()
    fixtures = []
    for name, role, content in json.loads(args.fixtures.read_text())["texts"]:
        fixtures.append({"id": name, "role": role, "text": prompts[role] + content, "raw_text": content})
    image_dir = args.output / "images"
    image_dir.mkdir()
    for name in ["01-oyakodon", "02-ginger-pork", "03-omelette", "04-gratin", "05-caprese"]:
        data = (args.images / f"{name}.png").read_bytes()
        (image_dir / f"{name}.png").write_bytes(data)
        fixtures.append({"id": name, "image": f"images/{name}.png", "source": "sample refrigerator image",
                         "sha256": hashlib.sha256(data).hexdigest()})
    for name, width, height in [("gradient_square", 144, 144), ("gradient_wide", 320, 96), ("gradient_tall", 96, 320)]:
        yy, xx = np.mgrid[:height, :width]
        pixels = np.stack([xx * 255 // max(1, width - 1), yy * 255 // max(1, height - 1), (xx + yy) % 256], -1).astype(np.uint8)
        path = image_dir / f"{name}.png"
        Image.fromarray(pixels).save(path)
        fixtures.append({"id": name, "image": f"images/{name}.png", "source": "deterministic pixel fixture", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    specification = {"model_revision": source.name, "text_length": 128, "vision_tokens": 70, "fixtures": fixtures,
                     "fp32_gate": {"cosine_min": 0.99999, "max_abs_max": 0.0001},
                     "low_precision_gate": {"cosine_min": 0.999, "max_abs_max": 0.01}}
    (args.output / "fixtures.json").write_text(json.dumps(specification, ensure_ascii=False, indent=2) + "\n")
    records = []
    with torch.inference_mode():
        table = model.get_input_embeddings().weight.detach().cpu().numpy()
        np.save(args.output / "token_embeddings.npy", table)
        for fixture in fixtures:
            name = fixture["id"]
            destination = args.output / name
            destination.mkdir()
            if "image" in fixture:
                image = Image.open(args.output / fixture["image"]).convert("RGB")
                batch = processor(images=[image], max_soft_tokens=70, return_tensors="pt", padding="max_length", max_length=128, truncation=False)
            else:
                batch = processor(text=fixture["text"], return_tensors="pt", padding="max_length", max_length=128, truncation=False)
            if batch["input_ids"].shape != (1, 128):
                raise ValueError(f"Unexpected token length for {name}: {batch['input_ids'].shape}")
            keys = ["input_ids", "attention_mask", "pixel_values", "image_position_ids"]
            inputs = {key: batch[key] for key in keys if key in batch}
            started = time.perf_counter()
            output = model(**inputs)
            vector = embedding(output.last_hidden_state, batch["attention_mask"])
            elapsed = time.perf_counter() - started
            for key, value in inputs.items():
                np.save(destination / f"{key}.npy", value.detach().cpu().numpy())
            np.save(destination / "reference.npy", vector.detach().cpu().numpy())
            embeddings = model.get_input_embeddings()(batch["input_ids"]).clone()
            if "pixel_values" in inputs:
                features = model.get_image_features(pixel_values=inputs["pixel_values"], image_position_ids=inputs["image_position_ids"]).pooler_output[0]
                selected = batch["input_ids"] == config.image_token_id
                embeddings[selected] = features
                np.save(destination / "soft_tokens.npy", features.detach().cpu().numpy())
                weights, valid = spatial_pooling_weights(inputs["image_position_ids"][0].numpy(), 70)
                np.save(destination / "pooling_weights.npy", weights[None])
                (destination / "pixel_values.f32").write_bytes(batch["pixel_values"].cpu().numpy().astype("<f4").tobytes())
                np.save(destination / "valid_soft_tokens.npy", valid)
            np.save(destination / "inputs_embeds.npy", embeddings.detach().cpu().numpy())
            record = {"id": name, "tokens": int(batch["attention_mask"].sum()), "seconds": elapsed,
                      "finite": bool(torch.isfinite(vector).all()), "norm": float(vector.norm()),
                      "inputs": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in inputs.items()}}
            records.append(record)
            print(json.dumps(record), flush=True)
    if not all(row["finite"] and abs(row["norm"] - 1) < 0.001 for row in records):
        raise RuntimeError("The original FP32 reference failed finite or normalization checks")
    (args.output / "reference-summary.json").write_text(json.dumps({"records": records, "complete": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
