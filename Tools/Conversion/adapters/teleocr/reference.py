import argparse
import resource
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .contracts import IMAGE_TOKEN, PROMPTS, SYSTEM, VISION_CASES, logit_metrics, window_inputs, write_json


def load(snapshot):
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(str(snapshot), trust_remote_code=True, use_fast=True, local_files_only=True)
    model = AutoModel.from_pretrained(str(snapshot), trust_remote_code=True, torch_dtype=torch.float32,
                                      local_files_only=True).eval()
    return processor, model


def case_image(fixture, case):
    image = Image.open(fixture).convert("RGB")
    if "crop" in case:
        image = image.crop(tuple(case["crop"]))
    return image.resize(tuple(case["size"]), Image.Resampling.BICUBIC)


def prompt_inputs(processor, image, prompt):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image], padding=True, return_tensors="pt")


def merged_rope(model, embeds, positions):
    config = model.config.text_config
    cos, sin = model.model.language_model.rotary_emb(embeds, positions)
    sections = config.rope_scaling["mrope_section"] * 2
    return tuple(torch.cat([part[index % 3] for index, part in enumerate(value.split(sections, dim=-1))], dim=-1)
                 for value in (cos, sin))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(8)
    processor, model = load(args.snapshot)
    visual = model.visual
    summary = {"vision": {}, "fixture": args.fixture.name}
    with torch.no_grad():
        for name, case in VISION_CASES.items():
            encoded = processor.image_processor(images=[case_image(args.fixture, case)], return_tensors="pt")
            pixels, grid = encoded["pixel_values"].float(), encoded["image_grid_thw"]
            _, grid_h, grid_w = (int(value) for value in grid[0])
            reference = visual(pixels, grid_thw=grid).numpy()
            inputs, order = window_inputs(pixels.numpy(), visual.rot_pos_emb(grid).numpy(), grid_h, grid_w)
            original, _ = visual.get_window_index(grid)
            if not np.array_equal(order[order >= 0], original.numpy()):
                raise RuntimeError(f"The padded window order differs from the original window index: {name}")
            np.savez(args.output / f"vision-{name}.npz", **inputs, order=order, reference=reference, grid=grid.numpy())
            summary["vision"][name] = {"size": case["size"], "grid": [grid_h, grid_w], "patches": len(inputs["key_valid"]),
                                       "valid_patches": int(inputs["key_valid"].sum()),
                                       "windows": len(inputs["key_valid"]) // 64, "image_tokens": int(reference.shape[0])}
            print(name, summary["vision"][name], flush=True)
        image = case_image(args.fixture, VISION_CASES["square448"])
        inputs = prompt_inputs(processor, image, PROMPTS["text"])
        ids, grid = inputs.input_ids, inputs.image_grid_thw
        logits = model(**inputs, use_cache=False).logits[:, -1, :].numpy()
        inner = model.model
        table = inner.get_input_embeddings()
        embedded = table(ids)
        features = torch.cat(inner.get_image_features(inputs.pixel_values, grid), dim=0)
        merged = embedded.masked_scatter((ids == IMAGE_TOKEN).unsqueeze(-1).expand_as(embedded), features)
        positions, deltas = inner.get_rope_index(ids, grid, None, attention_mask=inputs.attention_mask)
        cos, sin = merged_rope(model, merged, positions)
        count = ids.shape[1]
        mask = torch.full((count, count), float("-inf")).triu(1)[None, None]
        weight = table.weight
        exact = weight.half().float() == weight
        rows = weight[ids[0]]
    np.savez(args.output / "prefill-f1-448.npz", input_ids=ids.numpy().astype(np.int32), inputs_embeds=merged.numpy(),
             cos=cos.numpy(), sin=sin.numpy(), mask=mask.numpy(), embeddings=embedded.numpy(), logits=logits,
             positions=positions.numpy().astype(np.int32), rope_deltas=deltas.numpy().astype(np.int32))
    summary["prefill"] = {"tokens": count, "image_tokens": int((ids == IMAGE_TOKEN).sum()),
                          "rope_delta": int(deltas.reshape(-1)[0]), "position_max": [int(p.max()) for p in positions],
                          "reference_argmax_text": processor.tokenizer.decode([int(logits.argmax())]),
                          **{key: value for key, value in logit_metrics(logits, logits).items()
                             if key in ("reference_argmax", "reference_top5", "reference_top2_gap")}}
    summary["embedding_fp16"] = {"table_elements": int(weight.numel()), "table_inexact": int((~exact).sum()),
                                 "prompt_elements": int(rows.numel()),
                                 "prompt_inexact": int((rows.half().float() != rows).sum())}
    summary["seconds"] = time.perf_counter() - started
    summary["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(args.output / "reference.json", summary)
    print(summary["prefill"], summary["embedding_fp16"], flush=True)


if __name__ == "__main__":
    main()
