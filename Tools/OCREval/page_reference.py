import json
import resource
import sys
import time
from pathlib import Path

from .reference import LAYOUT_SIDE, MAX_TIME_S, PROMPTS, footprint, sha256, snapshot
from .score import BOX


def page_plan(layout, width, height):
    found, skipped = [], []
    for match in BOX.finditer(layout):
        values = [int(v) for v in match.groups()[:4]]
        products = [v * size for v, size in zip(values, (width, height, width, height))]
        if any(v > sys.maxsize - 999 for v in products):
            continue
        box = [products[0] // 1000, products[1] // 1000, (products[2] + 999) // 1000, (products[3] + 999) // 1000]
        block = {"label": match[5], "box_norm": values, "box_px": box}
        w, h = box[2] - box[0], box[3] - box[1]
        valid = 0 < w <= 32768 and 0 < h <= 32768 and w * h <= 100_000_000 and max(w, h) <= 200 * min(w, h)
        (found if valid else skipped).append(block)
    if not found:
        found = [{"label": None, "box_norm": None, "box_px": [0, 0, width, height]}]
    return found, skipped


def read_page(image, recognize):
    from PIL import Image

    layout = recognize(image.resize((LAYOUT_SIDE, LAYOUT_SIDE), Image.Resampling.BICUBIC), "layout")
    blocks, skipped = page_plan(layout["text"], *image.size)
    results = []
    for block in blocks:
        prompt = "table" if block["label"] == "table" else "text"
        result = recognize(image.crop(tuple(block["box_px"])), prompt)
        results.append(block | result | {"prompt": prompt, "otsl": result["text"] if prompt == "table" else None})
    return {"schema": "teleocr-reference-page/1", "image_px": list(image.size),
            "layout": layout | {"status": "parsed" if blocks[0]["label"] is not None else "unparsed", "skipped_boxes": skipped},
            "blocks": results}


def restore_weights(model, folder, expected):
    import torch
    from safetensors.torch import load_file
    from Tools.Conversion.adapters.teleocr.quantization import targets, tensor_record

    folder = Path(folder)
    record = json.loads((folder / "compression.json").read_text())
    if sha256(folder / "compression.json") != expected["compression_sha256"]:
        raise ValueError("Compression manifest differs from the fixed plan")
    if sha256(folder / "reference.safetensors") != expected["reference_sha256"] or record["reference_sha256"] != expected["reference_sha256"]:
        raise ValueError("Reference weights differ from the fixed plan")
    module = {"decoder": model.language_model, "vision": model.visual}[record["kind"]]
    names = set(targets(module, record["kind"], record["bits"]))
    layers = {layer["name"]: layer for layer in record["layers"]}
    weights = load_file(folder / "reference.safetensors")
    if names != set(layers) or len(layers) != len(record["layers"]) or set(weights) != {name + ".weight" for name in names}:
        raise ValueError("Reference target keys differ from the declared compression scope")
    for name, layer in layers.items():
        value = weights[name + ".weight"]
        target = module.get_submodule(name).weight
        if tensor_record(value) != layer["effective"] or value.shape != target.shape or target.dtype != torch.float32:
            raise ValueError(f"Invalid reference tensor: {name}")
    with torch.no_grad():
        for name in names:
            module.get_submodule(name).weight.copy_(weights[name + ".weight"])
    return {"kind": record["kind"], "bits": record["bits"], "targets": len(names), **expected}


def run(image_path, out, variant):
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[variant["dtype"]]
    source = snapshot()
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(source), trust_remote_code=True, use_fast=True)
    model = AutoModel.from_pretrained(str(source), trust_remote_code=True, torch_dtype=dtype).eval()
    restored = restore_weights(model, variant["weights"], variant["weight_hashes"]) if variant.get("weights") else None
    model.to("mps")
    torch.mps.synchronize()
    load_s = time.perf_counter() - started
    if model.generation_config.repetition_penalty != 1.05:
        raise ValueError("The reference requires repetition_penalty=1.05")

    def recognize(image, prompt):
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPTS[prompt]}]}]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat], images=[image], padding=True, return_tensors="pt").to(device="mps", dtype=dtype)
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, use_cache=True, max_new_tokens=4096, do_sample=False, max_time=MAX_TIME_S)
        torch.mps.synchronize()
        seconds = time.perf_counter() - start
        tokens = output.cpu().tolist()[0][inputs.input_ids.shape[1]:]
        text = processor.batch_decode([tokens], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        raw = processor.batch_decode([tokens], skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        finish = "eos" if tokens and tokens[-1] in (151645, 151643) else "max_new_tokens" if len(tokens) >= 4096 else "max_time"
        return {"text": text, "raw_text": raw, "token_ids": tokens, "finished_by": finish, "generate_s": seconds,
                "tokens": {"prompt": inputs.input_ids.shape[1], "output": len(tokens)}}

    image = Image.open(image_path).convert("RGB")
    started = time.perf_counter()
    result = read_page(image, recognize)
    phys, peak = footprint()
    result |= {"load_s": load_s, "generate_s": time.perf_counter() - started, "dtype": variant["dtype"],
               "reference": restored, "image_sha256": sha256(image_path), "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "footprint_bytes": phys, "footprint_peak_bytes": peak}
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
