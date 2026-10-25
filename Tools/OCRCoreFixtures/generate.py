import argparse
import base64
import hashlib
import importlib.util
import itertools
import json
import math
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import PIL
import torch
import torchvision
import transformers
from PIL import Image
from torchvision.transforms.v2 import functional as F
from transformers import AutoConfig, AutoProcessor
from transformers.generation.logits_process import LogitsProcessorList, RepetitionPenaltyLogitsProcessor
from transformers.generation.stopping_criteria import EosTokenCriteria, MaxLengthCriteria, StoppingCriteriaList
from transformers.image_processing_utils_fast import SizeDict
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

REVISION = "70a19ec6a3b036311af19ae5e762e650f41a5980"
SNAPSHOT = Path.home() / f".cache/huggingface/hub/models--XingChen-AGI--TeleOCR/snapshots/{REVISION}"
SYSTEM = "You are a helpful assistant."
PROMPTS = {
    "text": "Please output the text content from the image.",
    "table": "This is the image of a table. Please output the table in OTSL format.",
    "layout": "Analyze the image layout.",
}
SEGMENTS = ["system\n" + SYSTEM, "\n", "user\n", "assistant\n", *PROMPTS.values()]
STOP_TOKENS = [151645, 151643]
VOCABULARY = 151936
PENALTY = 1.05
OTSL_TOKENS = ["<nl>", "<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>"]
OTSL_CELLS = ("<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>")
BOX = re.compile(r"<box:(\d+) (\d+) (\d+) (\d+)><label:([^>]+)>(?:<([a-z]+)>)?")


def pattern(width, height):
    x = np.arange(width, dtype=np.int64)[None, :, None]
    y = np.arange(height, dtype=np.int64)[:, None, None]
    c = np.arange(3, dtype=np.int64)[None, None, :]
    return ((((x * 1103 + y * 2927 + c * 4111 + x * y * 17) * 2654435761) % (1 << 32)) >> 24).astype(np.uint8)


def bits(values):
    return np.ascontiguousarray(np.asarray(values, dtype=np.float32)).reshape(-1).view(np.uint32).tolist()


def float_digest(values):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values, dtype="<f4")).tobytes()).hexdigest()


def byte_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.uint8).tobytes()).hexdigest()


def load_module(snapshot):
    spec = importlib.util.spec_from_file_location("modeling_naviocr", snapshot / "modeling_naviocr.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def torch_resize(processor, array, width, height):
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)[None].contiguous()
    out = processor.image_processor.resize(
        image=tensor, size=SizeDict(height=height, width=width), interpolation=F.InterpolationMode.BICUBIC)
    return out[0].permute(1, 2, 0).contiguous().numpy()


def pillow_resize(array, width, height):
    return np.asarray(Image.fromarray(array).resize((width, height), Image.Resampling.BICUBIC))


def image_fixtures(processor):
    size = processor.image_processor.size
    smart = []
    shapes = [(448, 448), (1036, 1036), (1754, 1240), (200, 300), (10, 10), (14, 14), (56, 56), (42, 1000), (70, 98),
              (8000, 6000), (6048, 8064), (12000, 9000), (1, 200), (1, 201), (3, 700), (100, 20001), (20000, 101)]
    for height, width in shapes:
        try:
            h, w = smart_resize(height, width, factor=28, min_pixels=size["shortest_edge"], max_pixels=size["longest_edge"])
            smart.append({"width": width, "height": height, "resizedWidth": w, "resizedHeight": h})
        except ValueError:
            smart.append({"width": width, "height": height, "rejected": True})
    shared = [(33, 40, 56, 84), (150, 97, 56, 28), (61, 45, 56, 56), (64, 64, 64, 28), (50, 30, 28, 30),
              (300, 210, 28, 28), (9, 5, 28, 28), (40, 120, 28, 28), (120, 40, 28, 28), (20, 120, 40, 28),
              (30, 100, 20, 80), (30, 100, 20, 79), (20, 20, 20, 20)]
    resizes = []
    for flavor in ("torchVision", "pillow"):
        for sw, sh, dw, dh in shared:
            source = pattern(sw, sh)
            out = torch_resize(processor, source, dw, dh) if flavor == "torchVision" else pillow_resize(source, dw, dh)
            resizes.append({"resampling": flavor, "width": sw, "height": sh, "resizedWidth": dw, "resizedHeight": dh,
                            "rgb": base64.b64encode(np.ascontiguousarray(out).tobytes()).decode()})
    page = pattern(1240, 1754)
    digests = [
        {"resampling": "torchVision", "width": 1240, "height": 1754, "resizedWidth": 1232, "resizedHeight": 1764,
         "sha256": byte_digest(torch_resize(processor, page, 1232, 1764))},
        {"resampling": "pillow", "width": 1240, "height": 1754, "resizedWidth": 1036, "resizedHeight": 1036,
         "sha256": byte_digest(pillow_resize(page, 1036, 1036))},
    ]
    return {"smartResize": smart, "resize": resizes, "resizeDigests": digests}


def vision_stub(module, config):
    vision = config.vision_config
    return SimpleNamespace(
        spatial_merge_size=vision.spatial_merge_size, patch_size=vision.patch_size, window_size=vision.window_size,
        spatial_merge_unit=vision.spatial_merge_size ** 2,
        rotary_pos_emb=module.Qwen2_5_VisionRotaryEmbedding(vision.hidden_size // vision.num_heads // 2))


def window_order(rotary, window_index):
    return rotary.reshape(-1, 4, rotary.shape[-1])[window_index].reshape(-1, rotary.shape[-1])


def vision_case(name, processor, module, config, array, full):
    model = module.Qwen2_5_VisionTransformerPretrainedModel
    stub = vision_stub(module, config)
    out = processor.image_processor(images=[Image.fromarray(array)], return_tensors="pt")
    pixels, grid = out["pixel_values"], out["image_grid_thw"]
    t, gh, gw = grid[0].tolist()
    window_index, cu = model.get_window_index(stub, grid)
    bounds = torch.unique_consecutive(torch.tensor(cu, dtype=torch.int32))
    lengths = (bounds[1:] - bounds[:-1]).tolist()
    case = {"name": name, "width": array.shape[1], "height": array.shape[0], "grid": [t, gh, gw],
            "pixelValuesSHA256": float_digest(pixels.numpy()), "windowLengths": lengths}
    if not full:
        return case
    rotary = model.rot_pos_emb(stub, grid)
    emb = torch.cat((window_order(rotary, window_index),) * 2, dim=-1)
    cos, sin = emb.cos(), emb.sin()
    width = rotary.shape[-1] // 2
    tagged = SimpleNamespace(**vars(stub))
    tagged.rotary_pos_emb = lambda n: torch.arange(n, dtype=torch.float32)[:, None].repeat(1, width)
    positions = window_order(model.rot_pos_emb(tagged, grid), window_index)
    assert torch.equal(positions[:, :width], positions[:, :1].expand(-1, width))
    assert torch.equal(positions[:, width:], positions[:, width:width + 1].expand(-1, width))
    tagged.rotary_pos_emb = lambda n: torch.arange(width, dtype=torch.float32)[None, :].repeat(n, 1)
    frequencies = model.rot_pos_emb(tagged, grid)
    assert torch.equal(frequencies, torch.arange(width, dtype=torch.float32).repeat(2)[None, :].expand_as(frequencies))
    table = stub.rotary_pos_emb(max(gh, gw))
    first, second = positions[:, 0].long(), positions[:, width].long()
    rebuilt = torch.cat((table[first], table[second]), dim=-1).repeat(1, 2)
    difference = max(float((rebuilt.cos() - cos).abs().max()), float((rebuilt.sin() - sin).abs().max()))
    assert difference <= 1e-6
    rows = sorted({0, 1, 2, 3, 4, 63, 64, cos.shape[0] // 2, cos.shape[0] - 5, cos.shape[0] - 1})
    case.update({
        "pixelRows": [{"index": i, "values": bits(pixels[i])} for i in (0, pixels.shape[0] - 1)],
        "windowIndex": window_index.tolist(),
        "restore": torch.argsort(window_index).tolist(),
        "rotary": {
            "inverseFrequencyBits": bits(stub.rotary_pos_emb.inv_freq),
            "firstAxisPositions": first.tolist(),
            "secondAxisPositions": second.tolist(),
            "cosineTableBits": bits(table.cos()),
            "sineTableBits": bits(table.sin()),
            "tableReconstructionMaxDifference": difference,
            "rows": [{"index": i, "cosineBits": bits(cos[i]), "sineBits": bits(sin[i])} for i in rows],
        },
    })
    return case


def vision_fixtures(processor, module, config):
    return {"cases": [
        vision_case("square448", processor, module, config, pattern(448, 448), True),
        vision_case("square1036", processor, module, config, pattern(1036, 1036), True),
        vision_case("landscape300x200", processor, module, config, pattern(300, 200), True),
        vision_case("page1240x1754", processor, module, config, pattern(1240, 1754), False),
    ]}


def chat(processor, prompt):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def merged(module, cos3, sin3, section):
    shape = cos3.shape[:-1] + (cos3.shape[-1],)
    ones = torch.ones(shape[1], 1, shape[2], shape[3], dtype=torch.float32)
    cos = module.apply_multimodal_rotary_pos_emb(ones, ones, cos3, torch.zeros_like(sin3), section)[0][:, 0]
    sign = module.rotate_half(ones)[:, 0]
    sin = module.apply_multimodal_rotary_pos_emb(ones, ones, torch.zeros_like(cos3), sin3, section)[0][:, 0] * sign
    return cos[0], sin[0]


def rope(module, config, ids, grid):
    stub = SimpleNamespace(config=config)
    positions, deltas = module.Qwen2_5_VLModel.get_rope_index(
        stub, ids, grid, None, None, attention_mask=torch.ones_like(ids))
    return positions, int(deltas.reshape(-1)[0])


def prompt_case(name, processor, module, config, prompt, image):
    inputs = processor(text=[chat(processor, prompt)], images=[image], padding=True, return_tensors="pt")
    ids, grid = inputs.input_ids, inputs.image_grid_thw
    positions, delta = rope(module, config, ids, grid)
    image_slots = (ids[0] == config.image_token_id).nonzero().reshape(-1).tolist()
    assert image_slots == list(range(image_slots[0], image_slots[-1] + 1))
    return {"name": name, "prompt": prompt, "gridHeight": int(grid[0, 1]), "gridWidth": int(grid[0, 2]),
            "ids": ids[0].tolist(), "imageStart": image_slots[0], "imageCount": len(image_slots),
            "positions": positions[:, 0].tolist(), "ropeDelta": delta}, inputs, positions, delta


def text_rotary(module, config, positions, delta, length):
    text = config.text_config
    section = text.rope_scaling["mrope_section"]
    rotary = module.Qwen2_5_VLRotaryEmbedding(config=text)
    x = torch.zeros(1, dtype=torch.float32)
    cos3, sin3 = rotary(x, positions)
    assert torch.equal(cos3[..., :64], cos3[..., 64:]) and torch.equal(sin3[..., :64], sin3[..., 64:])
    cos, sin = merged(module, cos3, sin3, section)
    tags = torch.stack([torch.full((1, length, 128), float(a * 1000), dtype=torch.float32) for a in range(3)])
    tags = tags + torch.arange(128, dtype=torch.float32)
    tag_cos, _ = merged(module, tags, torch.zeros_like(tags), section)
    axes = (tag_cos[0] // 1000).long()
    source = (tag_cos[0] % 1000).long()
    assert torch.equal(tag_cos, tag_cos[:1].expand_as(tag_cos))
    top = int(positions.max())
    grid = torch.arange(top + 1, dtype=torch.long)[None, None, :].expand(3, 1, -1)
    table_cos, table_sin = rotary(x, grid)
    table_cos, table_sin = table_cos[0, 0, :, :64], table_sin[0, 0, :, :64]
    picked = positions[:, 0][axes, :].T
    rebuilt_cos = table_cos[picked, source % 64]
    rebuilt_sin = table_sin[picked, source % 64]
    difference = max(float((rebuilt_cos - cos).abs().max()), float((rebuilt_sin - sin).abs().max()))
    assert difference <= 1e-6
    rows = sorted({0, 14, 15, 16, 100, 270, length - 16, length - 1})
    decode = []
    for write in (length, length + 1, 1000, length + 4095):
        step = (torch.arange(1).view(1, 1, -1).expand(3, 1, -1) + write + delta)
        step_cos, step_sin = merged(module, *rotary(x, step), section)
        decode.append({"writePosition": write, "cosineBits": bits(step_cos[0]), "sineBits": bits(step_sin[0])})
    return {
        "inverseFrequencyBits": bits(rotary.inv_freq),
        "sectionAxes": axes.tolist(),
        "frequencyIndex": (source % 64).tolist(),
        "cosineTableBits": bits(table_cos),
        "sineTableBits": bits(table_sin),
        "tableReconstructionMaxDifference": difference,
        "rows": [{"index": i, "cosineBits": bits(cos[i]), "sineBits": bits(sin[i])} for i in rows],
        "decode": decode,
    }


def prompt_fixtures(processor, module, config, knowledge):
    tokenizer = processor.tokenizer
    segments = {s: tokenizer.encode(s, add_special_tokens=False) for s in SEGMENTS}
    f1 = Image.open(knowledge / "fixtures/F1.png").convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
    cases = []
    first, inputs, positions, delta = prompt_case("f1Text448", processor, module, config, PROMPTS["text"], f1)
    assert len(first["ids"]) == 286 and delta == -240 and int(positions.max()) == 45
    cases.append(first)
    layout = Image.fromarray(pattern(1036, 1036))
    cases.append(prompt_case("layout1036", processor, module, config, PROMPTS["layout"], layout)[0])
    table = Image.fromarray(pattern(300, 200))
    cases.append(prompt_case("table300x200", processor, module, config, PROMPTS["table"], table)[0])
    for case in cases:
        rebuilt = []
        special = {"<|im_start|>": 151644, "<|im_end|>": 151645, "<|vision_start|>": 151652, "<|vision_end|>": 151653}
        for part in ["<|im_start|>", "system\n" + SYSTEM, "<|im_end|>", "\n", "<|im_start|>", "user\n",
                     "<|vision_start|>", None, "<|vision_end|>", case["prompt"], "<|im_end|>", "\n", "<|im_start|>",
                     "assistant\n"]:
            rebuilt += [config.image_token_id] * case["imageCount"] if part is None else (
                [special[part]] if part in special else segments[part])
        assert rebuilt == case["ids"], case["name"]
    rotary = text_rotary(module, config, positions, delta, len(first["ids"]))
    return {"segments": [{"text": s, "ids": ids} for s, ids in segments.items()], "cases": cases,
            "textRotary": rotary}


def sparse_logits(step):
    scores = torch.full((1, VOCABULARY), step["base"], dtype=torch.float32)
    for index, value in step["overrides"]:
        scores[0, index] = value
    return scores


def encode_step(step):
    return {"baseBits": bits([step["base"]])[0],
            "overrides": [[i, bits([v])[0]] for i, v in step["overrides"]]}


def generation_fixtures():
    rng = np.random.default_rng(309)
    noisy = [(int(i), float(v)) for i, v in zip(rng.choice(VOCABULARY, 300, replace=False), rng.normal(0, 12, 300))]
    single = [
        ("seenTopPositiveLoses", {"base": -1.0, "overrides": [(11, 10.0), (12, 9.6), (151645, 3.0)]}, [11, 11, 5, 151655]),
        ("seenNegativeIsMultiplied", {"base": -9.0, "overrides": [(7, -0.5), (8, -0.52), (9, -3.0)]}, [7, 9]),
        ("zeroAndNegativeZero", {"base": -2.0, "overrides": [(3, 0.0), (4, -0.0), (5, -1e-30)]}, [3, 4, 5]),
        ("tieTakesFirstIndex", {"base": 0.25, "overrides": [(40, 2.0), (41, 2.1), (42, 2.0)]}, [41]),
        ("roundingAcrossValues", {"base": -50.0, "overrides": noisy}, [i for i, _ in noisy[::2]] + [i for i, _ in noisy[:9]]),
    ]
    penalty = []
    processor = RepetitionPenaltyLogitsProcessor(PENALTY)
    for name, step, seen in single:
        scores = sparse_logits(step)
        processed = processor(torch.tensor([seen]), scores.clone())
        indices = sorted({i for i, _ in step["overrides"]} | set(seen))
        penalty.append({"name": name, "logits": encode_step(step), "seen": seen,
                        "expected": [[i, bits([processed[0, i]])[0]] for i in indices],
                        "argmax": int(torch.argmax(processed, dim=-1))})
    sequences = [
        ("stopsAtImEnd", [151644, 100, 151645, 198], 4096,
         [{"base": -5.0, "overrides": [(300, 4.0), (301, 3.9)]},
          {"base": -5.0, "overrides": [(300, 4.0), (301, 3.9)]},
          {"base": -5.0, "overrides": [(151645, 9.0), (302, 8.6)]},
          {"base": -5.0, "overrides": [(151645, 9.0), (302, 8.5)]},
          {"base": -5.0, "overrides": [(303, 1.0)]}]),
        ("stopsAtEndOfText", [151644, 198], 4096,
         [{"base": -1.0, "overrides": [(500, 1.0)]}, {"base": -1.0, "overrides": [(151643, 2.0), (500, 2.05)]},
          {"base": -1.0, "overrides": [(500, 1.0)]}]),
        ("stopsAtBudget", [1, 2, 3], 3,
         [{"base": 0.0, "overrides": [(10, 1.0)]}, {"base": 0.0, "overrides": [(11, 1.0)]},
          {"base": 0.0, "overrides": [(12, 1.0)]}, {"base": 0.0, "overrides": [(13, 1.0)]}]),
        ("stopTokenOnLastBudgetStep", [1], 2,
         [{"base": 0.0, "overrides": [(10, 1.0)]}, {"base": 0.0, "overrides": [(151645, 1.0)]}]),
    ]
    steps = []
    for name, prompt, budget, logits in sequences:
        processors = LogitsProcessorList([RepetitionPenaltyLogitsProcessor(PENALTY)])
        criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=len(prompt) + budget),
                                         EosTokenCriteria(eos_token_id=STOP_TOKENS)])
        ids = torch.tensor([prompt])
        tokens, finish = [], None
        for step in logits:
            scores = processors(ids, sparse_logits(step))
            token = torch.argmax(scores, dim=-1)
            ids = torch.cat([ids, token[:, None]], dim=-1)
            tokens.append(int(token))
            if bool(criteria(ids, scores).all()):
                finish = "stopToken" if tokens[-1] in STOP_TOKENS else "maxNewTokens"
                break
        assert finish is not None, name
        steps.append({"name": name, "prompt": prompt, "maxNewTokens": budget,
                      "steps": [encode_step(s) for s in logits[:len(tokens)]], "tokens": tokens, "finish": finish})
    return {"penalty": penalty, "sequences": steps}


def layout_blocks(text, w, h):
    out = []
    for m in BOX.finditer(text):
        x0, y0, x1, y1 = (int(v) for v in m.groups()[:4])
        crop = [math.floor(x0 * w / 1000), math.floor(y0 * h / 1000), math.ceil(x1 * w / 1000), math.ceil(y1 * h / 1000)]
        out.append({"label": m.group(5), "orientation": m.group(6), "box": [x0, y0, x1, y1], "crop": crop})
    return out


def layout_fixtures(knowledge):
    outputs = knowledge / "records/outputs"
    size = Image.open(knowledge / "fixtures/F4.png").size
    assert Image.open(knowledge / "fixtures/F2.png").size == size
    texts = [
        ("f4", (outputs / "F4-layout-bf16.txt").read_text().rstrip("\n")),
        ("f2", (outputs / "F2-layout-bf16.txt").read_text().rstrip("\n")),
        ("probeSelfTest", "<box:068 071 848 103><label:title><up>"),
        ("withoutOrientation", "<box:1 2 3 4><label:text>tail<box:5 6 7 8><label:table><left>"),
        ("uppercaseOrientationIsIgnored", "<box:10 20 30 40><label:text><Up>"),
        ("malformedBoxesAreSkipped", "<box:1 2 3><label:x><box:1  2 3 4><label:x><box:1 2 3 4><label:><box:9 9 9 9><label:ok>"),
        ("labelKeepsSpacesAndNewlines", "noise<box:000 000 1000 1000><label:table caption\nx><down>more"),
        ("wideDigitsAndOverflowingBox", "<box:１２ 0 999 1200><label:figure><up>"),
        ("roundsOutward", "<box:001 999 333 667><label:text><up>"),
    ]
    cases = []
    for name, text in texts:
        cases.append({"name": name, "text": text, "width": size[0], "height": size[1],
                      "blocks": layout_blocks(text, size[0], size[1])})
    assert len(cases[0]["blocks"]) == 3
    return {"cases": cases}


def otsl_parts(text):
    pattern_ = "(" + "|".join(map(re.escape, OTSL_TOKENS)) + ")"
    return re.findall(pattern_, text), [part for part in re.split(pattern_, text) if part.strip()]


def count_right(rows, row, column, tokens):
    span = 0
    while row < len(rows) and column < len(rows[row]) and rows[row][column] in tokens:
        span += 1
        column += 1
    return span


def count_down(rows, row, column, tokens):
    span = 0
    while row < len(rows) and column < len(rows[row]) and rows[row][column] in tokens:
        span += 1
        row += 1
    return span


def otsl_table(text):
    tokens, parts = otsl_parts(text)
    rows = [list(row) for is_nl, row in itertools.groupby(tokens, lambda t: t == "<nl>") if not is_nl]
    widths = [len(row) for row in rows]
    if not rows:
        return {"rowWidths": [], "columnCount": 0, "cells": [], "grid": []}
    columns = max(widths)
    for row in rows:
        row.extend(["<ecel>"] * (columns - len(row)))
    cells, r, c = [], 0, 0
    for idx, part in enumerate(parts):
        if part in ("<fcel>", "<ecel>"):
            cell_text, offset = "", 1
            if part != "<ecel>" and idx + 1 < len(parts) and parts[idx + 1] not in OTSL_TOKENS:
                cell_text, offset = parts[idx + 1].strip(), 2
            right = parts[idx + offset] if idx + offset < len(parts) else ""
            below = rows[r + 1][c] if r + 1 < len(rows) and c < len(rows[r + 1]) else ""
            column_span = 1 + (count_right(rows, r, c + 1, {"<lcel>", "<xcel>"}) if right in {"<lcel>", "<xcel>"} else 0)
            row_span = 1 + (count_down(rows, r + 1, c, {"<ucel>", "<xcel>"}) if below in {"<ucel>", "<xcel>"} else 0)
            cells.append({"text": cell_text, "row": r, "column": c, "rowSpan": row_span, "columnSpan": column_span})
        if part in OTSL_CELLS:
            c += 1
        elif part == "<nl>":
            r, c = r + 1, 0
    grid = []
    if cells:
        grid = [["" for _ in range(len(rows[0]))] for _ in range(len(rows))]
        for cell in cells:
            for rr in range(cell["row"], min(cell["row"] + cell["rowSpan"], len(rows))):
                for cc in range(cell["column"], min(cell["column"] + cell["columnSpan"], len(rows[0]))):
                    grid[rr][cc] = cell["text"]
    return {"rowWidths": widths, "columnCount": columns, "cells": cells, "grid": grid}


def card_raises(text):
    tokens, parts = otsl_parts(text)
    rows = [list(row) for is_nl, row in itertools.groupby(tokens, lambda t: t == "<nl>") if not is_nl]
    r = 0
    for idx, part in enumerate(parts):
        if part in ("<fcel>", "<ecel>"):
            offset = 2 if part != "<ecel>" and idx + 1 < len(parts) and parts[idx + 1] not in OTSL_TOKENS else 1
            right = parts[idx + offset] if idx + offset < len(parts) else ""
            if right in {"<lcel>", "<xcel>"} and r >= len(rows):
                return True
        if part == "<nl>":
            r += 1
    return False


def table_fixtures(knowledge):
    scores = json.loads((knowledge / "records/scores.json").read_text())["fixtures"]["F2"]
    texts = [
        ("f2", (knowledge / "records/outputs/F2-table-bf16.txt").read_text().rstrip("\n")),
        ("raggedRows", "<fcel>A<fcel>B<fcel>C<nl><fcel>D<nl><fcel>E<fcel>F<nl>"),
        ("scoreHorizontalSpan", "<fcel>A<fcel>B<nl><fcel>C<lcel><nl>"),
        ("scoreShortRow", "<fcel>A<fcel>B<nl><fcel>C<nl>"),
        ("blockSpan", "<fcel>A<lcel><fcel>B<nl><ucel><xcel><ecel><nl><fcel>C<fcel>D<ucel><nl>"),
        ("whitespaceCells", "<fcel> <fcel>  x　<ecel>y<nl><fcel>\t<lcel><nl>"),
        ("textBeforeFirstToken", "head<fcel>A<fcel><nl>"),
        ("repeatedNewlines", "<nl><fcel>A<nl><nl><fcel>B<nl>"),
        ("spanAfterRepeatedNewlines", "<fcel>A<nl><nl><fcel>B<lcel><nl>"),
        ("noOtslTokens", "| A | B |"),
        ("onlyContinuationCells", "<lcel><ucel><nl><xcel><nl>"),
        ("unterminatedLastRow", "<fcel>1<fcel>2<nl><fcel>3"),
        ("pythonWhitespaceControls", "<fcel>\x1fA\x1c<fcel>\x85\u2028<nl><fcel>\u3000B\u205f<nl>"),
    ]
    cases = []
    for name, text in texts:
        cases.append({"name": name, "otsl": text, "cardRaises": card_raises(text)} | otsl_table(text))
    assert cases[0]["grid"] == scores["grid_card_parse"] and cases[0]["rowWidths"] == scores["rows_raw_widths"]
    assert cases[1]["rowWidths"] == [3, 1, 2]
    return {"cases": cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(str(args.snapshot), trust_remote_code=True, use_fast=True)
    config = AutoConfig.from_pretrained(str(args.snapshot), trust_remote_code=True)
    module = load_module(args.snapshot)
    files = {
        "images.json": image_fixtures(processor),
        "vision.json": vision_fixtures(processor, module, config),
        "prompts.json": prompt_fixtures(processor, module, config, args.knowledge),
        "generation.json": generation_fixtures(),
        "layout.json": layout_fixtures(args.knowledge),
        "tables.json": table_fixtures(args.knowledge),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = {"model": "XingChen-AGI/TeleOCR", "revision": REVISION, "python": sys.version.split()[0],
                  "torch": torch.__version__, "torchvision": torchvision.__version__,
                  "transformers": transformers.__version__, "pillow": PIL.__version__, "numpy": np.__version__,
                  "cpuCapability": torch.backends.cpu.get_cpu_capability(), "files": {}}
    for name, content in files.items():
        data = (json.dumps(content, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        (args.out / name).write_bytes(data)
        provenance["files"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    print(json.dumps(provenance, indent=1))


if __name__ == "__main__":
    main()
