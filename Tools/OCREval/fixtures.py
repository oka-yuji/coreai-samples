import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFilter, ImageFont

DATA = Path(__file__).resolve().parent / "data"
CORPUS = json.loads((DATA / "corpus.json").read_text())
W, H = 1240, 1754
SEED = 178
SET_SEED = 313
BLACK = (0, 0, 0)
GRAY = (0xB0, 0xB0, 0xB0)
FONTS = ("mincho", "gothic_w3", "gothic_w6")
SIZES = (28, 36, 44)
DEGRADATIONS = ("none", "blur", "perspective")
STRATA = ("body", "numbers", "table", "columns", "camera")
KINDS = {"body": "text", "numbers": "mixed", "table": "table", "columns": "two_column", "camera": "camera"}
RULES = {"body": "F1", "numbers": "F3", "table": "F2", "columns": "F4", "camera": "F5"}
PIPELINES = {"body": ["text"], "numbers": ["text"], "table": ["table"], "columns": ["two_stage"], "camera": ["text"]}
F_PIPELINES = {"F1": ["text"], "F2": ["table", "text"], "F3": ["text"], "F4": ["two_stage", "text"], "F5": ["text"]}
NO_START = set(CORPUS["no_line_start"])


def font(name, size):
    spec = CORPUS["fonts"][name]
    return ImageFont.truetype(spec["path"], size, index=spec["index"])


def page():
    img = Image.new("RGB", (W, H), (255, 255, 255))
    return img, ImageDraw.Draw(img)


def outward(box):
    return [math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])]


def fit(fnt, s, max_w):
    if fnt.getlength(s) > max_w:
        raise ValueError(f"text wider than {max_w}px: {s}")


def draw_lines(draw, lines, fnt, x, y0, pitch, max_w):
    boxes = []
    for i, s in enumerate(lines):
        fit(fnt, s, max_w)
        draw.text((x, y0 + i * pitch), s, font=fnt, fill=BLACK, anchor="ls")
        boxes.append(outward(draw.textbbox((x, y0 + i * pitch), s, font=fnt, anchor="ls")))
    return boxes


def wrap(text, fnt, max_w):
    lines, cur = [], ""
    for ch in text:
        if not cur or fnt.getlength(cur + ch) <= max_w:
            cur += ch
        elif ch in NO_START:
            lines.append(cur[:-1])
            cur = cur[-1] + ch
        else:
            lines.append(cur)
            cur = ch
    return lines + [cur] if cur else lines


def ruled_table(d, cells, fnt, head, x0, y0, col_w, row_h, left_cols):
    x1, y1 = x0 + sum(col_w), y0 + row_h * len(cells)
    for r, row in enumerate(cells):
        cx = x0
        for c, text in enumerate(row):
            f = head if r == 0 else fnt
            fit(f, text, col_w[c] - 32)
            base = y0 + r * row_h + row_h // 2
            if r == 0:
                tx, anchor = cx + col_w[c] // 2, "mm"
            elif c in left_cols:
                tx, anchor = cx + 16, "lm"
            else:
                tx, anchor = cx + col_w[c] - 16, "rm"
            d.text((tx, base), text, font=f, fill=BLACK, anchor=anchor)
            cx += col_w[c]
    for r in range(len(cells) + 1):
        d.line([(x0, y0 + r * row_h), (x1, y0 + r * row_h)], fill=BLACK, width=3 if r in (0, 1, len(cells)) else 2)
    cx = x0
    for c in range(len(col_w) + 1):
        d.line([(cx, y0), (cx, y1)], fill=BLACK, width=3 if c in (0, len(col_w)) else 2)
        if c < len(col_w):
            cx += col_w[c]
    return [x0 - 20, y0 - 20, x1 + 20 + 1, y1 + 20 + 1]


def make_f1(gt):
    img, d = page()
    draw_lines(d, gt["lines"], font("mincho", 36), 120, 200, 36 * 1.6, W - 240)
    return img, {}


def make_f2(gt):
    img, d = page()
    d.text((70, 150), gt["title"], font=font("gothic_w6", 44), fill=BLACK, anchor="ls")
    bbox = ruled_table(d, gt["cells"], font("gothic_w3", 32), font("gothic_w6", 32), 70, 220, [360, 230, 230, 280], 90, (0, 3))
    if bbox != gt["table_bbox"]:
        raise ValueError("F2 table_bbox differs from the #178 ground truth")
    return img, {}


def make_f3(gt):
    img, d = page()
    title, body = font("gothic_w6", 38), font("gothic_w3", 34)
    pitch = 34 * 1.9
    fit(title, gt["lines"][0], W - 240)
    d.text((120, 200), gt["lines"][0], font=title, fill=BLACK, anchor="ls")
    draw_lines(d, gt["lines"][1:], body, 120, 200 + 1.5 * pitch, pitch, W - 240)
    return img, {}


def make_f4(gt):
    img, d = page()
    title, body = font("gothic_w6", 44), font("gothic_w3", 30)
    margin, gutter = 90, 60
    col_w = (W - 2 * margin - gutter) // 2
    fit(title, gt["title"], W - 2 * margin)
    d.text((margin, 170), gt["title"], font=title, fill=BLACK, anchor="ls")
    boxes = [outward(d.textbbox((margin, 170), gt["title"], font=title, anchor="ls"))]
    for i, col in enumerate(gt["columns"]):
        boxes += draw_lines(d, col["lines"], body, margin + i * (col_w + gutter), 280, 30 * 1.7, col_w)
    return img, {"line_boxes": boxes}


def perspective_coeffs(dst, src):
    a = []
    for (x, y), (u, v) in zip(dst, src):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    b = np.array(src, dtype=np.float64).reshape(8)
    return np.linalg.solve(np.array(a, dtype=np.float64), b).tolist()


def corners(fr):
    return [(fr[0] * W, fr[1] * H), (W - fr[2] * W, fr[3] * H), (W - fr[4] * W, H - fr[5] * H), (fr[6] * W, H - fr[7] * H)]


def warp(img, fr):
    dst, src = corners(fr), [(0, 0), (W, 0), (W, H), (0, H)]
    out = img.transform((W, H), Image.Transform.PERSPECTIVE, perspective_coeffs(dst, src), Image.Resampling.BICUBIC, fillcolor=GRAY)
    return out, perspective_coeffs(src, dst), dst


def warp_box(box, fwd):
    a, b, c, d, e, f, g, h = fwd
    pts = [(x, y) for x in (box[0], box[2]) for y in (box[1], box[3])]
    xs = [(a * x + b * y + c) / (g * x + h * y + 1) for x, y in pts]
    ys = [(d * x + e * y + f) / (g * x + h * y + 1) for x, y in pts]
    return outward([min(xs), min(ys), max(xs), max(ys)])


def camera(img, fr, rotate):
    img, _, dst = warp(img, fr)
    img = img.rotate(rotate, resample=Image.Resampling.BICUBIC, fillcolor=GRAY)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    gain = np.linspace(0.85, 1.0, W, dtype=np.float32)[None, :, None]
    img = Image.fromarray(np.clip(np.asarray(img, dtype=np.float32) * gain + 0.5, 0, 255).astype(np.uint8))
    return img, {"corner_inset_fraction": [round(v, 4) for v in fr], "dst_corners_px": [[round(x, 1), round(y, 1)] for x, y in dst],
                 "rotate_deg_ccw": rotate, "gaussian_blur_radius": 1.0, "brightness_gradient": [0.85, 1.0],
                 "jpeg_quality": 80, "background": "#B0B0B0"}


def make_f5(gt, f3_img):
    rng = random.Random(SEED)
    fr = [rng.uniform(0.04, 0.06) for _ in range(8)]
    img, transform = camera(f3_img, fr, 2)
    if transform | {"seed": SEED} != gt["transform"]:
        raise ValueError("F5 transform differs from the #178 ground truth")
    return img, {}


def design():
    rng = random.Random(SET_SEED)
    plan = []
    for stratum in STRATA:
        fonts, sizes, degradations = rng.sample(FONTS * 2, 6), rng.sample(SIZES * 2, 6), rng.sample(DEGRADATIONS * 2, 6)
        for k in range(6):
            plan.append({"id": f"S{len(plan) + 1:02d}", "stratum": stratum, "index": k, "font": fonts[k], "size": sizes[k],
                         "degradation": "camera" if stratum == "camera" else degradations[k]})
    return plan


def paragraphs_from(start):
    pool = CORPUS["paragraphs"]
    return "".join(pool[(start + j) % len(pool)] for j in range(len(pool)))


def spec_page(sheet, fnt_name, size):
    img, d = page()
    title, body = font(fnt_name, round(size * 1.15)), font(fnt_name, size)
    pitch = size * 1.9
    fit(title, sheet["title"], W - 240)
    d.text((120, 200), sheet["title"], font=title, fill=BLACK, anchor="ls")
    draw_lines(d, sheet["lines"], body, 120, 200 + 1.5 * pitch, pitch, W - 240)
    return img, {"lines": [sheet["title"]] + sheet["lines"]}


def render(spec, rng):
    stratum, k, size = spec["stratum"], spec["index"], spec["size"]
    fnt = font(spec["font"], size)
    boxes = []
    if stratum == "body":
        img, d = page()
        lines = wrap(paragraphs_from(k * 3), fnt, W - 240)[:rng.randint(10, 14)]
        draw_lines(d, lines, fnt, 120, 200, size * 1.6, W - 240)
        gt = {"lines": lines}
    elif stratum in ("numbers", "camera"):
        img, gt = spec_page(CORPUS["spec_sheets"][k if stratum == "numbers" else (k + 3) % 6], spec["font"], size)
    elif stratum == "table":
        img, d = page()
        table = CORPUS["tables"][k]
        title = font(spec["font"], round(size * 1.2))
        fit(title, table["title"], W - 140)
        base = 140 + round(size * 1.2)
        d.text((70, base), table["title"], font=title, fill=BLACK, anchor="ls")
        col_w = [math.ceil(max(fnt.getlength(row[c]) for row in table["cells"])) + 32 for c in range(len(table["cells"][0]))]
        if sum(col_w) > W - 140:
            raise ValueError(f"{spec['id']} table wider than the page")
        last = len(col_w) - 1
        bbox = ruled_table(d, table["cells"], fnt, fnt, (W - sum(col_w)) // 2, base + 40, col_w, round(size * 2.5), (0, last))
        gt = {"title": table["title"], "cells": table["cells"], "table_bbox": bbox,
              "table_bbox_note": "outer ruled border expanded by 20px white margin; [left, top, right, bottom) px"}
    else:
        img, d = page()
        title_size = size + 8
        title = font(spec["font"], title_size)
        margin, gutter = 90, 60
        col_w = (W - 2 * margin - gutter) // 2
        name = CORPUS["article_titles"][k]
        fit(title, name, W - 2 * margin)
        d.text((margin, 170), name, font=title, fill=BLACK, anchor="ls")
        boxes = [outward(d.textbbox((margin, 170), name, font=title, anchor="ls"))]
        n = rng.randint(8, 10)
        lines = wrap(paragraphs_from(k * 3 + 1), fnt, col_w)
        columns = [lines[:n], lines[n:2 * n]]
        for i, col in enumerate(columns):
            boxes += draw_lines(d, col, fnt, margin + i * (col_w + gutter), 170 + title_size + 66, size * 1.7, col_w)
        gt = {"title": name, "columns": [{"lines": c} for c in columns], "gutter_px": gutter, "line_boxes": boxes}
    conditions = {"font": spec["font"], "font_name": CORPUS["fonts"][spec["font"]]["name"], "size_px": size,
                  "degradation": spec["degradation"]}
    degradation = spec["degradation"]
    if degradation == "blur":
        radius = round(rng.uniform(0.8, 1.4), 2)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        conditions["gaussian_blur_radius"] = radius
    elif degradation == "perspective":
        fr = [round(rng.uniform(0.02, 0.05), 4) for _ in range(8)]
        img, fwd, dst = warp(img, fr)
        conditions |= {"corner_inset_fraction": fr, "dst_corners_px": [[round(x, 1), round(y, 1)] for x, y in dst],
                       "background": "#B0B0B0"}
        if "table_bbox" in gt:
            gt["table_bbox"] = warp_box(gt["table_bbox"], fwd)
            gt["table_bbox_note"] = "warped outer border with 20px margin, bounding rectangle; [left, top, right, bottom) px"
        if "line_boxes" in gt:
            gt["line_boxes"] = [warp_box(b, fwd) for b in gt["line_boxes"]]
    elif degradation == "camera":
        fr = [round(rng.uniform(0.04, 0.06), 4) for _ in range(8)]
        img, transform = camera(img, fr, rng.choice((-2, 2)))
        conditions |= transform
    return img, gt, conditions


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(out, sid, img, gt, jpeg):
    name = f"{sid}.jpg" if jpeg else f"{sid}.png"
    if jpeg:
        img.save(out / name, quality=80)
    else:
        img.save(out / name)
    (out / f"{sid}.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1) + "\n")
    return {"id": sid, "stratum": gt["stratum"], "image": name, "image_sha256": sha256(out / name),
            "gt_sha256": sha256(out / f"{sid}.json")}


def make(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    f = {fid: json.loads((DATA / f"{fid}.json").read_text()) for fid in F_PIPELINES}
    f3_img, _ = make_f3(f["F3"])
    for fid, (img, extra) in (("F1", make_f1(f["F1"])), ("F2", make_f2(f["F2"])), ("F3", (f3_img, {})),
                              ("F4", make_f4(f["F4"])), ("F5", make_f5(f["F5"], f3_img))):
        gt = f[fid] | extra | {"stratum": "f178", "rule": fid, "pipelines": F_PIPELINES[fid]}
        rows.append(write(out, fid, img, gt, fid == "F5"))
    for spec in design():
        sid = spec["id"]
        rng = random.Random(f"{SET_SEED}:{sid}")
        img, content, conditions = render(spec, rng)
        gt = {"id": sid, "kind": KINDS[spec["stratum"]], "stratum": spec["stratum"], "rule": RULES[spec["stratum"]],
              "pipelines": PIPELINES[spec["stratum"]]} | content | {
                  "conditions": conditions, "size_px": [W, H], "seed": f"{SET_SEED}:{sid}",
                  "authored_by": "self (#313), fictional names"}
        rows.append(write(out, sid, img, gt, spec["stratum"] == "camera"))
    manifest = {"set": "teleocr-eval", "seeds": {"f178": SEED, "synthetic": SET_SEED},
                "versions": {"pillow": PIL.__version__, "numpy": np.__version__},
                "fonts": {k: {"name": v["name"], "sha256": sha256(v["path"])} for k, v in CORPUS["fonts"].items()},
                "samples": rows}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    return manifest
