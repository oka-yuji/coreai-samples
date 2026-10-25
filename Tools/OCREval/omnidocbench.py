import hashlib
import json
import math
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from .fixtures import sha256

SPEC = json.loads((Path(__file__).resolve().parent / "data" / "omnidocbench.json").read_text())
URL = "https://huggingface.co/datasets/{dataset}/resolve/{revision}/{path}"


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.cell = [], None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self.cell = {"text": "", "rowspan": int(a.get("rowspan") or 1), "colspan": int(a.get("colspan") or 1)}

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            if not self.rows:
                self.rows.append([])
            self.rows[-1].append(self.cell)
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell["text"] += data


def html_cells(text):
    parser = TableParser()
    parser.feed(text)
    grid = {}
    for r, row in enumerate(parser.rows):
        c = 0
        for cell in row:
            while (r, c) in grid:
                c += 1
            for dr in range(cell["rowspan"]):
                for dc in range(cell["colspan"]):
                    grid[(r + dr, c + dc)] = cell["text"].strip()
            c += cell["colspan"]
    if not grid:
        return []
    n_rows, n_cols = max(r for r, _ in grid) + 1, max(c for _, c in grid) + 1
    return [[grid.get((r, c), "") for c in range(n_cols)] for r in range(n_rows)]


def clean(strings):
    return all(s and not any(f in s for f in SPEC["rules"]["forbidden_substrings"]) for s in strings)


def candidate(block):
    rules = SPEC["rules"]
    if block.get("ignore"):
        return None
    if block["category_type"] == rules["text"]["category"]:
        text = block.get("text") or ""
        if clean([text]) and rules["text"]["min_chars"] <= len(text) <= rules["text"]["max_chars"]:
            return "text", {"lines": [text]}
    if block["category_type"] == rules["table"]["category"] and block.get("html"):
        cells = html_cells(block["html"])
        t = rules["table"]
        if cells and t["min_rows"] <= len(cells) <= t["max_rows"] and len(cells[0]) <= t["max_cols"]:
            if any(x for row in cells for x in row) and clean([x for row in cells for x in row if x] or [""]):
                return "table", {"title": "", "cells": cells}
    return None


def crop_box(poly, w, h):
    m = SPEC["rules"]["crop_margin_px"]
    xs, ys = poly[0::2], poly[1::2]
    return [max(0, math.floor(min(xs)) - m), max(0, math.floor(min(ys)) - m),
            min(w, math.ceil(max(xs)) + m), min(h, math.ceil(max(ys)) + m)]


def select(pages):
    rules = SPEC["rules"]
    groups = {}
    for page in pages:
        info = page["page_info"]
        attr = info["page_attribute"]
        lang = rules["languages"].get(attr.get("language"))
        if lang is None or attr.get("data_source") in rules["exclude_data_sources"]:
            continue
        for block in page["layout_dets"]:
            found = candidate(block)
            if found:
                key = hashlib.sha256(f"313|{info['image_path']}|{block['anno_id']}".encode()).hexdigest()
                groups.setdefault((found[0], lang), []).append((key, info, block, found[1]))
    chosen = []
    for kind, lang in sorted(groups):
        seen = set()
        for key, info, block, content in sorted(groups[(kind, lang)], key=lambda x: x[0]):
            if info["image_path"] in seen:
                continue
            seen.add(info["image_path"])
            n = len(seen)
            chosen.append({"id": f"OD-{kind}-{lang}-{n:02d}", "kind": kind, "lang": lang, "info": info, "block": block,
                           "content": content})
            if n == rules[kind]["per_language"]:
                break
    return chosen


def fetch(path, dest, expected=None):
    if dest.exists():
        digest = sha256(dest)
        if expected is None or digest == expected:
            return digest
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    url = URL.format(path=urllib.parse.quote(path), **SPEC["source"])
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=300) as response, open(part, "wb") as f:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            h.update(chunk)
            f.write(chunk)
    digest = h.hexdigest()
    if expected is not None and digest != expected:
        part.unlink()
        raise ValueError(f"{path}: SHA-256 {digest} differs from the pinned {expected}")
    part.rename(dest)
    return digest


def make(out_dir):
    out = Path(out_dir)
    source = out / "source"
    fetch(SPEC["source"]["annotation"], source / SPEC["source"]["annotation"], SPEC["source"]["annotation_sha256"])
    chosen = select(json.loads((source / SPEC["source"]["annotation"]).read_text()))
    pinned = {s["id"]: s for s in SPEC["samples"]}
    picked = [{"id": c["id"], "image_path": c["info"]["image_path"], "anno_id": c["block"]["anno_id"]} for c in chosen]
    if pinned and [(p["id"], p["image_path"], p["anno_id"]) for p in picked] != [
            (s["id"], s["image_path"], s["anno_id"]) for s in SPEC["samples"]]:
        raise ValueError("OmniDocBench selection differs from the pinned samples")
    rows = []
    for c, p in zip(chosen, picked):
        info = c["info"]
        digest = fetch(f"images/{info['image_path']}", out / info["image_path"],
                       pinned[c["id"]]["image_sha256"] if pinned else None)
        p["image_sha256"] = digest
        w, h = info["width"], info["height"]
        gt = {"id": c["id"], "kind": c["kind"], "stratum": f"omnidocbench_{c['kind']}_{c['lang']}",
              "rule": "F1" if c["kind"] == "text" else "F2", "pipelines": [c["kind"]]} | c["content"] | {
                  "crop_px": crop_box(c["block"]["poly"], w, h), "image": info["image_path"], "size_px": [w, h],
                  "source": {"dataset": SPEC["source"]["dataset"], "revision": SPEC["source"]["revision"],
                             "image_path": info["image_path"], "anno_id": c["block"]["anno_id"],
                             "category_type": c["block"]["category_type"], "page_attribute": info["page_attribute"]},
                  "redistribute": False}
        (out / f"{c['id']}.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1) + "\n")
        rows.append({"id": c["id"], "stratum": gt["stratum"], "image": info["image_path"], "image_sha256": digest,
                     "gt_sha256": sha256(out / f"{c['id']}.json")})
    (out / "selection.json").write_text(json.dumps(picked, indent=1) + "\n")
    manifest = {"set": "omnidocbench-subset", "source": SPEC["source"], "rules": SPEC["rules"], "samples": rows}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    return manifest
