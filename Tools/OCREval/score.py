import difflib
import itertools
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

L = json.loads((Path(__file__).resolve().parent / "data" / "labels.json").read_text())
MAP = {c: "-" for c in "\u2212\uff0d\u2010\u2011\u2012\u2013\u2014\u2015"} | {c: "~" for c in "\u301c\uff5e\u2053"}
MACRO_ARG = re.compile(r"\\[A-Za-z]+\{([^{}]*)\}")
DEGREE = ("^{\\circ}", "^\\circ", "\\degree", "\\circ")
MACRO = re.compile(r"\\[A-Za-z]+")
TAG = re.compile(r"<[^>]*>")
DECOR = re.compile(r"[#*_`|>\\${}^]")
SPACE = re.compile(r"\s+")
NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)*(?:/\d+(?:[.,]\d+)*)?(?:~[-+]?\d+(?:[.,]\d+)*)?(?:[A-Za-z%°Ω]{1,4})?")
OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL = "<nl>", "<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>"
OTSL_TOKENS = [OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]
OTSL_CELLS = (OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL)
BOX = re.compile(r"<box:(\d+) (\d+) (\d+) (\d+)><label:([^>]+)>(?:<([a-z]+)>)?")
OK = (L["good"], L["conditional"])
MEAN_KEYS = ("cer", "cer_raw", "numbers_match", "numbers_precision", "read_order_edit", "unmatched_rate", "cell_match",
             "cell_match_unordered", "layout_missing_rate", "cer_covered")
TEXT_KEYS = ("diff_ops_gt_vs_pred", "numbers_gt", "numbers_pred", "numbers_missing", "numbers_extra", "grid_card_parse",
             "pred_cell_texts")


def norm(s):
    s = unicodedata.normalize("NFKC", s)
    s = "".join(MAP.get(c, c) for c in s)
    while True:
        t = MACRO_ARG.sub(r"\1", s)
        if t == s:
            break
        s = t
    for d in DEGREE:
        s = s.replace(d, "°")
    s = MACRO.sub("", s)
    s = TAG.sub("", s)
    s = DECOR.sub("", s)
    return SPACE.sub("", s)


def raw(s):
    return SPACE.sub("", unicodedata.normalize("NFC", s))


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cers(pred, gt):
    return {"cer": lev(norm(pred), norm(gt)) / len(norm(gt)), "cer_raw": lev(raw(pred), raw(gt)) / len(raw(gt))}


def diff_ops(pred, gt, limit=40):
    p, g = norm(pred), norm(gt)
    ops = difflib.SequenceMatcher(None, g, p, autojunk=False).get_opcodes()
    return [[tag, g[i1:i2], p[j1:j2]] for tag, i1, i2, j1, j2 in ops if tag != "equal"][:limit]


def num_tokens(strings):
    return [t for s in strings for t in NUM.findall(norm(s))]


def number_match(gt_tokens, pred_tokens):
    both = Counter(gt_tokens) & Counter(pred_tokens)
    hit = sum(both.values())
    return {"numbers_match": hit / len(gt_tokens) if gt_tokens else None,
            "numbers_precision": hit / len(pred_tokens) if pred_tokens else None,
            "numbers_gt": gt_tokens, "numbers_pred": pred_tokens,
            "numbers_missing": sorted((Counter(gt_tokens) - Counter(pred_tokens)).elements()),
            "numbers_extra": sorted((Counter(pred_tokens) - Counter(gt_tokens)).elements())}


def locate(row, text):
    n = len(row)
    best_pos, best_r = 0, -1.0
    for pos in range(max(1, len(text) - n + 1)):
        r = difflib.SequenceMatcher(None, text[pos:pos + n], row, autojunk=False).ratio()
        if r > best_r:
            best_pos, best_r = pos, r
    return best_pos, best_r


def read_order(pred, gt_rows):
    p = norm(pred)
    found = []
    for i, row in enumerate(gt_rows, 1):
        pos, r = locate(norm(row), p)
        found.append({"id": i, "pos": pos, "ratio": round(r, 4), "matched": r >= 0.6})
    hit = [f for f in found if f["matched"]]
    order = [f["id"] for f in sorted(hit, key=lambda f: (f["pos"], f["id"]))]
    edit = lev(order, sorted(order)) / len(order) if order else None
    return {"read_order_edit": edit, "unmatched_rate": (len(found) - len(hit)) / len(found),
            "read_order_ids": order, "rows": found}


def otsl_rows_raw(text):
    rows = []
    for seg in text.split(OTSL_NL):
        parts = [p for p in re.split("(" + "|".join(map(re.escape, OTSL_TOKENS)) + ")", seg) if p.strip()]
        cells = []
        for idx, part in enumerate(parts):
            if part in OTSL_CELLS:
                nxt = parts[idx + 1] if idx + 1 < len(parts) and parts[idx + 1] not in OTSL_TOKENS else ""
                cells.append((part, nxt.strip() if part == OTSL_FCEL else ""))
        if cells:
            rows.append(cells)
    return rows


def card_grid(text):
    pattern = "(" + "|".join(map(re.escape, OTSL_TOKENS)) + ")"
    tokens = re.findall(pattern, text)
    parts = [p for p in re.split(pattern, text) if p.strip()]
    rows = [list(row) for is_nl, row in itertools.groupby(tokens, lambda t: t == OTSL_NL) if not is_nl]
    if not rows:
        return None
    max_cols = max(len(row) for row in rows)
    for row in rows:
        row.extend([OTSL_ECEL] * (max_cols - len(row)))

    def count(r, c, toks, dr, dc):
        span = 0
        while r < len(rows) and c < len(rows[r]) and rows[r][c] in toks:
            span, r, c = span + 1, r + dr, c + dc
        return span

    cells, r, c = [], 0, 0
    for idx, part in enumerate(parts):
        if part in (OTSL_FCEL, OTSL_ECEL):
            txt, off = "", 1
            if part != OTSL_ECEL and idx + 1 < len(parts) and parts[idx + 1] not in OTSL_TOKENS:
                txt, off = parts[idx + 1].strip(), 2
            nxt = parts[idx + off] if idx + off < len(parts) else ""
            below = rows[r + 1][c] if r + 1 < len(rows) and c < len(rows[r + 1]) else ""
            cs = 1 + (count(r, c + 1, {OTSL_LCEL, OTSL_XCEL}, 0, 1) if nxt in {OTSL_LCEL, OTSL_XCEL} else 0)
            rs = 1 + (count(r + 1, c, {OTSL_UCEL, OTSL_XCEL}, 1, 0) if below in {OTSL_UCEL, OTSL_XCEL} else 0)
            cells.append((txt, r, c, rs, cs))
        if part in OTSL_CELLS:
            c += 1
        elif part == OTSL_NL:
            r, c = r + 1, 0
    if not cells:
        return None
    grid = [["" for _ in range(len(rows[0]))] for _ in range(len(rows))]
    for txt, r0, c0, rs, cs in cells:
        for rr in range(r0, min(r0 + rs, len(rows))):
            for cc in range(c0, min(c0 + cs, len(rows[0]))):
                grid[rr][cc] = txt
    return grid


def table_scores(pred, gt_cells):
    n_rows, n_cols = len(gt_cells), len(gt_cells[0])
    flat_gt = [norm(x) for row in gt_cells for x in row]
    if not any(t in pred for t in OTSL_TOKENS):
        return {"otsl_found": False, "shape_strict_match": False, "cell_match": 0.0, "cell_match_unordered": 0.0}
    rows = otsl_rows_raw(pred)
    widths = [len(r) for r in rows]
    strict = len(rows) == n_rows and all(w == n_cols for w in widths)
    grid = card_grid(pred)
    pos = 0.0
    if strict and grid and len(grid) == n_rows and len(grid[0]) == n_cols:
        pos = sum(norm(grid[r][c]) == norm(gt_cells[r][c]) for r in range(n_rows) for c in range(n_cols)) / len(flat_gt)
    pred_cells = [norm(txt) for row in rows for tok, txt in row if tok in (OTSL_FCEL, OTSL_ECEL)]
    unordered = sum((Counter(flat_gt) & Counter(pred_cells)).values()) / len(flat_gt)
    return {"otsl_found": True, "rows_raw_widths": widths, "shape_gt": [n_rows, n_cols], "shape_strict_match": strict,
            "shape_card_parse": [len(grid), len(grid[0])] if grid else None, "cell_match": pos,
            "cell_match_unordered": unordered, "grid_card_parse": grid,
            "pred_cell_texts": [txt for row in rows for tok, txt in row if tok in (OTSL_FCEL, OTSL_ECEL)]}


def table_metrics(gt, pred):
    m = table_scores(pred, gt["cells"])
    pred_strings = m.get("pred_cell_texts") if m["otsl_found"] else [pred]
    return m | number_match(num_tokens(x for row in gt["cells"] for x in row), num_tokens(pred_strings))


def judge(fid, m):
    if m.get("scoring_error"):
        return L["scoring_failed"]
    c, n, ro = m.get("cer"), m.get("numbers_match"), m.get("read_order_edit")
    good, cond, bad = L["good"], L["conditional"], L["insufficient"]
    if fid == "F1":
        return good if c <= 0.02 else cond if c <= 0.10 else bad
    if fid == "F3":
        return good if c <= 0.03 and n >= 0.90 else cond if c <= 0.10 and n >= 0.70 else bad
    if fid == "F2":
        if not m["otsl_found"]:
            return bad
        return good if m["shape_strict_match"] and m["cell_match"] >= 0.90 else cond if m["cell_match_unordered"] >= 0.70 else bad
    if fid == "F4":
        return good if c <= 0.05 and ro is not None and ro <= 0.10 else cond if c <= 0.15 else bad
    if fid == "F5":
        return good if c <= 0.10 and n >= 0.80 else cond if c <= 0.25 else bad
    if fid == "F6":
        return good if c <= 0.10 and ro == 0 else cond if c <= 0.30 else bad
    raise ValueError(fid)


def gt_text(gt):
    if "cells" in gt:
        return gt["title"] + "".join(x for row in gt["cells"] for x in row)
    if gt["kind"] == "two_column":
        return gt["title"] + "".join(line for col in gt["columns"] for line in col["lines"])
    if gt["kind"] == "vertical":
        return "".join(gt["columns"])
    return "".join(gt["lines"])


def gt_rows(gt):
    if gt["kind"] == "two_column":
        return [gt["title"]] + [line for col in gt["columns"] for line in col["lines"]]
    return gt["columns"]


def text_metrics(fid, gt, pred):
    m = cers(pred, gt_text(gt)) | {"len_gt_norm": len(norm(gt_text(gt))), "len_pred_norm": len(norm(pred)),
                                   "diff_ops_gt_vs_pred": diff_ops(pred, gt_text(gt))}
    if fid in ("F3", "F5"):
        m |= number_match(num_tokens(["".join(gt["lines"])]), num_tokens([pred]))
    if fid in ("F4", "F6"):
        m |= read_order(pred, gt_rows(gt))
    return m


def blocks(text, w, h):
    out = []
    for m in BOX.finditer(text):
        x0, y0, x1, y1 = (int(v) for v in m.groups()[:4])
        crop = [math.floor(x0 * w / 1000), math.floor(y0 * h / 1000), math.ceil(x1 * w / 1000), math.ceil(y1 * h / 1000)]
        out.append({"label": m.group(5), "orientation": m.group(6), "box_norm": [x0, y0, x1, y1], "crop_px": crop})
    return out


def layout_metrics(gt, pred_blocks, pred_text):
    rows, boxes = gt_rows(gt), gt["line_boxes"]
    inside = [[b for b, blk in enumerate(pred_blocks)
               if blk["crop_px"][0] <= (x0 + x1) / 2 < blk["crop_px"][2] and blk["crop_px"][1] <= (y0 + y1) / 2 < blk["crop_px"][3]]
              for x0, y0, x1, y1 in boxes]
    covered = "".join(row for row, hit in zip(rows, inside) if hit)
    used = {b for hit in inside for b in hit}
    m = {"layout_blocks": len(pred_blocks), "layout_missing_rate": sum(not hit for hit in inside) / len(rows),
         "layout_missing_rows": [i for i, hit in enumerate(inside, 1) if not hit],
         "layout_blocks_without_gt": [b for b in range(len(pred_blocks)) if b not in used]}
    m["cer_covered"] = lev(norm(pred_text), norm(covered)) / len(norm(covered)) if norm(covered) else None
    return m


def read(path):
    return path.read_text().rstrip("\n") if path.exists() else None


def scored(fn):
    try:
        return fn()
    except Exception as e:
        return {"scoring_error": f"{type(e).__name__}: {e}"}


def usage_and_overall(j, model_ran=True):
    usage = {}
    body = [j.get(f) for f in ("F1", "F3", "F4")]
    if None in body:
        usage["digital_body_F1_F3_F4"] = L["too_few_scored"]
    elif j["F1"] == L["insufficient"]:
        usage["digital_body_F1_F3_F4"] = L["skip_f1"]
    elif all(x in OK for x in body):
        usage["digital_body_F1_F3_F4"] = L["candidate"]
    else:
        usage["digital_body_F1_F3_F4"] = L["outside_rules"]
    for key, f in (("table_F2", "F2"), ("camera_F5", "F5"), ("vertical_F6", "F6")):
        usage[key] = L["not_scored"] if f not in j else (L["excluded"].format(j[f]) if j[f] == L["insufficient"] else j[f])
    graded = [f for f in ("F1", "F2", "F3", "F4", "F5", "F6") if f in j and not j[f].startswith(L["undetermined"])]
    if not model_ran or "F1" not in graded or "F3" not in graded or len(graded) < 4:
        overall = L["undetermined"]
    elif L["insufficient"] in (j["F1"], j["F3"]):
        overall = L["skip"]
    elif j["F1"] == L["good"] and j["F3"] == L["good"] and j.get("F2") in OK and j.get("F4") in OK:
        overall = L["adopt"]
    else:
        out = [f for f in graded if j[f] == L["insufficient"]]
        overall = L["conditional_adopt"] + (L["excluded_list"].format(", ".join(out)) if out else "")
    return usage, overall


def load_set(set_dir):
    return {p.stem: json.loads(p.read_text()) for p in sorted(Path(set_dir).glob("*.json"))
            if p.name not in ("manifest.json", "selection.json")}


def reference_prediction(outputs, gt, dtype="bf16"):
    sid = gt["id"]
    pred = {"text": read(outputs / f"{sid}-text-{dtype}.txt"), "table": read(outputs / f"{sid}-table-{dtype}.txt")}
    layout = read(outputs / f"{sid}-layout-{dtype}.txt")
    if layout is not None:
        files = sorted(outputs.glob(f"{sid}-block[0-9][0-9]-text-{dtype}.txt"))
        pred["blocks"] = blocks(layout, *gt["size_px"])
        pred["blocks_text"] = "\n".join(p.read_text().strip() for p in files)
        pred["block_runs"] = [p.stem for p in files]
    return pred


def result_prediction(doc):
    found = doc["blocks"]
    text = "\n".join((b.get("text") or "").strip() for b in found)
    table = next((b["otsl"] for b in found if b.get("otsl")), None)
    return {"text": text, "table": table, "blocks_text": text,
            "blocks": [{"label": b.get("label"), "crop_px": b["box_px"]} for b in found],
            "layout_unparsed": doc.get("layout", {}).get("status") == "unparsed"}


def sample_scores(gt, pred):
    rule, pipeline = gt["rule"], gt["pipelines"][0]
    if pipeline == "table":
        if pred.get("table") is None:
            return {"missing": True, "judgment": L["not_scored"]}
        m = scored(lambda: table_metrics(gt, pred["table"]))
    else:
        text = pred.get("blocks_text") if pipeline == "two_stage" else pred.get("text")
        if text is None:
            return {"missing": True, "judgment": L["not_scored"]}
        m = scored(lambda: text_metrics(rule, gt, text))
        if pred.get("blocks") is not None and "line_boxes" in gt and "scoring_error" not in m:
            m |= scored(lambda: layout_metrics(gt, pred["blocks"], text))
    if pred.get("layout_unparsed"):
        m["layout_unparsed"] = True
    m["judgment"] = judge(rule, m)
    return m


def aggregate(samples, gts):
    out = {}
    for stratum in dict.fromkeys(g.get("stratum", "user") for g in gts.values()):
        ms = [samples[s] for s, g in gts.items() if g.get("stratum", "user") == stratum]
        agg = {"n": len(ms), "n_scored": sum(not m.get("missing") and "scoring_error" not in m for m in ms),
               "judgments": dict(Counter(m["judgment"] for m in ms))}
        for k in MEAN_KEYS:
            vals = [m[k] for m in ms if isinstance(m.get(k), (int, float)) and not isinstance(m.get(k), bool)]
            if vals:
                agg[f"mean_{k}"] = sum(vals) / len(vals)
        strict = [m["shape_strict_match"] for m in ms if "shape_strict_match" in m]
        if strict:
            agg["shape_strict_match_rate"] = sum(strict) / len(strict)
        out[stratum] = agg
    return out


def redact(m):
    return {k: v for k, v in m.items() if k not in TEXT_KEYS}


def compat_178(outputs, gts, dtype="bf16"):
    res, judg = {}, {}
    for fid in ("F1", "F3", "F4", "F5"):
        pred = read(outputs / f"{fid}-text-{dtype}.txt")
        if fid not in gts or pred is None:
            continue
        m = scored(lambda: text_metrics(fid, gts[fid], pred))
        m["run"] = f"{fid}-text-{dtype}"
        m["judgment"] = judg[fid] = judge(fid, m)
        res[fid] = m
    tbl = read(outputs / f"F2-table-{dtype}.txt")
    if "F2" in gts and tbl is not None:
        m = scored(lambda: table_metrics(gts["F2"], tbl))
        m["run"] = f"F2-table-{dtype}"
        aux = read(outputs / f"F2-text-{dtype}.txt")
        if aux is not None:
            cells = gts["F2"]["cells"]
            a = cers(aux, gt_text(gts["F2"])) | number_match(num_tokens(x for row in cells for x in row), num_tokens([aux]))
            m["full_page_text_aux"] = {"run": f"F2-text-{dtype}", "diff_ops_gt_vs_pred": diff_ops(aux, gt_text(gts["F2"]))} | a
        m["judgment"] = judg["F2"] = judge("F2", m)
        res["F2"] = m
    files = sorted(outputs.glob(f"F4-block*-text-{dtype}.txt"))
    if files and "F4" in res:
        pred = "\n".join(p.read_text().strip() for p in files)
        m = text_metrics("F4", gts["F4"], pred)
        m["runs"] = [p.stem for p in files]
        m["judgment_if_two_stage"] = judge("F4", m)
        res["F4"][L["two_stage_key"]] = m
    usage, overall = usage_and_overall(judg)
    j2 = dict(judg)
    if "F4" in res and L["two_stage_key"] in res["F4"]:
        j2["F4"] = res["F4"][L["two_stage_key"]]["judgment_if_two_stage"]
    usage2, overall2 = usage_and_overall(j2)
    return {"fixtures": res, "judgments": judg, "usage": usage, "overall": overall,
            "reference_only_if_F4_two_stage": {"judgments": j2, "usage": usage2, "overall": overall2}}


def score_set(set_dir, outputs=None, results=None, dtype="bf16"):
    gts = load_set(set_dir)
    samples = {}
    for sid, gt in gts.items():
        if results is not None:
            path = Path(results) / f"{sid}.json"
            pred = result_prediction(json.loads(path.read_text())) if path.exists() else {}
        else:
            pred = reference_prediction(Path(outputs), gt, dtype)
        m = sample_scores(gt, pred)
        samples[sid] = redact(m) if gt.get("redistribute") is False else m
    out = {"definition": L["definition"], "primary": f"{dtype} + MPS" if results is None else "results JSON",
           "conditional_meaning": L["conditional_meaning"], "samples": samples, "strata": aggregate(samples, gts)}
    if results is None and any(g.get("stratum") == "f178" for g in gts.values()):
        out["f178"] = compat_178(Path(outputs), {s: g for s, g in gts.items() if g.get("stratum") == "f178"}, dtype)
    return out


def selftest():
    assert norm("\u221210\u301c45 \u2103") == "-10~45\u00b0C"
    assert norm("$3.6\\mathrm{kg}$ and -10^{\\circ}C | **x** <td>y</td>") == "3.6kgand-10\u00b0Cxy"
    assert norm("\\textbf{\\textit{z}}\\quad{w}^2") == "zw2"
    c = cers("ab c", "abd")
    assert abs(c["cer"] - 1 / 3) < 1e-9 and abs(c["cer_raw"] - 1 / 3) < 1e-9
    assert cers("\uff21\uff22", "AB")["cer"] == 0 and cers("\uff21\uff22", "AB")["cer_raw"] == 1.0
    sample = "Model FX-210\u3001AC100 V\uff0850/60 Hz\uff09\u221210\u301c45 \u2103 1,800 mm 0.5 W"
    assert NUM.findall(norm(sample)) == ["-210", "100V", "50/60Hz", "-10~45\u00b0C", "1,800mm", "0.5W"]
    m = number_match(["5", "5", "10kg"], ["5", "10kg", "7"])
    assert m["numbers_match"] == 2 / 3 and m["numbers_precision"] == 2 / 3 and m["numbers_missing"] == ["5"], m
    assert number_match([], ["5"])["numbers_match"] is None and number_match([], ["5"])["numbers_precision"] == 0.0
    rows = ["abcde", "fghij", "klmno"]
    r = read_order("abcdefghijklmno", rows)
    assert r["read_order_edit"] == 0 and r["unmatched_rate"] == 0, r
    r = read_order("klmno\nfghij\nabcde", rows)
    assert r["read_order_ids"] == [3, 2, 1] and abs(r["read_order_edit"] - 2 / 3) < 1e-9, r
    r = read_order("abcde", rows)
    assert r["read_order_ids"] == [1] and r["read_order_edit"] == 0 and abs(r["unmatched_rate"] - 2 / 3) < 1e-9, r
    assert locate("ab", "xxabab") == (2, 1.0)
    t = table_scores("<fcel>A<fcel>B<nl><fcel>C<lcel><nl>", [["A", "B"], ["C", "C"]])
    assert t["rows_raw_widths"] == [2, 2] and t["shape_strict_match"] and t["cell_match"] == 1.0 and t["cell_match_unordered"] == 0.75, t
    t = table_scores("<fcel>A<fcel>B<nl><fcel>C<nl>", [["A", "B"], ["C", "D"]])
    assert t["rows_raw_widths"] == [2, 1] and not t["shape_strict_match"] and t["shape_card_parse"] == [2, 2] and t["cell_match"] == 0.0, t
    t = table_scores("| A | B |", [["A", "B"]])
    assert not t["otsl_found"] and judge("F2", t) == L["insufficient"]
    assert judge("F2", {"scoring_error": "x"}) == L["scoring_failed"]
    u, o = usage_and_overall({"F1": L["good"], "F3": L["good"], "F2": L["good"], "F4": L["conditional"]})
    assert o == L["adopt"], o
    u, o = usage_and_overall({"F1": L["good"], "F3": L["conditional"], "F2": L["insufficient"], "F4": L["good"], "F6": L["insufficient"]})
    assert o == L["conditional_adopt"] + L["excluded_list"].format("F2, F6"), o
    assert usage_and_overall({"F1": L["good"], "F3": L["insufficient"], "F2": L["good"], "F4": L["good"]})[1] == L["skip"]
    assert usage_and_overall({"F1": L["good"], "F2": L["good"], "F4": L["good"], "F5": L["good"]})[1] == L["undetermined"]
    assert usage_and_overall({"F1": L["good"], "F3": L["good"], "F2": L["good"], "F4": L["good"]}, model_ran=False)[1] == L["undetermined"]
    saved = dict(MAP)
    MAP.clear()
    try:
        broken = number_match(num_tokens(["\u221210\u301c45 \u2103"]), num_tokens(["-10~45\u00b0C"]))["numbers_match"]
    finally:
        MAP.update(saved)
    assert broken == 0.0, "mutation (no symbol map) must break the number match"
    assert number_match(num_tokens(["\u221210\u301c45 \u2103"]), num_tokens(["-10~45\u00b0C"]))["numbers_match"] == 1.0
    assert blocks("<box:068 071 848 103><label:title><up>", 1240, 1754) == [
        {"label": "title", "orientation": "up", "box_norm": [68, 71, 848, 103], "crop_px": [84, 124, 1052, 181]}]
    gt = {"kind": "two_column", "title": "ab", "columns": [{"lines": ["cd"]}, {"lines": ["ef"]}],
          "line_boxes": [[0, 0, 10, 10], [0, 20, 10, 30], [20, 20, 30, 30]]}
    lay = layout_metrics(gt, [{"crop_px": [0, 0, 12, 32]}, {"crop_px": [40, 0, 50, 10]}], "abcd")
    assert lay["layout_missing_rate"] == 1 / 3 and lay["layout_missing_rows"] == [3] and lay["cer_covered"] == 0, lay
    assert lay["layout_blocks_without_gt"] == [1], lay
    doc = {"blocks": [{"label": "title", "box_px": [0, 0, 12, 12], "text": "ab"}, {"label": "text", "box_px": [0, 15, 12, 32], "text": "cd "}]}
    assert result_prediction(doc)["blocks_text"] == "ab\ncd" and result_prediction(doc)["table"] is None
    print("selftest ok: norm (LaTeX/tags/symbols), CER_norm/CER_raw, number extractor + multiset, read-order search, "
          "OTSL strict shape / unordered multiset / no-OTSL, judge, overall rules, mutation detected, "
          "layout boxes, layout coverage, results JSON reader")
