import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys

from . import anchor, reference
from .anchor import parse, read
from .summary import HEX

SCHEMA = "gemma4-31b-numerics/1"
RECIPE = Path(__file__).resolve().parents[2] / "recipes/gemma4_31b"
BLOCK_ROWS = 256
NEAR_TIE = 0.01
BANDS = ((1, 512), (513, 2048), (2049, 4096), (4097, 16896))
MARGIN = 32
FILES = {"tokens": "json", "logits": "npy"}
REFERENCE_DTYPE = {"weights": "bfloat16", "compute": "bfloat16", "logits_saved": "float32"}
UNKNOWN = "the stage, weights, compute or embedding scale is not a known definition"
FIELDS = {reference.SCHEMA: ("id", "kind", "max_new_tokens"), anchor.SCHEMA: ("id", "kind"), anchor.LAYERS_SCHEMA: ("id",)}


def rounded(value, dtype):
    single = struct.unpack("<f", struct.pack("<f", value))[0]
    if dtype == "float32":
        return single
    if dtype == "float16":
        return struct.unpack("<e", struct.pack("<e", single))[0]
    bits = struct.unpack("<I", struct.pack("<f", single))[0]
    return struct.unpack("<f", struct.pack("<I", (bits + 0x7FFF + (bits >> 16 & 1)) & 0xFFFF0000))[0]


def expectations():
    data, fixtures = (RECIPE / "model.json").read_bytes(), (RECIPE / "fixtures.json").read_bytes()
    contract = parse(data)
    files, config = contract["files"], contract["config"]
    return {"model": {"id": contract["model_id"], "revision": contract["revision"],
                      "contract_sha256": hashlib.sha256(data).hexdigest(),
                      "weights_sha256": {name: entry["sha256"] for name, entry in files.items() if name.endswith(".safetensors")},
                      "verified_before_loading": {name: entry for name, entry in files.items()
                                                  if name in reference.LOADED_FILES or name.endswith(".safetensors")}},
            "fixtures_sha256": hashlib.sha256(fixtures).hexdigest(),
            "cases": [{key: case[key] for key in FIELDS[reference.SCHEMA]} for case in parse(fixtures)["cases"]],
            "device": contract["reference"]["device"], "vocabulary": config["vocab_size"], "window": config["sliding_window"],
            "scales": {dtype: rounded(config["hidden_size"] ** 0.5, dtype) for dtype in ("bfloat16", "float32", "float16")}}


def definition(stage, expected, weights=None, scale=None):
    scales = expected["scales"]
    if stage == "anchor":
        return {"stage": stage, "weights": weights, "compute": "float32", "embed_scale": scales[scale], "attention_rows": anchor.ROWS}
    if stage == "half":
        return {"stage": stage, "weights": "float16", "compute": "float16", "embed_scale": scales["bfloat16"], "attention_rows": None}
    if stage == "reference":
        return {"stage": stage, "weights": "bfloat16", "compute": "bfloat16", "embed_scale": scales["bfloat16"],
                "attention_rows": None}
    raise ValueError(f"Unknown stage: {stage}")


def provenance(record, expected):
    problems = [] if record.get("model") == expected["model"] else ["the model differs from the contract"]
    fields, cases = FIELDS.get(record.get("schema")), record.get("cases")
    if fields is None or not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases) \
            or [{key: case.get(key) for key in fields} for case in cases] \
            != [{key: case[key] for key in fields} for case in expected["cases"]]:
        problems.append("the cases differ from the fixtures")
    if record.get("schema") == reference.SCHEMA and record.get("fixtures_sha256") != expected["fixtures_sha256"]:
        problems.append("the fixtures differ from the fixture file")
    return problems


def identity(record, expected):
    if record.get("schema") == reference.SCHEMA:
        found = definition("reference", expected) if record.get("dtype") == REFERENCE_DTYPE else None
    else:
        known = [definition("anchor", expected, weights, scale) for weights in ("bfloat16", "float16")
                 for scale in expected["scales"]] + [definition("half", expected)]
        found = next((item for item in known if record.get("schema") == anchor.SCHEMA and record.get("stage") == item["stage"]
                      and record.get("compute") == {key: value for key, value in item.items() if key != "stage"}), None)
    known = found is not None and record.get("device") == expected["device"]
    return found, ([] if known else [UNKNOWN]) + provenance(record, expected)


def named_reference(record, other, other_sha256):
    if record.get("schema") == reference.SCHEMA:
        return []
    found = record.get("source")
    wanted = {"reference_sha256": other_sha256} if other.get("schema") == reference.SCHEMA else other.get("source")
    valid = isinstance(found, dict) and list(found) == ["reference_sha256"] and isinstance(found["reference_sha256"], str) \
        and HEX[64].fullmatch(found["reference_sha256"]) is not None
    return [] if valid and found == wanted else ["the forced inputs do not come from the same BF16 reference record"]


def load_case(folder, case, vocabulary):
    import numpy as np

    name = case.get("id")
    if not isinstance(name, str) or not reference.CASE_ID.fullmatch(name):
        return None, None, ["unexpected case ID"]
    data, problems = {}, []
    for kind, suffix in FILES.items():
        entry = (case.get("files") or {}).get(kind) or {}
        try:
            data[kind] = (Path(folder) / "cases" / f"{name}.{kind}.{suffix}").read_bytes()
        except OSError:
            data[kind] = None
        if data[kind] is None or len(data[kind]) != entry.get("bytes") \
                or hashlib.sha256(data[kind]).hexdigest() != entry.get("sha256"):
            problems.append(f"{name}: the {kind} file differs from the size and SHA-256 in its record")
    if problems:
        return None, None, problems
    try:
        tokens = parse(data["tokens"])
        logits = np.load(io.BytesIO(data["logits"]), allow_pickle=False)
        inputs, outputs = len(tokens["input_tokens"]), len(tokens["output_tokens"])
    except (ValueError, TypeError, KeyError):
        return None, None, [f"{name}: unreadable case files"]
    saved = case.get("saved_positions")
    if (case.get("input_tokens"), case.get("output_tokens"), saved, case.get("first_scoring_row"),
            case.get("scoring_positions")) != (inputs, outputs, inputs + outputs, inputs - 1, outputs):
        problems.append(f"{name}: the token counts or positions differ from its record")
    if logits.dtype != np.float32:
        problems.append(f"{name}: logits dtype {logits.dtype}")
    if logits.shape != (saved, vocabulary):
        problems.append(f"{name}: logits shape {list(logits.shape)} instead of {[saved, vocabulary]}")
    return tokens, logits, problems


def runner_up(values, top):
    import numpy as np

    masked = values.copy()
    masked[np.arange(len(values)), top] = -np.inf
    return masked.argmax(-1)


def row_metrics(baseline, candidate):
    import numpy as np

    rows = {key: [] for key in ("error", "kl", "logit", "match", "near", "gap")}
    for start in range(0, len(baseline), BLOCK_ROWS):
        x, y = (np.asarray(array[start:start + BLOCK_ROWS], dtype=np.float64) for array in (baseline, candidate))
        lx, ly = (values - values.max(-1, keepdims=True) for values in (x, y))
        lx, ly = (shifted - np.log(np.exp(shifted).sum(-1, keepdims=True)) for shifted in (lx, ly))
        px, py = np.exp(lx), np.exp(ly)
        top_x, top_y = x.argmax(-1), y.argmax(-1)
        second_x, second_y = runner_up(x, top_x), runner_up(y, top_y)
        index = np.arange(len(x))
        same = ((top_x == top_y) & (second_x == second_y)) | ((top_x == second_y) & (second_x == top_y))
        gap = px[index, top_x] - px[index, top_y]
        for key, value in (("error", np.abs(px - py).max(-1)), ("kl", (px * (lx - ly)).sum(-1)),
                           ("logit", np.abs(x - y).max(-1)), ("match", top_x == top_y),
                           ("near", (top_x != top_y) & same & (gap <= NEAR_TIE)), ("gap", gap)):
            rows[key].append(value)
    return {key: np.concatenate(value) for key, value in rows.items()}


def aggregate(rows):
    import numpy as np

    mismatched = ~rows["match"]
    return {"rows": len(rows["error"]), "max_probability_error_max": float(rows["error"].max()),
            "max_probability_error_mean": float(rows["error"].mean()), "kl_mean": float(rows["kl"].mean()),
            "kl_p99": float(np.percentile(rows["kl"], 99)), "kl_max": float(rows["kl"].max()),
            "max_abs_logit_difference": float(rows["logit"].max()), "top1_matches": int(rows["match"].sum()),
            "near_tie_mismatches": int(rows["near"].sum()),
            "other_mismatches": int((mismatched & ~rows["near"]).sum())}


def select(rows, indices):
    return {key: value[indices] for key, value in rows.items()}


def scoring(case):
    return slice(case["first_scoring_row"], case["first_scoring_row"] + case["scoring_positions"])


def band(tokens):
    return next(f"{low}-{high}" for low, high in BANDS if low <= tokens <= high)


def groups(compared, window):
    import numpy as np

    result, keys = [], {}
    for case, rows in compared:
        keys.setdefault((case["kind"], band(case["input_tokens"])), []).append((case, rows))
    for (kind, label), members in keys.items():
        for scope in ("all", "scoring"):
            parts = [rows if scope == "all" else select(rows, scoring(case)) for case, rows in members]
            joined = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
            result.append({"kind": kind, "band": label, "scope": scope, "cases": [case["id"] for case, _ in members],
                           **aggregate(joined)})
    for case, rows in compared:
        if case["saved_positions"] > window + MARGIN:
            result.append({"kind": case["kind"], "band": band(case["input_tokens"]), "scope": "window",
                           "cases": [case["id"]], "first_row": window - MARGIN, "last_row": window + MARGIN,
                           **aggregate(select(rows, slice(window - MARGIN, window + MARGIN + 1)))})
    return result


def compare_case(first, second, vocabulary):
    import numpy as np

    name = first[1].get("id")
    (left, a, left_problems), (right, b, right_problems) = (load_case(folder, case, vocabulary)
                                                            for folder, case in (first, second))
    result = {"id": name, "problems": left_problems + right_problems, "non_finite": {"baseline": 0, "candidate": 0},
              "all": None, "scoring": None, "mismatches": []}
    if left is None or right is None:
        return result, None
    result["problems"] += [f"{name}: both {kind} files are the same file" for kind, suffix in FILES.items()
                           if os.path.samefile(*(Path(folder) / "cases" / f"{name}.{kind}.{suffix}" for folder, _ in (first, second)))]
    if (left["input_tokens"], left["output_tokens"]) != (right["input_tokens"], right["output_tokens"]):
        result["problems"].append(f"{name}: the forced tokens differ")
    result["non_finite"] = {"baseline": reference.non_finite(a), "candidate": reference.non_finite(b)}
    if any(result["non_finite"].values()):
        result["problems"].append(f"{name}: non-finite logits (baseline {result['non_finite']['baseline']}, "
                                  f"candidate {result['non_finite']['candidate']})")
    if result["problems"]:
        return result, None
    rows, scored = row_metrics(a, b), scoring(first[1])
    result.update(all=aggregate(rows), scoring=aggregate(select(rows, scored)),
                  mismatches=[{"row": int(row), "scoring": bool(scored.start <= row < scored.stop),
                               "near_tie": bool(rows["near"][row]), "probability_gap": float(rows["gap"][row])}
                              for row in np.flatnonzero(~rows["match"])])
    return result, rows


def compare(baseline, candidate, expected=None):
    expected = expected or expectations()
    paths = [Path(folder) / "reference.json" for folder in (baseline, candidate)]
    (first, first_sha256, _), (second, second_sha256, _) = (read(path) for path in paths)
    identities, problems = {}, []
    for label, record in (("baseline", first), ("candidate", second)):
        identities[label], issues = identity(record, expected)
        problems += [f"{label}: {issue}" for issue in issues]
        if record.get("passed") is not True:
            problems.append(f"{label}: the record did not pass")
    for label, record, other, other_sha256 in (("baseline", first, second, second_sha256),
                                               ("candidate", second, first, first_sha256)):
        problems += [f"{label}: {issue}" for issue in named_reference(record, other, other_sha256)]
    if first_sha256 == second_sha256:
        problems.append("both artifacts are the same run")
    if paths[0].samefile(paths[1]):
        problems.append("both records are the same file")
    pairs = [] if problems else [compare_case((baseline, x), (candidate, y), expected["vocabulary"])
                                 for x, y in zip(first["cases"], second["cases"])]
    problems += [problem for case, _ in pairs for problem in case["problems"]]
    compared = [(case, rows) for case, (_, rows) in zip(first["cases"], pairs) if rows is not None] if pairs else []
    passed = not problems and len(compared) == len(expected["cases"])
    return {"schema": SCHEMA, "passed": passed, "problems": problems, "vocabulary": expected["vocabulary"],
            "window": expected["window"], "near_tie_margin": NEAR_TIE, "records_sha256": [first_sha256, second_sha256],
            "identities": identities, "cases": [case for case, _ in pairs],
            "groups": groups(compared, expected["window"]) if passed else []}


def main():
    parser = argparse.ArgumentParser(description="Compare the saved logits of two Gemma 4 31B diagnostic records.")
    for name in ("baseline", "candidate"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
