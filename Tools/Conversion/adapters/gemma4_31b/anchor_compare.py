import argparse
import hashlib
import io
from pathlib import Path
import sys

from ...storage import write_json
from . import reference
from .anchor import parse, read
from .prepare import load_contract

SCHEMA = "gemma4-31b-numerics/1"
CONTRACT = Path(__file__).resolve().parents[2] / "recipes/gemma4_31b/model.json"
BLOCK_ROWS = 256
NEAR_TIE = 0.01
BANDS = ((1, 512), (513, 2048), (2049, 4096), (4097, 16896))
MARGIN = 32
FILES = {"tokens": "json", "logits": "npy"}


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


def compare(baseline, candidate, vocabulary=None, window=None):
    config = load_contract(CONTRACT)["config"]
    vocabulary, window = vocabulary or config["vocab_size"], window or config["sliding_window"]
    (first, first_sha256, _), (second, second_sha256, _) = (read(Path(folder) / "reference.json")
                                                            for folder in (baseline, candidate))
    problems = [f"{label}: the record did not pass" for label, record in (("baseline", first), ("candidate", second))
                if record.get("passed") is not True]
    if first_sha256 == second_sha256:
        problems.append("both artifacts are the same run")
    if first.get("model") != second.get("model"):
        problems.append("different model")
    names = [[case.get("id") for case in record.get("cases", [])] for record in (first, second)]
    if names[0] != names[1] or not names[0]:
        problems.append("different cases")
    pairs = [] if problems else [compare_case((baseline, x), (candidate, y), vocabulary)
                                 for x, y in zip(first["cases"], second["cases"])]
    problems += [problem for case, _ in pairs for problem in case["problems"]]
    compared = [(case, rows) for case, (_, rows) in zip(first.get("cases", []), pairs) if rows is not None]
    passed = not problems and len(compared) == len(names[0])
    return {"schema": SCHEMA, "passed": passed, "problems": problems, "vocabulary": vocabulary, "window": window,
            "near_tie_margin": NEAR_TIE, "records_sha256": [first_sha256, second_sha256],
            "cases": [case for case, _ in pairs], "groups": groups(compared, window) if passed else []}


def main():
    parser = argparse.ArgumentParser(description="Compare the saved logits of two Gemma 4 31B diagnostic records.")
    for name in ("baseline", "candidate", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("The comparison must be written to a new file")
    result = compare(args.baseline, args.candidate)
    write_json(args.output, result)
    print(f"passed={result['passed']} problems={len(result['problems'])}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
