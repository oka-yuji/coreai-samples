import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from Tools.ModelReports.reports import summarize
from .compare import write
from .score import L, MEAN_KEYS


def read(path):
    return json.loads(path.read_text())


def require_keys(actual, expected, name):
    if set(actual) != set(expected):
        raise ValueError(f"Mismatched {name} keys")


def judgment_changes(left, right):
    ranks = {L["good"]: 2, L["conditional"]: 1, L["insufficient"]: 0}
    result = {"same": 0, "improved": 0, "degraded": 0, "missing": 0, "degraded_samples": []}
    for sid in left.keys() | right.keys():
        a, b = ranks.get(left.get(sid, {}).get("judgment")), ranks.get(right.get(sid, {}).get("judgment"))
        key = "missing" if a is None or b is None else "same" if a == b else "improved" if b > a else "degraded"
        result[key] += 1
        if key == "degraded":
            result["degraded_samples"].append(sid)
    result["degraded_samples"].sort()
    return result


def score_summary(score):
    samples = list(score["samples"].values())
    metrics = {}
    for key in [*MEAN_KEYS, "shape_strict_match"]:
        values = [row[key] for row in samples if isinstance(row.get(key), (int, float))]
        if values:
            metrics[key] = {"n": len(values), "mean": statistics.mean(values), "min": min(values), "max": max(values)}
    return {"expected": len(samples), "scored": sum(not row.get("missing") and "scoring_error" not in row for row in samples),
            "good_fraction": sum(row["judgment"] == L["good"] for row in samples) / len(samples) if samples else None,
            "judgments": dict(Counter(row["judgment"] for row in samples)), "metrics": metrics, "strata": score["strata"]}


def expected_runs(plan, inputs):
    quality = {(label, sid, v["id"], "quality", 0) for label, samples in inputs.items() for sid in samples for v in plan["variants"]}
    label, sid = plan["performance"]["set"], plan["performance"]["sample"]
    performance = {(label, sid, v["id"], phase, iteration) for v in plan["variants"] if v["engine"] == "coreai"
                   for phase, count in [("cold", 1), ("warmup", 1), ("measured", 20)] for iteration in range(count)}
    return quality | performance


def build(source, target, spec_path):
    invocation, plan, runs = (read(source / name) for name in ("invocation.json", "plan.json", "runs.json"))
    if invocation["aborted"] is not None or invocation.get("lock_retained") or invocation["only"] or not invocation["performance"]:
        raise ValueError("A complete full invocation is required; partial results remain private")
    inputs = read(source / "inputs/inputs.json")
    require_keys(inputs, plan["sets"], "input set")
    identities = [(row["set"], row["sample"], row["variant"], row["phase"], row["iteration"]) for row in runs]
    if len(set(identities)) != len(identities) or set(identities) != expected_runs(plan, inputs):
        raise ValueError("Missing, duplicate or unexpected runs")
    if any(row["context"]["plan_sha256"] != invocation["plan_sha256"] for row in runs):
        raise ValueError("Mixed plan identities")
    if any(row["status"] not in ("passed", "failed", "warmup") for row in runs):
        raise ValueError("Unknown run status")
    env, spec = read(source / "environment.json"), read(spec_path)
    result = {"schema": "teleocr-precision-evaluation/1", "plan_sha256": invocation["plan_sha256"], "source_commit": plan["source"],
              "cli_sha256": plan["cli_sha256"], "finished_at": invocation["finished_at"],
              "environment": {key: env[key] for key in ("os", "machine", "python", "packages", "model", "xcode", "swift")},
              "conditions": plan["conditions"], "fixture_revision": plan["fixture_revision"], "timing_boundary": plan["timing_boundary"],
              "counts": dict(Counter(row["status"] for row in runs)), "quality": {}, "comparisons": {}, "performance": {}, "variants": {}}
    scores = {}
    for label in inputs:
        scores[label] = {}
        result["quality"][label] = {}
        for variant in plan["variants"]:
            vid = variant["id"]
            scores[label][vid] = read(source / "scores" / label / f"{vid}.json")
            require_keys(scores[label][vid]["samples"], inputs[label], "score sample")
            result["quality"][label][vid] = score_summary(scores[label][vid])
            subset = [row for row in runs if row["phase"] == "quality" and row["set"] == label and row["variant"] == vid]
            result["quality"][label][vid]["execution"] = dict(Counter(row["status"] for row in subset))
    comparisons = read(source / "comparisons.json")
    require_keys(comparisons, inputs, "comparison set")
    for label, pairs in comparisons.items():
        require_keys(pairs, [pair["id"] for pair in plan["comparisons"]], "comparison pair")
        result["comparisons"][label] = {}
        for pair in plan["comparisons"]:
            require_keys(pairs[pair["id"]]["samples"], inputs[label], "comparison sample")
            rows = list(pairs[pair["id"]]["samples"].values())
            exact = sum(row.get("exact_prediction", False) and row.get("both_eos", False) for row in rows)
            present = sum(not row.get("missing", False) for row in rows)
            result["comparisons"][label][pair["id"]] = {
                "axis": pair["axis"], "expected": len(rows), "missing": sum(row.get("missing", False) for row in rows),
                "exact_and_eos": exact, "text_equal": sum(row.get("text_equal", False) for row in rows),
                "present": present, "mismatch_fraction": 1 - exact / present if present else None,
                "judgments": judgment_changes(scores[label][pair["left"]]["samples"], scores[label][pair["right"]]["samples"]),
                "output_agreement": "exact" if rows and exact == len(rows) else "differences_require_interpretation"}
        baseline = scores[label]["coreai-fp16"]["samples"]
        result["quality"][label]["relative_to_coreai_fp16"] = {
            variant["id"]: judgment_changes(baseline, scores[label][variant["id"]]["samples"])
            for variant in plan["variants"] if variant["engine"] == "coreai"}
    for variant in plan["variants"]:
        vid = variant["id"]
        result["variants"][vid] = {key: variant[key] for key in ("engine", "dtype", "asset_hashes", "weight_hashes") if key in variant}
        if variant["engine"] != "coreai":
            continue
        result["performance"][vid] = {}
        for cache, phases in [("cold", {"cold"}), ("prepared", {"warmup", "measured"})]:
            subset = [row for row in runs if row["variant"] == vid and row["phase"] in phases]
            canonical = [{key: row[key] for key in ("run_id", "context", "status", "metrics")} for row in subset]
            path = target / "runs" / f"{vid}-{cache}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in canonical))
            summary = summarize(path, spec) if any(row["status"] == "passed" for row in subset) else {"counts": dict(Counter(row["status"] for row in subset)), "metrics": {}}
            summary["thermal_observations"] = sorted({row[key]["thermal"] for row in subset for key in ("conditions_before", "conditions_after") if key in row})
            summary["power_sources"] = sorted({"AC Power" if "AC Power" in row[key]["power"] else "Battery Power" if "Battery Power" in row[key]["power"] else "unrecorded"
                                              for row in subset for key in ("conditions_before", "conditions_after") if key in row})
            result["performance"][vid][cache] = summary
    result["raw_runs_sha256"] = hashlib.sha256((source / "runs.json").read_bytes()).hexdigest()
    write(target / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Export aggregate TeleOCR evaluation evidence without private images, output text or paths.")
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    spec = Path(__file__).resolve().parents[2] / "docs/model-reports/metrics.json"
    result = build(args.source, args.target, spec)
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
