import hashlib
import json
from pathlib import Path
import shutil

from Tools.LLMEvaluation.items import FILES, lines, read_json, sha256
from Tools.LLMEvaluation.scoring import folded, frozen_set, generated, partition, read_records, standard

REASONS = ("hash_mismatch", "missing_output", "malformed_record", "not_in_set", "duplicate_id", "input_differs",
           "token_count_differs", "baseline_error", "top2_differs", "unscored_item", "score_inconsistent")
CHECKS = {"missing_output": "missing", "not_in_set": "not_in_set", "duplicate_id": "duplicate",
          "top2_differs": "top2_differs"}
SIDES = ("baseline", "candidate")


def repeated(ids, rule):
    length, times = rule["min_length"], rule["min_consecutive"]
    for period in range(length, len(ids) // times + 1):
        run = 0
        for left, right in zip(ids, ids[period:]):
            run = run + 1 if left == right else 0
            if run == (times - 1) * period:
                return True
    return False


def flags(ids, rules):
    return {"empty_output": not ids, "control_token": not set(ids).isdisjoint(rules["control_token_ids"]),
            "repetition": repeated(ids, rules["repetition"])}


def diverging(before, after):
    left, right = generated(before), generated(after)
    if left == right:
        return None
    return next((index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
                min(len(left), len(right)))


def gap(record, position):
    if position is None or position >= len(record["top2"]):
        return None
    return float(record["top2"][position][1] - record["top2"][position][3])


def compare(document, layer2):
    items, contract = document["items"], document["contract"]
    runs = {side: document[side] for side in SIDES}
    summaries = [runs[side]["summary"] for side in SIDES]
    records = {side: {} for side in SIDES}
    for side in SIDES:
        for record in runs[side]["outputs"]:
            records[side].setdefault(record["id"], record)
    rows = {side: {row["id"]: row for row in runs[side]["scores"]} for side in SIDES}

    def union(key):
        return sorted(set(summaries[0]["checks"][key]) | set(summaries[1]["checks"][key]))

    mismatch = (any(run[part].get("contract") != contract for run in runs.values() for part in ("run", "summary"))
                or any(run["run"].get("set") != document["set"] for run in runs.values())
                or any(summaries[0].get(key) != summaries[1].get(key) for key in ("items", "tokenizer", "scorer")))
    found = {"hash_mismatch": union("item_hash_differs"), **{reason: union(key) for reason, key in CHECKS.items()},
             "malformed_record": list(dict.fromkeys(summaries[0]["checks"]["malformed"]
                                                    + summaries[1]["checks"]["malformed"])),
             "input_differs": sorted(item["id"] for item in items if all(item["id"] in records[side] for side in SIDES)
                                     and records["baseline"][item["id"]]["input_sha256"]
                                     != records["candidate"][item["id"]]["input_sha256"]),
             "token_count_differs": sorted(item["id"] for item in items if "token_count" in item and any(
                 records[side][item["id"]]["input_tokens"] != item["token_count"]
                 for side in SIDES if item["id"] in records[side])),
             "baseline_error": sorted(item["id"] for item in items
                                      if "error" in records["baseline"].get(item["id"], {})),
             "unscored_item": sorted(item["id"] for item in items if item["kind"] != "unscored" and any(
                 rows[side][item["id"]]["correct"] is None
                 and rows[side][item["id"]]["state"] not in ("missing", "malformed")
                 for side in SIDES))}

    rules = layer2["critical_failures"]
    failures, quality = [], {"empty_output": [], "control_token": [], "repetition": []}
    regressions, improvements, changed, inconsistent, categories = [], [], [], [], {}
    for item in items:
        name = item["id"]
        before, after = records["baseline"].get(name), records["candidate"].get(name)
        marks = flags(before["output_ids"], rules) if before else {}
        for kind in (kind for kind, flag in marks.items() if flag):
            quality[kind].append(name)
        if after:
            kinds = ["non_finite_logits"] if after["non_finite"] > 0 else ["runtime_error"] if "error" in after else []
            if before:
                kinds += [kind for kind, flag in flags(after["output_ids"], rules).items() if flag and not marks[kind]]
            failures += [{"id": name, "kind": kind} for kind in kinds]
        category = categories.setdefault(item["category"], {"kind": item["kind"], "items": 0, "baseline_correct": 0,
                                                            "candidate_correct": 0, "regressions": 0,
                                                            "improvements": 0})
        old, new = rows["baseline"][name]["correct"], rows["candidate"][name]["correct"]
        category["items"] += 1
        category["baseline_correct"] += old is True
        category["candidate_correct"] += new is True
        if item["kind"] == "unscored" or old is None or new is None or old == new:
            continue
        (regressions if old else improvements).append(name)
        category["regressions" if old else "improvements"] += 1
        position = diverging(before, after) if before and after else None
        if before and after and before["output_ids"] == after["output_ids"]:
            inconsistent.append(name)
        changed.append({"id": name, "position": position, "baseline_gap": gap(before, position)})
    found["score_inconsistent"] = inconsistent

    incomplete = [{"reason": reason, "ids": found[reason]} for reason in REASONS
                  if found[reason] or (reason == "hash_mismatch" and mismatch)]
    present = {"incomplete": incomplete, "critical_failure": failures, "regression_found": regressions,
               "no_regression_found": True}
    verdict = next(label for label in layer2["precedence"] if present[label])
    weak = sorted(name for name, value in categories.items()
                  if value["kind"] != "unscored" and value["baseline_correct"] < layer2["min_baseline_correct"])
    rerun = document.get("rerun")
    if rerun is not None:
        again = [(rows["candidate"][row["id"]]["correct"], row["correct"]) for row in rerun["scores"]
                 if row["id"] in rows["candidate"]]
        rerun = {"correct_to_incorrect": sum(old is True and new is False for old, new in again),
                 "incorrect_to_correct": sum(old is False and new is True for old, new in again)}
    return {
        "verdict": verdict, "incomplete": incomplete, "critical_failures": failures,
        "regressions": sorted(regressions), "improvements": sorted(improvements),
        "changed": sorted(changed, key=lambda entry: entry["id"]), "categories": categories, "not_evaluable": weak,
        "upper_bounds": {name: 1 - 0.05 ** (1 / categories[name]["items"])
                         for name in layer2["upper_bound"]["categories"]
                         if name in categories and categories[name]["regressions"] == 0},
        "baseline_quality": quality,
        "code": {"baseline": summaries[0].get("code"), "candidate": summaries[1].get("code"),
                 "scorer": summaries[0].get("scorer")},
        "rerun": rerun, "layer1": document.get("layer1"),
        "claim": (layer2["claim"] + ("; not evaluated: " + ", ".join(weak) if weak else "")
                  if verdict == "no_regression_found" else None),
        "contract": contract, "runs": {side: runs[side]["run"] for side in SIDES},
    }


def checked_pair(contract, baseline, candidate, reason):
    pairs = [tuple(pair) for pair in contract["pairs"]]
    rule = contract["substitute_baseline"]
    substitutes = [(rule["with"], after) for before, after in pairs if before == rule["replaces"]]
    if (baseline, candidate) not in (substitutes if reason else pairs):
        shown = {name: ", ".join(f"{before} -> {after}" for before, after in found)
                 for name, found in (("pairs", pairs), ("substitutes", substitutes))}
        raise ValueError(f"--substitute-reason applies only to {shown['substitutes']}, not to {baseline} -> {candidate}"
                         if reason else f"{baseline} -> {candidate} is not a pair of the contract ({shown['pairs']}); "
                         f"{shown['substitutes']} need --substitute-reason")


def item_rows(bundle, path):
    built = lines(Path(path).read_bytes())
    if len({item["id"] for item in built}) != len(built):
        raise ValueError("Duplicate item ids in the items file")
    name = frozen_set(bundle, {item["id"] for item in built})
    entries = {entry["id"]: entry for entry in bundle["sets"][name]["items"]}
    differs = [item["id"] for item in built if (item["category"], item.get("token_count"))
               != (entries[item["id"]]["category"], entries[item["id"]].get("token_count"))]
    if differs:
        raise ValueError(f"{len(differs)} items differ from the {name} set in category or token count, first: "
                         f"{differs[:5]}")
    return name, [{"id": item["id"], "category": item["category"], "kind": item["scorer"]["kind"],
                   **({"token_count": item["token_count"]} if "token_count" in item else {})} for item in built]


def run_files(folder, scores, rows):
    folder = Path(folder).resolve()
    scored = Path(scores) / folder.name
    summary = read_json(scored / "summary.json")
    found = lines((scored / "scores.jsonl").read_bytes())
    if sorted((row["id"], row["category"], row["kind"]) for row in found) != rows:
        raise ValueError(f"The scores in {scored.name} are not one row per item of the items file")
    data, run = (folder / "outputs.jsonl").read_bytes(), read_json(folder / "run.json")
    recorded = summary.get("outputs", {}).get("sha256")
    differs = [text for text, failed in (
        ("the scoring records no SHA-256 of outputs.jsonl", recorded is None),
        ("outputs.jsonl differs", recorded is not None and recorded != hashlib.sha256(data).hexdigest()),
        ("run.json differs", summary.get("run") != standard(run))) if failed]
    if differs:
        raise ValueError(f"{folder.name} is not the run scored in {scored.name}: {'; '.join(differs)}. "
                         "Score the pair again")
    return {"run": run, "summary": summary, "scores": found,
            "outputs": partition(read_records(data), summary["tokenizer"]["vocabulary"])[0]}


def listed(values):
    return ", ".join(f"`{value}`" for value in values) or "none"


def summary(record):
    runs, layer1, rerun, bounds = record["runs"], record["layer1"], record["rerun"], record["upper_bounds"]
    table = ["| Category | Kind | Items | Baseline correct | Candidate correct | Regressions | Improvements "
             "| Upper bound |", "|---|---|---|---|---|---|---|---|"]
    table += [f"| `{name}` | `{value['kind']}` | {value['items']} | {value['baseline_correct']} | "
              f"{value['candidate_correct']} | {value['regressions']} | {value['improvements']} | "
              + (f"{bounds[name]:.2%}" if name in bounds else "-") + " |"
              for name, value in record["categories"].items()]
    changed = ["| Item | First diverging position | Baseline top-2 gap |", "|---|---|---|"]
    changed += [f"| `{entry['id']}` | " + " | ".join("-" if entry[key] is None else str(entry[key])
                                                     for key in ("position", "baseline_gap")) + " |"
                for entry in record["changed"]]
    text = [
        "# Paired comparison", "",
        f"- Verdict: `{record['verdict']}`",
        f"- Baseline `{runs['baseline'].get('role')}`, candidate `{runs['candidate'].get('role')}`",
        f"- Contract `{record['contract']['version']}`, `contract.json` SHA-256 `{record['contract']['sha256']}`",
        f"- Substitute baseline: {record['substitute_reason'] or 'none'}",
        "- Layer 1: " + (f"`{layer1['file']}`, SHA-256 `{layer1['sha256']}`" if layer1 else "not given"),
        "", "## Claim", "", record["claim"] or "None. A claim is written only for `no_regression_found`.",
        "", "## Reasons for incomplete", "",
        *([f"- `{entry['reason']}`: {listed(entry['ids'])}" for entry in record["incomplete"]] or ["None."]),
        "", "## Critical failures", "",
        *([f"- `{entry['id']}`: `{entry['kind']}`" for entry in record["critical_failures"]] or ["None."]),
        "", "## Categories", "", *table, "",
        f"- Regressions: {listed(record['regressions'])}",
        f"- Improvements: {listed(record['improvements'])}",
        f"- Categories that could not be evaluated: {listed(record['not_evaluable'])}",
        "", "## Changed items", "", *(changed if record["changed"] else ["None."]),
        "", "## Baseline quality", "",
        *[f"- `{kind}`: {listed(names)}" for kind, names in record["baseline_quality"].items()],
        "", "## Code scoring and rerun", "",
        *[f"- `{name}`: `{json.dumps(value, sort_keys=True)}`" for name, value in record["code"].items()],
        "- Rerun with another call shape: " + (f"{rerun['correct_to_incorrect']} correct to incorrect, "
                                               f"{rerun['incorrect_to_correct']} incorrect to correct"
                                               if rerun else "not given"),
    ]
    return "\n".join(text) + "\n"


def compare_runs(bundle, items_path, baseline, candidate, scores, output, rerun=None, layer1=None,
                 substitute_reason=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output must be a new folder: {output}")
    if folded(Path(baseline).resolve().name) == folded(Path(candidate).resolve().name):
        raise ValueError("The baseline and candidate run folders need different names, as in one score call")
    reason = substitute_reason if substitute_reason and substitute_reason.strip() else None
    name, rows = item_rows(bundle, items_path)
    keys = sorted((row["id"], row["category"], row["kind"]) for row in rows)
    document = {"set": name, "items": rows,
                "contract": {"version": bundle["contract"]["version"], "sha256": bundle["hashes"][FILES["contract"]]},
                "baseline": run_files(baseline, scores, keys), "candidate": run_files(candidate, scores, keys),
                "rerun": {"scores": lines((Path(rerun) / "scores.jsonl").read_bytes())} if rerun else None,
                "layer1": {"file": Path(layer1).name, "sha256": sha256(layer1)} if layer1 else None}
    checked_pair(bundle["contract"], document["baseline"]["run"].get("role"),
                 document["candidate"]["run"].get("role"), reason)
    record = standard({**compare(document, bundle["contract"]["layer2"]), "substitute_reason": reason})
    output.mkdir(parents=True)
    try:
        (output / "verdict.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (output / "summary.md").write_text(summary(record), encoding="utf-8")
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return {"verdict": record["verdict"],
            "incomplete": {entry["reason"]: len(entry["ids"]) for entry in record["incomplete"]},
            **{key: len(record[key]) for key in ("critical_failures", "regressions", "improvements")},
            "not_evaluable": record["not_evaluable"]}
