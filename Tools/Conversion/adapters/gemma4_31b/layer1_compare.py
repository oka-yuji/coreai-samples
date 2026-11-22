import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

from Tools.LLMEvaluation import checks, items
from Tools.LLMEvaluation.items import FILES
from Tools.LLMEvaluation.layer1 import SET, band, chosen, inconsistencies, label
from ...storage import write_json
from ..verify_llm import compare as judged, probabilities
from .anchor import parse, read
from .compare import CONTRACT
from .native import valid
from .prepare import load_contract

SCHEMA = "gemma4-31b-layer1/1"
LOGITS = "gemma4-31b-layer1-logits/1"
SIDES = ("baseline", "candidate")
DTYPES = {"float16": "<f2", "float32": "<f4"}
SIDE = {"schema": "text", "role": "text", "weights": "text", "forced_sha256": "text"}
ENTRY = {"id": "text", "file": "text", "dtype": "text", "shape": "shape", "sha256": "text", "tokens_sha256": "text"}
LIMITS = ("non_finite_values", "max_probability_error", "band_mean_max_probability_error", "significant_top1_mismatches",
          "near_tie_probability_margin")
REPORT = ("kl_mean", "kl_p99", "kl_max", "max_abs_logit_difference", "top1_agreement", "near_tie_count")


def failed(problem):
    return {"schema": SCHEMA, "passed": False, "problems": [problem]}


def validated(record, table):
    if not isinstance(record, dict):
        return {}, list(table)
    malformed = [key for key, kind in table.items() if key not in record or not valid(kind, record[key])]
    return {key: record[key] for key in table if key not in malformed}, malformed


def forced_input(folder, expected, identity, rule, problems):
    data = (folder / "forced.jsonl").read_bytes()
    summary, summary_sha256, _ = read(folder / "forced.json")
    summary, digest = summary if isinstance(summary, dict) else {}, hashlib.sha256(data).hexdigest()
    counted, malformed = validated(summary.get("forced"), {"count": "count", "sha256": "text"})
    if malformed or summary.get("contract") != identity or summary.get("set") != SET \
            or counted != {"count": len(expected), "sha256": digest}:
        problems.append("forced.json is not the summary of forced.jsonl for the layer 1 items of this contract")
    records, lines = {}, data.split(b"\n")
    for number, line in enumerate(lines[:-1] if lines[-1] == b"" else lines, 1):
        try:
            record = parse(line)
        except Exception as error:
            problems.append(f"forced.jsonl line {number} is not JSON ({type(error).__name__})")
            continue
        name = record.get("id") if isinstance(record, dict) else None
        if not isinstance(name, str):
            problems.append(f"forced.jsonl line {number} has no item id")
        elif name in records:
            problems.append(f"forced.jsonl repeats {name}")
        elif name not in expected:
            problems.append(f"forced.jsonl holds {name}, which is not a layer 1 item")
        else:
            records[name] = record, inconsistencies(record, rule)
    return {"sha256": digest, "summary_sha256": summary_sha256}, records


def side(name, folder, digest, roles, expected, problems):
    found = {"folder": folder, "role": None, "weights": None, "sha256": None, "entries": {}}
    try:
        record, found["sha256"], _ = read(folder / "logits.json")
    except Exception as error:
        problems.append(f"the {name} logits.json is unreadable ({type(error).__name__})")
        return found
    kept, malformed = validated(record, SIDE)
    found.update(role=kept.get("role"), weights=kept.get("weights"))
    problems += [text for text, wrong in (
        (f"the {name} record is malformed ({', '.join(malformed)})", malformed),
        (f"the {name} record is not a {LOGITS} record", "schema" in kept and kept["schema"] != LOGITS),
        (f"the {name} role is not a role of the contract", "role" in kept and kept["role"] not in roles),
        (f"the {name} logits were made from another forced input",
         "forced_sha256" in kept and kept["forced_sha256"] != digest)) if wrong]
    listed = record.get("items") if isinstance(record, dict) else None
    if not isinstance(listed, list):
        problems.append(f"the {name} record has no item list")
        return found
    for index, value in enumerate(listed):
        entry, malformed = validated(value, ENTRY)
        if malformed:
            problems.append(f"item {index} of the {name} record is malformed ({', '.join(malformed)})")
        elif entry["id"] in found["entries"]:
            problems.append(f"the {name} lists {entry['id']} twice")
        elif entry["id"] not in expected:
            problems.append(f"the {name} lists {entry['id']}, which is not a layer 1 item")
        else:
            found["entries"][entry["id"]] = entry
    return found


def listed(name, found, record, vocabulary, problems):
    entry, shape = found["entries"].get(record["id"]), [record["rows"], vocabulary]
    if entry is None:
        problems.append(f"the {name} has no logits for the item")
        return None
    wrong = [text for text, hit in (
        (f"the {name} read other tokens", entry["tokens_sha256"] != record["tokens_sha256"]),
        (f"the {name} dtype {entry['dtype']!r} is not float16 or float32", entry["dtype"] not in DTYPES),
        (f"the {name} logits have the shape {entry['shape']}, not {shape}", entry["shape"] != shape),
        (f"the {name} file name is not a name in its folder",
         entry["file"] in ("", "..") or Path(entry["file"]).name != entry["file"])) if hit]
    problems += wrong
    return None if wrong else entry


def contents(path, size):
    with open(path, "rb") as handle:
        return handle.read(size + 1)


def loaded(name, folder, entry, problems):
    import numpy as np

    dtype, path = np.dtype(DTYPES[entry["dtype"]]), folder / entry["file"]
    size = entry["shape"][0] * entry["shape"][1] * dtype.itemsize
    try:
        status = os.stat(path)
        if not stat.S_ISREG(status.st_mode):
            problems.append(f"the {name} logits are not a regular file")
            return None
        if status.st_size != size:
            problems.append(f"the {name} file holds {status.st_size} B, not {size} B")
            return None
        data = contents(path, size)
    except Exception as error:
        problems.append(f"the {name} logits are unreadable ({type(error).__name__})")
        return None
    if len(data) != size:
        problems.append(f"the {name} file changed while it was read")
    elif hashlib.sha256(data).hexdigest() != entry["sha256"]:
        problems.append(f"the {name} file differs from its SHA-256")
    else:
        return np.frombuffer(data, dtype=dtype).reshape(entry["shape"])
    return None


def per_row(baseline, candidate):
    import numpy as np

    error = np.abs(probabilities(baseline) - probabilities(candidate)).max(axis=-1)
    x, y = baseline.astype(np.float64), candidate.astype(np.float64)
    logit = np.abs(x - y).max(axis=-1)
    x, y = x - x.max(axis=-1, keepdims=True), y - y.max(axis=-1, keepdims=True)
    x, y = x - np.log(np.exp(x).sum(axis=-1, keepdims=True)), y - np.log(np.exp(y).sum(axis=-1, keepdims=True))
    return {"error": error, "kl": (np.exp(x) * (x - y)).sum(axis=-1), "logit": logit}


def blank(name):
    return {"id": name, "band": None, "input_tokens": None, "rows": None, "problems": [],
            "non_finite": dict.fromkeys(SIDES), "max_probability_error": None, "top1_matches": None,
            "near_tie_mismatches": None, "significant_mismatches": None}


def item(name, records, sides, vocabulary, rule):
    import numpy as np

    entry, (record, wrong) = blank(name), records.get(name, (None, []))
    if record is None:
        entry["problems"].append("the item is not in the forced input")
        return entry, None
    if wrong:
        entry["problems"] += [f"the forced record is inconsistent: {text}" for text in wrong]
        return entry, None
    entry.update(band=band(rule["bands"], record["input_tokens"]), input_tokens=record["input_tokens"],
                 rows=record["rows"])
    entries = {key: listed(key, sides[key], record, vocabulary, entry["problems"]) for key in SIDES}
    arrays = {}
    for key in SIDES:
        if entry["problems"]:
            return entry, None
        arrays[key] = loaded(key, sides[key]["folder"], entries[key], entry["problems"])
    if entry["problems"]:
        return entry, None
    entry["non_finite"] = {key: int(array.size - np.count_nonzero(np.isfinite(array))) for key, array in arrays.items()}
    if any(entry["non_finite"].values()):
        entry["problems"].append(f"non-finite values (baseline {entry['non_finite']['baseline']}, "
                                 f"candidate {entry['non_finite']['candidate']})")
        return entry, None
    found = judged(arrays["baseline"], arrays["candidate"], rule)
    entry.update({key: found[key] for key in ("max_probability_error", "top1_matches", "near_tie_mismatches",
                                              "significant_mismatches")})
    return entry, per_row(arrays["baseline"], arrays["candidate"])


def summarized(members):
    import numpy as np

    if not members:
        return {"positions": 0, "mean_max_probability_error": None, "max_probability_error": None, **dict.fromkeys(REPORT)}
    values = {key: np.concatenate([rows[key] for _, rows in members]) for key in ("error", "kl", "logit")}
    positions = len(values["error"])
    return {"positions": positions, "mean_max_probability_error": float(values["error"].mean()),
            "max_probability_error": float(values["error"].max()), "kl_mean": float(values["kl"].mean()),
            "kl_p99": float(np.percentile(values["kl"], 99)), "kl_max": float(values["kl"].max()),
            "max_abs_logit_difference": float(values["logit"].max()),
            "top1_agreement": sum(entry["top1_matches"] for entry, _ in members) / positions,
            "near_tie_count": sum(entry["near_tie_mismatches"] for entry, _ in members)}


def measured(bundle, vocabulary, forced, folders):
    contract, problems = bundle["contract"], []
    rule, expected = contract["layer1"], [entry["id"] for entry in chosen(bundle)]
    identity = {"version": contract["version"], "sha256": bundle["hashes"][FILES["contract"]]}
    hashes, records = forced_input(forced, expected, identity, rule, problems)
    sides = {name: side(name, folders[name], hashes["sha256"], contract["roles"], expected, problems) for name in SIDES}
    if all(sides[name]["sha256"] for name in SIDES) and (
            os.path.samefile(folders["baseline"], folders["candidate"])
            or sides["baseline"]["sha256"] == sides["candidate"]["sha256"]):
        problems.append("the baseline and the candidate are the same folder or the same record")
    results, compared = [], []
    for name in expected:
        try:
            entry, rows = item(name, records, sides, vocabulary, rule)
        except Exception as error:
            entry, rows = blank(name), None
            entry["problems"].append(f"the item could not be compared ({type(error).__name__})")
        results.append(entry)
        compared += [(entry, rows)] if rows is not None else []
    bands = []
    for low, high in rule["bands"]:
        name = label(low, high)
        members = [(entry, rows) for entry, rows in compared if entry["band"] == name]
        value = {"band": name, "items": sum(entry["band"] == name for entry in results), "compared": len(members),
                 **summarized(members)}
        value["state"] = "not_evaluated" if not value["items"] else "passed" if value["compared"] == value["items"] \
            and value["mean_max_probability_error"] <= rule["band_mean_max_probability_error"] else "failed"
        bands.append(value)
    largest = max((entry["max_probability_error"] for entry, _ in compared), default=None)
    counts = {"non_finite_values": sum(value or 0 for entry in results for value in entry["non_finite"].values()),
              "significant_top1_mismatches": sum(entry["significant_mismatches"] or 0 for entry in results)}
    verdict = {name: {"value": value, "passed": value <= rule[name]} for name, value in counts.items()}
    verdict["max_probability_error"] = {"value": largest,
                                        "passed": largest is not None and largest <= rule["max_probability_error"]}
    verdict["band_mean_max_probability_error"] = {"value": [band["mean_max_probability_error"] for band in bands],
                                                  "passed": all(band["state"] != "failed" for band in bands)
                                                  and any(band["state"] == "passed" for band in bands)}
    problems += [f"{entry['id']}: {problem}" for entry in results for problem in entry["problems"]]
    return {"schema": SCHEMA, "passed": not problems and all(check["passed"] for check in verdict.values()),
            "problems": problems, "contract": identity, "limits": {key: rule[key] for key in LIMITS}, "forced": hashes,
            **{name: {key: sides[name][key] for key in ("role", "weights", "sha256")} for name in SIDES},
            "vocabulary": vocabulary, "checks": verdict, "bands": bands, "overall": summarized(compared),
            "items": results}


def recorded(path, result):
    try:
        write_json(path, result)
    except Exception as error:
        result = failed(f"the comparison could not be recorded ({type(error).__name__})")
        write_json(path, result)
    return result


def compare(forced, baseline, candidate, output, contract_path=CONTRACT):
    output = Path(output)
    if os.path.lexists(output):
        raise FileExistsError(f"The layer 1 result must be written to a new folder: {output}")
    bundle = items.load()
    found = checks.problems(bundle)
    if found:
        raise ValueError(f"The frozen evaluation files do not validate: {found[0]}")
    vocabulary = load_contract(contract_path)["config"]["vocab_size"]
    output.mkdir(parents=True)
    result = failed("the comparison did not finish")
    try:
        result = measured(bundle, vocabulary, Path(forced), {"baseline": Path(baseline), "candidate": Path(candidate)})
    except Exception as error:
        result = failed(f"the comparison could not be made ({type(error).__name__})")
    finally:
        result = recorded(output / "layer1.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare the logits of two sides of the layer 1 check of the quality "
                                                 "contract on the same forced input.")
    for name in ("forced", "baseline", "candidate", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.forced, args.baseline, args.candidate, args.output)
    print(json.dumps({"passed": result["passed"], "problems": result["problems"],
                      "checks": {name: check["passed"] for name, check in result.get("checks", {}).items()}}, indent=2),
          flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
