import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

from .typed_head import FIXED_OPTIONS, LABELS, SLOTS

FORMAT = "typed-decision-v1"
DECISION = "decision.json"
SUMS = "SHA256SUMS"
INPUTS = ("input_ids", "position_ids")
OUTPUT = "hidden_states"
STATES = ("keyCache", "valueCache", "convStates", "recurrentStates")
HEAD_TENSORS = ("proj.weight", "proj.bias")
HASH_SCHEMA = "typed-decision-quantized-hashes/1"
HASH_SETTINGS = ("model_id", "revision", "model_type", "device", "versions", "recipe_sha256", "compute_precision",
                 "compression", "context_length")
SHA256 = re.compile(r"[0-9a-f]{64}")
TEMPLATE = {
    "version": "bare-v1",
    "text": "[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n{options}\n[decision]:",
    "option_separator": "\n",
    "choice_line": "{label}) {option}",
    "labels": LABELS,
    "fixed_options": {kind: list(options) for kind, options in FIXED_OPTIONS.items()},
    "add_special_tokens": False,
    "last_token": "]:",
}
SCHEMA = {
    "format": str, "backbone": dict, "head": dict, "slots": dict, "verbalizers": list, "verbalizer_ids": list,
    "template": dict, "temperatures": dict, "limits": dict, "ties": str, "source": dict, "files": list,
}
BACKBONE = {"asset": str, "function": str, "inputs": list, "output": str, "hidden": dict, "states": list,
            "context_length": int}
HEAD = {"file": str, "weight": str, "bias": str, "weight_shape": list, "bias_shape": list, "dtype": str, "sha256": str}
TEMPERATURES = {"per_kind": dict, "calibration_id": str, "calibration_sha256": str, "source": str}
LIMITS = ("options", "input_tokens")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def listing(root, excluded=(DECISION, SUMS)):
    root = Path(root)
    entries = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"Bundle files must not be links: {relative}")
        if path.is_file() and relative not in excluded:
            entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return sorted(entries, key=lambda entry: entry["path"])


def write_sums(root):
    entries = listing(root, (SUMS,))
    Path(root, SUMS).write_text("".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries), encoding="utf-8")
    return entries


def document(contract, judge, calibration, digests, merged_revision, asset, hidden_size, files):
    rows = SLOTS["choice"][1]
    return {
        "format": FORMAT,
        "backbone": {"asset": asset, "function": "main", "inputs": list(INPUTS), "output": OUTPUT,
                     "hidden": {"dtype": "float16", "size": hidden_size}, "states": list(STATES),
                     "context_length": contract["context_length"]},
        "head": {"file": contract["head"], "weight": HEAD_TENSORS[0], "bias": HEAD_TENSORS[1],
                 "weight_shape": [rows, hidden_size], "bias_shape": [rows], "dtype": "float32", "sha256": digests["head"]},
        "slots": {kind: list(span) for kind, span in SLOTS.items()},
        "verbalizers": judge["slots"]["verbalizers"],
        "verbalizer_ids": judge["verbalizer_ids"],
        "template": TEMPLATE,
        "temperatures": {"per_kind": calibration["per_kind"],
                         "calibration_id": f"{contract['model_id']}@{contract['revision']}/calibration.json",
                         "calibration_sha256": digests["calibration"], "source": "card"},
        "limits": dict(contract["limits"]),
        "ties": "lowest-index",
        "source": {"model": {"id": contract["model_id"], "revision": contract["revision"]},
                   "base_model": dict(contract["base_model"]), "adapter_sha256": digests["adapter"],
                   "adapter_config_sha256": digests["adapter_config"], "head_sha256": digests["head"],
                   "merged_model_id": contract["merged_model_id"], "merged_revision": merged_revision,
                   "upstream_revision": contract["upstream_revision"], "compression": contract["compression"],
                   "compute_precision": contract["precision"]},
        "files": files,
    }


def compare_hashes(first, second):
    left, right = ({entry["name"]: entry for entry in side["tensors"]} for side in (first, second))
    fields = ("kind", "shape", "dtype", "sha256")
    differing = sorted(name for name in left.keys() & right.keys() if any(left[name][f] != right[name][f] for f in fields))
    result = {"items": len(left), "compared": len(left.keys() & right.keys()), "differing": differing,
              "only_first": sorted(left.keys() - right.keys()), "only_second": sorted(right.keys() - left.keys()),
              "order": [entry["name"] for entry in first["tensors"]] == [entry["name"] for entry in second["tensors"]],
              "layout": first["layout"] == second["layout"],
              "settings": sorted(key for key in HASH_SETTINGS if first.get(key) != second.get(key)),
              "schema": first.get("schema") == second.get("schema") == HASH_SCHEMA,
              "unreadable": sorted(name for name in left if left[name]["sha256"] is None)}
    result["matching"] = result["compared"] - len(differing)
    result["hashes_match"] = not (differing or result["only_first"] or result["only_second"] or result["settings"]
                                  or result["unreadable"]) and result["order"] and result["layout"] and result["schema"]
    return result


def typed(value, kind):
    return type(value) is kind


def fields(document, schema, label, errors):
    if not isinstance(document, dict):
        errors.append(f"{label} must be an object")
        return False
    missing = sorted(set(schema) - document.keys())
    wrong = sorted(key for key, kind in schema.items() if key in document and not typed(document[key], kind))
    errors.extend([f"{label}.{key} is missing" for key in missing] + [f"{label}.{key} has the wrong type" for key in wrong])
    return not missing and not wrong


def positive(value):
    return typed(value, float) and math.isfinite(value) and value > 0


def check_slots(slots, errors):
    covered = []
    for kind in SLOTS:
        span = slots.get(kind)
        if not (isinstance(span, list) and len(span) == 2 and all(typed(value, int) for value in span)
                and 0 <= span[0] < span[1]):
            errors.append(f"slots.{kind} must be [start, end)")
            return
        covered.extend(range(*span))
    if set(slots) != set(SLOTS):
        errors.append("slots must name noul, score and choice")
    if sorted(covered) != list(range(SLOTS["choice"][1])) or len(covered) != len(set(covered)):
        errors.append("slots must be disjoint and cover every head row")


def check_files(files, root, errors):
    paths = []
    for entry in files:
        if not (isinstance(entry, dict) and set(entry) == {"path", "bytes", "sha256"} and typed(entry["path"], str)
                and typed(entry["bytes"], int) and entry["bytes"] >= 0 and typed(entry["sha256"], str)
                and SHA256.fullmatch(entry["sha256"])):
            errors.append("files entries need path, bytes and sha256")
            return {}
        path = PurePosixPath(entry["path"])
        if path.is_absolute() or ".." in path.parts or entry["path"] in (DECISION, SUMS) or not entry["path"]:
            errors.append(f"files has an invalid path: {entry['path']}")
        paths.append(entry["path"])
    if paths != sorted(set(paths)):
        errors.append("files must be unique and sorted by path")
    if root is not None:
        actual = {entry["path"]: entry for entry in listing(root)}
        recorded = {entry["path"]: entry for entry in files}
        if actual.keys() != recorded.keys():
            errors.append(f"files differ from the bundle: {sorted(actual.keys() ^ recorded.keys())}")
        errors.extend(f"file differs: {path}" for path in sorted(actual.keys() & recorded.keys())
                      if actual[path] != recorded[path])
    return {entry["path"]: entry for entry in files}


def validate(document, root=None):
    errors = []
    if not fields(document, SCHEMA, "decision", errors):
        return errors
    if document["format"] != FORMAT:
        errors.append(f"format must be {FORMAT}")
    backbone, head, temperatures, limits = (document[key] for key in ("backbone", "head", "temperatures", "limits"))
    size = None
    if fields(backbone, BACKBONE, "backbone", errors):
        hidden = backbone["hidden"]
        if set(hidden) == {"dtype", "size"} and typed(hidden["size"], int) and hidden["size"] > 0 \
                and hidden["dtype"] in ("float16", "float32"):
            size = hidden["size"]
        else:
            errors.append("backbone.hidden needs a float dtype and a positive size")
        if backbone["function"] != "main" or backbone["inputs"] != list(INPUTS) or backbone["output"] != OUTPUT \
                or backbone["states"] != list(STATES) or backbone["context_length"] <= 0:
            errors.append("backbone must describe main with the hidden-state contract and a positive context length")
    if fields(head, HEAD, "head", errors):
        rows = SLOTS["choice"][1]
        if size is None or head["weight_shape"] != [rows, size] or head["bias_shape"] != [rows] \
                or head["dtype"] != "float32" or (head["weight"], head["bias"]) != HEAD_TENSORS:
            errors.append("head must hold proj.weight [24, hidden] and proj.bias [24] in float32")
        if not SHA256.fullmatch(head["sha256"]):
            errors.append("head.sha256 must be a SHA-256 digest")
    check_slots(document["slots"], errors)
    if len(document["verbalizers"]) != SLOTS["choice"][1] or not all(typed(v, str) for v in document["verbalizers"]):
        errors.append("verbalizers must name every head row")
    if len(document["verbalizer_ids"]) != SLOTS["choice"][1] or not all(typed(v, int) for v in document["verbalizer_ids"]):
        errors.append("verbalizer_ids must give a token for every head row")
    if document["template"] != TEMPLATE:
        errors.append("template differs from bare-v1")
    if fields(temperatures, TEMPERATURES, "temperatures", errors):
        values = temperatures["per_kind"]
        if set(values) != set(SLOTS) or not all(positive(value) for value in values.values()):
            errors.append("temperatures.per_kind must give every kind a positive finite value")
    if not isinstance(limits, dict) or set(limits) != set(LIMITS) \
            or not all(typed(limits[key], int) and limits[key] > 0 for key in LIMITS if key in limits):
        errors.append("limits must give positive integer options and input_tokens")
    elif limits["options"] > SLOTS["choice"][1] - SLOTS["choice"][0]:
        errors.append("limits.options exceeds the choice slots")
    if document["ties"] != "lowest-index":
        errors.append("ties must be lowest-index")
    files = check_files(document["files"], root, errors)
    if files and isinstance(head, dict) and isinstance(backbone, dict):
        if head.get("file") not in files or files[head["file"]]["sha256"] != head.get("sha256"):
            errors.append("head.file must be listed with the head digest")
        if not any(path.startswith(f"{backbone.get('asset')}/") for path in files):
            errors.append("backbone.asset must be listed")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate a typed-decision bundle description.")
    parser.add_argument("decision", type=Path)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    document = json.loads(args.decision.read_text(encoding="utf-8"))
    errors = validate(document, args.decision.parent if args.check_files else None)
    print(json.dumps({"passed": not errors, "errors": errors, "checked_files": args.check_files}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
