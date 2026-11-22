import hashlib
import json
import math
import os
from pathlib import Path
import shutil

from Tools.LLMEvaluation.items import FILES, canonical
from Tools.LLMEvaluation.scoring import generated, nonstandard, partition, read_records, record_checks

SET, ROLE = "primary", "reference"


def strict(data):
    def pairs(entries):
        if len({key for key, _ in entries}) != len(entries):
            raise ValueError("Duplicate JSON key")
        return dict(entries)

    return json.loads(data, object_pairs_hook=pairs, parse_constant=nonstandard)


def tokens_sha256(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode("utf-8")).hexdigest()


def label(low, high):
    return f"{low}-{high}"


def band(bands, tokens):
    return next((label(low, high) for low, high in bands if low <= tokens <= high), None)


def chosen(bundle):
    rule = bundle["contract"]["layer1"]["items"]
    counts = {name: count for family in rule["families"].values() for name, count in family.items()}
    groups = {}
    for entry in bundle["sets"][SET]["items"]:
        groups.setdefault(entry["category"], []).append(entry)
    return [entry for name, group in groups.items()
            for entry in sorted(group, key=lambda entry: entry["id"])[:counts.get(name, rule["per_category"])]]


def same(value, expected):
    return canonical(value) == canonical(expected)


def token_ids(value):
    return isinstance(value, list) and bool(value) and all(type(token) is int and token >= 0 for token in value)


def inconsistencies(record, rule):
    if not isinstance(record, dict):
        return ["it is not a JSON object"]
    if not (all(isinstance(record.get(key), str) for key in ("id", "input_sha256", "tokens_sha256"))
            and all(type(record.get(key)) is int and record[key] >= 0 for key in ("start", "rows", "input_tokens"))
            and all(isinstance(record.get(key), list) and all(type(token) is int and token >= 0 for token in record[key])
                    for key in ("tokens", "targets"))):
        return ["it does not have the fields of a forced record"]
    tokens, targets, rows, length = record["tokens"], record["targets"], record["rows"], record["input_tokens"]
    return [text for text, failed in (
        ("its tokens_sha256 is not that of its tokens", record["tokens_sha256"] != tokens_sha256(tokens)),
        ("its input_sha256 is not that of its first input_tokens tokens",
         record["input_sha256"] != tokens_sha256(tokens[:length])),
        ("its start is not input_tokens - 1", record["start"] != length - 1),
        ("its tokens are not input_tokens + rows - 1", len(tokens) != length + rows - 1),
        ("its targets are not one per row", len(targets) != rows),
        ("its tokens after the input are not the first rows - 1 targets", tokens[length:] != targets[:max(rows - 1, 0)]),
        ("its rows are outside 1 to max_per_item", not 1 <= rows <= rule["positions"]["max_per_item"]),
        ("its input length is outside the bands", band(rule["bands"], length) is None)) if failed]


def by_id(data, name):
    rows = data.split(b"\n")
    if rows[-1] == b"":
        rows.pop()
    found = {}
    for number, row in enumerate(rows, 1):
        record = strict(row)
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or record["id"] in found:
            raise ValueError(f"{name} line {number} is not a record with a new id")
        found[record["id"]] = record
    return found


def checked_sources(bundle, summary, run, known, files):
    if not isinstance(run, dict):
        raise ValueError("run.json is not a JSON object")
    identity = {"version": bundle["contract"]["version"], "sha256": bundle["hashes"][FILES["contract"]]}
    failed = [text for text, wrong in (
        ("inputs.json is not the summary of inputs.jsonl under this contract",
         not same(summary, {"contract": identity, **files})),
        ("the run was made under another contract", not same(run.get("contract"), identity)),
        (f"the run is not of the {SET} set", run.get("set") != SET),
        (f"the run is not of the {ROLE} role", run.get("role") != ROLE),
        ("the run did not read these input records", not same(run.get("inputs"), files)),
        ("the run did not finish", type(run.get("finished")) is not str or not same(run.get("items"), files["count"])),
        ("the run has no id", type(run.get("id")) is not str),
        (f"the input records are not one per item of the {SET} set",
         set(known) != {entry["id"] for entry in bundle["sets"][SET]["items"]})) if wrong]
    if failed:
        raise ValueError("; ".join(failed))
    return identity


def forced_record(entry, source, output, rule):
    ids = source.get("input_ids")
    if not token_ids(ids) or source.get("input_sha256") != tokens_sha256(ids):
        return None, ["the input record does not hold token ids with their input_sha256"]
    targets = generated(output)
    wrong = [text for text, failed in (
        ("the output ended with an error", "error" in output),
        ("the output saw non-finite logits", output["non_finite"] != 0),
        ("the output was generated from another input",
         (output["input_sha256"], output["input_tokens"]) != (source["input_sha256"], len(ids))),
        ("the input length differs from the token_count of the set", entry.get("token_count", len(ids)) != len(ids)),
        ("the output has no generated token", not targets),
        ("the input length is outside the bands", band(rule["bands"], len(ids)) is None)) if failed]
    if wrong:
        return None, wrong
    count = min(rule["positions"]["max_per_item"], len(targets))
    forced = ids + targets[:count - 1]
    return {"id": entry["id"], "input_sha256": source["input_sha256"], "tokens": forced,
            "tokens_sha256": tokens_sha256(forced), "start": len(ids) - 1, "rows": count, "input_tokens": len(ids),
            "targets": targets[:count]}, []


def write(bundle, inputs, run, output):
    inputs, run, output = Path(inputs), Path(run), Path(output)
    if os.path.lexists(output):
        raise FileExistsError(f"Output must be a new folder: {output}")
    data, outputs = (inputs / "inputs.jsonl").read_bytes(), (run / "outputs.jsonl").read_bytes()
    summary, stored = strict((inputs / "inputs.json").read_bytes()), strict((run / "run.json").read_bytes())
    known = by_id(data, "inputs.jsonl")
    files = {"count": len(known), "sha256": hashlib.sha256(data).hexdigest()}
    identity = checked_sources(bundle, summary, stored, known, files)
    good, broken = partition(read_records(outputs), math.inf)
    found = {key: value for key, value in record_checks(bundle["sets"][SET]["items"], good, broken, set()).items()
             if value}
    if found:
        raise ValueError("The output records fail the record checks: "
                         + "; ".join(f"{key} {value[:5]}" for key, value in found.items()))
    rule, produced, selection = bundle["contract"]["layer1"], {value["id"]: value for value in good}, chosen(bundle)
    rows, problems, categories = [], [], {}
    for entry in selection:
        made, wrong = forced_record(entry, known[entry["id"]], produced[entry["id"]], rule)
        problems += [f"{entry['id']}: {text}" for text in wrong]
        rows += [made] if made else []
        categories[entry["category"]] = categories.get(entry["category"], 0) + 1
    problems += [f"{row['id']}: the written record is inconsistent: {text}" for row in rows
                 for text in inconsistencies(row, rule)]
    if problems:
        raise ValueError(f"{len(problems)} findings stop the forced input: {'; '.join(problems[:5])}")
    content = "".join(canonical(row) + "\n" for row in rows).encode("utf-8")
    result = {"contract": identity, "set": SET, "inputs": files,
              "run": {"id": stored["id"], "outputs_sha256": hashlib.sha256(outputs).hexdigest()},
              "forced": {"count": len(rows), "sha256": hashlib.sha256(content).hexdigest()},
              "bands": {label(low, high): sum(band(rule["bands"], row["input_tokens"]) == label(low, high) for row in rows)
                        for low, high in rule["bands"]},
              "categories": categories}
    output.mkdir(parents=True)
    try:
        (output / "forced.jsonl").write_bytes(content)
        (output / "forced.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return result
