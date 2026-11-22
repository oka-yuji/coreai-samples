from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import unicodedata

from Tools.LLMEvaluation import execution
from Tools.LLMEvaluation.items import FILES, ROOT, SETS, canonical, digest, lines, read_json

CONTRACT = read_json(ROOT / FILES["contract"])
RULES = CONTRACT["scorers"]


def nfkc(text):
    return unicodedata.normalize("NFKC", text)


def reading(text):
    rule = RULES["reading_exact"]
    text = nfkc(text).strip()
    if text.endswith(rule["trailing"]):
        text = text[:-len(rule["trailing"])].strip()
    if any(text.startswith(opening) and text.endswith(closing) for opening, closing in rule["enclosing"]):
        text = text[1:-1]
    return text.strip()


def overlap(output, answer):
    if not output or not answer:
        return Fraction(0)
    return Fraction(2 * sum((Counter(output) & Counter(answer)).values()), len(output) + len(answer))


def final_integer(text):
    marker = RULES["final_integer"]["marker"]
    text = nfkc(text)
    if marker not in text:
        return None, "no_marker"
    found = re.match(r"\s*(-?\d+)", text[text.rindex(marker) + len(marker):])
    return (int(found.group(1)), "scored") if found else (None, "no_integer")


def nonstandard(name):
    raise ValueError(f"Not a JSON value: {name}")


def decoded(text, start):
    repeated = []

    def pairs(entries):
        if len({key for key, _ in entries}) != len(entries):
            repeated.append(entries)
        return dict(entries)

    value, _ = json.JSONDecoder(object_pairs_hook=pairs, parse_constant=nonstandard).raw_decode(text, start)
    return value, bool(repeated)


def first_object(text):
    for start in (index for index, character in enumerate(text) if character == "{"):
        try:
            value, repeated = decoded(text, start)
        except (ValueError, RecursionError):
            continue
        return (None, "duplicate_key") if repeated else (value, "scored")
    return None, "no_json"


def strings(value):
    return isinstance(value, list) and all(isinstance(entry, str) for entry in value)


def entities(value, answer):
    keys = CONTRACT["entity_keys"]
    predicted = set() if value is None else {(key, name) for key in keys if strings(value.get(key))
                                             for name in value[key]}
    expected = {(key, name) for key in keys for name in answer[key]}
    correct = (value is not None and set(value) == set(keys) and all(strings(value[key]) for key in keys)
               and predicted == expected)
    return correct, {"tp": len(predicted & expected), "fp": len(predicted - expected), "fn": len(expected - predicted)}


def formatted(text, rules):
    found = [line.strip() for line in text.splitlines() if line.strip()]
    texts = [line[len(rules["bullet"]):].strip() for line in found]
    return found, (len(found) == rules["items"] and all(line.startswith(rules["bullet"]) for line in found)
                   and all(len(value) <= rules["max_chars"] and value.endswith(tuple(rules["endings"]))
                           for value in texts)
                   and all(any(word in value for value in texts) for word in rules["required"]))


def score(item, text):
    kind, answer = item["scorer"]["kind"], item["answer"]
    if kind == "unscored":
        return {"kind": kind, "correct": None, "state": "unscored", "parsed": None}
    if kind == "python_tests":
        return {"kind": kind, **execution.run(execution.program(item["scorer"], execution.extract(text))),
                "state": "scored", "parsed": None}
    extra, state = {}, "scored"
    if kind == "choice_digit":
        parsed = next((int(character) for character in nfkc(text) if character in RULES[kind]["digits"]), None)
        correct, state = parsed == answer, "scored" if parsed is not None else "no_digit"
    elif kind == "final_integer":
        parsed, state = final_integer(text)
        correct = parsed == answer
    elif kind == "reading_exact":
        parsed, answers = reading(text), [reading(value) for value in answer]
        correct, extra = parsed in answers, {"f1": max(overlap(parsed, value) for value in answers)}
    elif kind == "format_rules":
        parsed, correct = formatted(text, item["scorer"])
    elif kind == "json_exact":
        parsed, state = first_object(text)
        correct = parsed is not None and canonical(parsed) == canonical(answer)
    elif kind == "entity_sets":
        parsed, state = first_object(text)
        correct, extra = entities(parsed, answer)
    else:
        raise ValueError(f"Unknown scorer kind: {kind}")
    return {"kind": kind, "correct": correct, "state": state, "parsed": parsed, **extra}


def committed(bundle):
    return {entry["id"]: entry["item_sha256"] for name in SETS for entry in bundle["sets"][name]["items"]}


def generated(record):
    return record["output_ids"] + ([record.get("stop_token")] if record.get("stop") == "stop_token" else [])


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def well_formed(record, vocabulary):
    def token(value):
        return integer(value) and 0 <= value < vocabulary

    def probability(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1

    def entry(value):
        return (isinstance(value, list) and len(value) == 4 and token(value[0]) and token(value[2])
                and value[0] != value[2] and probability(value[1]) and probability(value[3])
                and value[1] >= value[3])

    return (isinstance(record.get("id"), str) and isinstance(record.get("input_sha256"), str)
            and integer(record.get("input_tokens")) and record["input_tokens"] >= 1
            and integer(record.get("non_finite")) and record["non_finite"] >= 0
            and isinstance(record.get("output_ids"), list) and all(map(token, record["output_ids"]))
            and record.get("stop") in ("stop_token", "limit")
            and (record["stop"] == "limit" or token(record.get("stop_token")))
            and isinstance(record.get("top2"), list) and all(map(entry, record["top2"]))
            and isinstance(record.get("error", ""), str))


def read_records(data):
    found = []
    for number, line in enumerate(data.split(b"\n"), 1):
        if line.strip():
            try:
                found.append((number, json.loads(line.decode("utf-8"), parse_constant=nonstandard)))
            except (ValueError, RecursionError):
                found.append((number, None))
    return found


def partition(records, vocabulary):
    good, broken = [], []
    for number, record in records:
        if isinstance(record, dict) and well_formed(record, vocabulary):
            good.append(record)
        else:
            name = record.get("id") if isinstance(record, dict) else None
            broken.append(name if isinstance(name, str) else number)
    return good, broken


def differing(items, hashes):
    return {item["id"] for item in items if digest(item) != hashes.get(item["id"])}


def record_checks(items, good, broken, differs):
    ids = {item["id"] for item in items}
    seen = Counter([record["id"] for record in good] + [name for name in broken if isinstance(name, str)])
    return {"missing": sorted(ids - set(seen)), "not_in_set": sorted(set(seen) - ids),
            "duplicate": sorted(key for key, count in seen.items() if count > 1),
            "top2_differs": sorted({record["id"] for record in good
                                    if [entry[0] for entry in record["top2"]] != generated(record)}),
            "item_hash_differs": sorted(differs), "malformed": broken}


def ratio(numerator, denominator):
    return None if denominator == 0 else float(Fraction(numerator, denominator))


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["category"], []).append(row)
    categories = {}
    for name, group in groups.items():
        value = {"kind": group[0]["kind"], "items": len(group), "correct": sum(row["correct"] is True for row in group),
                 "unscored": sum(row["correct"] is None for row in group),
                 "states": dict(Counter(row["state"] for row in group))}
        if value["kind"] == "entity_sets":
            value.update({key: sum(row.get(key, 0) for row in group) for key in ("tp", "fp", "fn")})
            value["micro_f1"] = ratio(2 * value["tp"], 2 * value["tp"] + value["fp"] + value["fn"])
        if value["kind"] == "reading_exact":
            found = [row["f1"] for row in group if "f1" in row]
            value["mean_f1"] = ratio(sum(found, Fraction(0)), len(found))
        categories[name] = value
    return categories


def pinned(path, contract):
    data = Path(path).read_bytes()
    pin = contract["model"]["files"]["tokenizer.json"]
    if len(data) != pin["bytes"] or hashlib.sha256(data).hexdigest() != pin["sha256"]:
        raise ValueError("tokenizer.json differs from the contract")
    import tokenizers

    return tokenizers.Tokenizer.from_str(data.decode("utf-8")), pin["sha256"]


def standard(row):
    return json.loads(json.dumps(row), parse_constant=str)


def frozen_set(bundle, ids):
    sets = {name: {entry["id"] for entry in bundle["sets"][name]["items"]} for name in SETS}
    name = max(SETS, key=lambda key: len(sets[key] & ids))
    if sets[name] != ids:
        raise ValueError(f"The items are not the {name} set: {len(sets[name] - ids)} missing, "
                         f"{len(ids - sets[name])} extra")
    return name


def folded(name):
    return unicodedata.normalize("NFC", name).casefold()


def score_runs(bundle, items_path, runs, output, tokenizer, tokenizer_sha256, commit=None):
    output, runs = Path(output), [Path(run).resolve() for run in runs]
    if output.exists():
        raise FileExistsError(f"Output must be a new folder: {output}")
    if not all(run.name for run in runs):
        raise ValueError("Each run folder needs a name")
    if len({folded(run.name) for run in runs}) != len(runs):
        raise ValueError("Run folder names must differ after NFC normalization and case folding")
    listed = Path(items_path).read_bytes()
    items, items_sha256 = lines(listed), hashlib.sha256(listed).hexdigest()
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("Duplicate item ids in the items file")
    frozen = frozen_set(bundle, {item["id"] for item in items})
    execution.available()
    differs, results, done = differing(items, committed(bundle)), {}, {}
    vocabulary = tokenizer.get_vocab_size(with_added_tokens=True)

    def scored(item, ids):
        if item["scorer"]["kind"] == "python_tests" and item["id"] in differs:
            return {"kind": "python_tests", "correct": None, "state": "item_hash_differs", "parsed": None}
        key = (item["id"], tuple(ids))
        if key not in done:
            done[key] = score(item, tokenizer.decode(ids, skip_special_tokens=True))
        return done[key]

    for run in runs:
        data = (run / "outputs.jsonl").read_bytes()
        good, broken = partition(read_records(data), vocabulary)
        first, damaged = {}, set(broken)
        for record in good:
            first.setdefault(record["id"], record)
        rows = [{"id": item["id"], "category": item["category"],
                 **(scored(item, first[item["id"]]["output_ids"])
                    if item["id"] in first else
                    {"kind": item["scorer"]["kind"], "correct": None,
                     "state": "malformed" if item["id"] in damaged else "missing", "parsed": None})}
                for item in items]
        results[run.name] = rows, {
            "contract": {"version": bundle["contract"]["version"], "sha256": bundle["hashes"][FILES["contract"]]},
            "items": {"file": Path(items_path).name, "sha256": items_sha256, "set": frozen},
            "tokenizer": {"sha256": tokenizer_sha256, "vocabulary": vocabulary},
            "scorer": {"commit": commit, "python": platform.python_version()}, "run": read_json(run / "run.json"),
            "outputs": {"sha256": hashlib.sha256(data).hexdigest()},
            "checks": record_checks(items, good, broken, differs), "categories": summarize(rows),
            "code": {"timeouts": sum(row.get("rerun", False) for row in rows),
                     "rerun_passed": sum(row.get("rerun", False) and row["correct"] for row in rows)}}
    output.mkdir(parents=True)
    try:
        for name in results:
            (output / name).mkdir()
        for name, (rows, summary) in results.items():
            (output / name / "scores.jsonl").write_text(
                "".join(json.dumps(standard({**row, **({"f1": float(row["f1"])} if "f1" in row else {})}),
                                   allow_nan=False) + "\n" for row in rows), encoding="utf-8")
            (output / name / "summary.json").write_text(json.dumps(standard(summary), indent=2, allow_nan=False) + "\n",
                                                        encoding="utf-8")
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return {name: {"checks": {key: len(value) for key, value in summary["checks"].items()},
                   "categories": {key: {field: value[field] for field in ("items", "correct", "unscored")}
                                  for key, value in summary["categories"].items()}}
            for name, (rows, summary) in results.items()}
