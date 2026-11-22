import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ...storage import digest, file_hash, write_json
from ..pinned import sha256
from .anchor import read
from .compare import BLOCK_ROWS, CONTRACT
from .export import BUNDLE, FUNCTIONS, SCHEMA as EXPORT_SCHEMA
from .layers import CONDITIONS, CONTEXT_SCHEMA, check_layers, kept_rows, tokens
from .prepare import load_contract
from .reference import LOADED_FILES

SCHEMA = "gemma4-31b-native-comparison/1"
MAX_ABS_LOGIT = 2.0
NEAR_TIE = 0.01
KNOWN_ENDS, KNOWN_SHORTEST = range(97, 113), 8
TIMEOUT_S = 1800
LARGEST = 2**63 - 1
VERIFIER = {"record": "text", "token_count": "count", "start": "count", "chunk": "count", "tokens_sha256": "text",
            "shape": "shape", "dtype": "text", "byte_order": "text", "main_calls": "count", "main_hash": "text",
            "prefix_function": "text", "prefix_calls": "count", "prefix_chunk": "count", "cached": "flag",
            "non_finite_count": "count", "load_s": "seconds", "prefix_s": "seconds", "scoring_s": "seconds",
            "specialize_s": "seconds"}
OPTIONAL = {"prefix_function"}
PUBLIC = ("cached", "prefix_function", "prefix_calls", "main_calls", "non_finite_count", "load_s", "prefix_s", "scoring_s",
          "specialize_s")


def read_once(path):
    data = Path(path).read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def pinned_model(contract, contract_path):
    return {"id": contract["model_id"], "revision": contract["revision"], "contract_sha256": sha256(contract_path),
            "verified_before_loading": {name: entry for name, entry in contract["files"].items()
                                        if name in LOADED_FILES or name.endswith(".safetensors")}}


def valid(kind, value):
    if kind == "count":
        return type(value) is int and 0 <= value <= LARGEST
    if kind == "seconds":
        return valid("count", value) or type(value) is float and math.isfinite(value) and value >= 0
    if kind == "shape":
        return type(value) is list and all(valid("count", size) for size in value)
    return type(value) is {"flag": bool, "text": str}[kind]


def validated(record):
    kept, malformed = {}, []
    for key, kind in VERIFIER.items():
        if key in record and valid(kind, record[key]):
            kept[key] = record[key]
        elif key in record or key not in OPTIONAL:
            malformed.append(key)
    return kept, malformed


def known_rows(start, length, chunk):
    rows = []
    for first in range(start, length, chunk):
        end = min(first + chunk, length)
        if end - first >= KNOWN_SHORTEST and end in KNOWN_ENDS:
            rows.extend(range(first, end))
    return rows


def prefill_chunk():
    return 4096 if os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") >= 64 << 30 else 2048


def call_lengths(position, count, limit):
    lengths, end = [], position
    while end < position + count:
        length = min(limit, position + count - end)
        while length >= KNOWN_SHORTEST and end + length in KNOWN_ENDS:
            length -= 1
        lengths.append(length)
        end += length
    return lengths


def prefix_record(start, chunk, prefill):
    if start == 0:
        return {"prefix_calls": 0, "prefix_chunk": chunk}
    lengths = call_lengths(0, start, chunk)
    unsplit = lengths == [min(chunk, start - first) for first in range(0, start, chunk)]
    return {"prefix_function": "prefill" if prefill and unsplit else "main", "prefix_calls": len(lengths),
            "prefix_chunk": chunk}


def measure(swift, reference, indices, known):
    import numpy as np

    counts = dict.fromkeys(("non_finite", "reference_non_finite", "constant_rows", "near_tie", "outside"), 0)
    largest = known_largest = 0.0
    for start in range(0, len(swift), BLOCK_ROWS):
        x = np.asarray(swift[start:start + BLOCK_ROWS], dtype=np.float64)
        y = np.asarray(reference[indices[start:start + BLOCK_ROWS]], dtype=np.float64)
        excluded = known[start:start + BLOCK_ROWS]
        counts["non_finite"] += int(x.size - np.isfinite(x).sum())
        counts["reference_non_finite"] += int(y.size - np.isfinite(y).sum())
        counts["constant_rows"] += int((x.max(axis=-1) == x.min(axis=-1)).sum())
        difference = np.abs(x - y).max(axis=-1)
        largest = float(np.maximum(largest, difference[~excluded].max(initial=0.0)))
        known_largest = float(np.maximum(known_largest, difference[excluded].max(initial=0.0)))
        top = np.partition(y, -2, axis=-1)[:, -2:]
        near = top[:, 1] - top[:, 0] <= NEAR_TIE
        changed = (x.argmax(axis=-1) != y.argmax(axis=-1)) & ~excluded
        counts["near_tie"] += int((changed & near).sum())
        counts["outside"] += int((changed & ~near).sum())
    finite = not counts["non_finite"] and not counts["reference_non_finite"]
    return {"rows": len(swift), "non_finite": counts["non_finite"], "reference_non_finite": counts["reference_non_finite"],
            "constant_rows": counts["constant_rows"], "max_abs_logit": largest if finite else None,
            "top1_mismatches": {"near_tie": counts["near_tie"], "outside": counts["outside"]},
            "known": {"rows": int(known.sum()), "max_abs_logit": known_largest if finite and known.any() else None},
            "checks": {"finite": finite, "no_constant_rows": not counts["constant_rows"],
                       "max_abs_logit": finite and largest < MAX_ABS_LOGIT, "top1": not counts["outside"]}}


def layer_problems(record, contract, label):
    try:
        if record.get("layer_types") != check_layers(contract["config"], record.get("layers")):
            return [f"the {label} recorded layer types that differ from the contract"]
    except ValueError as error:
        return [f"the {label}: {error}"]
    return []


def load_reference(folder, contract, contract_path):
    import numpy as np

    record, hashed, _ = read(folder / "record.json")
    problems = [] if (record.get("schema"), record.get("implementation")) == (CONTEXT_SCHEMA, "original") else [
        "the reference is not a context run of the original implementation"]
    problems += layer_problems(record, contract, "reference")
    if record.get("model") != pinned_model(contract, contract_path):
        problems.append("the reference recorded another model or contract")
    arrays = {}
    for name in ("short", "context"):
        entry, ids, rows = (record.get("inputs") or {}).get(name) or {}, tokens(name, contract["config"]), kept_rows(name)
        if (entry.get("positions"), entry.get("tokens_sha256"), entry.get("rows")) != (len(ids), digest(ids), rows):
            problems.append(f"the reference did not keep the rows of the {name} input of the conditions")
        content, content_hash = read_once(folder / f"{name}.npy")
        logits = entry.get("logits") or {}
        if (len(content), content_hash) != (logits.get("bytes"), logits.get("sha256")):
            problems.append(f"the {name} logits differ from the size and SHA-256 in the reference record")
            continue
        array = np.load(io.BytesIO(content))
        if array.dtype != np.float32 or array.shape != (len(rows), contract["config"]["vocab_size"]):
            problems.append(f"the {name} logits are {array.dtype} {list(array.shape)}")
            continue
        arrays[name] = array
    return record, hashed, arrays, problems


def load_export(folder, contract, contract_path):
    record, hashed, _ = read(folder / "export.json")
    problems = [] if record.get("schema") == EXPORT_SCHEMA and record.get("passed") is True else [
        "the export is not a passed FP16 export"]
    problems += layer_problems(record, contract, "export")
    if record.get("model") != pinned_model(contract, contract_path):
        problems.append("the export recorded another model or contract")
    bundle, recorded = folder / BUNDLE, record.get("bundle") or {}
    files = recorded.get("files") or {}
    present = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_symlink() or not path.is_dir()}
    if recorded.get("folder") != BUNDLE or present != set(files):
        problems.append("the bundle folder holds other files than its export record")
    else:
        for relative, entry in sorted(files.items()):
            if (bundle / relative).is_symlink() or file_hash(bundle / relative) != entry:
                problems.append(f"a bundle file differs from the export record: {relative}")
    main = bundle / f"{recorded.get('name')}.aimodel" / "main.hash"
    if not main.is_file() or main.read_bytes().hex() != recorded.get("main_hash"):
        problems.append("main.hash of the bundle differs from the export record")
    if record.get("functions") not in [list(functions) for functions in FUNCTIONS.values()]:
        problems.append("the export recorded other functions than main or main and prefill")
    facts = {"bundle": bundle, "main_hash": recorded.get("main_hash"), "prefill": "prefill" in (record.get("functions") or [])}
    return record, hashed, facts, problems


def run_verifier(command, timeout, log):
    with log.open("xb") as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, start_new_session=True)
        try:
            stdout, _ = process.communicate(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise
    return process.returncode, stdout


def printed(stdout):
    lines = [line for line in stdout.decode(errors="replace").splitlines() if line.strip()]
    try:
        value = json.loads(lines[0]) if len(lines) == 1 else None
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def condition(spec, config, arrays, facts, verifier, output):
    name, source, length, start, chunk = spec
    fixed = {"name": name, "input": source, "tokens": length, "start": start, "chunk": chunk, "rows": length - start}
    entry = {**fixed, "problems": [], "checks": {}}
    try:
        measured(entry, config, arrays, facts, verifier, output)
        json.dumps(entry, allow_nan=False)
    except Exception as error:
        entry = {**fixed, "problems": [f"the condition could not be compared ({type(error).__name__})"], "checks": {}}
    entry["passed"] = not entry["problems"] and bool(entry["checks"]) and all(entry["checks"].values())
    return entry


def measured(entry, config, arrays, facts, verifier, output):
    import numpy as np

    name, source, length, start, chunk, rows = (entry[key] for key in ("name", "input", "tokens", "start", "chunk", "rows"))
    ids, vocabulary, problems = tokens(source, config)[:length], config["vocab_size"], entry["problems"]
    (output / f"{name}.tokens.json").write_text(json.dumps(ids))
    command = [str(verifier), "logits", "--model", str(facts["bundle"]), "--tokens", str(output / f"{name}.tokens.json"),
               "--start", str(start), "--chunk", str(chunk), "--output", str(output / f"{name}.f16")]
    began = time.perf_counter()
    try:
        code, stdout = run_verifier(command, TIMEOUT_S, output / f"{name}.log")
    except subprocess.TimeoutExpired:
        problems.append(f"the verifier did not finish within {TIMEOUT_S} s")
    else:
        if code:
            problems.append(f"the verifier exited with {code}")
        record = printed(stdout)
        if record is None:
            problems.append("the verifier did not print one JSON object")
        else:
            record, malformed = validated(record)
            entry["verifier"] = record
            problems += [f"the verifier record is malformed ({key})" for key in malformed]
            if not malformed:
                expected = {"record": "logits", "token_count": length, "start": start, "chunk": chunk,
                            "tokens_sha256": hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest(),
                            "shape": [rows, vocabulary], "dtype": "float16", "byte_order": "little",
                            "main_calls": math.ceil(rows / chunk), "main_hash": facts["main_hash"],
                            **prefix_record(start, prefill_chunk(), facts["prefill"])}
                problems += [f"the verifier recorded {key} {record.get(key)!r}, not {value!r}"
                             for key, value in expected.items() if record.get(key) != value]
                if start == 0 and "prefix_function" in record:
                    problems.append("the verifier recorded a prefix function for a condition that starts at position 0")
    entry["seconds"] = time.perf_counter() - began
    if problems:
        return
    data, content_hash = read_once(output / f"{name}.f16")
    entry["output"] = {"bytes": len(data), "sha256": content_hash}
    if len(data) != rows * vocabulary * 2:
        problems.append(f"the logits file holds {len(data)} B, not {rows * vocabulary * 2} B")
        return
    index = {row: position for position, row in enumerate(kept_rows(source))}
    known = np.zeros(rows, dtype=bool)
    known[[row - start for row in known_rows(start, length, chunk)]] = True
    entry.update(measure(np.frombuffer(data, dtype="<f2").reshape(rows, vocabulary), arrays[source],
                         np.array([index[position] for position in range(start, length)]), known))
    if entry["verifier"]["non_finite_count"] != entry["non_finite"]:
        problems.append(f"the verifier counted {entry['verifier']['non_finite_count']} non-finite values, not the "
                        f"{entry['non_finite']} of its file")


def compare(export_folder, reference_folder, verifier, output, contract_path=CONTRACT):
    if os.path.lexists(output):
        raise FileExistsError(f"The comparison must be written to a new folder: {output}")
    contract = load_contract(contract_path)
    output.mkdir(parents=True)
    problems, reference, reference_hash, arrays, exported, export_hash, facts = [], {}, None, {}, {}, None, None
    try:
        reference, reference_hash, arrays, found = load_reference(reference_folder, contract, contract_path)
        problems += found
    except Exception as error:
        problems.append(f"the reference record is unreadable ({type(error).__name__})")
    try:
        exported, export_hash, facts, found = load_export(export_folder, contract, contract_path)
        problems += found
    except Exception as error:
        problems.append(f"the export record is unreadable ({type(error).__name__})")
    if None not in (reference_hash, export_hash) and exported.get("layers") != reference.get("layers"):
        problems.append("the export and the reference are of other layer counts")
    verifier_record = None
    if not problems:
        try:
            data, verifier_hash = read_once(verifier)
            verifier_record = {"bytes": len(data), "sha256": verifier_hash}
        except Exception as error:
            problems.append(f"the verifier is unreadable ({type(error).__name__})")
    conditions = [] if problems else [condition(spec, contract["config"], arrays, facts, verifier, output)
                                      for spec in CONDITIONS]
    result = {"schema": SCHEMA, "problems": problems,
              "passed": not problems and len(conditions) == len(CONDITIONS) and all(item["passed"] for item in conditions),
              "criteria": {"max_abs_logit": MAX_ABS_LOGIT, "near_tie": NEAR_TIE},
              "known_call": {"ends": [KNOWN_ENDS[0], KNOWN_ENDS[-1]], "shortest": KNOWN_SHORTEST},
              "reference": {"sha256": reference_hash, "layers": reference.get("layers")},
              "export": {"sha256": export_hash, "layers": exported.get("layers"), "functions": exported.get("functions"),
                         "main_hash": facts["main_hash"] if facts else None},
              "verifier": verifier_record, "conditions": conditions}
    try:
        write_json(output / "comparison.json", result)
    except Exception as error:
        result = {"schema": SCHEMA, "passed": False, "conditions": [],
                  "problems": [f"the comparison could not be recorded ({type(error).__name__})"]}
        write_json(output / "comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare the all-position logits that coreai-verify writes for the "
                                                 "twelve conditions with the FP32 reference of the same first layers.")
    for name in ("export", "reference", "verifier", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.export, args.reference, args.verifier, args.output)
    print(json.dumps({"passed": result["passed"], "problems": result["problems"],
                      "conditions": {item["name"]: {"passed": item["passed"], "max_abs_logit": item.get("max_abs_logit")}
                                     for item in result["conditions"]}}, indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
