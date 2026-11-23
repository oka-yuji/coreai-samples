import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from Tools.LLMEvaluation import layer1
from ...storage import write_json
from . import native
from .compare import CONTRACT
from .layer1_anchor import LOGITS, forced, interruptible
from .prepare import load_contract

SCHEMA = "gemma4-31b-layer1-candidate/1"
ROLES = {"float16": "fp16", "8bit": "w8"}
CHUNK = 1


def launched(entry, record, facts, vocabulary, verifier, output):
    name, ids, start, rows, problems = record["id"], record["tokens"], record["start"], record["rows"], entry["problems"]
    tokens = output / f"{name}.tokens.json"
    with tokens.open("x") as handle:
        json.dump(ids, handle)
    command = [str(verifier), "logits", "--model", str(facts["bundle"]), "--tokens", str(tokens), "--start", str(start),
               "--chunk", str(CHUNK), "--output", str(output / f"{name}.f16")]
    began = time.perf_counter()
    try:
        code, stdout = native.run_verifier(command, native.TIMEOUT_S, output / f"{name}.log")
    except subprocess.TimeoutExpired:
        problems.append(f"the verifier did not finish within {native.TIMEOUT_S} s")
        return
    finally:
        entry["seconds"] = time.perf_counter() - began
    if code:
        problems.append(f"the verifier exited with {code}")
    printed = native.printed(stdout)
    if printed is None:
        problems.append("the verifier did not print one JSON object")
        return
    kept, malformed = native.validated(printed)
    entry["verifier"] = kept
    if malformed:
        problems += [f"the verifier record is malformed ({key})" for key in malformed]
        return
    expected = {"record": "logits", "token_count": len(ids), "start": start, "chunk": CHUNK,
                "tokens_sha256": record["tokens_sha256"], "shape": [rows, vocabulary], "dtype": "float16",
                "byte_order": "little", "main_calls": math.ceil(rows / CHUNK), "main_hash": facts["main_hash"],
                **native.prefix_record(start, native.prefill_chunk(), facts["prefill"])}
    problems += [f"the verifier recorded {key} {kept.get(key)!r}, not {value!r}" for key, value in expected.items()
                 if kept.get(key) != value]
    if start == 0 and "prefix_function" in kept:
        problems.append("the verifier recorded a prefix function for an item that starts at position 0")
    if problems:
        return
    try:
        data = layer1.file_bytes(output / f"{name}.f16", rows * vocabulary * 2)
    except ValueError as error:
        problems.append(f"the logits file is refused: {error}")
        return
    entry["output"] = {"file": f"{name}.f16", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def attempted(record, facts, vocabulary, verifier, output):
    fixed = {key: record[key] for key in ("id", "input_tokens", "start", "rows")}
    entry = {**fixed, "tokens": len(record["tokens"]), "problems": []}
    try:
        launched(entry, record, facts, vocabulary, verifier, output)
        json.dumps(entry, allow_nan=False)
    except Exception as error:
        entry = {**fixed, "tokens": len(record["tokens"]),
                 "problems": [f"the item could not be run ({type(error).__name__})"]}
    return entry


def recorded(output, result, handoff):
    try:
        write_json(output / "logits.json", handoff)
        write_json(output / "run.json", result)
    except Exception as error:
        result = {"schema": SCHEMA, "passed": False, "items": [],
                  "problems": [f"the run could not be recorded ({type(error).__name__})"]}
        write_json(output / "run.json", result)
    return result


def run(export_folder, forced_folder, verifier, output, contract_path=CONTRACT):
    output = Path(output)
    if os.path.lexists(output):
        raise FileExistsError(f"The candidate logits must be written to a new folder: {output}")
    contract = load_contract(contract_path)
    vocabulary = contract["config"]["vocab_size"]
    exported, export_hash, facts, problems = native.load_export(Path(export_folder), contract, contract_path)
    precision, context, name = exported.get("precision"), exported.get("context"), (exported.get("bundle") or {}).get("name")
    if not native.valid("text", name):
        problems.append("the export record has no bundle name")
    role = ROLES.get(precision) if isinstance(precision, str) else None
    if not problems and role is None:
        problems.append(f"no role is known for the precision {precision!r}")
    hashes, records = forced(forced_folder, vocabulary)
    if not native.valid("count", context) or any(len(record["tokens"]) > context for record in records):
        problems.append("an item of the forced input is longer than the context of the export")
    if problems:
        raise ValueError(f"The export is refused for this forced input: {'; '.join(problems)}")
    data = layer1.file_bytes(verifier)
    verifier_hash = hashlib.sha256(data).hexdigest()
    weights = f"{name} main.hash {facts['main_hash']}"
    output.mkdir(parents=True)
    result = {"schema": SCHEMA, "role": role, "weights": weights, "passed": False,
              "export": {"sha256": export_hash, "precision": precision, "layers": exported.get("layers"),
                         "functions": exported.get("functions"), "context": context, "main_hash": facts["main_hash"]},
              "verifier": {"bytes": len(data), "sha256": verifier_hash},
              "forced": {**hashes, "items": [record["id"] for record in records]}, "chunk": CHUNK,
              "prefill_chunk": native.prefill_chunk(), "items": [], "problems": []}
    began = time.perf_counter()
    try:
        for record in records:
            result["items"].append(attempted(record, facts, vocabulary, verifier, output))
            print(record["id"], result["items"][-1].get("seconds"), result["items"][-1]["problems"][:1], flush=True)
    finally:
        done = [(entry, record) for entry, record in zip(result["items"], records) if not entry["problems"]]
        result["problems"] = [f"{entry['id']}: {problem}" for entry in result["items"] for problem in entry["problems"]]
        result["passed"] = not result["problems"] and len(done) == len(records)
        result["timing"] = {"items_s": time.perf_counter() - began}
        handoff = {"schema": LOGITS, "role": role, "weights": weights, "forced_sha256": hashes["sha256"],
                   "items": [{"id": entry["id"], "file": entry["output"]["file"], "dtype": "float16",
                              "shape": [entry["rows"], vocabulary], "sha256": entry["output"]["sha256"],
                              "tokens_sha256": record["tokens_sha256"]} for entry, record in done]}
        result = recorded(output, result, handoff)
    return result


def main():
    parser = argparse.ArgumentParser(description="Write the logits of the scored rows of the layer 1 forced input that "
                                                 "coreai-verify reads from a Gemma 4 31B export, one launch per item.")
    for name in ("export", "forced", "verifier", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    interruptible()
    result = run(args.export, args.forced, args.verifier, args.output)
    print(json.dumps({"passed": result["passed"], "items": len(result["items"]), "problems": result["problems"]},
                     indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
