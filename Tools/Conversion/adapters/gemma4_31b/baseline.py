import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import time
import uuid

from ...storage import digest, read_json, write_json
from ..pinned import sha256
from .inputs import line, strict, tokens_sha256
from .prepare import environment, load_contract
from .reference import (finalize, load_model, loading_clean, loading_record, memory, non_finite, probabilities, supervise,
                        verify_inputs)

SCHEMA = "gemma4-31b-baseline/1"
TOOL = "Tools.Conversion.adapters.gemma4_31b.baseline"
CHUNK = 1024
LAST = ("long-ja-4k", "long-ja-16k", "synth-long-4k", "synth-long-16k")
SESSION = ("id", "items", "started", "finished")
EMPTY = hashlib.sha256(b"").hexdigest()
CHANGED = "the run folder was changed outside this tool"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_inputs(data, vocabulary):
    rows = data.split(b"\n")
    if rows[-1] == b"":
        rows.pop()
    found, seen = [], set()
    for number, row in enumerate(rows, 1):
        record = strict(row)
        ids = record.get("input_ids") if isinstance(record, dict) else None
        limit = record.get("max_new_tokens") if isinstance(record, dict) else None
        if not isinstance(ids, list) or not ids or any(type(token) is not int or not 0 <= token < vocabulary
                                                       for token in ids) \
                or type(limit) is not int or limit < 1 or not isinstance(record.get("id"), str) or record["id"] in seen:
            raise ValueError(f"inputs.jsonl line {number} is not an input record with a new id")
        if record.get("input_sha256") != tokens_sha256(ids):
            raise ValueError(f"{record['id']}: input_sha256 differs from the input_ids")
        seen.add(record["id"])
        found.append(record)
    return found


def ordered(inputs):
    return sorted(inputs, key=lambda entry: entry["id"].rsplit("-", 1)[0] in LAST)


def conditions(contract, contract_sha256, inputs_sha256, count, model, commit, versions):
    return {"contract": {"version": contract["version"], "sha256": contract_sha256}, "set": "primary", "role": "reference",
            "asset": {"model": model["model_id"], "revision": model["revision"],
                      "weights": {name: entry["sha256"] for name, entry in sorted(model["files"].items())
                                  if name.endswith(".safetensors")}},
            "inputs": {"sha256": inputs_sha256, "count": count},
            "implementation": {"tool": TOOL, "commit": commit,
                               "environment": {key: versions[key] for key in ("python", "packages", "os_build")}},
            "call_shape": {"prefill_chunk": CHUNK}, "order": {"last": list(LAST)}}


def identity(run):
    return digest({key: value for key, value in run.items() if key not in SESSION[1:]})


def empty(path):
    try:
        status = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISREG(status.st_mode) and status.st_size == 0


def unstarted(folder):
    names = sorted(path.name for path in folder.iterdir())
    if not names:
        return True
    if names not in (["outputs.jsonl"], ["outputs.jsonl", "progress.json"]) or not empty(folder / "outputs.jsonl"):
        return False
    if names == ["outputs.jsonl"]:
        return True
    try:
        checkpoint = read_json(folder / "progress.json")
    except (ValueError, OSError):
        return False
    return isinstance(checkpoint, dict) and type(checkpoint.get("lines")) is int and checkpoint["lines"] == 0 \
        and checkpoint.get("sha256") == EMPTY and "next_sha256" not in checkpoint


def resume(folder, run, inputs):
    stored = read_json(folder / "run.json")
    if not isinstance(stored, dict) or not isinstance(stored.get("id"), str):
        raise ValueError("run.json has no run id; a run folder of an earlier version cannot be resumed")
    differing = sorted(key for key in (set(run) | set(stored)) - set(SESSION) if run.get(key) != stored.get(key))
    if differing:
        raise ValueError(f"--output holds another run; these differ from its run.json: {', '.join(differing)}")
    if not (folder / "progress.json").exists():
        raise ValueError(f"progress.json is missing; {CHANGED}")
    checkpoint = read_json(folder / "progress.json")
    if not isinstance(checkpoint, dict) or checkpoint.get("run") != identity(stored):
        raise ValueError(f"progress.json belongs to another run; {CHANGED}")
    items = stored.get("items")
    if type(items) is not int or not 0 <= items <= len(inputs):
        raise ValueError(f"run.json records {items!r} finished items, not a count from 0 to {len(inputs)}; {CHANGED}")
    data = (folder / "outputs.jsonl").read_bytes()
    complete = data.rfind(b"\n") + 1
    lines = data[:complete].split(b"\n")[:-1]
    if len(lines) < items:
        raise ValueError(f"outputs.jsonl has {len(lines)} complete lines, fewer than the {items} finished items that "
                         f"run.json records; {CHANGED}")
    hasher = hashlib.sha256(data[:complete])
    written, recorded = hasher.hexdigest(), checkpoint.get("lines")
    counted = type(recorded) is int and 0 <= recorded <= len(lines)
    if not (counted and (len(lines) == recorded and written == checkpoint.get("sha256")
                         or len(lines) == recorded + 1 and written == checkpoint.get("next_sha256"))):
        raise ValueError(f"outputs.jsonl has {len(lines)} complete lines, which match neither the {recorded!r} lines that "
                         f"progress.json records nor the line that it announced; {CHANGED}")
    known, done = {entry["id"] for entry in inputs}, set()
    for number, row in enumerate(lines, 1):
        try:
            name = strict(row).get("id")
        except (ValueError, AttributeError, RecursionError):
            name = None
        if not isinstance(name, str) or name not in known:
            raise ValueError(f"outputs.jsonl line {number} has no id of the input records")
        if name in done:
            raise ValueError(f"outputs.jsonl line {number} repeats an id")
        done.add(name)
    if complete < len(data):
        with (folder / "outputs.jsonl").open("r+b") as handle:
            handle.truncate(complete)
    return stored, done, hasher


def start(folder, run, inputs):
    folder = Path(folder)
    if (folder / "run.json").exists():
        stored, done, hasher = resume(folder, run, inputs)
        stored["items"] = len(done)
        stored.pop("finished", None)
    else:
        if folder.exists() and not (folder.is_dir() and unstarted(folder)):
            raise ValueError(f"--output exists but is not a run folder with run.json: {folder}")
        folder.mkdir(exist_ok=True)
        stored, done, hasher = {**run, "id": str(uuid.uuid4()), "items": 0, "started": now()}, set(), hashlib.sha256()
        (folder / "outputs.jsonl").write_bytes(b"")
        write_json(folder / "progress.json", {"lines": 0, "sha256": EMPTY, "run": identity(stored)})
    write_json(folder / "run.json", stored)
    return stored, done, hasher


def run_items(folder, run, inputs, done, hasher, produce, deadline=None, rows=None):
    folder, key = Path(folder), identity(run)
    with (folder / "outputs.jsonl").open("ab") as handle:
        for entry in ordered(inputs):
            if entry["id"] in done:
                continue
            record, info = produce(entry)
            data = line(record)
            following = hasher.copy()
            following.update(data)
            write_json(folder / "progress.json", {"lines": run["items"], "sha256": hasher.hexdigest(),
                                                  "next_sha256": following.hexdigest(), "run": key})
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            hasher = following
            run["items"] += 1
            done.add(entry["id"])
            row = {"id": entry["id"], "input_tokens": record["input_tokens"], "output_tokens": len(record["output_ids"]),
                   "stop": "error" if "error" in record else record["stop"], **info}
            if rows is not None:
                rows.append(row)
            print(json.dumps({"record": "item", **row}, sort_keys=True), flush=True)
            if "error" in record:
                raise RuntimeError(f"{entry['id']} ended with an error; the run stops")
            if deadline is not None and time.monotonic() >= deadline:
                break
    return hasher


def finish(folder, run, hasher):
    write_json(Path(folder) / "progress.json", {"lines": run["items"], "sha256": hasher.hexdigest(), "run": identity(run)})
    run["finished"] = now()
    write_json(Path(folder) / "run.json", run)


def leaders(row):
    first = int(row.argmax())
    value, row[first] = row[first], -math.inf
    second = int(row.argmax())
    row[first] = value
    return first, second


def release(device, info):
    import torch

    if device == "mps":
        torch.mps.synchronize()
        info["mps_driver_peak_bytes"] = max(info.get("mps_driver_peak_bytes", 0), torch.mps.driver_allocated_memory())
        torch.mps.empty_cache()


def forward(model, tokens, cache, device):
    import torch

    return model(input_ids=torch.tensor([tokens], device=device), past_key_values=cache, use_cache=True,
                 logits_to_keep=1).logits[0, -1]


def decode(model, entry, stops, device, chunk, record, info, started):
    from transformers import DynamicCache

    ids, cache = entry["input_ids"], DynamicCache(config=model.config)
    for position in range(0, len(ids), chunk):
        logits = forward(model, ids[position:position + chunk], cache, device)
        release(device, info)
    info["first_token_s"] = time.perf_counter() - started
    while True:
        row = logits.float().cpu().numpy()
        count = non_finite(row)
        if count:
            record.update(non_finite=count, error=f"non-finite logits at output position {len(record['top2'])}")
            return
        first, second = leaders(row)
        chance, other = probabilities(row, [first, second])
        record["top2"].append([first, chance, second, other])
        if first in stops:
            record.update(stop="stop_token", stop_token=first)
            return
        record["output_ids"].append(first)
        if len(record["output_ids"]) == entry["max_new_tokens"]:
            return
        logits = forward(model, [first], cache, device)


def generate(model, entry, stops, device, chunk=CHUNK):
    started, info = time.perf_counter(), {}
    record = {"id": entry["id"], "input_sha256": entry["input_sha256"], "input_tokens": len(entry["input_ids"]),
              "output_ids": [], "top2": [], "non_finite": 0}
    try:
        decode(model, entry, stops, device, chunk, record, info, started)
        release(device, info)
        info.update(memory(device))
    except Exception as error:
        record.setdefault("error", f"{type(error).__name__}: {error}")
    record.setdefault("stop", "limit")
    info["seconds"] = time.perf_counter() - started
    return record, info


def main():
    begun = time.monotonic()
    parser = argparse.ArgumentParser(description="Generate the BF16 MPS reference outputs of quality evaluation inputs.")
    for name in ("model", "snapshot", "prepare", "contract", "inputs", "output", "log"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--max-minutes", type=float)
    args = parser.parse_args()
    if args.max_minutes is not None and not (math.isfinite(args.max_minutes) and args.max_minutes > 0):
        parser.error("--max-minutes must be a positive number")
    model = load_contract(args.model)
    ready = read_json(args.prepare / "ready.json")
    if ready.get("ready") is not True or ready.get("contract_sha256") != sha256(args.model):
        raise ValueError("Run prepare with this contract first")
    data, inputs_data = args.contract.read_bytes(), args.inputs.read_bytes()
    contract, inputs = strict(data), read_inputs(inputs_data, model["config"]["vocab_size"])
    device, stops = model["reference"]["device"], set(contract["generation"]["eos_token_id"])
    run = conditions(contract, hashlib.sha256(data).hexdigest(), hashlib.sha256(inputs_data).hexdigest(), len(inputs),
                     model, args.commit, environment(args.model.parent))
    args.log.mkdir()
    run, done, hasher = start(args.output, run, inputs)
    before = run["items"]
    deadline = None if args.max_minutes is None else begun + args.max_minutes * 60

    def work():
        import torch

        began = time.perf_counter()
        verify_inputs(args.snapshot, model)
        verified = time.perf_counter()
        loaded, info = load_model(args.snapshot, device)
        loaded.eval()
        loading = loading_record(loaded, info, device)
        if not loading_clean(loading, device):
            raise ValueError(f"The model did not load cleanly: {loading['info']}")
        ready_at, rows = time.perf_counter(), []
        with torch.no_grad():
            final = run_items(args.output, run, inputs, done, hasher,
                              lambda entry: generate(loaded, entry, stops, device), deadline, rows)
        return final, rows, loading, {"verify_s": verified - began, "load_s": ready_at - verified,
                                      "items_s": time.perf_counter() - ready_at}

    def conclude(result, observed, started):
        final, rows, loading, timing = result
        record = {"schema": SCHEMA, "run": run["id"], "checks": {}, "loading": loading, "resources": {"cases": rows},
                  "items": {"before": before, "after": run["items"], "count": len(inputs)}, "timing": timing}
        finalize(record, observed, memory(device))
        if record["passed"]:
            finish(args.output, run, final)
        record["timing"]["total_s"] = time.perf_counter() - started
        write_json(args.log / "session.json", record)
        print(json.dumps({"record": "session", "items": run["items"], "count": len(inputs),
                          "complete": run["items"] == len(inputs), "passed": record["passed"]}, sort_keys=True), flush=True)
        return 0 if record["passed"] else 1

    return supervise(args.log, SCHEMA, work, conclude)


if __name__ == "__main__":
    sys.exit(main())
