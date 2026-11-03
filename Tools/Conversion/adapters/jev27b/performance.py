import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys

from ...storage import read_json, write_json
from .compare import tokens_sha256
from .decision import sha256
from .fixtures import KINDS, SEED, Builder, Sources, eligible, serialize
from .prepare import check_integrity, distributed_tokenizer, load_contract
from .typed_head import render, tokenize

SCHEMA = "typed-decision-performance-rows/1"
LENGTHS = ((100, "noul"), (500, "choice"), (2000, "score"), (8192, "noul"))
P_SEED = f"{SEED}:P"
ROUNDS = 3
REQUEST = ("id", "kind", "state", "question", "options")
MEMORY = {"peak_rss_bytes": "maximum resident set size", "peak_footprint_bytes": "peak memory footprint"}


def used(fixtures):
    rows = {name: read_json(Path(fixtures) / f"{name}.json")["rows"] for name in ("F", "Q")}
    return {row["id"] for name in rows for row in rows[name] if row["source"] == "test_set_30k"} | {
        part for row in rows["F"] for part in row["parts"] or []}


def build(builder, sources, excluded):
    rng = random.Random(P_SEED)
    pool = [row for kind in KINDS for row in eligible(sources.split("test_set_30k"), kind) if row["id"] not in excluded]
    rng.shuffle(pool)
    entries = []
    for target, kind in LENGTHS:
        base = next(row for row in pool if row["kind"] == kind and builder.tokens({**row, "state": ""}) < target)
        pool.remove(base)
        entry = builder.finish(builder.exact(f"performance:tokens{target}", base, pool, target))
        if entry["kind"] != kind or entry["parts"][0] != base["id"] or entry["tokens"] != target:
            raise ValueError(f"The {target}-token row did not keep its base row")
        entries.append(entry)
    return entries


def schedule(rows):
    order = [(row, "warmup") for row in rows] + [(row, str(index)) for index in range(1, ROUNDS + 1) for row in rows]
    return [{**row, "id": f"{row['id']}@{label}"} for row, label in order]


def write_rows(args):
    if (args.output / "p.json").exists():
        raise FileExistsError("p.json must be written to a new directory")
    contract = load_contract(args.model)
    check_integrity(args.corpus, contract["dataset"]["id"], contract["dataset"]["revision"])
    sources = Sources(args.corpus)
    tokenizer = distributed_tokenizer(args.snapshot)
    builder = Builder(sources, tokenizer)
    excluded = used(args.fixtures)
    entries = build(builder, sources, excluded)
    rows = [sources.restore(entry) for entry in entries]
    for entry, row in zip(entries, rows, strict=True):
        entry["tokens_sha256"] = tokens_sha256(tokenize(tokenizer, render(row["kind"], row["state"], row["question"],
                                                                         row["options"])))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, chosen in (("input-P.json", rows), ("input-P-latency.json", schedule(rows))):
        (args.output / name).write_bytes(serialize({"set": "P", "rows": [{key: row[key] for key in REQUEST}
                                                                          for row in chosen]}).encode("utf-8"))
    write_json(args.output / "p.json",
               {"schema": SCHEMA, "seed": P_SEED, "lengths": [list(item) for item in LENGTHS], "rounds": ROUNDS,
                "excluded_parts": len(excluded), "dataset": contract["dataset"],
                "fixtures": {name: sha256(args.fixtures / f"{name}.json") for name in ("F", "Q")},
                "tokenizer_sha256": sha256(args.snapshot / "tokenizer.json"),
                "inputs": {name: sha256(args.output / name) for name in ("input-P.json", "input-P-latency.json")},
                "rows": entries})
    return 0


def context(plan, plan_sha256, workload, cache):
    item = plan["context"]
    return {"model": plan["model_id"], "variant": item["variant"], "device": item["device"], "os": item["os"],
            "workload": workload, "cache": cache, "plan_sha256": plan_sha256, "power": item["power"],
            "thermal": item["thermal"]}


def run(identifier, identity, status, metrics, **extra):
    return {"run_id": identifier, "context": identity, "status": status,
            "metrics": {name: {"value": value, "unit": unit} for name, (value, unit) in metrics.items()}, **extra}


def total_load(document):
    return sum(document["bundle"]["load_seconds"][part] for part in ("bundle", "tokenizer", "head", "backbone"))


def same_run(plan, manifest, document, name):
    return document["bundle"]["version"] == plan["context"]["artifact_revision"] \
        and document.get("input_sha256") == manifest["inputs"][name]


def latency_runs(plan, plan_sha256, manifest, document):
    expected = {entry["id"]: entry for entry in manifest["rows"]}
    origin = same_run(plan, manifest, document, "input-P-latency.json")
    cache = f"specialization {document['bundle']['specialization']}; one process for every length"
    runs = {}
    for row in document["rows"]:
        identifier, label = row["id"].split("@")
        entry = expected[identifier]
        valid = origin and "error" not in row and row["token_count"] == entry["tokens"] \
            and row["tokens_sha256"] == entry["tokens_sha256"]
        status = "failed" if not valid else "warmup" if label == "warmup" else "passed"
        identity = context(plan, plan_sha256, f"P row of {entry['tokens']} tokens ({entry['kind']}), lengths interleaved "
                           f"in one process, {ROUNDS} measured rounds after one warmup", cache)
        runs.setdefault(entry["tokens"], []).append(run(
            f"latency:{entry['tokens']}:{label}", identity, status,
            {"latency_s": (row["elapsed_seconds"], "s")} if valid else {}, row_id=row["id"],
            resident_bytes_max=row.get("resident_bytes_max"), valid_tokens=valid))
    return runs


def load_runs(plan, plan_sha256, manifest, documents, expected, metric):
    cache = {"fresh": "specialization cache entry deleted before each load (--clear-cache)",
             "cached": "specialization cache entry present; new process"}[expected]
    identity = context(plan, plan_sha256, "new process per load, --limit 1 on the 100-token P row", cache)
    return [run(f"{metric}:{index}", identity,
                "passed" if document["bundle"]["specialization"] == expected
                and same_run(plan, manifest, document, "input-P.json") else "failed",
                {metric: (total_load(document), "s")}, specialization=document["bundle"]["specialization"],
                load_seconds=document["bundle"]["load_seconds"])
            for index, document in enumerate(documents, start=1)]


def memory_run(plan, plan_sha256, text, latency):
    found = {name: int(re.search(rf"^\s*(\d+)\s+{label}$", text, re.M).group(1)) for name, label in MEMORY.items()}
    resident = [item["resident_bytes_max"] for item in latency[8192] if item["status"] == "passed"]
    complete = len(resident) == plan["measured_runs"] and all(item["samples"] > 0 and item["failed_samples"] == 0
                                                               for item in resident)
    identity = context(plan, plan_sha256, "the latency process: /usr/bin/time -l for the process lifetime, 100 ms "
                       "sampling during the measured 8,192-token rows", "specialization cached; one process")
    metrics = {name: (value, "B") for name, value in found.items()}
    if complete:
        metrics["resident_bytes"] = (max(item["rss_bytes"] for item in resident), "B")
    return [run("memory:latency-process", identity, "passed" if complete else "failed", metrics,
                resident_footprint_bytes=max((item["footprint_bytes"] for item in resident), default=None),
                resident_samples=[item["samples"] for item in resident],
                resident_failed_samples=[item["failed_samples"] for item in resident])]


def cancel_runs(plan, plan_sha256, metrics):
    items, uncancelled = metrics["cancellations"], metrics["uncancelledSeconds"]
    identity = context(plan, plan_sha256, f"TypedDecisionInferenceTests: a {items[0]['tokens']}-token request cancelled "
                       "after the test backbone reported the start of its forward call; the backbone runs the forward "
                       "in a task that the cancellation does not reach", "specialization cached; test process")
    return [run(f"cancel:{item['attempt']}", identity,
                "passed" if item["entered"] and item["cancelled"] and item["request_to_error_seconds"] > uncancelled / 2
                else "failed",
                {"cancel_s": (item["request_to_error_seconds"], "s")}, start_to_error_seconds=item["start_to_error_seconds"],
                entered=item["entered"], cancelled=item["cancelled"], uncancelled_seconds=uncancelled) for item in items]


def bundle_run(plan, plan_sha256, bundle):
    files = [path for path in sorted(Path(bundle).rglob("*")) if path.is_symlink() or path.is_file()]
    if any(path.is_symlink() for path in files):
        raise ValueError("The bundle must not contain links")
    identity = context(plan, plan_sha256, f"file sizes of the calibrated bundle ({len(files)} files)",
                       "not applicable (no model load)")
    return [run("bundle:calibrated", identity,
                "passed" if sha256(Path(bundle) / "decision.json") == plan["context"]["artifact_revision"] else "failed",
                {"bundle_bytes": (sum(path.stat().st_size for path in files), "B")}, files=len(files))]


def counted(plan, latency, cancel):
    labels = {tokens: [row["run_id"].rsplit(":", 1)[1] for row in rows] for tokens, rows in latency.items()}
    if sorted(labels) != [tokens for tokens, _ in LENGTHS] or len(cancel) != plan["measured_runs"] \
            or any(items.count("warmup") != plan["warmup_runs"]
                   or len(items) - items.count("warmup") != plan["measured_runs"] for items in labels.values()):
        raise ValueError("The outputs do not hold the planned warmup and measured runs")


def write_lines(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))


def write_records(args):
    plan, manifest = read_json(args.plan), read_json(args.manifest)
    plan_sha256 = sha256(args.plan)
    latency = latency_runs(plan, plan_sha256, manifest, read_json(args.latency))
    files = {f"latency-{tokens}.jsonl": rows for tokens, rows in sorted(latency.items())}
    files["first-load.jsonl"] = load_runs(plan, plan_sha256, manifest, [read_json(path) for path in args.fresh],
                                          "fresh", "first_load_s")
    files["warm-load.jsonl"] = load_runs(plan, plan_sha256, manifest, [read_json(path) for path in args.cached],
                                         "cached", "warm_load_s")
    files["memory.jsonl"] = memory_run(plan, plan_sha256, args.time.read_text(), latency)
    files["cancel.jsonl"] = cancel_runs(plan, plan_sha256, read_json(args.cancel))
    files["bundle.jsonl"] = bundle_run(plan, plan_sha256, args.bundle)
    counted(plan, latency, files["cancel.jsonl"])
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in files.items():
        if (args.output / name).exists():
            raise FileExistsError(f"{name} must be written to a new directory")
        write_lines(args.output / name, rows)
    return 0 if all(row["status"] != "failed" for rows in files.values() for row in rows) else 1


def main():
    parser = argparse.ArgumentParser(description="Performance rows P of JEV-27B and the run records of their measurement.")
    commands = parser.add_subparsers(dest="command", required=True)
    rows = commands.add_parser("rows")
    for name in ("model", "snapshot", "corpus", "fixtures", "output"):
        rows.add_argument(f"--{name}", type=Path, required=True)
    records = commands.add_parser("records")
    for name in ("plan", "manifest", "latency", "time", "cancel", "bundle", "output"):
        records.add_argument(f"--{name}", type=Path, required=True)
    for name in ("fresh", "cached"):
        records.add_argument(f"--{name}", type=Path, nargs=ROUNDS, required=True)
    args = parser.parse_args()
    return write_rows(args) if args.command == "rows" else write_records(args)


if __name__ == "__main__":
    sys.exit(main())
