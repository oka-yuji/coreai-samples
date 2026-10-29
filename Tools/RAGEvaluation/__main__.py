from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

from Tools.Conversion.runner import execute
from Tools.Conversion.storage import cleanup_failure, directory_lock, file_hash, tree_hash, write_json


def make_plan(manifest_path, split, strategy, mode):
    raw = Path(manifest_path).read_bytes()
    manifest = json.loads(raw)
    cases = [item["id"] for item in manifest["cases"] if item["split"] == split]
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("Empty split or duplicate case identity")
    return {
        "schemaVersion": 1,
        "manifestHash": hashlib.sha256(raw).hexdigest(),
        "manifest": str(Path(manifest_path).resolve()),
        "strategy": strategy,
        "mode": mode,
        "split": split,
        "cases": cases,
        "settings": manifest["settings"],
    }


def initial_result(plan, case_id, status, reason):
    return {
        "schemaVersion": 1, "caseID": case_id, "manifestHash": plan["manifestHash"],
        "requestedStrategy": plan["strategy"], "mode": plan["mode"],
        "status": status, "reason": reason,
    }


def summarize(plan, output):
    rows = []
    for index, case_id in enumerate(plan["cases"]):
        path = Path(output) / f"case-{index:03}" / "result.json"
        row = json.loads(path.read_text()) if path.exists() else initial_result(plan, case_id, "notRun", "missing result")
        for key, expected in {
            "caseID": case_id, "manifestHash": plan["manifestHash"],
            "requestedStrategy": plan["strategy"], "mode": plan["mode"],
        }.items():
            if row.get(key) != expected:
                raise ValueError(f"Result identity differs: {path}: {key}")
        if row.get("status") not in {"notRun", "running", "completed", "failed", "cancelled"}:
            raise ValueError(f"Invalid result state: {path}")
        control = Path(output) / f"case-{index:03}.control.json"
        if control.exists():
            state = json.loads(control.read_text())
            if state.get("caseID") != case_id:
                raise ValueError(f"Driver identity differs: {control}")
            if state.get("driverState") != "settled":
                row["nativeStatus"] = row["status"]
                row["status"] = "running"
                row["reason"] = state.get("reason", "process completion unconfirmed")
        elif path.exists():
            row["nativeStatus"] = row["status"]
            row["status"] = "running"
            row["reason"] = "driver completion record missing"
        rows.append(row)
    counts = Counter(row["status"] for row in rows)
    return {
        "requested": len(plan["cases"]),
        "counts": {state: counts[state] for state in ["completed", "failed", "cancelled", "running", "notRun"]},
        "factualSupport": "ungraded",
        "rows": rows,
    }


@contextmanager
def interrupt_signals():
    previous = {}

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    def relay(signum, frame):
        os.kill(os.getpid(), signal.SIGTERM)

    try:
        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
            prior = signal.getsignal(sig)
            previous[sig] = prior
            if prior != signal.SIG_IGN:
                signal.signal(sig, relay if sig == signal.SIGHUP else stop)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_cases(plan, output, app, model, executor=execute):
    for index, case_id in enumerate(plan["cases"]):
        directory = Path(output) / f"case-{index:03}"
        control = Path(output) / f"case-{index:03}.control.json"
        write_json(control, {"caseID": case_id, "driverState": "attempted"})
        command = [str(app), "--rag-eval", plan["manifest"], "--case", case_id,
                   "--mode", plan["mode"], "--rag-strategy", plan["strategy"],
                   "--model", str(model), "--output", str(directory)]
        try:
            executor(command, Path.cwd(), os.environ.copy(), Path(output) / f"case-{index:03}.log",
                     plan["settings"]["deadlineSeconds"] + 120)
            if not (directory / "result.json").exists():
                raise RuntimeError("Process succeeded without a result")
            write_json(control, {"caseID": case_id, "driverState": "settled"})
        except BaseException as error:
            if cleanup_failure(error) is not None:
                write_json(control, {"caseID": case_id, "driverState": "cleanupUnconfirmed", "reason": str(error)})
                raise
            directory.mkdir(exist_ok=True)
            result_path = directory / "result.json"
            row = json.loads(result_path.read_text()) if result_path.exists() else initial_result(plan, case_id, "notRun", "")
            if row.get("status") not in {"failed", "cancelled"}:
                row["status"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
            row["driverError"] = str(error)
            write_json(result_path, row)
            write_json(control, {"caseID": case_id, "driverState": "settled"})
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            write_json(Path(output) / "summary.json", summarize(plan, output))


def active_models():
    probe = subprocess.run(
        ["pgrep", "-x", r"CoreAISamples|coreai-chat|coreai-verify"],
        capture_output=True, text=True, check=False)
    if probe.returncode not in [0, 1]:
        raise RuntimeError("Could not inspect existing model processes")
    return [int(pid) for pid in probe.stdout.split() if int(pid) != os.getpid()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", default="flat", choices=["flat", "structure-bm25", "structure-llm"])
    parser.add_argument("--mode", default="work", choices=["work", "chat"])
    parser.add_argument("--split", default="evaluation", choices=["evaluation", "development"])
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.summary_only:
        plan = json.loads((output / "plan.json").read_text())
        write_json(output / "summary.json", summarize(plan, output))
        return
    if not args.app or not args.manifest or not args.model:
        parser.error("--app, --manifest and --model are required")
    plan = make_plan(args.manifest, args.split, args.strategy, args.mode)
    if not plan["settings"].get("expectedModelHash"):
        parser.error("Freeze the verified 4bit main.hash in settings.expectedModelHash first")
    app, model = args.app.resolve(strict=True), args.model.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    frozen = args.manifest.read_bytes()
    if hashlib.sha256(frozen).hexdigest() != plan["manifestHash"]:
        raise RuntimeError("Manifest changed while preparing the evaluation")
    (output / "manifest.json").write_bytes(frozen)
    plan["manifest"] = str(output / "manifest.json")
    write_json(output / "plan.json", plan)
    write_json(output / "summary.json", summarize(plan, output))
    with interrupt_signals():
        execute([str(app), "--rag-eval", plan["manifest"], "--rag-strategy", args.strategy, "--validate-only"],
                Path.cwd(), os.environ.copy(), output / "validation.log", 120)
        if active_models():
            raise RuntimeError("Another model process exists; do not stop it or take its lock")
        lock = Path(os.environ.get("COREAI_HEAVY_JOB_LOCK", Path.home() / "Library/Caches/coreai-samples/heavy-job"))
        owner = {"session": "rag-evaluation", "job": args.strategy,
                 "started": datetime.now(timezone.utc).isoformat(), "scope": plan["cases"]}
        with directory_lock(lock, owner):
            try:
                plan["executableHash"] = file_hash(app)
                plan["bundleFiles"] = tree_hash(model)
                plan["xcode"] = subprocess.check_output(["xcodebuild", "-version"], text=True).strip()
                write_json(output / "plan.json", plan)
                run_cases(plan, output, app, model)
            finally:
                if cleanup_failure(sys.exc_info()[1]) is None:
                    lock.rename(lock.with_name("heavy-job-done-rag-" + uuid.uuid4().hex))
    summary = summarize(plan, output)
    if summary["counts"]["completed"] != summary["requested"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
