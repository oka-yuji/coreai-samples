from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import time
import uuid

from .recipe import executable_path, expand, preflight, runtime_environment, source_paths
from .storage import (
    PipelineError, ProcessCleanupError, blocked_signals, check_results, cleanup_failure, contained, digest, directory_lock, file_hash,
    read_json, tree_hash, write_json,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def host_identity(recipe, environment):
    result = {"system": platform.platform(), "machine": platform.machine(), "environment": environment}
    if platform.system() == "Darwin":
        result["os_build"] = subprocess.check_output(["/usr/bin/sw_vers", "-buildVersion"], text=True).strip()
    if recipe["python"]:
        probe = (
            "import sys,sysconfig,json,importlib.metadata as m;"
            "print(json.dumps({'version':sys.version,'executable':sys.executable,"
            "'libraries':sorted(set([sysconfig.get_path('purelib'),sysconfig.get_path('platlib')])),"
            "'packages':sorted((d.metadata['Name'],d.version) for d in m.distributions())}))"
        )
        completed = subprocess.run([recipe["python"], "-c", probe], env=environment,
                                   capture_output=True, text=True, check=True, timeout=30)
        python = json.loads(completed.stdout)
        python["binary"] = file_hash(Path(python["executable"]).resolve())
        python["library_hashes"] = {p: digest(tree_hash(Path(p), allow_empty=True)) for p in python.pop("libraries")}
        result["python"] = python
    return result


def inventory(stage, variables):
    return {str(path): tree_hash(path) for path in source_paths(stage, variables)}


def outputs(stage, root):
    for relative in stage["outputs"]:
        path = contained(root, relative)
        if not path.exists() or (path.is_dir() and not any(path.iterdir())):
            raise PipelineError(f"Required artifact is missing or empty: {path}")
    return {"required": stage["outputs"], "artifacts": tree_hash(root, output=True)}


@contextlib.contextmanager
def deferred_interrupts():
    previous = {}
    pending = []
    failure = None
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda signum, _frame: pending.append(signum))
        yield pending
    except BaseException as error:
        failure = error
        if pending:
            error.add_note(f"Deferred interruption signals: {pending}")
        raise
    finally:
        with blocked_signals(preserving=failure):
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    if pending:
        raise KeyboardInterrupt("Termination requested")


def running_group(group):
    result = subprocess.run(["/bin/ps", "-ax", "-o", "pgid=,stat="], capture_output=True, text=True,
                            check=True, timeout=5)
    rows = [line.split() for line in result.stdout.splitlines() if line.strip()]
    if not rows or any(len(row) != 2 or not row[0].isdigit()
                       or re.fullmatch(r"[RSDITZWUX?][A-Za-z+<>=?-]*", row[1]) is None for row in rows):
        raise PipelineError("Could not inspect process-group completion")
    return any(int(pgid) == group and not state.startswith("Z") for pgid, state in rows)


def signal_group(group, signum):
    try:
        os.killpg(group, signum)
    except ProcessLookupError:
        return True
    except PermissionError as error:
        if error.errno != errno.EPERM:
            raise
        try:
            active = running_group(group)
        except Exception as inspection_error:
            error.add_note(f"Process-group inspection failed: {inspection_error}")
            raise error from inspection_error
        if active:
            raise
        return True
    return False


def cleanup_process(child, primary, state):
    errors = []
    with blocked_signals():
        if primary is not None:
            try:
                signal_group(child.pid, signal.SIGTERM)
            except OSError as error:
                errors.append(error)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        ended = False
        try:
            ended = signal_group(child.pid, signal.SIGKILL)
        except OSError as error:
            errors.append(error)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired as error:
            errors.append(error)
        deadline = time.monotonic() + 5
        while not ended:
            try:
                ended = not running_group(child.pid)
            except Exception as error:
                errors.append(error)
                break
            if ended:
                break
            if time.monotonic() >= deadline:
                errors.append(PipelineError("A process-group member is still running"))
                break
            time.sleep(0.05)
        if not ended or child.returncode is None:
            raise ProcessCleanupError(primary, child.pid, errors, child.returncode)
        state["confirmed"] = True
        if primary is not None:
            for error in errors:
                primary.add_note(f"Cleanup recovered after {type(error).__name__}: {error}")


def execute(command, cwd, environment, log, timeout):
    with log.open("xb") as handle:
        child = None
        primary = None
        cleanup = {"confirmed": False}
        with deferred_interrupts() as pending:
            try:
                child = subprocess.Popen(command, cwd=cwd, env=environment, stdout=handle,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + timeout
                while True:
                    if pending:
                        raise KeyboardInterrupt("Termination requested")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    try:
                        code = child.wait(timeout=min(0.1, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        pass
                if code:
                    raise PipelineError(f"Command exited {code}; see {log}")
            except BaseException as error:
                if child is None:
                    raise
                primary = error
            try:
                cleanup_process(child, primary, cleanup)
            except ProcessCleanupError:
                raise
            except BaseException as cleanup_error:
                if not cleanup["confirmed"]:
                    raise ProcessCleanupError(primary or cleanup_error, child.pid, [cleanup_error], child.returncode) from cleanup_error
                if primary is None:
                    raise
                primary.add_note(f"Further interruption after confirmed cleanup: {cleanup_error}")
            if primary is not None:
                raise primary


def reusable(stage, object_dir, key):
    if not stage["cache"] or object_dir is None or not (object_dir / "manifest.json").is_file():
        return None
    try:
        manifest = read_json(object_dir / "manifest.json")
        if (not isinstance(manifest, dict) or manifest.get("version") != 1 or manifest.get("key") != key
                or manifest.get("status") != "completed" or not isinstance(manifest.get("run_id"), str)):
            return None
        observed = outputs(stage, object_dir / "artifacts")
        if observed != manifest.get("outputs"):
            return None
        check_results(object_dir / "artifacts", stage["checks"])
        return manifest
    except (OSError, ValueError, TypeError, KeyError):
        return None


def cached_object(work, key):
    try:
        entry = read_json(work / "index" / f"{key}.json")
        name = entry["object"]
        if not isinstance(name, str) or not re.fullmatch(re.escape(key) + r"-[0-9]{8}T[0-9]{6}-[a-f0-9]{8}", name):
            return None
        target = work / "objects" / name
        return None if target.is_symlink() else target
    except (OSError, ValueError, TypeError, KeyError):
        return None


def save_report(path, report):
    write_json(path, report)
    rows = ["# Core AI pipeline run", "",
            f"- Recipe: {report['recipe']}",
            f"- Status: {report['status']}",
            f"- Started: {report['started']}",
            f"- Verification: {report['verification']}",
            "- A completed pipeline is not a production-readiness certification.", "",
            "| Stage | Result | Prepare/hash seconds | Execute seconds | Origin |",
            "|---|---|---:|---:|---|"]
    for stage in report["stages"]:
        rows.append(f"| {stage['id']} | {stage['status']} | {stage.get('prepare_s', 0):.3f} | "
                    f"{stage.get('execute_s', 0):.3f} | {stage.get('origin', '')} |")
    if "error" in report:
        rows.extend(["", "Error:", "", "    " + report["error"].replace("\n", "\n    ")])
    path.with_suffix(".md").write_text("\n".join(rows) + "\n")


def run(recipe, work_dir, *, lock_dir=None, refresh=False):
    preflight(recipe)
    work = Path(work_dir).resolve()
    for stage in recipe["steps"]:
        for source in source_paths(stage, recipe["variables"]):
            if source.resolve().is_relative_to(work) or work.is_relative_to(source.resolve()):
                raise PipelineError("Work directory and source inputs must not overlap")
    needs_machine_lock = any(s["heavy"] for s in recipe["steps"])
    if needs_machine_lock and lock_dir is None:
        raise PipelineError("Heavy stages require the shared --lock-dir")
    with directory_lock(work / "run.lock", {"recipe": recipe["name"], "started": now()}):
        machine = directory_lock(lock_dir, {"recipe": recipe["name"], "started": now(), "scope": "pipeline"})\
            if needs_machine_lock else contextlib.nullcontext()
        with machine:
            return run_locked(recipe, work, refresh=refresh)


def run_locked(recipe, work, *, refresh):
    environment = runtime_environment(recipe)
    started = time.perf_counter()
    host_started = time.perf_counter()
    host = host_identity(recipe, environment)
    host_s = time.perf_counter() - host_started
    engine = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = work / "runs" / identifier
    run_dir.mkdir(parents=True)
    report_path = run_dir / "report.json"
    report = {"version": 1, "recipe": recipe["name"], "run_id": identifier, "started": now(),
              "status": "running", "verification": "not_run", "host": host,
              "host_hash_s": host_s, "stages": []}
    bindings = dict(recipe["variables"])
    manifests = {}
    failure = None
    try:
        for stage in recipe["steps"]:
            label = stage["id"]
            stage_started = time.perf_counter()
            source = inventory(stage, bindings)
            dependency_data = {
                d: {"key": manifests[d]["key"], "outputs": manifests[d]["outputs"]}
                for d in stage["depends"]
            }
            template_bindings = {**bindings, "out": "<output>", "scratch": "<scratch>"}
            command = expand(stage["command"], template_bindings)
            executable = executable_path(command[0], environment)
            executable_hash = file_hash(executable)
            definition = expand(stage, template_bindings)
            key = digest({"definition": definition, "inputs": source, "dependencies": dependency_data,
                          "host": host, "engine": engine, "executable": executable_hash})
            object_dir = cached_object(work, key)
            row = {"id": label, "role": stage["role"], "key": key, "status": "running"}
            report["stages"].append(row)
            save_report(report_path, report)
            with directory_lock(work / "locks" / key, {"run": identifier, "stage": label}):
                cached = None if refresh else reusable(stage, object_dir, key)
                row["prepare_s"] = time.perf_counter() - stage_started
                if cached is not None:
                    manifests[label] = cached
                    bindings[label] = str(object_dir / "artifacts")
                    row.update(status="reused", execute_s=0, origin=cached["run_id"],
                               artifacts=bindings[label])
                    print(f"{label}: reused ({row['prepare_s']:.2f}s verification)", flush=True)
                else:
                    attempt = run_dir / label
                    artifacts, scratch = attempt / "artifacts", attempt / "scratch"
                    artifacts.mkdir(parents=True)
                    scratch.mkdir()
                    actual = {**bindings, "out": str(artifacts), "scratch": str(scratch)}
                    command = expand(stage["command"], actual)
                    command[0] = str(executable)
                    cwd = expand(stage.get("cwd", "$out"), actual)
                    child_env = {**environment, "TMPDIR": str(scratch) + "/"}
                    execution_started = time.perf_counter()
                    print(f"{label}: executing", flush=True)
                    if shutil.disk_usage(work).free < stage.get("min_free_bytes", 0):
                        raise PipelineError(f"Insufficient free disk for {label}")
                    execute(command, cwd, child_env, attempt / "process.log", stage.get("timeout_s", 3600))
                    row["execute_s"] = time.perf_counter() - execution_started
                    check_results(artifacts, stage["checks"])
                    artifact_inventory = outputs(stage, artifacts)
                    if (inventory(stage, bindings) != source or host_identity(recipe, environment) != host
                            or file_hash(executable) != executable_hash):
                        raise PipelineError("Inputs or toolchain changed during execution; result not cached")
                    for dependency in stage["depends"]:
                        if outputs(next(s for s in recipe["steps"] if s["id"] == dependency),
                                   Path(bindings[dependency])) != manifests[dependency]["outputs"]:
                            raise PipelineError(f"Dependency changed during execution: {dependency}")
                    manifest = {"version": 1, "key": key, "status": "completed", "run_id": identifier,
                                "stage": label, "completed": now(), "outputs": artifact_inventory,
                                "fingerprint": {"definition": definition, "inputs": source,
                                                "dependencies": dependency_data},
                                "command": command, "execute_s": row["execute_s"]}
                    write_json(attempt / "manifest.json", manifest)
                    if object_dir is not None and not refresh:
                        write_json(work / "quarantine" / f"{key}-{identifier}.json",
                                   {"object": str(object_dir), "reason": "reuse validation failed"})
                    shutil.rmtree(scratch)
                    object_dir = work / "objects" / f"{key}-{identifier}"
                    object_dir.parent.mkdir(parents=True, exist_ok=True)
                    attempt.rename(object_dir)
                    if stage["cache"]:
                        write_json(work / "index" / f"{key}.json", {"object": object_dir.name})
                    manifests[label] = manifest
                    bindings[label] = str(object_dir / "artifacts")
                    row.update(status="executed", origin=identifier, artifacts=bindings[label])
                row["total_s"] = time.perf_counter() - stage_started
                verified = {r["id"] for r in report["stages"]
                            if r["role"] == "verify" and r["status"] in {"reused", "executed"}}
                required = {s["id"] for s in recipe["steps"] if s["role"] == "verify"}
                if verified:
                    report["verification"] = "passed" if verified == required else "incomplete"
            save_report(report_path, report)
        report["status"] = "completed"
        report["verification"] = "passed" if any(s["role"] == "verify" for s in recipe["steps"]) else "not_configured"
    except BaseException as error:
        failure = cleanup_failure(error) or error
        primary = failure.primary if isinstance(failure, ProcessCleanupError) and failure.primary is not None else failure
        report["status"] = "cancelled" if isinstance(primary, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(primary).__name__}: {primary}"
        if isinstance(failure, ProcessCleanupError):
            report["cleanup"] = failure.details
        if getattr(failure, "__notes__", None):
            report["error_notes"] = failure.__notes__
        if report["stages"] and report["stages"][-1]["status"] == "running":
            report["stages"][-1]["status"] = report["status"]
            if report["stages"][-1]["role"] == "verify":
                report["verification"] = "incomplete" if isinstance(primary, KeyboardInterrupt) else "failed"
        raise failure
    finally:
        report["elapsed_s"] = time.perf_counter() - started
        report["finished"] = now()
        try:
            save_report(report_path, report)
            print(f"Report: {report_path}", flush=True)
        except BaseException as reporting_error:
            if failure is None:
                raise
            failure.add_note(f"Could not publish the terminal report: {reporting_error}")
    return report_path
