from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import signal
import uuid


class PipelineError(ValueError):
    pass


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise PipelineError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise PipelineError(f"Non-finite JSON number: {value}")

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_hash(path):
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise PipelineError(f"Not a regular file: {path}")
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            result.update(block)
    after = path.stat()
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if signature(before) != signature(after):
        raise PipelineError(f"File changed while hashing: {path}")
    return {"bytes": after.st_size, "sha256": result.hexdigest()}


def tree_hash(path, *, output=False, allow_empty=False):
    path = Path(path)
    if output and path.is_symlink():
        raise PipelineError(f"Output symlink is not allowed: {path}")
    if not path.exists():
        raise PipelineError(f"Missing input or output: {path}")
    if path.is_file():
        return {"file": file_hash(path)}
    if not path.is_dir():
        raise PipelineError(f"Unsupported file type: {path}")
    entries = {}
    for child in sorted(path.rglob("*")):
        relative = str(child.relative_to(path))
        if not output and ("__pycache__" in child.parts or ".git" in child.parts or child.suffix in {".pyc", ".pyo"}):
            continue
        if child.is_symlink() and (output or child.is_dir()):
            raise PipelineError(f"Directory/output symlink is not allowed: {child}")
        if child.is_dir():
            entries[relative] = {"directory": True}
        else:
            entries[relative] = file_hash(child)
    if not entries and not allow_empty:
        raise PipelineError(f"Empty directory cannot be a required artifact: {path}")
    return {"tree": entries}


def contained(root, relative):
    path = Path(relative)
    if not relative or path.is_absolute() or ".." in path.parts or relative == ".":
        raise PipelineError(f"Expected a nonempty relative artifact path: {relative}")
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise PipelineError(f"Artifact escapes its stage: {target}")
    return target


def check_results(root, checks):
    for check in checks:
        value = read_json(contained(root, check["file"]))
        try:
            for key in check["key"]:
                value = value[key]
        except (KeyError, IndexError, TypeError) as error:
            raise PipelineError(f"Missing result field: {check}") from error
        if "equals" in check:
            expected = check["equals"]
            passed = type(value) is type(expected) and value == expected
        else:
            passed = type(value) in (int, float) and math.isfinite(value)
            if passed and "min" in check:
                passed = value >= check["min"]
            if passed and "max" in check:
                passed = value <= check["max"]
        if not passed:
            raise PipelineError(f"Verification failed: {check}; observed {value!r}")


@contextlib.contextmanager
def directory_lock(path, owner):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    marker = path / "owner.txt"
    identity = None
    try:
        with blocked_signals():
            try:
                path.mkdir()
            except FileExistsError as error:
                raise PipelineError(f"Busy lock; inspect its owner, never steal it: {path}") from error
            identity = path.stat().st_ino
            write_json(marker, {"token": token, "pid": os.getpid(), **owner})
        yield
    finally:
        with blocked_signals():
            if identity is not None and marker.exists() and read_json(marker).get("token") == token:
                marker.unlink()
                path.rmdir()
            elif identity is not None and not marker.exists() and path.exists() and path.stat().st_ino == identity:
                path.rmdir()


@contextlib.contextmanager
def blocked_signals():
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
