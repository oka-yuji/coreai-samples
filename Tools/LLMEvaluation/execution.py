import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile

from Tools.LLMEvaluation.items import FILES, ROOT, read_json

RULES = read_json(ROOT / FILES["contract"])["scorers"]["python_tests"]
LIMIT = RULES["wall_clock_seconds"]
SANDBOX = "/usr/bin/sandbox-exec"
PROFILE = '(version 1) (deny default) (allow process-exec file-read*) (allow file-write* (subpath (param "WORK")))'
DRIVER = """
import json
import os
import sys
import types


def drive():
    marker_fd, result_fd, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    with os.fdopen(marker_fd, "rb") as handle:
        marker = handle.read()
    del sys.argv[1:]
    with open(path, encoding="utf-8") as handle:
        units = json.load(handle)
    module = types.ModuleType("program")
    sys.modules[module.__name__] = module
    run, build, write, namespace = exec, compile, os.write, module.__dict__
    for name, source in units:
        run(build(source, name, "exec", dont_inherit=True), namespace)
    write(result_fd, marker)


drive()
"""
QUIET = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}


def extract(text):
    lines = text.splitlines(keepends=True)
    fences = [index for index, line in enumerate(lines) if line.startswith("```")]
    start = next((index for index in fences if lines[index][3:].split()[:1] == ["python"]),
                 fences[0] if fences else None)
    if start is None:
        return text
    end = next((index for index in fences if index > start), len(lines))
    return "".join(lines[start + 1:end])


def program(scorer, code):
    parts = {**scorer, "code": code}
    return [[name, parts[name]] for name in RULES["units"]]


def command(work):
    return [SANDBOX, "-p", PROFILE, "-D", f"WORK={work}", sys.executable, "-I", "-S", "-B"]


def available():
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder).resolve()
        if subprocess.run([*command(work), "-c", ""], cwd=work, **QUIET).returncode:
            raise RuntimeError(f"{SANDBOX} does not run Python with the isolation profile, so no code is run")


def attempt(units):
    marker = secrets.token_hex(16).encode("ascii")
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder).resolve()
        work, path = root / "work", root / "units.json"
        work.mkdir()
        path.write_text(json.dumps(units), encoding="utf-8")
        marker_read, marker_write = os.pipe()
        os.write(marker_write, marker)
        os.close(marker_write)
        result_read, result_write = os.pipe()
        try:
            try:
                child = subprocess.Popen([*command(work), "-c", DRIVER, str(marker_read), str(result_write), str(path)],
                                         cwd=work, pass_fds=(marker_read, result_write), **QUIET)
            finally:
                os.close(marker_read)
                os.close(result_write)
            try:
                status = child.wait(LIMIT)
            except subprocess.TimeoutExpired:
                return "timeout"
            finally:
                child.kill()
                child.wait()
            received = os.read(result_read, len(marker) + 1)
        finally:
            os.close(result_read)
    if received != marker:
        return "not_finished"
    return "passed" if status == 0 else "exit_status"


def run(units):
    first = attempt(units)
    last = attempt(units) if first == "timeout" else first
    return {"correct": last == "passed", "failure": None if last == "passed" else last, "rerun": first == "timeout"}
