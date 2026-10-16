from __future__ import annotations

import glob
import math
import os
from pathlib import Path
import re
import shutil
from string import Template

from .storage import PipelineError, contained, read_json


IDENTIFIER = re.compile(r"[a-zA-Z_][a-zA-Z_0-9]*\Z")
STEP_ID = re.compile(r"[a-z][a-z0-9_]*\Z")
ROLES = {"prepare", "reference", "export", "verify", "benchmark"}
RESERVED = {"out", "scratch", "repo", "recipe_dir"}
ENVIRONMENT = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "DEVELOPER_DIR")


def keys(value, required, optional, label):
    if not isinstance(value, dict):
        raise PipelineError(f"{label} must be an object")
    missing, extra = set(required) - value.keys(), value.keys() - set(required) - set(optional)
    if missing or extra:
        raise PipelineError(f"{label}: missing={sorted(missing)}, unknown={sorted(extra)}")


def strings(value, label, *, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise PipelineError(f"{label} must be a {'nonempty ' if nonempty else ''}list")
    if any(not isinstance(v, str) or not v for v in value):
        raise PipelineError(f"{label} must contain nonempty strings")


def expand(value, variables):
    if isinstance(value, str):
        try:
            return Template(value).substitute(variables)
        except (KeyError, ValueError) as error:
            raise PipelineError(f"Unresolved or malformed variable in {value!r}: {error}") from error
    if isinstance(value, list):
        return [expand(v, variables) for v in value]
    if isinstance(value, dict):
        return {k: expand(v, variables) for k, v in value.items()}
    return value


def load_recipe(path, bindings, repo):
    path = Path(path).resolve()
    data = read_json(path)
    keys(data, {"version", "name", "steps"}, {"variables", "environment", "python", "description"}, "recipe")
    if type(data["version"]) is not int or data["version"] != 1:
        raise PipelineError("Only recipe version 1 is supported")
    if not isinstance(data["name"], str) or not STEP_ID.fullmatch(data["name"]):
        raise PipelineError("Recipe name must be a lowercase identifier")
    variables = data.get("variables", {})
    if not isinstance(variables, dict):
        raise PipelineError("variables must be an object")
    supplied = {**variables, **bindings}
    if any(not IDENTIFIER.fullmatch(k) or k in RESERVED for k in supplied):
        raise PipelineError("Invalid or reserved variable name")
    if any(not isinstance(v, str) for v in supplied.values()):
        raise PipelineError("Variable values must be strings")
    resolved = {"repo": str(Path(repo).resolve()), "recipe_dir": str(path.parent)}
    while supplied:
        advanced = False
        for name, value in list(supplied.items()):
            try:
                resolved[name] = Template(value).substitute(resolved)
            except KeyError:
                continue
            except ValueError as error:
                raise PipelineError(f"Malformed variable: {name}") from error
            del supplied[name]
            advanced = True
        if not advanced:
            raise PipelineError(f"Unresolved/cyclic variables: {sorted(supplied)}")
    environment = data.get("environment", {})
    if not isinstance(environment, dict) or any(
        not IDENTIFIER.fullmatch(k) or not isinstance(v, str) for k, v in environment.items()
    ):
        raise PipelineError("environment must map names to strings")
    if not isinstance(data["steps"], list) or not data["steps"]:
        raise PipelineError("steps must be nonempty")
    seen = set()
    steps = []
    for stage in data["steps"]:
        keys(stage, {"id", "role", "command", "inputs", "outputs"},
             {"depends", "checks", "heavy", "timeout_s", "cache", "min_free_bytes", "cwd"}, "step")
        name = stage["id"]
        if not isinstance(name, str) or not STEP_ID.fullmatch(name):
            raise PipelineError("Step id must be a lowercase identifier")
        if name in seen or name in resolved or name in RESERVED:
            raise PipelineError(f"Duplicate/reserved step id: {name}")
        if stage["role"] not in ROLES:
            raise PipelineError(f"Unknown role: {stage['role']}")
        for field in ("command", "inputs", "outputs"):
            strings(stage[field], field, nonempty=True)
        dependencies = stage.get("depends", [])
        strings(dependencies, "depends")
        if len(set(dependencies)) != len(dependencies) or not set(dependencies) <= seen:
            raise PipelineError(f"Unknown, cyclic or future dependency in {name}: {dependencies}")
        for field in ("heavy", "cache"):
            if field in stage and type(stage[field]) is not bool:
                raise PipelineError(f"{field} must be boolean")
        for field in ("timeout_s", "min_free_bytes"):
            value = stage.get(field, 3600 if field == "timeout_s" else 0)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise PipelineError(f"Invalid {field}")
            if field == "timeout_s" and value == 0:
                raise PipelineError("timeout_s must be positive")
        if stage["role"] == "benchmark" and stage.get("cache", False):
            raise PipelineError("Benchmark measurements cannot be cached")
        for relative in stage["outputs"]:
            contained(Path("/unused"), relative)
            if "$" in relative:
                raise PipelineError("Output declarations must be literal relative paths")
        checks = stage.get("checks", [])
        if not isinstance(checks, list) or (stage["role"] == "verify" and not checks):
            raise PipelineError("Verification stages require explicit checks")
        for check in checks:
            keys(check, {"file", "key"}, {"equals", "min", "max"}, "check")
            contained(Path("/unused"), check["file"])
            if not isinstance(check["key"], list) or any(type(k) not in (str, int) for k in check["key"]):
                raise PipelineError("check.key must be an array of JSON keys/indices")
            if ("equals" in check) == bool({"min", "max"} & check.keys()):
                raise PipelineError("check requires equals or numeric min/max")
            for limit in ("min", "max"):
                if limit in check and (type(check[limit]) not in (int, float) or not math.isfinite(check[limit])):
                    raise PipelineError("Check limits must be finite numbers")
            if not any(Path(check["file"]).is_relative_to(Path(v)) for v in stage["outputs"]):
                raise PipelineError("Every checked result must be included in required outputs")
        placeholders = {**resolved, **{d: f"/dependency/{d}" for d in dependencies},
                        "out": "/stage/output", "scratch": "/stage/scratch"}
        expand(stage, placeholders)
        for pattern in stage["inputs"]:
            if any(v in Template(pattern).get_identifiers() for v in [*seen, "out", "scratch"]):
                raise PipelineError("inputs are source files; declare stage artifacts through depends")
        seen.add(name)
        steps.append({**stage, "depends": dependencies, "checks": checks,
                      "cache": stage.get("cache", stage["role"] != "benchmark"),
                      "heavy": stage.get("heavy", False)})
    python = expand(data.get("python", ""), resolved)
    if python and not isinstance(python, str):
        raise PipelineError("python must be an executable path")
    return {"name": data["name"], "steps": steps, "variables": resolved,
            "environment": expand(environment, resolved), "python": python}


def source_paths(stage, variables):
    paths = []
    for pattern in expand(stage["inputs"], variables):
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            raise PipelineError(f"No input matches: {pattern}")
        paths.extend(Path(p).absolute() for p in matches)
    return sorted(set(paths))


def runtime_environment(recipe):
    base = {name: os.environ[name] for name in ENVIRONMENT if name in os.environ}
    base.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    base.update(recipe["environment"])
    return base


def executable_path(command, environment):
    if os.sep in command and not Path(command).is_absolute():
        raise PipelineError(f"Use an absolute executable path instead of {command!r}")
    found = shutil.which(command, path=environment.get("PATH"))
    if not found:
        raise PipelineError(f"Executable is unavailable: {command}")
    return Path(found).absolute()


def inspect_source(path):
    if not path.exists():
        raise PipelineError(f"Missing source: {path}")
    candidates = path.rglob("*") if path.is_dir() else [path]
    found = False
    for candidate in candidates:
        if "__pycache__" in candidate.parts or ".git" in candidate.parts or candidate.suffix in {".pyc", ".pyo"}:
            continue
        found = True
        if not candidate.exists():
            raise PipelineError(f"Missing source or dangling link: {candidate}")
        if candidate.is_symlink() and candidate.is_dir():
            raise PipelineError(f"Directory symlink is not supported: {candidate}")
        if candidate.is_file():
            with candidate.open("rb"):
                pass
        elif not candidate.is_dir():
            raise PipelineError(f"Unsupported source type: {candidate}")
    if not found:
        raise PipelineError(f"Empty source directory: {path}")


def preflight(recipe):
    variables = recipe["variables"]
    environment = runtime_environment(recipe)
    for stage in recipe["steps"]:
        paths = source_paths(stage, variables)
        for path in paths:
            inspect_source(path)
        command = expand(stage["command"], {**variables, **{d: "/dependency/" + d for d in stage["depends"]},
                                            "out": "/stage/output", "scratch": "/stage/scratch"})
        executable_path(command[0], environment)
        if "cwd" in stage and stage["cwd"] not in ("$out", "$" + "{out}") and not Path(expand(stage["cwd"], variables)).is_dir():
            raise PipelineError(f"Working directory is unavailable: {stage['cwd']}")
    if recipe["python"]:
        executable_path(recipe["python"], environment)
