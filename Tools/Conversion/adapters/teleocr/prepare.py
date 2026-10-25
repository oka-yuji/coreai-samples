import argparse
import importlib.metadata
import platform
import subprocess
import sys
import tomllib
from pathlib import Path

from .contracts import MODEL_ID, REVISION, digest, read_json, write_json


def validate(snapshot, specification):
    if (specification["model_id"], specification["revision"]) != (MODEL_ID, REVISION):
        raise ValueError("The recipe supports only the pinned TeleOCR revision")
    if Path(snapshot).resolve().name != REVISION:
        raise ValueError("Use the immutable snapshots/<revision> directory")
    for relative, expected in specification["files"].items():
        path = Path(snapshot) / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.is_file():
            raise ValueError(f"Missing or unsafe pinned source file: {relative}")
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Source content does not match the pinned revision: {relative}")


def versions(recipe):
    project = tomllib.loads((recipe / "pyproject.toml").read_text())
    pinned = dict(dependency.split("==") for dependency in project["project"]["dependencies"])
    installed = {name: importlib.metadata.version(name) for name in pinned}
    if installed != pinned:
        raise ValueError(f"Use the recipe environment from its lock: {installed}")
    return installed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specification = read_json(args.model)
    validate(args.snapshot, specification)
    fixture = args.model.parent / specification["fixture"]["file"]
    if digest(fixture) != specification["fixture"]["sha256"]:
        raise ValueError("The recipe fixture differs from its pinned digest")
    installed = versions(args.model.parent)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "source.json").exists():
        raise FileExistsError("Source identity must be written to a new directory")
    write_json(args.output / "source.json", {
        "verified": True, "model_id": MODEL_ID, "revision": REVISION, "files": specification["files"],
        "versions": installed, "python": sys.version, "platform": platform.platform(),
        "os_build": subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip(),
        "xcode": subprocess.check_output(["xcodebuild", "-version"], text=True).strip()})


if __name__ == "__main__":
    main()
