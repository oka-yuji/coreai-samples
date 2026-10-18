import argparse
import importlib.metadata
import platform
from pathlib import Path
import subprocess
import sys

from .contracts import DIFFUSERS_REVISION, PROFILE, REVISION, digest, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specification = read_json(args.model)
    if specification["revision"] != REVISION or specification["diffusers_revision"] != DIFFUSERS_REVISION:
        raise ValueError("The source specification is not the supported revision")
    required = {"torch": "2.9.0", "transformers": "5.18.0", "diffusers": "0.41.0.dev0",
                "coreai-torch": "0.4.3", "coreai-core": "1.0.0b3", "numpy": "2.4.6"}
    versions = {name: importlib.metadata.version(name) for name in required}
    if versions != required:
        raise ValueError(f"Use the frozen research environment: {versions}")
    distribution = importlib.metadata.distribution("diffusers")
    import json
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != DIFFUSERS_REVISION:
        raise ValueError("Diffusers must be installed from the fixed Git revision using the frozen lock")
    files = {}
    for relative, expected in specification["files"].items():
        name = Path(relative)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Unsafe source path in the specification")
        path = args.snapshot / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Source content does not match the pinned revision: {relative}")
        files[relative] = expected
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "source.json").exists():
        raise FileExistsError("Source identity must be written to a new directory")
    write_json(args.output / "source.json", {"verified": True, "profile": PROFILE, "files": files,
               "versions": versions, "python": sys.version, "platform": platform.platform(),
               "os_build": subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip(),
               "xcode_version": subprocess.check_output(["xcodebuild", "-version"], text=True).strip()})


if __name__ == "__main__":
    main()
