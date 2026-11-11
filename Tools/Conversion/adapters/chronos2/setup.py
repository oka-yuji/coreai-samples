import argparse
import importlib
import importlib.metadata
from pathlib import Path
import platform
import sys
import tomllib
import uuid

from ...storage import PipelineError, directory_lock, file_hash, write_json
from ..pinned import apple_toolchain
from .artifacts import load_contract, read_artifact, reject_links, require_equal, validate_source, write_new


ROOT = Path(__file__).resolve().parents[4]
IMPORTS = ("chronos", "coreai.runtime", "coreai_torch", "numpy", "torch", "torchao", "transformers",
           "huggingface_hub", "safetensors")


def environment_inputs(recipe):
    return {name: file_hash(reject_links(recipe / name)) for name in ("pyproject.toml", "uv.lock")}


def source_inputs(recipe):
    paths = [*Path(__file__).parent.glob("*.py"), *(ROOT / "Tools/Conversion").glob("*.py"),
             ROOT / "Tools/Conversion/adapters/pinned.py"]
    sources = {path.relative_to(ROOT).as_posix(): file_hash(reject_links(path)) for path in sorted(paths)}
    sources.update({"recipe/" + name: file_hash(reject_links(recipe / name))
                    for name in ("recipe.json", "model.json", "fixtures.json", "README.md", "NOTICE", "LICENSE")})
    return sources


def installed_environment(recipe):
    project = tomllib.loads((recipe / "pyproject.toml").read_text())["project"]
    if sys.version_info[:2] != (3, 11):
        raise PipelineError("The pinned environment requires Python 3.11")
    packages = {}
    for dependency in project["dependencies"]:
        name, expected = dependency.split("==")
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise PipelineError(f"Installed dependency differs: {name}; run setup again")
        packages[name] = actual
    return {"python": platform.python_version(), "binary": file_hash(Path(sys.executable).resolve()),
            "packages": packages}


def toolchain_identity():
    toolchain = apple_toolchain()
    return {key: toolchain[key] for key in ("xcode", "macos_sdk", "coreai_build_sha256")} | {
        "system": platform.system(), "release": platform.release(), "version": platform.version(),
        "machine": platform.machine(),
    }


def check_setup(recipe, stamp):
    record = read_artifact(stamp)
    expected = {"format": "coreai-samples-chronos2-setup/1", "environment_inputs": environment_inputs(recipe),
                "sources": source_inputs(recipe), "environment": installed_environment(recipe),
                "toolchain": toolchain_identity()}
    require_equal(record, expected, "setup identity; run setup again")
    return record


def setup(workspace, recipe, snapshot, lock=None):
    from ...local import command, default_lock, default_snapshot

    workspace = reject_links(workspace)
    recipe = reject_links(recipe)
    contract = load_contract(recipe.parent / "model.json")
    snapshot = reject_links(snapshot or default_snapshot(contract))
    profile = workspace / "chronos2"
    stamp = profile / "setup.json"
    bindings = workspace / "chronos2.paths.json"
    python = profile / ".venv/bin/python"
    for path in (profile, profile / ".venv", python.parent, stamp, bindings, workspace / "setup.lock"):
        reject_links(path)
    lock = reject_links(lock or default_lock())
    with directory_lock(workspace / "setup.lock", {"operation": "chronos2-setup"}), \
            directory_lock(lock, {"operation": "chronos2-setup"}):
        validate_source(snapshot, contract)
        inputs = environment_inputs(recipe.parent)
        sources = source_inputs(recipe.parent)
        toolchain = toolchain_identity()
        profile.mkdir(parents=True, exist_ok=True)
        stamp.unlink(missing_ok=True)
        bindings.unlink(missing_ok=True)
        command("uv", "sync", "--project", recipe.parent, "--frozen", "--no-dev", "--python", "3.11",
                env={"UV_PROJECT_ENVIRONMENT": str(profile / ".venv")})
        for path in (profile / ".venv", python.parent):
            reject_links(path)
        if not python.is_file():
            raise PipelineError("Setup did not produce the dedicated Python environment")
        probe = profile / ("probe-" + uuid.uuid4().hex + ".json")
        try:
            command(python, "-m", "Tools.Conversion.adapters.chronos2.setup", "--probe", probe,
                    "--recipe", recipe.parent, cwd=ROOT,
                    env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
            environment = read_artifact(probe)
        finally:
            probe.unlink(missing_ok=True)
        require_equal(environment_inputs(recipe.parent), inputs, "environment inputs during setup")
        require_equal(source_inputs(recipe.parent), sources, "source inputs during setup")
        require_equal(toolchain_identity(), toolchain, "toolchain during setup")
        write_new(stamp, {"format": "coreai-samples-chronos2-setup/1", "environment_inputs": inputs,
                          "sources": sources, "environment": environment, "toolchain": toolchain})
        write_json(bindings, {"python": str(python), "snapshot": str(snapshot), "setup_stamp": str(stamp)})
        print(f"Local bindings: {bindings}")
    return bindings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    environment = installed_environment(args.recipe)
    for module in IMPORTS:
        importlib.import_module(module)
    write_new(args.probe, environment)


if __name__ == "__main__":
    main()
