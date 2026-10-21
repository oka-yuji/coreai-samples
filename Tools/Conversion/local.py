import argparse
import os
from pathlib import Path
import subprocess
import sys

from .adapters.pinned import load_contract
from .recipe import ENVIRONMENT
from .storage import PipelineError, directory_lock, read_json, write_json


ROOT = Path(__file__).resolve().parents[2]
TOOL = Path(__file__).resolve().parent
UPSTREAM = "https://github.com/apple/coreai-models.git"


def default_lock():
    override = os.environ.get("COREAI_HEAVY_JOB_LOCK")
    return Path(override).expanduser() if override else Path.home() / "Library/Caches/coreai-samples/heavy-job"


def recipe_path(name, precision=None):
    if name == "qwen_image21_mobile":
        candidate = precision or "w4"
        if candidate not in ("fp16", "bf16", "mixed", "w8", "w4", "mixed-f32", "w4-f32"):
            raise PipelineError("Select fp16, bf16, mixed, w8, w4, mixed-f32 or w4-f32 for the mobile image recipe")
        filename = "recipe.json" if candidate == "w4" else f"recipe-{candidate}.json"
        path = TOOL / "recipes" / name / filename
        if not path.is_file():
            raise PipelineError("The selected mobile image recipe is unavailable")
        return path
    precision = precision or "fp16_safe"
    if precision not in ("fp16_safe", "int4_int8") or (precision == "int4_int8" and name != "embeddinggemma2"):
        raise PipelineError("INT4/INT8 is available only for embeddinggemma2")
    filename = "recipe-int4-int8.json" if precision == "int4_int8" else "recipe.json"
    path = TOOL / "recipes" / name / filename
    if Path(name).name != name or not path.is_file():
        raise PipelineError("Select a recipe directory from Tools/Conversion/recipes")
    return path


def default_snapshot(contract):
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    hub = Path(os.environ.get("HF_HUB_CACHE", home / "hub"))
    return hub / ("models--" + contract["model_id"].replace("/", "--")) / "snapshots" / contract["revision"]


def command(*args, **kwargs):
    environment = {key: os.environ[key] for key in ENVIRONMENT if key in os.environ}
    environment.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    environment.update(kwargs.pop("env", {}))
    kwargs["env"] = environment
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def check_checkout(path, revision):
    if path.is_symlink():
        raise PipelineError("Use a dedicated checkout, not a symlink")
    origin = command("git", "-C", path, "remote", "get-url", "origin", capture_output=True, text=True).stdout.strip()
    head = command("git", "-C", path, "rev-parse", "HEAD", capture_output=True, text=True).stdout.strip()
    dirty = command("git", "-C", path, "status", "--porcelain", "--untracked-files=no",
                    capture_output=True, text=True).stdout.strip()
    if origin != UPSTREAM or head != revision or dirty:
        raise PipelineError("Existing exporter must be a clean checkout of the recipe's public revision")


def check_workspace_paths(workspace, upstream):
    for path in (upstream, upstream / ".venv", upstream / ".build"):
        if path.is_symlink() or not path.resolve().is_relative_to(workspace.resolve()):
            raise PipelineError("Exporter, Python and build directories must belong to this workspace")


def setup(workspace, recipe, snapshot, lock=None):
    contract_path = recipe.parent / "model.json"
    contract = read_json(contract_path)
    if contract.get("model_type") == "embedding_gemma2":
        return setup_embedding(workspace, recipe, snapshot, lock)
    if contract.get("model_type") == "qwen_image21":
        return setup_image(workspace, recipe, snapshot, lock)
    snapshot = (snapshot or default_snapshot(contract)).expanduser().resolve()
    if not snapshot.is_dir():
        raise PipelineError(
            f"Download the pinned input first: hf download {contract['model_id']} --revision {contract['revision']}"
        )
    load_contract(contract_path, snapshot)
    upstream = workspace / "upstream/coreai-models"
    check_workspace_paths(workspace, upstream)
    with directory_lock(workspace / "setup.lock", {"operation": "setup"}), \
            directory_lock(lock or default_lock(), {"operation": "conversion-setup"}):
        if not upstream.exists():
            upstream.parent.mkdir(parents=True, exist_ok=True)
            command("git", "clone", "--no-checkout", "--", UPSTREAM, upstream)
            command("git", "-C", upstream, "checkout", "--detach", contract["upstream_revision"])
        check_checkout(upstream, contract["upstream_revision"])
        command("uv", "sync", "--project", upstream, "--frozen", "--no-dev", "--package", "coreai-models",
                "--python", "3.11", env={"UV_PROJECT_ENVIRONMENT": str(upstream / ".venv")})
        command("swift", "build", "--package-path", upstream, "-c", "release", "--product", "llm-runner",
                "--disable-automatic-resolution")
        check_checkout(upstream, contract["upstream_revision"])
        python = upstream / ".venv/bin/python"
        runner = upstream / ".build/release/llm-runner"
        if not python.is_file() or not runner.is_file():
            raise PipelineError("Setup did not produce the Python environment and Swift runner")
        bindings = workspace / f"{recipe.parent.name}.paths.json"
        write_json(bindings, {"upstream": str(upstream), "snapshot": str(snapshot)})
        print(f"Local bindings: {bindings}")


def setup_embedding(workspace, recipe, snapshot, lock):
    from .adapters.embeddinggemma2.prepare import environment_inputs, native_sources, validate_source
    from .adapters.pinned import apple_toolchain, sha256
    contract = read_json(recipe.parent / "model.json")
    copied = workspace / "inputs/embeddinggemma2" / contract["revision"]
    snapshot = (snapshot or (copied if copied.is_dir() else default_snapshot(contract))).expanduser().resolve()
    validate_source(snapshot, contract)
    profile = workspace / "embeddinggemma2"
    check_workspace_paths(workspace, profile)
    with directory_lock(workspace / "setup.lock", {"operation": "setup"}), \
            directory_lock(lock or default_lock(), {"operation": "embedding-setup"}):
        profile.mkdir(parents=True, exist_ok=True)
        environment = environment_inputs(recipe.parent)
        command("uv", "sync", "--project", recipe.parent, "--frozen", "--no-dev", "--python", "3.11",
                env={"UV_PROJECT_ENVIRONMENT": str(profile / ".venv")})
        command(profile / ".venv/bin/python", "-c",
                "from transformers import AutoProcessor, EmbeddingGemma2Model; import coreai_torch; import coreai.runtime")
        sources = native_sources(ROOT)
        toolchain = apple_toolchain()
        command("swift", "build", "--package-path", ROOT / "CoreAISamplesKit", "--scratch-path", profile / ".build",
                "-c", "release", "--product", "embedding-verify", "--disable-automatic-resolution")
        runner = profile / ".build/release/embedding-verify"
        if sources != native_sources(ROOT) or toolchain != apple_toolchain() or environment != environment_inputs(recipe.parent):
            raise PipelineError("Native inputs changed during setup; retry after edits finish")
        stamp = profile / "native-build.json"
        write_json(stamp, {"sources": sources, "toolchain": toolchain, "runner_sha256": sha256(runner),
                           "environment_inputs": environment})
        bindings = workspace / "embeddinggemma2.paths.json"
        write_json(bindings, {"python": str(profile / ".venv/bin/python"), "snapshot": str(snapshot),
                              "runner": str(runner), "native_build": str(stamp)})
        print(f"Local bindings: {bindings}")


def setup_image(workspace, recipe, snapshot, lock):
    import hashlib
    import uuid

    contract = read_json(recipe.parent / "model.json")
    snapshot = (snapshot or default_snapshot(contract)).expanduser().resolve()
    if not snapshot.is_dir():
        raise PipelineError(f"Download the pinned input first: {contract['model_id']}@{contract['revision']}")
    profile = workspace / recipe.parent.name
    check_workspace_paths(workspace, profile)
    environment_paths = [recipe.parent / name for name in ("pyproject.toml", "uv.lock")]
    environment = [hashlib.sha256(path.read_bytes()).hexdigest() for path in environment_paths]
    with directory_lock(workspace / "setup.lock", {"operation": "setup"}), \
            directory_lock(lock or default_lock(), {"operation": "image-setup"}):
        profile.mkdir(parents=True, exist_ok=True)
        command("uv", "sync", "--project", recipe.parent, "--frozen", "--no-dev", "--python", "3.11",
                env={"UV_PROJECT_ENVIRONMENT": str(profile / ".venv")})
        python = profile / ".venv/bin/python"
        validation = profile / "setup" / uuid.uuid4().hex
        command(python, "-m", "Tools.Conversion.adapters.qwen_image21.prepare", "--snapshot", snapshot,
                "--model", recipe.parent / "model.json", "--output", validation, cwd=ROOT)
        if environment != [hashlib.sha256(path.read_bytes()).hexdigest() for path in environment_paths]:
            raise PipelineError("Environment inputs changed during setup")
        bindings = workspace / f"{recipe.parent.name}.paths.json"
        write_json(bindings, {"python": str(python), "snapshot": str(snapshot),
                              "heavy_lock": str(lock or default_lock())})
        print(f"Local bindings: {bindings}")


def pipeline_command(action, recipe, workspace, lock, refresh):
    bindings = workspace / f"{recipe.parent.name}.paths.json"
    if not bindings.is_file():
        raise PipelineError("Run setup for this workspace and recipe first")
    result = [sys.executable, "-m", "Tools.Conversion", action, str(recipe), "--bindings", str(bindings)]
    if recipe.parent.name in ("qwen_image21", "qwen_image21_mobile"):
        result += ["--set", f"heavy_lock={lock}"]
    if action == "run":
        profile = recipe.parent.name + ("-int4-int8" if recipe.stem == "recipe-int4-int8" else "")
        result += ["--work-dir", str(workspace / "work" / profile), "--lock-dir", str(lock)]
        if refresh:
            result.append("--refresh")
    elif refresh:
        raise PipelineError("--refresh is only available for run")
    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare and use the maintainer's local conversion workspace.")
    parser.add_argument("action", choices=("setup", "plan", "run"))
    parser.add_argument("--recipe", default="qwen3_06b")
    parser.add_argument("--precision", choices=("fp16_safe", "int4_int8", "fp16", "bf16", "mixed", "w8", "w4", "mixed-f32", "w4-f32"))
    parser.add_argument("--workspace", type=Path, default=ROOT / ".local/conversion")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--lock-dir", type=Path, default=default_lock())
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    try:
        recipe = recipe_path(args.recipe, args.precision)
        workspace = args.workspace.expanduser().resolve()
        if args.action == "setup":
            if args.refresh:
                raise PipelineError("--refresh is only available for run")
            setup(workspace, recipe, args.snapshot, args.lock_dir.expanduser().absolute())
            return 0
        if args.snapshot:
            raise PipelineError("--snapshot belongs to setup")
        executable = pipeline_command(args.action, recipe, workspace, args.lock_dir.expanduser().absolute(), args.refresh)
        os.chdir(ROOT)
        os.execv(sys.executable, executable)
    except (PipelineError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Conversion workspace error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
