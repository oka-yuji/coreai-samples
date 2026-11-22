import time

from ...storage import PipelineError, directory_lock, write_json
from ..pinned import sha256
from .prepare import ENVIRONMENT_FILES, check_record, load_contract


def setup(workspace, recipe, snapshot, lock=None):
    from ...local import check_workspace_paths, command, default_lock

    contract = load_contract(recipe.parent / "model.json")
    snapshot = (snapshot or workspace / "models" / contract["model_id"].replace("/", "--")).expanduser().resolve()
    check_record(snapshot, contract)
    profile = workspace / "gemma4_31b"
    check_workspace_paths(workspace, profile)
    with directory_lock(workspace / "setup.lock", {"operation": "setup"}), \
            directory_lock(lock or default_lock(), {"operation": "gemma4-31b-setup"}):
        started = time.perf_counter()
        inputs = {name: sha256(recipe.parent / name) for name in ENVIRONMENT_FILES}
        (profile / "hf-cache/hub").mkdir(parents=True, exist_ok=True)
        command("uv", "sync", "--project", recipe.parent, "--frozen", "--no-dev", "--python", "3.11",
                env={"UV_PROJECT_ENVIRONMENT": str(profile / ".venv")})
        python = profile / ".venv/bin/python"
        command(python, "-c", "import accelerate, torch, transformers; assert torch.backends.mps.is_available()")
        if inputs != {name: sha256(recipe.parent / name) for name in ENVIRONMENT_FILES}:
            raise PipelineError("Environment inputs changed during setup")
        record = profile / "setup.json"
        write_json(record, {"environment_inputs": inputs, "reference_environment_s": round(time.perf_counter() - started, 1)})
        bindings = workspace / "gemma4_31b.paths.json"
        write_json(bindings, {"python": str(python), "hf_cache": str(profile / "hf-cache"), "setup_record": str(record),
                              "snapshot": str(snapshot)})
        print(f"Local bindings: {bindings}")
