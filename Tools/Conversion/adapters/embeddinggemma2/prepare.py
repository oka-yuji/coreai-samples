import argparse
import importlib.metadata
import json
from pathlib import Path
import tomllib

from ..pinned import apple_toolchain, sha256


def validate_source(snapshot, contract):
    snapshot = Path(snapshot).resolve()
    if contract.get("model_id") != "google/embeddinggemma-2" or contract.get("model_type") != "embedding_gemma2":
        raise ValueError("Unexpected embedding source model")
    if snapshot.name != contract["revision"]:
        raise ValueError("Use an immutable directory named with the complete source revision")
    if (contract["text_length"], contract["vision_tokens"], contract["precision"], contract["vector_dimensions"]) != (
        128, 70, "fp16_safe", 768
    ):
        raise ValueError("This recipe supports only the measured S128/vision70 safe FP16 contract")
    for name, identity in contract["source_files"].items():
        path = snapshot / name
        if Path(name).is_absolute() or ".." in Path(name).parts or not path.is_file():
            raise ValueError("Missing or invalid pinned source file")
        if path.stat().st_size != identity["bytes"] or sha256(path) != identity["sha256"]:
            raise ValueError(f"Pinned source differs: {name}")
    if json.loads((snapshot / "config.json").read_text())["model_type"] != contract["model_type"]:
        raise ValueError("Source architecture differs")
    return snapshot


def native_sources(repo):
    kit = Path(repo) / "CoreAISamplesKit"
    files = [kit / "Package.swift", kit / "Package.resolved"]
    for name in ("EmbeddingCore", "EmbeddingRuntime", "embedding-verify"):
        files.extend(sorted((kit / "Sources" / name).rglob("*.swift")))
    return {str(path.relative_to(repo)): sha256(path) for path in files}


def environment_inputs(recipe):
    return {name: sha256(recipe / name) for name in ("pyproject.toml", "uv.lock")}


def check_environment(recipe, expected):
    if expected != environment_inputs(recipe):
        raise ValueError("Environment requirements changed; run setup again")
    project = tomllib.loads((recipe / "pyproject.toml").read_text())
    for dependency in project["project"]["dependencies"]:
        name, version = dependency.split("==")
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Installed dependency differs: {name}; run setup again")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    validate_source(args.snapshot, contract)
    repo = Path(__file__).resolve().parents[4]
    build = json.loads(args.build.read_text())
    check_environment(args.contract.parent, build.get("environment_inputs"))
    toolchain = apple_toolchain()
    if build["sources"] != native_sources(repo) or build["toolchain"] != toolchain or build["runner_sha256"] != sha256(args.runner):
        raise ValueError("Native verifier changed; run the recipe setup again")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ready.json").write_text(json.dumps(
        {"ready": True, "contract": contract, "toolchain": toolchain, "native_sources": build["sources"]},
        indent=2) + "\n")


if __name__ == "__main__":
    main()
