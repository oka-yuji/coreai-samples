from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import subprocess


def load_contract(path, snapshot):
    contract = json.loads(Path(path).read_text())
    expected = {"model_id", "revision", "model_type", "precision", "compression", "context_length",
                "output_name", "upstream_revision"}
    if set(contract) != expected:
        raise ValueError("Model contract has missing or unknown fields")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", contract["model_id"]):
        raise ValueError("Expected an owner/model Hugging Face ID")
    if not re.fullmatch(r"[0-9a-f]{40}", contract["revision"]):
        raise ValueError("A full source revision is required")
    snapshot = Path(snapshot).resolve()
    if snapshot.name != contract["revision"]:
        raise ValueError("Use the immutable snapshots/<revision> directory")
    config = json.loads((snapshot / "config.json").read_text())
    if config.get("model_type") != contract["model_type"]:
        raise ValueError("Snapshot architecture does not match the recipe")
    if contract["model_type"] != "qwen3":
        raise ValueError("This adapter is validated for Qwen3 only; add a model-specific adapter")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (snapshot / name).is_file():
            raise ValueError(f"Snapshot is incomplete: {name}")
    if not list(snapshot.glob("*.safetensors")):
        raise ValueError("No checkpoint weights in the snapshot")
    if contract["precision"] not in {"float16", "bfloat16", "float32"}:
        raise ValueError("Unsupported compute precision")
    if contract["compression"] != "none":
        raise ValueError("This reference recipe covers unquantized exports only")
    if type(contract["context_length"]) is not int or contract["context_length"] <= 0:
        raise ValueError("context_length must be positive")
    if not re.fullmatch(r"[a-z][a-z0-9_]+", contract["output_name"]):
        raise ValueError("Invalid bundle name")
    return contract, snapshot


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def check_upstream(upstream, contract):
    upstream = Path(upstream).resolve()
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != contract["upstream_revision"]:
        raise ValueError(f"Unexpected exporter revision: {revision}")
    return revision


def apple_toolchain():
    def output(*command):
        return subprocess.check_output(command, text=True, timeout=30).strip()
    compiler = Path(output("xcrun", "--find", "coreai-build"))
    return {"xcode": output("xcodebuild", "-version"),
            "macos_sdk": output("xcrun", "--sdk", "macosx", "--show-sdk-version"),
            "coreai_build": str(compiler), "coreai_build_sha256": sha256(compiler)}


@contextmanager
def pinned_hub(snapshot, contract):
    variables = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {key: os.environ.get(key) for key in variables}
    with tempfile.TemporaryDirectory(prefix="coreai-pinned-hub-") as folder:
        home = Path(folder)
        hub = home / "hub"
        repository = hub / ("models--" + contract["model_id"].replace("/", "--"))
        target = repository / "snapshots" / contract["revision"]
        target.mkdir(parents=True)
        for source in sorted(snapshot.rglob("*")):
            if source.is_file():
                destination = target / source.relative_to(snapshot)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(source.resolve(strict=True))
            elif source.is_symlink():
                raise ValueError(f"Invalid snapshot link: {source}")
        (repository / "refs").mkdir()
        (repository / "refs" / "main").write_text(contract["revision"])
        os.environ.update({"HF_HOME": str(home), "HF_HUB_CACHE": str(hub), "HUGGINGFACE_HUB_CACHE": str(hub),
                           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        try:
            yield hub
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check pinned inputs before any model job.")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    contract, snapshot = load_contract(args.contract, args.snapshot)
    revision = check_upstream(args.upstream, contract)
    toolchain = apple_toolchain()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ready.json").write_text(json.dumps(
        {"ready": True, "contract": contract, "snapshot": str(snapshot), "upstream_revision": revision,
         "apple_toolchain": toolchain},
        indent=2) + "\n")
