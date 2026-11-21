import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import tomllib

from ...storage import canonical, read_json, write_json
from ..jev27b.prepare import tokenizer_sections
from ..pinned import apple_toolchain, sha256

HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
CONTRACT = {"model_id", "revision", "model_type", "architectures", "files", "config", "tokenizer", "reference",
            "upstream_revision"}
CONFIG = {"num_hidden_layers", "layer_pattern", "hidden_size", "intermediate_size", "vocab_size", "sliding_window",
          "num_key_value_heads", "num_global_key_value_heads", "head_dim", "global_head_dim", "attention_k_eq_v",
          "final_logit_softcapping", "tie_word_embeddings", "bos_token_id", "eos_token_id"}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
SPECIAL_TOKENS = ("bos_token", "eos_token", "eot_token", "pad_token")
REFERENCE = {"weights": "bfloat16", "compute": "bfloat16", "device": "mps"}
ENVIRONMENT_FILES = ("pyproject.toml", "uv.lock")
PACKAGES = ("huggingface-hub", "jinja2", "numpy", "safetensors", "tokenizers", "torch", "transformers")
DOWNLOAD_STATE = ".cache"


def load_contract(path):
    contract = read_json(path)
    if set(contract) != CONTRACT or set(contract["config"]) != CONFIG \
            or set(contract["tokenizer"]) != {"policy", "files"}:
        raise ValueError("Model contract has missing or unknown fields")
    if not REPOSITORY.fullmatch(contract["model_id"]) \
            or not all(HEX40.fullmatch(contract[key]) for key in ("revision", "upstream_revision")):
        raise ValueError("Expected an owner/name ID and full commit revisions")
    files = contract["files"]
    if not files or any(Path(name).is_absolute() or ".." in Path(name).parts or Path(name).parts[0] == DOWNLOAD_STATE
                        or set(entry) != {"bytes", "sha256"} or type(entry["bytes"]) is not int or entry["bytes"] <= 0
                        or not HEX64.fullmatch(entry["sha256"]) for name, entry in files.items()):
        raise ValueError("Every pinned file needs a relative path, its size and its SHA-256")
    tokenizer = contract["tokenizer"]
    if tokenizer["policy"] != "keep-distributed" \
            or tokenizer["files"] != {name: files.get(name, {}).get("sha256") for name in TOKENIZER_FILES}:
        raise ValueError("The tokenizer must be the distributed files pinned in files")
    config = contract["config"]
    pattern, layers, stops = config["layer_pattern"], config["num_hidden_layers"], config["eos_token_id"]
    if not pattern or type(layers) is not int or layers % len(pattern) or not stops or len(set(stops)) != len(stops):
        raise ValueError("The layer pattern and the stop tokens must be well formed")
    if contract["reference"] != REFERENCE:
        raise ValueError("This adapter covers the BF16 MPS reference only")
    return contract


def check_record(snapshot, contract):
    snapshot = Path(snapshot)
    record = read_json(Path(f"{snapshot}.integrity.json"))
    files = contract["files"]
    rows = record.get("files") or []
    entries = {row.get("path"): row for row in rows}
    outside = record.get("local_files_not_in_scope")
    if record.get("repo_id") != contract["model_id"] or record.get("revision") != contract["revision"] \
            or set(entries) != set(files) or len(entries) != len(rows) or record.get("files_checked") != len(files) \
            or record.get("ok_true") != len(files) or record.get("ok_false") != 0 or not isinstance(outside, list) \
            or any(not isinstance(name, str) or Path(name).parts[:1] != (DOWNLOAD_STATE,) for name in outside):
        raise ValueError(f"Incomplete integrity record or another source: {snapshot.name}")
    for name, pinned in files.items():
        row = entries[name]
        if row.get("ok") is not True or row.get("expected_size") != pinned["bytes"] \
                or row.get("actual_size") != pinned["bytes"] or row.get("actual_sha256") != pinned["sha256"]:
            raise ValueError(f"The integrity record differs from the contract: {name}")
    present = {path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*")
               if (path.is_symlink() or not path.is_dir()) and path.name != ".DS_Store"
               and path.relative_to(snapshot).parts[0] != DOWNLOAD_STATE}
    if present != set(files):
        raise ValueError(f"Snapshot files differ from the contract: missing {sorted(set(files) - present)}, "
                         f"unexpected {sorted(present - set(files))}")
    for name, pinned in files.items():
        path = snapshot / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != pinned["bytes"]:
            raise ValueError(f"Missing, linked or resized file: {name}")
    return {"repo_id": record["repo_id"], "revision": record["revision"], "checked_at": record.get("checked_at"),
            "ok_definition": record.get("ok_definition"), "files": len(files),
            "bytes": sum(pinned["bytes"] for pinned in files.values()), "download_state_files": len(outside)}


def check_hashes(snapshot, contract):
    observed = {}
    for name, pinned in sorted(contract["files"].items()):
        digest = sha256(Path(snapshot) / name)
        if digest != pinned["sha256"]:
            raise ValueError(f"SHA-256 differs from the contract: {name}")
        observed[name] = {"bytes": pinned["bytes"], "sha256": digest}
    return observed


def identifiers(value):
    return value if isinstance(value, list) else [value]


def check_config(snapshot, contract):
    config = read_json(Path(snapshot) / "config.json")
    text = config.get("text_config") or {}
    generation = read_json(Path(snapshot) / "generation_config.json")
    expected = dict(contract["config"])
    pattern = expected.pop("layer_pattern")
    stops = expected.pop("eos_token_id")
    expected["layer_types"] = pattern * (expected["num_hidden_layers"] // len(pattern))
    observed = {key: text.get(key) for key in expected}
    different = [key for key in sorted(expected) if canonical(observed[key]) != canonical(expected[key])]
    top = {"model_type": (config.get("model_type"), contract["model_type"]),
           "architectures": (config.get("architectures"), contract["architectures"]),
           "dtype": (text.get("dtype"), contract["reference"]["weights"]),
           "generation_config.bos_token_id": (generation.get("bos_token_id"), expected["bos_token_id"]),
           "generation_config.eos_token_id": (generation.get("eos_token_id"), stops)}
    different += [key for key, (found, wanted) in top.items() if canonical(found) != canonical(wanted)]
    if not set(identifiers(config.get("eos_token_id"))) <= set(stops):
        different.append("config.eos_token_id")
    if not set(identifiers(text.get("eos_token_id"))) <= set(stops):
        different.append("text_config.eos_token_id")
    if different:
        raise ValueError(f"config.json or generation_config.json differs from the contract: {different}")
    return {"model_type": config["model_type"], "text_model_type": text.get("model_type"),
            "architectures": config["architectures"], "dtype": text["dtype"],
            **{key: observed[key] for key in sorted(expected) if key != "layer_types"},
            "layer_types": {name: observed["layer_types"].count(name) for name in sorted(set(pattern))},
            "eos_token_id": {"generation_config": generation["eos_token_id"], "config": config.get("eos_token_id"),
                             "text_config": text.get("eos_token_id")}}


def check_special_tokens(snapshot, contract):
    data = read_json(Path(snapshot) / "tokenizer.json")
    settings = read_json(Path(snapshot) / "tokenizer_config.json")
    added = {token["content"]: token["id"] for token in data["added_tokens"]}
    vocabulary = data["model"]["vocab"]
    names = {key: settings.get(key) for key in SPECIAL_TOKENS}
    found = {key: added.get(text, vocabulary.get(text)) for key, text in names.items()}
    config = contract["config"]
    largest = max([*vocabulary.values(), *added.values()])
    if found["bos_token"] != config["bos_token_id"] or found["eos_token"] not in config["eos_token_id"] \
            or found["eot_token"] not in config["eos_token_id"] or largest >= config["vocab_size"]:
        raise ValueError(f"Special tokens differ from the contract: {found}")
    text = {identifier: content for content, identifier in added.items()}
    return {"special": {key: {"text": names[key], "id": found[key]} for key in SPECIAL_TOKENS},
            "stop_tokens": {str(identifier): text.get(identifier) for identifier in config["eos_token_id"]},
            "added_tokens": len(added), "vocabulary": len(vocabulary), "largest_id": largest,
            "post_processor": data.get("post_processor")}


def runtime_tokenizer(snapshot):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot))
    different = tokenizer_sections(json.loads(tokenizer.backend_tokenizer.to_str()),
                                   read_json(Path(snapshot) / "tokenizer.json"))
    if different:
        raise ValueError(f"The reference tokenizer differs from the distributed tokenizer.json: {different}")
    return tokenizer


def environment(recipe):
    import torch

    project = tomllib.loads((Path(recipe) / "pyproject.toml").read_text())
    pins = dict(dependency.split("==") for dependency in project["project"]["dependencies"])
    installed = {name: importlib.metadata.version(name) for name in sorted(set(PACKAGES) | set(pins))}
    if any(installed[name] != version for name, version in pins.items()):
        raise ValueError(f"Use the recipe environment from its lock: {installed}")
    if not torch.backends.mps.is_available():
        raise ValueError("The BF16 reference requires MPS")
    return {"python": platform.python_version(), "packages": installed, "mps_available": True,
            "apple_toolchain": apple_toolchain(), "platform": platform.platform(),
            "os_build": subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip()}


def check_setup(setup, recipe):
    inputs = {name: sha256(Path(recipe) / name) for name in ENVIRONMENT_FILES}
    if setup.get("environment_inputs") != inputs:
        raise ValueError("The environment was built from other recipe inputs; run setup again")
    return setup


def main():
    parser = argparse.ArgumentParser(description="Check the pinned Gemma 4 31B snapshot and the reference environment.")
    for name in ("model", "snapshot", "setup", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    output = args.output / "ready.json"
    if output.exists():
        raise FileExistsError("ready.json must be written to a new directory")
    contract = load_contract(args.model)
    record = check_record(args.snapshot, contract)
    config = check_config(args.snapshot, contract)
    tokens = check_special_tokens(args.snapshot, contract)
    files = check_hashes(args.snapshot, contract)
    tokenizer = runtime_tokenizer(args.snapshot)
    tokens["runtime"] = {"class": type(tokenizer).__name__, "uses_distributed_file": True}
    versions = environment(args.model.parent)
    setup = check_setup(read_json(args.setup), args.model.parent)
    write_json(output, {"ready": True, "contract": contract, "contract_sha256": sha256(args.model), "integrity": record,
                        "files": files, "config": config, "tokenizer": tokens, "environment": versions, "setup": setup})


if __name__ == "__main__":
    sys.exit(main())
