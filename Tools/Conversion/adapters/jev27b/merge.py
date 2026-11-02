import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import shutil
import sys
import time

from ...storage import read_json, write_json
from ..pinned import sha256
from .prepare import ADAPTER_KEY, TOKENIZER_FILES, check_adapter, check_integrity, load_contract

SOURCE_PREFIX = "model."
SAVED_PREFIX = "model.language_model."
HEAD = "lm_head.weight"
FORMULA = "W' = bf16(fp32(W) + scale * (fp32(B) @ fp32(A)))"
METADATA = {"format": "pt"}
PACKAGES = ("torch", "transformers", "peft", "safetensors")
EXAMPLE_LAYERS = (0, 3)


def lora_pairs(keys):
    pairs = {}
    for key in sorted(keys):
        match = ADAPTER_KEY.fullmatch(key)
        if match is None:
            raise ValueError(f"Not a LoRA tensor: {key}")
        pairs.setdefault(f"{match[1]}.weight", {})[match[2]] = key
    if any(set(pair) != {"A", "B"} for pair in pairs.values()):
        raise ValueError("Every LoRA projection needs one A and one B tensor")
    return {name: (pair["A"], pair["B"]) for name, pair in pairs.items()}


def saved_name(name):
    if name == HEAD:
        return name
    if not name.startswith(SOURCE_PREFIX) or name.startswith(SAVED_PREFIX):
        raise ValueError(f"Unexpected backbone tensor: {name}")
    return SAVED_PREFIX + name[len(SOURCE_PREFIX):]


def targeted(names, targets):
    return sorted(name for name in names if name.endswith(".weight") and name.split(".")[-2] in targets)


def merge_projection(weight, a, b, scale):
    return (weight.float() + scale * (b.float() @ a.float())).to(weight.dtype)


def saved_index(weight_map, total_size):
    return {"metadata": {"total_size": total_size},
            "weight_map": dict(sorted((saved_name(name), shard) for name, shard in weight_map.items()))}


def manifest_files(root):
    files = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in Path(root).rglob("*") if path.is_file()]
    return sorted(files, key=lambda entry: entry["path"])


def merged_revision(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:40]


def examples(pairs, layer_types):
    return {str(layer): {"layer_type": layer_types[layer],
                         "pairs": [{"lora_A": first, "lora_B": second, "backbone": name, "saved": saved_name(name)}
                                   for name, (first, second) in pairs.items() if name.startswith(f"model.layers.{layer}.")]}
            for layer in EXAMPLE_LAYERS}


def merge(snapshot, checkpoint, weight_map, pairs, lora, scale):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    consumed, source, saved = Counter(), Counter(), Counter()
    total = 0
    with torch.no_grad():
        for shard in sorted(set(weight_map.values())):
            names = sorted(name for name, owner in weight_map.items() if owner == shard)
            tensors = {}
            with safe_open(str(snapshot / shard), framework="pt") as handle:
                if sorted(handle.keys()) != names:
                    raise ValueError(f"Shard differs from the index: {shard}")
                for name in names:
                    tensor = handle.get_tensor(name)
                    source[str(tensor.dtype)] += 1
                    if name in pairs:
                        first, second = pairs[name]
                        tensor = merge_projection(tensor, lora[first], lora[second], scale)
                        consumed[name] += 1
                    saved[str(tensor.dtype)] += 1
                    total += tensor.numel() * tensor.element_size()
                    tensors[saved_name(name)] = tensor
            save_file(tensors, str(checkpoint / shard), metadata=METADATA)
            print(shard, len(names), flush=True)
    return consumed, total, {"source": dict(source), "saved": dict(saved)}


def main():
    from safetensors.torch import load_file

    parser = argparse.ArgumentParser(description="Merge the JEV-27B System 1 LoRA into its text backbone.")
    for name in ("model", "snapshot", "base", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    checkpoint = args.output / "checkpoint"
    checkpoint.mkdir(parents=True)
    contract = load_contract(args.model)
    snapshot = args.snapshot.resolve()
    check_integrity(snapshot, contract["model_id"], contract["revision"])
    adapter = check_adapter(snapshot, contract)
    scale = contract["lora"]["scale"]
    if adapter["scale"] != scale:
        raise ValueError("The adapter scale differs from the contract")
    index = read_json(snapshot / "model.safetensors.index.json")
    lora = load_file(str(snapshot / "adapter/adapter_model.safetensors"))
    pairs = lora_pairs(lora)
    consumed, total, dtypes = merge(snapshot, checkpoint, index["weight_map"], pairs, lora, scale)
    if set(consumed) != set(pairs) or set(consumed.values()) != {1} or len(pairs) != contract["lora"]["pairs"]:
        raise ValueError("Every LoRA pair must be merged exactly once")
    if total != index["metadata"]["total_size"]:
        raise ValueError("The merged shards differ in size from the source index")
    write_json(checkpoint / "model.safetensors.index.json", saved_index(index["weight_map"], total))
    shutil.copyfile(args.base / "config.json", checkpoint / "config.json")
    for name in TOKENIZER_FILES:
        shutil.copyfile(snapshot / name, checkpoint / name)
    files = manifest_files(checkpoint)
    revision = merged_revision(files)
    write_json(args.output / "manifest.json", {"merged_revision": revision, "files": files})
    record = read_json(Path(f"{snapshot}.integrity.json"))
    shards = set(index["weight_map"].values())
    write_json(args.output / "provenance.json", {
        "merged_revision": revision,
        "source": {"model_id": contract["model_id"], "revision": contract["revision"], "base_model": contract["base_model"],
                   "shards_sha256": {entry["path"]: entry["actual_sha256"] for entry in record["files"] if entry["path"] in shards},
                   "shards_sha256_from": {"record": f"{snapshot.name}.integrity.json", "checked_at": record["checked_at"]},
                   "adapter_model_sha256": sha256(snapshot / "adapter/adapter_model.safetensors"),
                   "adapter_config_sha256": sha256(snapshot / "adapter/adapter_config.json"),
                   "head_sha256": sha256(snapshot / contract["head"]),
                   "base_config_sha256": sha256(args.base / "config.json"),
                   "tokenizer_sha256": {name: sha256(snapshot / name) for name in TOKENIZER_FILES}},
        "formula": FORMULA, "scale": scale,
        "adapter": {key: adapter[key] for key in ("r", "alpha", "use_rslora", "use_dora")},
        "dtypes": {**dtypes, "adapter": dict(Counter(str(tensor.dtype) for tensor in lora.values())), "compute": "torch.float32"},
        "pairs": {"total": len(pairs), "per_target": adapter["per_target"]},
        "tensors": {"total": len(index["weight_map"]), "merged": len(consumed), "copied": len(index["weight_map"]) - len(consumed)},
        "examples": examples(pairs, read_json(snapshot / "config.json")["layer_types"]),
        "elapsed_s": round(time.perf_counter() - started, 1),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "versions": {name: importlib.metadata.version(name) for name in PACKAGES}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
