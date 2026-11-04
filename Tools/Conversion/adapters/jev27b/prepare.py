import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import tomllib

from ...storage import read_json, write_json
from ..pinned import apple_toolchain, sha256
from .typed_head import BANDS, FIXED_OPTIONS, LABELS, SLOTS

HEX40 = re.compile(r"[0-9a-f]{40}")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
CONTRACT = {"model_id", "revision", "model_type", "base_model", "merged_model_id", "lora", "head", "template_version",
            "precision", "compression", "context_length", "limits", "tokenizer", "dataset", "upstream_revision"}
LORA = {"r", "alpha", "scale", "use_rslora", "use_dora", "pairs", "targets"}
ADAPTER_KEY = re.compile(r"base_model\.model\.(.+)\.lora_([AB])\.weight")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
REFERENCE_PACKAGES = ("accelerate", "huggingface-hub", "numpy", "peft", "safetensors", "tokenizers", "torch", "transformers")
UPSTREAM_PACKAGES = ("coreai-models", "coreai-torch", "coreai-opt", "coreai-core", "torch", "transformers")
CONTENT_LIMIT = 32 * 1024 * 1024
RECORDED_DIGESTS = {"lfs-sha256": ("expected_sha256", "actual_sha256"),
                    "git-blob-sha1": ("expected_git_blob_sha1", "actual_git_blob_sha1")}
ENVIRONMENT_FILES = ("pyproject.toml", "uv.lock")
VERSION_PROBE = """import importlib.metadata as m, json, sys
found = {}
for name in sys.argv[1:]:
    try:
        found[name] = m.version(name)
    except m.PackageNotFoundError:
        found[name] = None
print(json.dumps(found))"""


def load_contract(path):
    contract = read_json(path)
    if set(contract) != CONTRACT or set(contract["lora"]) != LORA \
            or set(contract["base_model"]) != {"id", "revision"} or set(contract["dataset"]) != {"id", "revision"}:
        raise ValueError("Model contract has missing or unknown fields")
    identifiers = (contract["model_id"], contract["merged_model_id"], contract["base_model"]["id"], contract["dataset"]["id"])
    revisions = (contract["revision"], contract["base_model"]["revision"], contract["dataset"]["revision"],
                 contract["upstream_revision"])
    if not all(REPOSITORY.fullmatch(value) for value in identifiers) or not all(HEX40.fullmatch(value) for value in revisions):
        raise ValueError("Expected owner/name IDs and full commit revisions")
    lora = contract["lora"]
    if lora["use_rslora"] is not False or lora["use_dora"] is not False or type(lora["r"]) is not int or lora["r"] <= 0 \
            or type(lora["alpha"]) is not int or lora["scale"] != lora["alpha"] / lora["r"] \
            or len(set(lora["targets"])) != len(lora["targets"]):
        raise ValueError("The contract covers plain LoRA scaled by alpha / r")
    if contract["limits"] != {"options": len(LABELS), "input_tokens": BANDS[-1][1]}:
        raise ValueError("Limits must match the decision head and the evaluated bands")
    if contract["tokenizer"].get("policy") != "keep-distributed":
        raise ValueError("Only the distributed tokenizer is supported")
    return contract


def git_blob(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def check_integrity(directory, repository, revision):
    directory = Path(directory)
    record = read_json(Path(f"{directory}.integrity.json"))
    files = record.get("files")
    if record.get("repo_id") != repository or record.get("revision") != revision or not files \
            or record.get("files_checked") != len(files) or record.get("ok_true") != len(files) \
            or record.get("ok_false") != 0 or record.get("local_files_not_in_scope") \
            or any(entry.get("ok") is not True for entry in files):
        raise ValueError(f"Incomplete integrity record or another source: {directory.name}")
    present = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
               if (path.is_symlink() or not path.is_dir()) and path.name != ".DS_Store"
               and ".cache" not in path.relative_to(directory).parts}
    if present != {entry["path"] for entry in files}:
        raise ValueError(f"Files differ from the integrity record: {sorted(present ^ {entry['path'] for entry in files})}")
    checked = datetime.fromisoformat(record["checked_at"]).timestamp()
    total = verified = 0
    for entry in files:
        relative = Path(entry["path"])
        path = directory / relative
        if relative.is_absolute() or ".." in relative.parts or path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or unsafe file: {entry['path']}")
        names = RECORDED_DIGESTS.get(entry["check"])
        if names is None or entry.get(names[0]) is None or entry.get(names[0]) != entry.get(names[1]):
            raise ValueError(f"Integrity record entry is inconsistent: {entry['path']}")
        status = path.stat()
        if status.st_size != entry["expected_size"] or status.st_size != entry["actual_size"] or status.st_mtime > checked + 1:
            raise ValueError(f"File changed after its integrity check: {entry['path']}")
        total += status.st_size
        if status.st_size <= CONTENT_LIMIT:
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            observed = digest if entry["check"] == "lfs-sha256" else git_blob(data)
            if observed != entry[names[0]] or digest != entry["actual_sha256"]:
                raise ValueError(f"File content differs from its integrity record: {entry['path']}")
            verified += 1
    return {"revision": revision, "files": len(files), "bytes": total, "content_rehashed": verified,
            "checked_at": record["checked_at"], "checks": dict(sorted(Counter(entry["check"] for entry in files).items()))}


def safetensors_header(path):
    with Path(path).open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    return header


def check_config(snapshot, contract):
    config = read_json(snapshot / "config.json")
    expected = {"model_type": contract["model_type"], "dtype": contract["precision"],
                "max_position_embeddings": contract["context_length"]}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("config.json differs from the contract")
    names = list(read_json(snapshot / "model.safetensors.index.json")["weight_map"])
    return {"model_type": config["model_type"], "hidden_size": config["hidden_size"], "vocab_size": config["vocab_size"],
            "layers": dict(sorted(Counter(config["layer_types"]).items())), "tensors": len(names),
            "language_model_prefix": any(name.startswith("model.language_model.") for name in names),
            "mtp_declared": config.get("mtp_num_hidden_layers"), "mtp_tensors": sum("mtp" in name for name in names),
            "visual_tensors": sum("visual" in name for name in names)}, config


def check_adapter(snapshot, contract):
    adapter = read_json(snapshot / "adapter/adapter_config.json")
    lora = contract["lora"]
    expected = {"peft_type": "LORA", "r": lora["r"], "lora_alpha": lora["alpha"], "use_rslora": lora["use_rslora"],
                "use_dora": lora["use_dora"], "modules_to_save": None, "trainable_token_indices": None,
                "target_parameters": None, "layers_to_transform": None, "layers_pattern": None, "exclude_modules": None,
                "rank_pattern": {}, "alpha_pattern": {}, "bias": "none", "lora_bias": False, "fan_in_fan_out": False,
                "layer_replication": None, "use_qalora": False, "alora_invocation_tokens": None, "arrow_config": None,
                "use_bdlora": None, "velora_config": None, "monteclora_config": None, "kasa_config": None}
    differences = sorted(key for key, value in expected.items() if adapter.get(key, "missing") != value)
    if sorted(adapter.get("target_modules", [])) != sorted(lora["targets"]):
        differences.append("target_modules")
    if differences:
        raise ValueError(f"adapter_config.json differs from the contract: {differences}")
    pairs = {}
    for key, info in safetensors_header(snapshot / "adapter/adapter_model.safetensors").items():
        match = ADAPTER_KEY.fullmatch(key)
        if not match or info["dtype"] != "F32" or len(info["shape"]) != 2:
            raise ValueError(f"Unexpected adapter tensor: {key}")
        pairs.setdefault(match[1], {})[match[2]] = info["shape"]
    index = read_json(snapshot / "model.safetensors.index.json")["weight_map"]
    targeted = {name[:-len(".weight")] for name in index
                if name.endswith(".weight") and name.split(".")[-2] in lora["targets"]}
    if set(pairs) != targeted or len(pairs) != lora["pairs"] or any(
            set(shapes) != {"A", "B"} or shapes["A"][0] != lora["r"] or shapes["B"][1] != lora["r"]
            for shapes in pairs.values()):
        raise ValueError("Adapter pairs do not cover exactly the targeted backbone projections")
    return {"r": adapter["r"], "alpha": adapter["lora_alpha"], "scale": adapter["lora_alpha"] / adapter["r"],
            "use_rslora": adapter["use_rslora"], "use_dora": adapter["use_dora"],
            "targets": sorted(adapter["target_modules"]), "modules_to_save": adapter["modules_to_save"],
            "pairs": len(pairs), "per_target": dict(sorted(Counter(name.split(".")[-1] for name in pairs).items())),
            "dtype": "F32", "dropout": adapter.get("lora_dropout"), "peft_version": adapter.get("peft_version"),
            "base_model_name_or_path": adapter.get("base_model_name_or_path")}


def check_decision(snapshot, contract, config):
    judge = read_json(snapshot / "judge_config.json")
    slots = judge["slots"]
    verbalizers = [*FIXED_OPTIONS["noul"], *FIXED_OPTIONS["score"], *LABELS]
    vocab = read_json(snapshot / "tokenizer.json")["model"]["vocab"]
    hidden = config["hidden_size"]
    head = {name: (info["dtype"], info["shape"]) for name, info in safetensors_header(snapshot / contract["head"]).items()}
    calibration = read_json(snapshot / "calibration.json")
    temperatures = calibration["per_kind"]
    checks = {
        "ranges": slots["ranges"] == {kind: list(span) for kind, span in SLOTS.items()},
        "num_slots": slots["num_slots"] == SLOTS["choice"][1],
        "verbalizers": slots["verbalizers"] == verbalizers,
        "template_version": slots["template_version"] == contract["template_version"],
        "verbalizer_ids": judge["verbalizer_ids"] == [vocab.get(token) for token in verbalizers],
        "kinds": sorted(judge["kinds"]) == sorted(SLOTS),
        "hidden_size": judge["hidden_size"] == hidden,
        "unmerged_adapter": judge["weights_mode"] == "unmerged" and judge["adapter_subfolder"] == "adapter",
        "no_softcap": "softcap" in judge and judge["softcap"] is None,
        "adapter_summary": (judge["lora"]["r"], judge["lora"]["alpha"], sorted(judge["lora"]["target_modules"]))
        == (contract["lora"]["r"], contract["lora"]["alpha"], sorted(contract["lora"]["targets"])),
        "head": head == {"proj.weight": ("F32", [SLOTS["choice"][1], hidden]), "proj.bias": ("F32", [SLOTS["choice"][1]])},
        "temperatures": set(temperatures) == set(SLOTS) and all(
            isinstance(value, float) and math.isfinite(value) and value > 0 for value in temperatures.values()),
        "placeholder_rows_excluded": calibration.get("d1_excluded") is True,
        "per_kind_only": calibration.get("per_kind_family") == {},
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"Decision files differ from the contract: {failed}")
    return {"slots": slots["ranges"], "verbalizers": verbalizers, "verbalizer_ids": judge["verbalizer_ids"],
            "template_version": slots["template_version"], "temperatures": temperatures,
            "calibration_rows": calibration.get("fit_rows"), "head": {name: shape for name, (_, shape) in head.items()},
            "model_version": judge.get("model_version")}


def regexes(node):
    if isinstance(node, dict):
        own = [node["Regex"]] if isinstance(node.get("Regex"), str) else []
        return own + [found for value in node.values() for found in regexes(value)]
    if isinstance(node, list):
        return [found for value in node for found in regexes(value)]
    return []


def merges(model):
    return [" ".join(item) if isinstance(item, list) else item for item in model.get("merges", [])]


def tokenizer_sections(first, second):
    different = []
    for key in sorted(set(first) | set(second)):
        if key == "model":
            left, right = first.get(key, {}), second.get(key, {})
            for field in sorted(set(left) | set(right)):
                same = merges(left) == merges(right) if field == "merges" else left.get(field) == right.get(field)
                if not same:
                    different.append(f"model.{field}")
        elif first.get(key) != second.get(key):
            different.append(key)
    return different


def compare_tokenizers(snapshot, base):
    files = {}
    for name in TOKENIZER_FILES:
        left, right = (Path(root) / name for root in (snapshot, base))
        files[name] = {"jev": {"bytes": left.stat().st_size, "sha256": sha256(left)},
                       "base": {"bytes": right.stat().st_size, "sha256": sha256(right)}}
        files[name]["identical"] = files[name]["jev"]["sha256"] == files[name]["base"]["sha256"]
    jev, original = (read_json(Path(root) / "tokenizer.json") for root in (snapshot, base))
    different = tokenizer_sections(jev, original)
    files["tokenizer.json"]["different_sections"] = different
    files["tokenizer.json"]["same"] = {"vocab": "model.vocab" not in different, "merges": "model.merges" not in different,
                                       "added_tokens": "added_tokens" not in different,
                                       "normalizer": "normalizer" not in different}
    files["tokenizer.json"]["detail"] = {key: {"jev": jev.get(key), "base": original.get(key)}
                                         for key in different if key in ("pre_tokenizer", "decoder", "post_processor")}
    files["tokenizer.json"]["mark_class"] = {name: any("\\p{M}" in regex for regex in regexes(data.get("pre_tokenizer")))
                                             for name, data in (("jev", jev), ("base", original))}
    jev_config, base_config = (read_json(Path(root) / "tokenizer_config.json") for root in (snapshot, base))
    regex = jev_config.get("pretokenize_regex")
    files["tokenizer_config.json"].update({
        "only_jev": sorted(set(jev_config) - set(base_config)), "only_base": sorted(set(base_config) - set(jev_config)),
        "different_values": sorted(key for key in set(jev_config) & set(base_config) if jev_config[key] != base_config[key]),
        "pretokenize_regex": regex,
        "pretokenize_regex_matches_tokenizer_json": regex in regexes(jev.get("pre_tokenizer"))})
    return files


def distributed_tokenizer(snapshot):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot))
    different = tokenizer_sections(json.loads(tokenizer.backend_tokenizer.to_str()), read_json(Path(snapshot) / "tokenizer.json"))
    if different:
        raise ValueError(f"The reference tokenizer differs from the distributed tokenizer.json: {different}")
    return tokenizer


def check_setup(setup, recipe, revision, runner_sha256):
    inputs = {name: sha256(Path(recipe) / name) for name in ENVIRONMENT_FILES}
    if setup.get("environment_inputs") != inputs or setup.get("upstream_revision") != revision \
            or setup.get("llm_runner_sha256") != runner_sha256:
        raise ValueError("The environments were built from other recipe inputs or revisions; run setup again")
    return setup


def environment(recipe, upstream, runner):
    import torch

    project = tomllib.loads((recipe / "pyproject.toml").read_text())
    pins = dict(dependency.split("==") for dependency in project["project"]["dependencies"])
    installed = {name: importlib.metadata.version(name) for name in sorted(set(REFERENCE_PACKAGES) | set(pins))}
    if any(installed[name] != version for name, version in pins.items()):
        raise ValueError(f"Use the recipe environment from its lock: {installed}")

    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(upstream), *arguments], text=True, timeout=30).strip()

    packages = subprocess.check_output([str(upstream / ".venv/bin/python"), "-I", "-c", VERSION_PROBE, *UPSTREAM_PACKAGES],
                                       text=True, timeout=60)
    return {"python": platform.python_version(), "reference": installed, "mps_available": torch.backends.mps.is_available(),
            "upstream": {"revision": git("rev-parse", "HEAD"), "clean": not git("status", "--porcelain", "--untracked-files=no"),
                         "packages": json.loads(packages), "llm_runner_sha256": sha256(runner),
                         "llm_runner_bytes": Path(runner).stat().st_size},
            "apple_toolchain": apple_toolchain(), "platform": platform.platform(),
            "os_build": subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip()}


def main():
    parser = argparse.ArgumentParser(description="Check the pinned JEV-27B inputs and environments.")
    for name in ("model", "snapshot", "base", "upstream", "runner", "setup", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    output = args.output / "ready.json"
    if output.exists():
        raise FileExistsError("ready.json must be written to a new directory")
    contract = load_contract(args.model)
    snapshot = args.snapshot.resolve()
    integrity = {"snapshot": check_integrity(snapshot, contract["model_id"], contract["revision"]),
                 "base": check_integrity(args.base, contract["base_model"]["id"], contract["base_model"]["revision"])}
    summary, config = check_config(snapshot, contract)
    adapter = check_adapter(snapshot, contract)
    decision = check_decision(snapshot, contract, config)
    tokenizer = compare_tokenizers(snapshot, args.base)
    tokenizer["runtime"] = {"class": type(distributed_tokenizer(snapshot)).__name__, "uses_distributed_file": True}
    tokenizer["policy"] = contract["tokenizer"]
    versions = environment(args.model.parent, args.upstream, args.runner)
    if versions["upstream"]["revision"] != contract["upstream_revision"] or not versions["upstream"]["clean"]:
        raise ValueError("The converter checkout must be a clean copy of the contract revision")
    setup = check_setup(read_json(args.setup), args.model.parent, contract["upstream_revision"],
                        versions["upstream"]["llm_runner_sha256"])
    write_json(output, {"ready": True, "contract": contract, "config": summary, "integrity": integrity, "adapter": adapter,
                        "decision": decision, "tokenizer": tokenizer, "environment": versions, "setup": setup})


if __name__ == "__main__":
    sys.exit(main())
