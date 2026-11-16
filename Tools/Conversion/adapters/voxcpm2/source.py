import copy
from pathlib import Path
import re
import subprocess

import numpy as np

from ...storage import file_hash, read_json


def expected_config(contract):
    architecture = contract["architecture"]
    return {
        ("architecture",): "voxcpm2",
        ("lm_config", "hidden_size"): architecture["hidden_size"],
        ("lm_config", "num_hidden_layers"): architecture["tslm_layers"],
        ("lm_config", "num_key_value_heads"): architecture["kv_heads"],
        ("lm_config", "kv_channels"): architecture["head_dim"],
        ("lm_config", "vocab_size"): architecture["vocab_size"],
        ("lm_config", "use_mup"): False,
        ("residual_lm_num_layers",): architecture["ralm_layers"],
        ("residual_lm_no_rope",): True,
        ("patch_size",): architecture["patch_size"],
        ("feat_dim",): architecture["feat_dim"],
        ("audio_vae_config", "out_sample_rate"): architecture["sample_rate"],
    }


def load_contract(path):
    contract = read_json(path)
    if contract.get("version") != 1 or contract.get("model_id") != "openbmb/VoxCPM2":
        raise ValueError("Use the pinned VoxCPM2 contract")
    for key in ("revision", "upstream_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", contract.get(key, "")):
            raise ValueError(f"A full {key} is required")
    return contract


def with_capacity(contract, capacity):
    contract = copy.deepcopy(contract)
    contract["execution"]["capacity"] = capacity
    for asset in ("tslm", "ralm"):
        for function in contract["assets"][asset]["functions"].values():
            for entry in function["states"]:
                entry[2][2] = capacity
    return contract


def validate_source(snapshot, contract):
    snapshot = Path(snapshot).resolve(strict=True)
    if snapshot.name != contract["revision"]:
        raise ValueError("Use the pinned source snapshot directory")
    for path in snapshot.rglob("*"):
        relative = path.relative_to(snapshot)
        if path.is_symlink() and not path.resolve().is_relative_to(snapshot):
            raise ValueError(f"Source link escapes the snapshot: {relative}")
        if path.is_file() and relative.parts[:2] != (".cache", "huggingface") and str(relative) not in contract["files"]:
            raise ValueError(f"Unmanaged source file: {relative}")
    for name, expected in contract["files"].items():
        path = snapshot / name
        if not path.resolve().is_relative_to(snapshot) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Source identity mismatch: {name}")
    config = read_json(snapshot / "config.json")
    for keys, expected in expected_config(contract).items():
        value = config
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        if value != expected or type(value) is not type(expected):
            raise ValueError(f"The model configuration does not match the contract: {'.'.join(keys)}")
    return snapshot


def check_upstream(upstream, contract):
    upstream = Path(upstream).resolve(strict=True)
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip()
    import voxcpm

    if revision != contract["upstream_revision"] or dirty or not Path(voxcpm.__file__).resolve().is_relative_to(upstream):
        raise ValueError("Use the clean, pinned VoxCPM checkout")
    return upstream


def load_model(snapshot, upstream, contract):
    import torch
    from safetensors.torch import load_file
    from transformers import LlamaTokenizerFast
    from voxcpm.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig
    from voxcpm.modules.audiovae import AudioVAEV2

    snapshot = validate_source(snapshot, contract)
    check_upstream(upstream, contract)
    torch.set_num_threads(4)
    config = VoxCPMConfig.model_validate_json((snapshot / "config.json").read_text())
    config.dtype = "float32"
    tokenizer = LlamaTokenizerFast.from_pretrained(str(snapshot), local_files_only=True)
    model = VoxCPM2Model(config, tokenizer, AudioVAEV2(config=config.audio_vae_config), device="cpu")
    state = load_file(snapshot / "model.safetensors")
    vae = torch.load(snapshot / "audiovae.pth", map_location="cpu", weights_only=True)
    state.update({f"audio_vae.{key}": value for key, value in vae.get("state_dict", vae).items()})
    model.load_state_dict(state, strict=True)
    model = model.to(torch.float32).eval().requires_grad_(False)
    if any(value.dtype != torch.float32 for value in model.state_dict().values() if value.is_floating_point()):
        raise ValueError("The reference must run in float32")
    return model


def compare(reference, actual, limit):
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape or not reference.size or not np.isfinite(reference).all():
        raise ValueError("The reference must be finite, nonempty and have the output shape")
    nonfinite = int((~np.isfinite(actual)).sum())
    if nonfinite:
        return {"passed": False, "nonfinite": nonfinite}
    difference = np.abs(reference - actual)
    peak = float(np.max(np.abs(reference)))
    normalized = float(np.max(difference)) / max(peak, 1e-8)
    return {"passed": normalized <= limit, "nonfinite": 0, "normalized_max": normalized,
            "max_abs": float(np.max(difference)), "reference_max_abs": peak}


def summarize(rows):
    measured = [row for row in rows if "normalized_max" in row]
    return {"count": len(rows), "passed": sum(bool(row["passed"]) for row in rows),
            "nonfinite": sum(row["nonfinite"] for row in rows),
            "normalized_max": max((row["normalized_max"] for row in measured), default=None),
            "max_abs": max((row["max_abs"] for row in measured), default=None)}


def add_source_arguments(parser):
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
