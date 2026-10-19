import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from ...storage import file_hash
from .inputs import EXECUTION_SHAPES


def validate_source(snapshot, contract):
    snapshot = Path(snapshot).resolve(strict=True)
    if snapshot.name != contract["revision"]:
        raise ValueError("Use the pinned source snapshot directory")
    expected_files = set(contract["files"])
    for path in snapshot.rglob("*"):
        relative = path.relative_to(snapshot)
        if path.is_symlink() and not path.resolve().is_relative_to(snapshot):
            raise ValueError(f"Source link escapes the snapshot: {relative}")
        if path.is_file() and relative.parts[:2] != (".cache", "huggingface") and str(relative) not in expected_files:
            raise ValueError(f"Unmanaged source file: {relative}")
    for name, expected in contract["files"].items():
        path = snapshot / name
        if not path.resolve().is_relative_to(snapshot) or file_hash(path) != expected:
            raise ValueError(f"Source identity mismatch: {name}")
    config = json.loads((snapshot / "config.json").read_text())
    if (config.get("model_type"), config.get("tts_model_type"), config.get("tts_model_size")) != ("qwen3_tts", "custom_voice", "0b6"):
        raise ValueError("This profile requires Qwen3-TTS 0.6B CustomVoice")
    return snapshot


def load_model(snapshot, contract_path, upstream):
    from qwen_tts import Qwen3TTSModel
    import qwen_tts
    from safetensors.torch import load_file

    contract = json.loads(Path(contract_path).read_text())
    if contract["version"] != 1 or contract["precision"] != "float32" or contract.get("execution_shapes") != EXECUTION_SHAPES:
        raise ValueError("Only the fixed FP32 research profile is supported")
    snapshot = validate_source(snapshot, contract)
    upstream = Path(upstream).resolve(strict=True)
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if revision != contract["upstream_revision"] or dirty or not Path(qwen_tts.__file__).resolve().is_relative_to(upstream):
        raise ValueError("Use the clean, pinned Qwen3-TTS checkout")
    torch.set_num_threads(4)
    wrapper = Qwen3TTSModel.from_pretrained(str(snapshot), device_map="cpu", dtype=torch.float32,
                                           attn_implementation="eager", local_files_only=True)
    wrapper.model.load_state_dict(load_file(snapshot / "model.safetensors"), strict=True)
    codec = wrapper.model.speech_tokenizer.model
    codec.load_state_dict(load_file(snapshot / "speech_tokenizer/model.safetensors"), strict=True)
    for model in (wrapper.model, codec):
        model.eval().requires_grad_(False)
        for module in model.modules():
            if hasattr(module, "config"):
                module.config._attn_implementation = "eager"
    codec.decoder.pre_transformer.config.use_cache = False
    return wrapper, contract


def generation_settings(specification):
    sampling = specification.get("sampling", {})
    return {"speaker": specification["speaker"], "language": "Japanese", "non_streaming_mode": True,
            "do_sample": sampling.get("enabled", False), "subtalker_dosample": sampling.get("enabled", False),
            "repetition_penalty": sampling.get("repetition_penalty", 1.0),
            "top_k": sampling.get("top_k", 50), "subtalker_top_k": sampling.get("top_k", 50),
            "top_p": sampling.get("top_p", 1.0), "subtalker_top_p": sampling.get("top_p", 1.0),
            "temperature": sampling.get("temperature", 0.9), "subtalker_temperature": sampling.get("temperature", 0.9),
            "max_new_tokens": specification["max_new_tokens"]}


def array(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    value = np.asarray(tensor)
    return value.astype(np.float32 if np.issubdtype(value.dtype, np.floating) or value.dtype == np.bool_ else np.int32)


def canonical_positions(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if not np.isfinite(value).all() or np.any(value < 0) or np.any(value >= 32768) or np.any(value != np.floor(value)):
        raise ValueError("Position IDs must be exact integers within the source range")
    return value.astype(np.int32)


def compare_tensor(reference, actual, limit=0.001):
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape or not reference.size or not np.isfinite(reference).all():
        raise ValueError("The reference must be finite, nonempty and have the output shape")
    if not np.isfinite(actual).all():
        return {"passed": False, "finite": False}
    difference = reference - actual
    maximum = float(np.max(np.abs(difference)))
    normalized = maximum / max(float(np.max(np.abs(reference))), 1e-8)
    norm = float(np.linalg.norm(reference.ravel()) * np.linalg.norm(actual.ravel()))
    return {"passed": normalized <= limit, "finite": True, "max_abs": maximum,
            "normalized_max": normalized, "rms": float(np.sqrt(np.mean(difference**2))),
            "cosine": float(np.dot(reference.ravel(), actual.ravel()) / norm) if norm else None}


def add_source_arguments(parser):
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)


def model_from_arguments(args):
    return load_model(args.snapshot, args.contract, args.upstream)
