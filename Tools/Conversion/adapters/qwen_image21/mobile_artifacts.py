import importlib.metadata
import json
from pathlib import Path

from .contracts import DIFFUSERS_REVISION, REVISION, digest
from .evaluation import canonical_hash
from .fixtures import file_record
from .mobile_profile import validate_mobile_profile


CONVERTER_FILES = ("mobile_artifacts.py", "mobile_convert.py", "mobile_models.py", "mobile_profile.py", "quantization.py",
                   "denoiser.py", "encoders.py", "profile.py", "contracts.py", "evaluation.py")


def converter_identity():
    root = Path(__file__).parent
    return {name: digest(root / name) for name in CONVERTER_FILES}


def validate_environment():
    expected = {"torch": "2.9.0", "transformers": "5.18.0", "diffusers": "0.41.0.dev0",
                "coreai-torch": "0.4.3", "coreai-core": "1.0.0b3", "coreai-opt": "0.3.0",
                "torchao": "0.15.0", "numpy": "2.4.6"}
    actual = {name: importlib.metadata.version(name) for name in expected}
    direct = json.loads(importlib.metadata.distribution("diffusers").read_text("direct_url.json") or "{}")
    if actual != expected or direct.get("vcs_info", {}).get("commit_id") != DIFFUSERS_REVISION:
        raise ValueError("Use the pinned mobile conversion environment")
    return actual


def validate_source(snapshot, kind=None):
    specification = json.loads((Path(__file__).parents[2] / "recipes/qwen_image21/model.json").read_text())
    if specification["revision"] != REVISION or specification["diffusers_revision"] != DIFFUSERS_REVISION:
        raise ValueError("The source specification changed")
    groups = {"text": "text_encoder/", "vision": "text_encoder/", "denoiser": "transformer/",
              "vae_encoder": "vae/", "vae_decoder": "vae/"}
    if kind is not None and kind not in groups:
        raise ValueError("Unknown source component")
    selected = {name: record for name, record in specification["files"].items()
                if kind is None or name.startswith(groups[kind]) or name.startswith(("processor/", "scheduler/"))
                or name in ("LICENSE", "model_index.json")}
    for name, expected in selected.items():
        path = Path(snapshot) / name
        if not path.is_file() or file_record(path) != expected:
            raise ValueError("Source content differs from the pinned revision: " + name)
    return selected


def read_profile(path, dependency_lock):
    profile = json.loads(Path(path).read_text())
    validate_mobile_profile(profile)
    if profile["source"]["converter_sha256"] != converter_identity():
        raise ValueError("The profile belongs to different converter code")
    if profile["source"]["dependency_lock_sha256"] != digest(dependency_lock):
        raise ValueError("The profile belongs to a different dependency lock")
    return profile


def inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("An asset must be a real directory")
    records = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("Asset entries must be ordinary files or directories")
        if path.is_file():
            records[path.relative_to(root).as_posix()] = file_record(path)
    if not records:
        raise ValueError("An asset must contain files")
    return records


def read_component(directory, profile, kind, *, check_assets=True):
    validate_mobile_profile(profile)
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Expected a real component directory")
    record = json.loads((directory / "component.json").read_text())
    specification = profile["components"][kind]
    if (record.get("format") != 2 or record.get("kind") != kind or canonical_hash(record.get("profile")) != canonical_hash(profile)
            or record.get("profile_sha256") != canonical_hash(profile) or record.get("conversion_complete") is not True
            or record.get("asset") != kind + ".aimodel" or record.get("inputs") != specification["inputs"]
            or record.get("outputs") != specification["outputs"]):
        raise ValueError("The component belongs to a different source, precision or physical contract")
    compression_path = directory / "compression.json"
    if file_record(compression_path) != record.get("compression_file"):
        raise ValueError("The quantized reference identity changed")
    compression = json.loads(compression_path.read_text())
    if compression.get("profile_sha256") != canonical_hash(profile) or compression.get("kind") != kind:
        raise ValueError("The weight reference belongs to another component or profile")
    if check_assets:
        if inventory(directory / record["asset"]) != record["asset_inventory"]:
            raise ValueError("The complete asset file set or its contents changed")
    return record, compression["compression"]
