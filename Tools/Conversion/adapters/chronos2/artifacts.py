import copy
import math
from pathlib import Path, PurePosixPath
import re
import struct

from ...storage import PipelineError, canonical, file_hash, read_json, write_json
from .contracts import MODEL_ID, QUANTILES, REVISION, compare_predictions, prepare_context, validate_predictions


CASE_IDS = (
    "trend-1", "trend-15", "trend-16", "trend-17", "trend-32", "trend-64", "trend-1008", "trend-8192",
    "constant-zero", "constant-positive", "periodic", "negative-scale", "missing-16", "missing-1008",
    "missing-8192", "single-observed", "large-counts", "leading-missing",
)
ABI = {
    "asset": "Model.aimodel", "entrypoint": "main",
    "inputs": [{"name": "context", "dtype": "float32", "shape": [1, -1],
                "min_length": 16, "max_length": 8192, "length_multiple": 16}],
    "outputs": [{"name": "quantiles", "dtype": "float32", "shape": [1, 21, 16]}], "state_names": [],
}
FORECAST_CONFIG = {
    "context_length": 8192, "input_patch_size": 16, "input_patch_stride": 16, "max_output_patches": 64,
    "output_patch_size": 16, "quantiles": list(QUANTILES), "time_encoding_scale": 8192,
    "use_arcsinh": True, "use_reg_token": True,
}
CORE_CONFIG = {
    "architectures": ["Chronos2Model"], "chronos_pipeline_class": "Chronos2Pipeline", "model_type": "t5",
    "d_ff": 3072, "d_kv": 64, "d_model": 768, "dense_act_fn": "relu", "dropout_rate": 0.1,
    "feed_forward_proj": "relu", "initializer_factor": 0.05, "is_gated_act": False,
    "layer_norm_epsilon": 1e-6, "num_heads": 12, "num_layers": 12, "pad_token_id": 0,
    "reg_token_id": 1, "rope_theta": 10000.0, "use_cache": True, "vocab_size": 2,
}


def require_equal(actual, expected, label):
    if canonical(actual) != canonical(expected):
        raise PipelineError(f"Unexpected {label}")


def relative_path(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
        raise PipelineError("Expected a safe relative artifact path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise PipelineError("Expected a safe relative artifact path")
    return Path(*path.parts)


def reject_links(path):
    path = Path(path).expanduser().absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise PipelineError("Symlink artifact or setup path is not supported")
    return path


def new_output(path):
    path = reject_links(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise PipelineError("Output must be a new or empty directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_new(path, value):
    path = reject_links(path)
    if path.exists():
        raise PipelineError("Refusing to overwrite an artifact")
    write_json(path, value)


def read_artifact(path):
    return read_json(reject_links(path))


def validate_inventory(entries):
    if not isinstance(entries, dict) or not entries:
        raise PipelineError("Expected a nonempty file inventory")
    for name, identity in entries.items():
        relative_path(name)
        if (not isinstance(identity, dict) or set(identity) != {"bytes", "sha256"}
                or type(identity["bytes"]) is not int or identity["bytes"] < 0
                or not isinstance(identity["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]) is None):
            raise PipelineError("Invalid file identity")


def inventory(root):
    root = reject_links(root)
    if not root.is_dir():
        raise PipelineError("Expected an artifact directory")
    result = {}
    for path in sorted(root.rglob("*")):
        reject_links(path)
        name = path.relative_to(root).as_posix()
        relative_path(name)
        if path.is_dir():
            if not any(path.iterdir()):
                raise PipelineError("Empty artifact directories are not supported")
        else:
            result[name] = file_hash(path)
    validate_inventory(result)
    return result


def load_contract(path):
    contract = read_artifact(path)
    expected = {
        "schema_version": 1, "model_id": MODEL_ID, "revision": REVISION, "model_type": "chronos2",
        "precision": "float32", "library": {"distribution": "chronos-forecasting", "version": "2.3.2"},
        "abi": ABI, "raw_context": {"min_length": 1, "max_length": 8192,
                                   "padding": "NaN on the left", "missing": "NaN"},
        "quantiles": list(QUANTILES),
    }
    if not isinstance(contract, dict) or set(contract) != {*expected, "files"}:
        raise PipelineError("Invalid Chronos-2 model contract")
    require_equal({key: contract[key] for key in expected}, expected, "Chronos-2 model contract")
    validate_inventory(contract["files"])
    if set(contract["files"]) != {"config.json", "README.md", "model.safetensors"}:
        raise PipelineError("Unexpected source inventory")
    return contract


def validate_config(config):
    if not isinstance(config, dict):
        raise PipelineError("Expected a model configuration")
    require_equal({key: config.get(key) for key in CORE_CONFIG}, CORE_CONFIG, "model configuration")
    require_equal(config.get("chronos_config"), FORECAST_CONFIG, "forecast configuration")
    if config.get("torch_dtype", config.get("dtype")) != "float32":
        raise PipelineError("Only the FP32 source is supported")
    if "dtype" in config and config["dtype"] != "float32":
        raise PipelineError("Conflicting model precision")


def validate_source(snapshot, contract):
    snapshot = reject_links(snapshot)
    if not snapshot.is_dir() or snapshot.name != REVISION:
        raise PipelineError("Use the pinned snapshot directory named with its complete revision")
    for name, expected in contract["files"].items():
        path = snapshot / relative_path(name)
        if not path.is_file() or file_hash(path) != expected:
            raise PipelineError(f"Pinned source differs: {name}")
    validate_config(read_json(snapshot / "config.json"))
    return snapshot


def nullable_context(values):
    if not isinstance(values, list):
        raise PipelineError("Context must be a JSON array")
    if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value)) for value in values):
        raise PipelineError("Context must contain finite numbers or null")
    return prepare_context([math.nan if value is None else value for value in values])


def load_fixtures(path):
    fixtures = read_artifact(path)
    if not isinstance(fixtures, dict) or set(fixtures) != {"schema_version", "atol", "rtol", "cases"}:
        raise PipelineError("Invalid fixture document")
    require_equal({key: fixtures[key] for key in ("schema_version", "atol", "rtol")},
                  {"schema_version": 1, "atol": 1e-4, "rtol": 1e-4}, "fixture tolerance")
    cases = fixtures["cases"]
    validate_case_ids(cases)
    for case in cases:
        if set(case) != {"id", "context"}:
            raise PipelineError("Unexpected fixture fields")
        nullable_context(case["context"])
    return fixtures


def validate_case_ids(cases):
    if (not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases)
            or [case.get("id") for case in cases] != list(CASE_IDS)):
        raise PipelineError("Expected all 18 unique fixture IDs in their pinned order")


def input_identity(model_path, fixture_path):
    return {"source": {"model_id": MODEL_ID, "revision": REVISION},
            "model": file_hash(reject_links(model_path)), "fixtures": file_hash(reject_links(fixture_path))}


def vector_document(kind, identity, fixtures, values):
    document = copy.deepcopy({
        "format": f"coreai-samples-chronos2-{kind}/1", "identity": identity, "precision": "float32",
        "abi": ABI, "quantiles": list(QUANTILES), "atol": 1e-4, "rtol": 1e-4,
        "cases": [{**case, "quantiles": output} for case, output in zip(fixtures["cases"], values, strict=True)],
    })
    validate_vectors(document, kind, identity, fixtures)
    return document


def validate_vectors(document, kind, identity, fixtures):
    expected = {"format": f"coreai-samples-chronos2-{kind}/1", "identity": identity, "precision": "float32",
                "abi": ABI, "quantiles": list(QUANTILES), "atol": 1e-4, "rtol": 1e-4}
    if not isinstance(document, dict) or set(document) != {*expected, "cases"}:
        raise PipelineError("Invalid prediction document")
    require_equal({key: document[key] for key in expected}, expected, "prediction identity or ABI")
    validate_case_ids(document["cases"])
    for case, fixture in zip(document["cases"], fixtures["cases"], strict=True):
        if set(case) != {"id", "context", "quantiles"}:
            raise PipelineError("Missing prediction vectors or unknown case fields")
        require_equal(case["context"], fixture["context"], "raw fixture context")
        validate_predictions(case["quantiles"])
        try:
            for row in case["quantiles"][0]:
                for value in row:
                    struct.pack("!f", value)
        except OverflowError as error:
            raise PipelineError("Predictions must be representable as float32") from error


def compare_documents(reference, actuals, identity, fixtures, kind="actuals"):
    validate_vectors(reference, "reference", identity, fixtures)
    validate_vectors(actuals, kind, identity, fixtures)
    cases = [{"id": expected["id"], **compare_predictions(actual["quantiles"], expected["quantiles"])}
             for expected, actual in zip(reference["cases"], actuals["cases"], strict=True)]
    return {"passed": all(case["passed"] for case in cases), "count": len(cases),
            "elements": sum(case["count"] for case in cases), "atol": 1e-4, "rtol": 1e-4,
            "identity": identity, "cases": cases}


def require_passed(comparison):
    if comparison["passed"] is not True or comparison["count"] != 18 or comparison["elements"] != 6048:
        raise PipelineError("Chronos-2 predictions failed the fixed comparison")
