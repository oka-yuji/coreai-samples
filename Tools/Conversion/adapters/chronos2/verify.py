import argparse
import asyncio
from pathlib import Path
import subprocess

from ...storage import PipelineError, file_hash
from .artifacts import ABI, REVISION, compare_documents, inventory, new_output, nullable_context, read_artifact
from .artifacts import require_equal, require_passed, validate_vectors, vector_document, write_new
from .convert import validate_export
from .prepare import check_ready


RUNTIME_ABI = {"function_names": ["main"], "input_names": ["context"],
               "output_names": ["quantiles"], "state_names": []}
STORAGE_TYPES = {"Float32", "Bool", "Int32", "Int64", "UInt32", "UInt64"}


def validate_inspection(report):
    if not isinstance(report, dict) or not isinstance(report.get("summary"), dict):
        raise PipelineError("Missing native inspection summary")
    summary = report["summary"]
    expected_function = {
        "name": "main", "inputs": [{"name": "context", "type": "NDArray (Float32, 1 × ?)"}],
        "outputs": [{"name": "quantiles", "type": "NDArray (Float32, 1 × 21 × 16)"}],
    }
    require_equal(summary.get("functions"), [expected_function], "native stateless ABI")
    if summary.get("compatibilityVersion") != "27.0":
        raise PipelineError("Unexpected native compatibility version")
    for container in (summary, report):
        for key in ("states", "state_names", "stateNames"):
            if key in container and container[key] != []:
                raise PipelineError("Native state is not supported")
    storage = summary.get("storageTypes")
    if not isinstance(storage, list) or not storage:
        raise PipelineError("Missing storage precision inspection")
    names = []
    for row in storage:
        if (not isinstance(row, dict) or set(row) != {"name", "count"} or row["name"] not in STORAGE_TYPES
                or type(row["count"]) is not int or row["count"] <= 0 or row["name"] in names):
            raise PipelineError("Unexpected native storage precision")
        names.append(row["name"])
    compute = summary.get("computeTypes")
    if ("Float32" not in names or not isinstance(compute, list) or not compute
            or any(not isinstance(name, str) or name not in STORAGE_TYPES for name in compute)
            or len(set(compute)) != len(compute) or "Float32" not in compute):
        raise PipelineError("Unexpected native compute precision")
    metadata = report.get("metadata")
    if not isinstance(metadata, dict):
        raise PipelineError("Missing native attribution")
    custom = metadata.get("creatorDefinedMetadata", {})
    expected = {"author": "Amazon", "license": "Apache-2.0", "producer": "coreai-core 1.0.0b3"}
    require_equal({key: metadata.get(key) for key in expected}, expected, "native attribution")
    if not isinstance(custom, dict) or custom.get("source_revision") != REVISION:
        raise PipelineError("Unexpected native source revision")
    return {"abi": ABI, "compatibility_version": "27.0", "storage_types": storage, "compute_types": compute,
            "metadata": {**expected, "source_revision": REVISION}}


def inspect_asset(asset, out):
    raw = out / "inspection-raw.json"
    output = subprocess.check_output([
        "xcrun", "coreai-build", "inspect", str(asset), "--io", "--metadata", "--storage", "--compute", "--json",
    ], text=True, timeout=600)
    with raw.open("x") as handle:
        handle.write(output)
    inspection = validate_inspection(read_artifact(raw))
    write_new(out / "inspection.json", inspection)
    return inspection


async def execute_cases(function, fixtures, ndarray):
    import numpy as np

    values = []
    for case in fixtures["cases"]:
        context = np.asarray([nullable_context(case["context"])], dtype=np.float32)
        outputs = await function(inputs={"context": ndarray(context)}, state={})
        if set(outputs) != {"quantiles"}:
            raise PipelineError("Unexpected native output names")
        actual = outputs["quantiles"].numpy().copy()
        if actual.dtype != np.float32 or actual.shape != (1, 21, 16) or not np.isfinite(actual).all():
            raise PipelineError("Native predictions must be finite Float32[1, 21, 16]")
        values.append(actual.tolist())
    return values


async def run_native(asset, fixtures):
    from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions

    native = await AIModel.load(asset, specialization_options=SpecializationOptions.from_preferred_compute_unit_kind(
        ComputeUnitKind.gpu()))
    function = native.load_function("main")
    observed = {"function_names": native.function_names, "input_names": function.desc.input_names,
                "output_names": function.desc.output_names, "state_names": function.desc.state_names}
    require_equal(observed, RUNTIME_ABI, "native runtime ABI")
    return await execute_cases(function, fixtures, NDArray)


def verification_record(reference_path, reference, actuals, identity, fixtures, asset, inspection):
    return {**compare_documents(reference, actuals, identity, fixtures), "reference": file_hash(reference_path),
            "asset": asset, "inspection": inspection, "runtime": RUNTIME_ABI}


def verify(model_path, fixture_path, ready, reference_path, export, out):
    out = new_output(out)
    _, fixtures, identity = check_ready(ready, model_path, fixture_path)
    reference = read_artifact(reference_path)
    validate_vectors(reference, "reference", identity, fixtures)
    asset = validate_export(export, reference_path, reference, identity, fixtures)
    inspection = inspect_asset(Path(export) / "Model.aimodel", out)
    values = asyncio.run(run_native(Path(export) / "Model.aimodel", fixtures))
    actuals = vector_document("actuals", identity, fixtures, values)
    write_new(out / "actuals.json", actuals)
    comparison = verification_record(reference_path, reference, actuals, identity, fixtures, asset, inspection)
    write_new(out / "comparison.json", comparison)
    require_equal(inventory(Path(export) / "Model.aimodel"), asset, "asset during verification")
    require_passed(comparison)
    return comparison


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "fixtures", "ready", "reference", "export", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    verify(args.model, args.fixtures, args.ready, args.reference, args.export, args.out)


if __name__ == "__main__":
    main()
