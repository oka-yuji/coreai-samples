import argparse
from pathlib import Path

from ...storage import PipelineError, file_hash
from .artifacts import compare_documents, inventory, new_output, read_artifact, require_equal, require_passed
from .artifacts import validate_vectors, vector_document, write_new
from .prepare import check_ready
from .reference import context_tensor, load_official, prediction_values


def compare_wrapper(module, reference, identity, fixtures):
    import torch

    with torch.no_grad():
        values = [prediction_values(module(context_tensor(case["context"]))) for case in fixtures["cases"]]
    actuals = vector_document("wrapper", identity, fixtures, values)
    comparison = compare_documents(reference, actuals, identity, fixtures, kind="wrapper")
    return actuals, comparison


def validate_export(folder, reference_path, reference, identity, fixtures):
    folder = Path(folder)
    record = read_artifact(folder / "export.json")
    actuals = read_artifact(folder / "wrapper-actuals.json")
    comparison = compare_documents(reference, actuals, identity, fixtures, kind="wrapper")
    require_passed(comparison)
    require_equal(read_artifact(folder / "wrapper-comparison.json"), comparison, "wrapper comparison")
    asset = inventory(folder / "Model.aimodel")
    expected = {"format": "coreai-samples-chronos2-export/1", "identity": identity,
                "reference": file_hash(reference_path), "asset": asset,
                "wrapper_actuals": file_hash(folder / "wrapper-actuals.json"),
                "wrapper_comparison": file_hash(folder / "wrapper-comparison.json")}
    require_equal(record, expected, "export identity")
    if {path.name for path in folder.iterdir()} != {
        "Model.aimodel", "export.json", "wrapper-actuals.json", "wrapper-comparison.json",
    }:
        raise PipelineError("Unknown export artifacts")
    return asset


def convert(model_path, fixture_path, snapshot, ready, reference_path, out):
    out = new_output(out)
    _, fixtures, identity = check_ready(ready, model_path, fixture_path, snapshot)
    reference = read_artifact(reference_path)
    validate_vectors(reference, "reference", identity, fixtures)
    from .model_wrapper import Chronos2Wrapper, export_program
    import coreai_torch
    from coreai.runtime import AIModelAssetMetadata

    module = Chronos2Wrapper(load_official(snapshot)).eval()
    actuals, comparison = compare_wrapper(module, reference, identity, fixtures)
    write_new(out / "wrapper-actuals.json", actuals)
    write_new(out / "wrapper-comparison.json", comparison)
    require_passed(comparison)
    exported = export_program(module, coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(exported, input_names=["context"], output_names=["quantiles"],
                                   state_names=[], entrypoint_name="main")
    program = converter.to_coreai()
    metadata = AIModelAssetMetadata()
    metadata.author = "Amazon"
    metadata.license = "Apache-2.0"
    metadata.model_description = "Chronos-2 FP32 univariate 16-step forecast"
    metadata.set_custom("source_revision", identity["source"]["revision"])
    asset = out / "Model.aimodel"
    if asset.exists() or asset.is_symlink():
        raise PipelineError("Refusing to overwrite the exported asset")
    program.save_asset(asset, metadata)
    record = {"format": "coreai-samples-chronos2-export/1", "identity": identity,
              "reference": file_hash(reference_path), "asset": inventory(asset),
              "wrapper_actuals": file_hash(out / "wrapper-actuals.json"),
              "wrapper_comparison": file_hash(out / "wrapper-comparison.json")}
    write_new(out / "export.json", record)
    return record


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "fixtures", "snapshot", "ready", "reference", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    convert(args.model, args.fixtures, args.snapshot, args.ready, args.reference, args.out)


if __name__ == "__main__":
    main()
