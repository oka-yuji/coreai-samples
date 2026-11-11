import argparse
from pathlib import Path
import shutil
import uuid

from ...storage import PipelineError, file_hash
from .artifacts import ABI, QUANTILES, inventory, new_output, read_artifact, reject_links, require_equal, require_passed
from .artifacts import validate_inventory, validate_vectors, write_new
from .convert import validate_export
from .prepare import check_ready
from .verify import validate_inspection, verification_record


def validate_bundle(folder, expected):
    record = read_artifact(Path(folder) / "chronos2.json")
    require_equal(record, expected, "bundle manifest")
    validate_inventory(record["files"])
    observed = inventory(folder)
    observed.pop("chronos2.json")
    require_equal(observed, record["files"], "bundle inventory")


def bundle(model_path, fixture_path, snapshot, ready, reference_path, export, verified, out):
    out = new_output(out)
    contract, fixtures, identity = check_ready(ready, model_path, fixture_path, snapshot)
    reference = read_artifact(reference_path)
    validate_vectors(reference, "reference", identity, fixtures)
    asset = validate_export(export, reference_path, reference, identity, fixtures)
    verified = reject_links(verified)
    actuals = read_artifact(verified / "actuals.json")
    inspection = validate_inspection(read_artifact(verified / "inspection-raw.json"))
    require_equal(read_artifact(verified / "inspection.json"), inspection, "sanitized inspection")
    comparison = verification_record(reference_path, reference, actuals, identity, fixtures, asset, inspection)
    require_passed(comparison)
    require_equal(read_artifact(verified / "comparison.json"), comparison, "native comparison")
    if {path.name for path in verified.iterdir()} != {
        "actuals.json", "inspection-raw.json", "inspection.json", "comparison.json",
    }:
        raise PipelineError("Unknown verification artifacts")
    target = out / "bundle"
    temporary = out / (".bundle-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        shutil.copytree(Path(export) / "Model.aimodel", temporary / "Model.aimodel")
        copies = {"reference.json": Path(reference_path), "SOURCE_README.md": Path(snapshot) / "README.md",
                  "LICENSE": Path(model_path).parent / "LICENSE", "NOTICE": Path(model_path).parent / "NOTICE"}
        expected_files = {"Model.aimodel/" + name: value for name, value in asset.items()}
        for name, path in copies.items():
            if name != "SOURCE_README.md":
                reject_links(path)
            expected_files[name] = file_hash(path)
            shutil.copyfile(path, temporary / name)
        require_equal(expected_files["SOURCE_README.md"], contract["files"]["README.md"], "official README")
        summary = {key: comparison[key] for key in ("passed", "count", "elements", "atol", "rtol", "cases")}
        summary.update({"inspection": inspection, "runtime": comparison["runtime"], "identity": identity})
        write_new(temporary / "verification.json", summary)
        expected_files["verification.json"] = file_hash(temporary / "verification.json")
        require_equal(inventory(temporary), expected_files, "copied bundle files")
        manifest = {"format": "coreai-samples-chronos2/1", "source": identity["source"], "precision": "float32",
                    "abi": ABI, "quantiles": list(QUANTILES), "raw_context": contract["raw_context"],
                    "files": expected_files}
        write_new(temporary / "chronos2.json", manifest)
        validate_bundle(temporary, manifest)
        if target.exists() or target.is_symlink():
            raise PipelineError("Refusing to overwrite a bundle")
        temporary.rename(target)
        try:
            write_new(out / "inventory.json", {"format": "coreai-samples-chronos2-inventory/1",
                                               "files": inventory(target)})
        except BaseException:
            shutil.rmtree(target)
            raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "fixtures", "snapshot", "ready", "reference", "export", "verified", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    bundle(args.model, args.fixtures, args.snapshot, args.ready, args.reference, args.export, args.verified, args.out)


if __name__ == "__main__":
    main()
