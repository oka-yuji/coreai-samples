import argparse
import json
from pathlib import Path
import signal

import numpy as np

from .contracts import require_heavy_job
from .evaluation import canonical_hash
from .fixtures import file_record
from .measurement import write_metrics
from .mobile_artifacts import inventory, read_profile, validate_environment, validate_source
from .mobile_evidence import POLICY_SHA256, mobile_fixtures, source_suite, validate_run_result
from .mobile_examples import fixed_examples, tensor
from .mobile_reference import component_paths, reference_preflight, selected_runs
from .mobile_runtime import CandidateComponent
from .numeric_policy import numeric_metrics


def compare_example(kind, example, actual, profile):
    expected = example["expected"]
    specification = profile["components"][kind]["outputs"]
    if len(expected) != len(actual) or len(actual) != len(specification):
        raise ValueError("Every declared component output must be compared")
    dtype = profile["components"][kind]["compute_dtype"]
    records = []
    for index, (reference, observed) in enumerate(zip(expected, actual, strict=True)):
        if list(observed.shape) != specification[index]["shape"] or observed.dtype.name != specification[index]["dtype"]:
            raise ValueError("A saved Core AI output differs from the full physical contract")
        if kind in ("text", "denoiser"):
            observed = observed[:, :reference.shape[1]]
        records.append({"output": index, **numeric_metrics(reference, observed, compute_dtype=dtype, boundary="component")})
        if kind == "denoiser":
            target = example["target_indices"]
            records.append({"output": index, **numeric_metrics(reference[:, target], observed[:, target],
                            compute_dtype=dtype, boundary="fixed-step-target")})
    return {"id": example["id"], "comparisons": records, "passed": all(row["metrics"]["passed"] for row in records)}


def completed_run(record, operation):
    expected = {"text", "vae_decoder", *["denoiser-" + str(index) for index in range(8)]}
    if operation == "edit":
        expected.update(("vision", "vae_encoder"))
    rows = record.get("fixed", [])
    fixed = len(rows) == len(expected) and {row.get("id") for row in rows} == expected
    for row in rows:
        pairs = {(0, "component"), (0, "fixed-step-target")} if row["id"].startswith("denoiser-") else {
            (index, "component") for index in range(4 if row["id"] == "vision" else 1)}
        comparisons = row.get("comparisons", [])
        fixed = fixed and row.get("passed") is True and len(comparisons) == len(pairs) and {
            (value.get("output"), value.get("boundary")) for value in comparisons} == pairs and all(
                value.get("policy_sha256") == POLICY_SHA256 and value.get("metrics", {}).get("passed") is True for value in comparisons)
    rollout = record.get("rollout", {})
    return fixed and set(rollout) == {"latent", "rgba"} and all(
        row.get("policy_sha256") == POLICY_SHA256 and row.get("boundary") == "final-" + name
        and row.get("metrics", {}).get("passed") is True for name, row in rollout.items())


def audit_numeric(directory, reference, snapshot, fixtures, manifest, profile, component_records, source_identity, versions):
    directory, reference = Path(directory), Path(reference)
    report = json.loads((directory / "verification.json").read_text())
    identity = report["identity"]
    code = ("mobile_verify.py", "mobile_runtime.py", "mobile_generate.py", "mobile_evidence.py",
            "mobile_examples.py", "preprocessing.py", "numeric_policy.py")
    if (identity.get("profile") != profile or identity.get("profile_sha256") != canonical_hash(profile)
            or identity.get("policy_sha256") != POLICY_SHA256 or identity.get("versions") != versions
            or identity.get("exported_components") != component_records
            or identity.get("source_identity") != source_identity
            or identity.get("source_summary") != file_record(reference / "summary.json")
            or identity.get("fixtures_file") != file_record(Path(fixtures) / "fixtures.json")
            or identity.get("verification_code") != {name: file_record(Path(__file__).with_name(name)) for name in code}
            or report.get("numeric_complete") is not True or report.get("numeric_passed") is not True):
        raise ValueError("The numeric report is incomplete or belongs to different inputs, code or components")
    actual = {name: value for name, value in inventory(directory).items() if name != "verification.json"}
    if actual != report.get("evidence_files"):
        raise ValueError("The numeric output file set or contents changed")
    expected = {row["run_id"]: row for row in manifest["runs"]}
    rows = report.get("runs", [])
    if len(rows) != 60 or {row.get("run_id") for row in rows} != set(expected):
        raise ValueError("Numerical acceptance requires all sixty unique inputs")
    from .examples import arrays
    for row in rows:
        key = row["run_id"]
        validate_run_result(row, expected[key])
        metadata = json.loads((Path(fixtures) / expected[key]["fixture_id"] / "input.json").read_text())
        comparisons = []
        for kind in profile["components"]:
            for example in fixed_examples(kind, reference / key, snapshot, metadata):
                captured = arrays(directory / "fixed" / key / (example["id"] + ".npz"))
                names = [f"output_{index}" for index in range(len(example["expected"]))]
                if set(captured) != set(names):
                    raise ValueError("A fixed comparison lost a required physical output")
                comparisons.append(compare_example(kind, example, [captured[name] for name in names], profile))
        if comparisons != row["fixed"] or not completed_run(row, expected[key]["operation"]):
            raise ValueError("Saved fixed-input metrics do not reproduce their outputs or full coverage")
        for name, filename, shape, kind in (("latent", "latent-7.npy", (1, 1024, 64), "denoiser"),
                                            ("rgba", "rgba.npy", (1, 4, 1, 512, 512), "vae_decoder")):
            metrics = numeric_metrics(tensor(reference / key / filename, shape),
                                      tensor(directory / "rollout" / key / filename, shape),
                                      compute_dtype=profile["components"][kind]["compute_dtype"], boundary="final-" + name)
            if metrics != row["rollout"][name] or metrics["metrics"]["passed"] is not True:
                raise ValueError("Saved rollout metrics do not reproduce their final outputs")
    return report


def execute(args):
    from PIL import Image
    from .examples import arrays
    from .mobile_generate import generate

    require_heavy_job()
    versions = validate_environment()
    profile = read_profile(args.profile, args.dependency_lock)
    manifest = json.loads((args.fixtures / "fixtures.json").read_text())
    mobile_fixtures(args.fixtures, manifest)
    expected, selected = selected_runs(manifest, args.run_ids)
    paths = component_paths(args.component)
    _, component_records = reference_preflight(profile, paths)
    if set(paths) != set(profile["components"]):
        raise ValueError("Core AI validation requires all five converted components")
    source_files = validate_source(args.snapshot)
    source_identity, _ = source_suite(args.reference, profile, args.fixtures, manifest, selected, component_records, source_files, versions)
    if args.output.is_symlink() or (args.output.exists() and any(args.output.iterdir())):
        raise FileExistsError("Candidate verification requires a new empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {"format": 2, "profile": profile, "profile_sha256": canonical_hash(profile),
                "policy_sha256": POLICY_SHA256, "fixtures_file": file_record(args.fixtures / "fixtures.json"),
                "source_identity": source_identity, "source_summary": file_record(args.reference / "summary.json"),
                "exported_components": component_records, "versions": versions,
                "verification_code": {name: file_record(Path(__file__).with_name(name)) for name in
                    ("mobile_verify.py", "mobile_runtime.py", "mobile_generate.py", "mobile_evidence.py",
                     "mobile_examples.py", "preprocessing.py", "numeric_policy.py")}}
    records = {key: {"run_id": key, "input": value, "status": "not_run", "fixed": [], "rollout": {},
                    "not_applicable": [] if value["operation"] == "edit" else [
                        {"component": kind, "reason": "generation has no reference image"} for kind in ("vision", "vae_encoder")]}
               for key, value in expected.items()}
    report = {"identity": identity, "runs": list(records.values()), "numeric_complete": False,
              "numeric_passed": False, "quality_status": "unreviewed", "device_status": "not_run", "verified": False}
    failure = None
    active = None

    def publish():
        report["numeric_complete"] = all(row["status"] in ("completed", "numeric_failed") for row in records.values())
        report["numeric_passed"] = report["numeric_complete"] and all(completed_run(row, expected[key]["operation"])
                                                                     for key, row in records.items())
        write_metrics(args.output / "verification.json", report, preserving=failure)

    publish()
    try:
        for kind in profile["components"]:
            with CandidateComponent(paths[kind], profile, kind) as component:
                for key in selected:
                    active = key
                    metadata = json.loads((args.fixtures / expected[key]["fixture_id"] / "input.json").read_text())
                    for example in fixed_examples(kind, args.reference / key, args.snapshot, metadata):
                        records[key]["status"] = "running"
                        actual = component.call(example["values"])
                        directory = args.output / "fixed" / key
                        directory.mkdir(parents=True, exist_ok=True)
                        np.savez(directory / (example["id"] + ".npz"), **{f"output_{index}": value for index, value in enumerate(actual)})
                        records[key]["fixed"].append(compare_example(kind, example, actual, profile))
                    publish()
        for key in selected:
            active = key
            record = expected[key]
            metadata = json.loads((args.fixtures / record["fixture_id"] / "input.json").read_text())
            encoded = arrays(args.fixtures / record["fixture_id"] / "encoded.npz")
            image = None
            if metadata["reference"]:
                with Image.open(args.fixtures / metadata["reference"]) as original:
                    image = original.convert("RGBA")
            noise = np.load(args.fixtures / record["noise"], allow_pickle=False)
            latent, rgba = generate(paths, profile, args.snapshot, args.output / "rollout" / key, encoded, metadata, noise, image)
            records[key]["rollout"] = {
                "latent": numeric_metrics(tensor(args.reference / key / "latent-7.npy", (1, 1024, 64)), latent,
                                          compute_dtype=profile["components"]["denoiser"]["compute_dtype"], boundary="final-latent"),
                "rgba": numeric_metrics(tensor(args.reference / key / "rgba.npy", (1, 4, 1, 512, 512)), rgba,
                                        compute_dtype=profile["components"]["vae_decoder"]["compute_dtype"], boundary="final-rgba")}
            records[key]["status"] = "completed" if completed_run(records[key], record["operation"]) else "numeric_failed"
            publish()
            print(json.dumps({"run_id": key, "status": records[key]["status"]}), flush=True)
        report["evidence_files"] = {name: value for name, value in inventory(args.output).items() if name != "verification.json"}
        if any(records[key]["status"] != "completed" for key in selected):
            raise ValueError("Candidate fidelity failed the frozen numerical policy; inspect every retained comparison")
    except BaseException as error:
        failure = error
        report.update(error_type=type(error).__name__, error=str(error))
        if active is not None and records[active]["status"] == "running":
            records[active]["status"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        raise
    finally:
        publish()


def main():
    parser = argparse.ArgumentParser()
    for name in ("snapshot", "fixtures", "profile", "dependency-lock", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--component", action="append", default=[])
    parser.add_argument("--run-ids")
    args = parser.parse_args()
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    execute(args)


if __name__ == "__main__":
    main()
