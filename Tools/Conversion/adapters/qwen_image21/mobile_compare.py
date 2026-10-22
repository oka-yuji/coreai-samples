import argparse
import json
from pathlib import Path

import numpy as np

from Tools.Conversion.storage import write_json
from .contracts import tensor_metrics
from .evaluation import canonical_hash, quality_summary
from .fixtures import file_record, quality_axes
from .mobile_artifacts import inventory, read_profile, validate_environment, validate_source
from .mobile_evidence import mobile_fixtures, source_suite, validate_run_result
from .mobile_examples import tensor
from .mobile_profile import comparison_candidate, validate_mobile_profile
from .mobile_reference import component_paths, reference_preflight
from .mobile_verify import audit_numeric


def baseline_suite(root, expected):
    root = Path(root)
    report = Path(__file__).parents[4] / "docs/research/qwen-image21-evaluation-20261021.json"
    cohorts = json.loads(report.read_text())["source_cohorts"]
    pinned = next(row for row in cohorts if row["identity"]["precision"] == "fp32")
    if file_record(root / "summary.json") != pinned["summary"]:
        raise ValueError("The Float32 baseline differs from the accepted source-cohort record")
    summary = json.loads((root / "summary.json").read_text())
    identity = json.loads((root / "identity.json").read_text())
    if (summary["identity"] != pinned["identity"] or summary.get("complete") is not True
            or canonical_hash(identity) != canonical_hash(pinned["identity"])):
        raise ValueError("The Float32 source baseline identity or completion changed")
    rows = summary.get("runs", [])
    if len(rows) != 60 or {row["run_id"] for row in rows} != set(expected):
        raise ValueError("The baseline must retain all sixty frozen inputs")
    for row in rows:
        validate_run_result(row, expected[row["run_id"]])
        directory = root / row["run_id"]
        actual = {name: value for name, value in inventory(directory).items() if name != "result.json"}
        if actual != row["files"] or json.loads((directory / "result.json").read_text()) != row:
            raise ValueError("The Float32 baseline output files changed")
    return {"summary_file": file_record(root / "summary.json"), "identity": summary["identity"]}


def quality_template(cohorts):
    axes = quality_axes()
    return {"format": 2, "cohorts": {name: {"identity": identity,
            "ratings": [{"run_id": key, "axes": {axis: None for axis in values}} for key, values in axes.items()]}
            for name, identity in cohorts.items()}}


def score_quality(value, cohorts):
    if value.get("format") != 2 or set(value.get("cohorts", {})) != set(cohorts):
        raise ValueError("Every compared quality cohort is required")
    scores = {}
    axes = quality_axes()
    for name, identity in cohorts.items():
        cohort = value["cohorts"][name]
        if canonical_hash(cohort.get("identity")) != canonical_hash(identity):
            raise ValueError("Quality ratings belong to a different output cohort")
        rows = cohort.get("ratings", [])
        if len(rows) != 60 or {row.get("run_id") for row in rows} != set(axes):
            raise ValueError("Quality ratings must retain all sixty unique output identities")
        scores[name] = {row["run_id"]: row["axes"] for row in rows}
    results = {left + "-to-" + right: quality_summary(scores[left], scores[right], axes)
               for left, right in (("A0", "A1"), ("A1", "B"), ("A0", "C"), ("B", "C"))}
    return {"complete": all(row["complete"] for row in results.values()), "comparisons": results,
            "passed": all(row["candidate_meets_provisional_target"] for row in results.values())}


def execute(args):
    versions = validate_environment()
    profile = read_profile(args.profile, args.dependency_lock)
    source_files = validate_source(args.snapshot)
    manifest = json.loads((args.fixtures / "fixtures.json").read_text())
    mobile_fixtures(args.fixtures, manifest)
    expected = {row["run_id"]: row for row in manifest["runs"]}
    _, components = reference_preflight(profile, component_paths(args.component))
    source_identity, _ = source_suite(args.reference, profile, args.fixtures, manifest, list(expected), components, source_files, versions)
    mixed_identity = json.loads((args.mixed_reference / "identity.json").read_text())
    mixed_profile = mixed_identity["profile"]
    validate_mobile_profile(mixed_profile)
    if mixed_profile["id"] != "qmi-mobile-" + comparison_candidate(profile) + "-v1" or mixed_profile["source"] != profile["source"]:
        raise ValueError("A1 must be the declared unquantized mixed-precision source counterpart")
    source_suite(args.mixed_reference, mixed_profile, args.fixtures, manifest, list(expected), {}, source_files, versions)
    baseline = baseline_suite(args.baseline, expected)
    audit_numeric(args.verification, args.reference, args.snapshot, args.fixtures, manifest, profile, components, source_identity, versions)
    cohorts = {"A0": baseline,
               "A1": {"summary_file": file_record(args.mixed_reference / "summary.json"), "profile_sha256": canonical_hash(mixed_profile)},
               "B": {"summary_file": file_record(args.reference / "summary.json"), "profile_sha256": canonical_hash(profile)},
               "C": {"verification_file": file_record(args.verification / "verification.json"), "profile_sha256": canonical_hash(profile)}}
    roots = {"A0": args.baseline, "A1": args.mixed_reference, "B": args.reference, "C": args.verification / "rollout"}
    rows = []
    for key, record in expected.items():
        comparisons = {}
        for left, right in (("A0", "A1"), ("A1", "B"), ("A0", "C")):
            metrics = {}
            for name, shape in (("latent-7.npy", (1, 1024, 64)), ("rgba.npy", (1, 4, 1, 512, 512))):
                value = tensor_metrics(tensor(roots[left] / key / name, shape), tensor(roots[right] / key / name, shape),
                                       max_limit=np.inf, rmse_limit=np.inf)
                value.pop("passed")
                metrics[name] = value
            comparisons[left + "-to-" + right] = metrics
        rows.append({"run_id": key, "input": record, "comparisons": comparisons})
    if args.output.is_symlink() or (args.output.exists() and any(args.output.iterdir())):
        raise FileExistsError("Cohort comparison requires a new empty directory")
    args.output.mkdir(parents=True, exist_ok=True)
    template = quality_template(cohorts)
    ratings = json.loads(args.quality.read_text()) if args.quality else template
    quality = score_quality(ratings, cohorts)
    write_json(args.output / "quality-ratings.json", ratings)
    write_json(args.output / "comparison.json", {"format": 2, "cohorts": cohorts, "runs": rows, "quality": quality,
                "scope": "A0/A1/B changes are descriptive precision and quantization differences, not conversion-error gates",
                "quality_file": file_record(args.output / "quality-ratings.json"), "quality_passed": quality["passed"]})


def main():
    parser = argparse.ArgumentParser()
    for name in ("snapshot", "fixtures", "profile", "dependency-lock", "reference", "mixed-reference", "baseline", "verification", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--component", action="append", default=[])
    parser.add_argument("--quality", type=Path)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
