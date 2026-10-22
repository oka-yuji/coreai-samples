import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

from Tools.Conversion.storage import write_json
from .contracts import require_heavy_job, validate_bundle
from .evaluation import canonical_hash, validate_manifest
from .fixtures import file_record
from .mobile_artifacts import inventory, read_component, read_profile
from .mobile_compare import execute as compare_cohorts
from .mobile_reference import component_paths
from .pack import ATTRIBUTION, files, regular


def copy_file(source, target, expected):
    source = regular(source, follow=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Refusing to replace a bundle file: " + str(target))
    if sys.platform == "darwin" and source.stat().st_dev == target.parent.stat().st_dev:
        subprocess.run(["/bin/cp", "-c", "-n", str(source), str(target)], check=True)
    else:
        with source.open("rb") as left, target.open("xb") as right:
            shutil.copyfileobj(left, right, length=1024 * 1024)
    if file_record(target) != expected:
        raise ValueError("A copied bundle file differs from its verified source: " + str(target))


def compression_summary(components, baseline):
    rows = {}
    for kind, (record, compression) in components.items():
        layers = compression["layers"]
        elements = sum(row["shape"][0] * row["shape"][1] for row in layers)
        bits = sum(row["payload_bits_scenario"] for row in layers)
        original = sum(value["bytes"] for value in inventory(Path(baseline) / kind / (kind + ".aimodel")).values())
        rows[kind] = {"asset_bytes": record["asset_bytes"], "baseline_fp32_asset_bytes": original,
                      "asset_size_ratio": record["asset_bytes"] / original, "quantized_linear_count": len(layers),
                      "quantized_weight_elements": elements,
                      "selected_linear_payload_bits_scenario": bits if elements else None,
                      "selected_linear_bits_per_weight_scenario": bits / elements if elements else None,
                      "scope": "Asset bytes include serialization; grouped q/scale/zero scenario excludes other weights, IR overhead and runtime memory"}
        retained = compression.get("dense_layers", [])
        if retained:
            dense_elements = sum(row["weight"]["shape"][0] * row["weight"]["shape"][1] for row in retained)
            dense_bits = sum(row["payload_bits"] for row in retained)
            rows[kind].update(retained_dense_linear_count=len(retained), retained_dense_weight_elements=dense_elements,
                              retained_dense_weight_payload_bits=dense_bits,
                              declared_linear_weight_bits_per_weight_scenario=(bits + dense_bits) / (elements + dense_elements))
    return rows


def pack(args):
    require_heavy_job()
    profile = read_profile(args.profile, args.dependency_lock)
    paths = component_paths(args.component)
    if set(paths) != set(profile["components"]):
        raise ValueError("Packing requires all five converted components")
    if args.output.is_symlink() or (args.output.exists() and any(args.output.iterdir())):
        raise FileExistsError("Packing requires a new empty output directory")
    if any(args.output.resolve().is_relative_to(path.resolve()) for path in
           (args.snapshot, args.fixtures, args.reference, args.mixed_reference, args.baseline, args.baseline_bundle, args.verification, *paths.values())):
        raise ValueError("The output must not overlap source model or evidence directories")
    args.output.mkdir(parents=True, exist_ok=True)
    comparison = args.output / "cohort-comparison"
    compare_cohorts(SimpleNamespace(**{**vars(args), "output": comparison}))
    result = json.loads((comparison / "comparison.json").read_text())
    numeric_identity = json.loads((args.verification / "verification.json").read_text())["identity"]
    source_files = numeric_identity["source_identity"]["source_files"]
    components = {kind: read_component(path, profile, kind) for kind, path in paths.items()}
    validate_bundle(args.baseline_bundle)
    sizes = compression_summary(components, args.baseline_bundle)
    license_path = regular(args.snapshot / "LICENSE", follow=True)
    if ATTRIBUTION not in license_path.read_text():
        raise ValueError("The source Qwen Research License attribution changed")
    bundle = args.output / "bundle"
    bundle.mkdir()
    marker = bundle / ".packing"
    marker.write_text("Candidate bundle creation in progress\n")
    planned = []
    for kind, (record, _) in components.items():
        path = paths[kind]
        planned += [(path / "component.json", Path(kind) / "component.json", numeric_identity["exported_components"][kind]),
                    (path / "compression.json", Path(kind) / "compression.json", record["compression_file"])]
        planned += [(path / record["asset"] / name, Path(kind) / record["asset"] / name, value)
                    for name, value in record["asset_inventory"].items()]
    planned += [(source, Path("config/processor") / relative, source_files["processor/" + relative.as_posix()])
                for source, relative in files(args.snapshot / "processor", {".json", ".txt", ".jinja"})]
    for name in ("text_encoder/config.json", "transformer/config.json", "vae/config.json", "scheduler/scheduler_config.json"):
        planned.append((args.snapshot / name, Path("config") / name, source_files[name]))
    planned += [(license_path, Path("LICENSE"), source_files["LICENSE"]),
                (args.dependency_lock, Path("uv.lock"), {"bytes": args.dependency_lock.stat().st_size,
                                                        "sha256": profile["source"]["dependency_lock_sha256"]}),
                (args.fixtures / "fixtures.json", Path("evidence/fixtures.json"), numeric_identity["fixtures_file"]),
                (args.verification / "verification.json", Path("evidence/numeric.json"), result["cohorts"]["C"]["verification_file"])]
    for name in ("comparison.json", "quality-ratings.json"):
        planned.append((comparison / name, Path("evidence") / name, file_record(comparison / name)))
    for cohort, root in (("A0", args.baseline), ("A1", args.mixed_reference), ("B", args.reference)):
        for name in ("identity.json", "summary.json"):
            expected = result["cohorts"][cohort]["summary_file"] if name == "summary.json" else file_record(root / name)
            planned.append((root / name, Path("evidence") / cohort / name, expected))
    required = sum(value["bytes"] for _, _, value in planned) + 10 * 1024**3
    if shutil.disk_usage(bundle).free < required:
        raise OSError("Insufficient free space for the complete bundle and reserve")
    for source, destination, expected in planned:
        copy_file(source, bundle / destination, expected)
    changes = "\n".join(kind + ": converted for the declared Core AI compute and storage profile." for kind in paths)
    (bundle / "NOTICE.txt").write_text("Qwen Image 2.1 Turbo mobile candidate\nBuilt with Qwen.\n\n" + ATTRIBUTION
                                     + "\n\nModified model assets:\n" + changes
                                     + "\n\nOriginal source configuration and LICENSE are copied unchanged.\n")
    (bundle / "README.txt").write_text(
        "Qwen Image 2.1 Turbo mobile candidate\nBuilt with Qwen.\n\n"
        "Fixed 512-square generation and single-image editing, eight steps, CFG one, no prefix KV cache.\n"
        "Read bundle.json for numerical, human-quality and device validation states; these are separate gates.\n"
        "A candidate with verified=false is for local review and must not be registered as a verified model.\n"
        "Model assets retain the Qwen Research License; read LICENSE and NOTICE.txt. The sample-code license\n"
        "does not relicense these model assets. Raw numerical traces remain in the maintainer evaluation archive.\n")
    write_json(bundle / "compression.json", sizes)
    marker.unlink()
    manifest = {"format": 2, "profile": profile, "assets": inventory(bundle), "numeric_verified": True,
                "quality_verified": result["quality_passed"], "verified": result["quality_passed"],
                "target_minimum_os": "27.0", "device_verification": "pending",
                "source_revision": profile["source"]["revision"], "profile_sha256": canonical_hash(profile),
                "evidence": {"numeric": "evidence/numeric.json", "comparison": "evidence/comparison.json",
                             "quality": "evidence/quality-ratings.json"}}
    total = validate_manifest(manifest)
    write_json(bundle / "bundle.json", manifest)
    if {name: value for name, value in inventory(bundle).items() if name != "bundle.json"} != manifest["assets"]:
        raise ValueError("The complete bundle inventory changed during packing")
    write_json(args.output / "receipt.json", {"format": 2, "bundle_ready": True, "manifest": file_record(bundle / "bundle.json"),
               "profile_sha256": canonical_hash(profile), "asset_bytes": total, "numeric_passed": True,
               "quality_passed": result["quality_passed"], "verified": result["quality_passed"], "device_verification": "pending"})


def main():
    parser = argparse.ArgumentParser()
    for name in ("snapshot", "fixtures", "profile", "dependency-lock", "reference", "mixed-reference", "baseline",
                 "baseline-bundle", "verification", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--component", action="append", default=[])
    parser.add_argument("--quality", type=Path)
    pack(parser.parse_args())


if __name__ == "__main__":
    main()
