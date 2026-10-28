import argparse
import hashlib
import json
from pathlib import Path
import re

from Tools.Conversion.storage import write_json
from .contracts import require_heavy_job
from .evaluation import validate_manifest
from .fixtures import file_record
from .mobile_artifacts import inventory
from .mobile_pack import copy_file


def pack(args):
    require_heavy_job()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("AOT packing requires a new output directory")
    if not re.fullmatch(r"[a-z0-9]{1,32}", args.architecture):
        raise ValueError("Invalid architecture identifier")
    sources = {kind: Path(path) for kind, path in args.compiled}
    if not sources or len(sources) != len(args.compiled):
        raise ValueError("Compiled components must be nonempty and unique")
    for source in [args.bundle, *sources.values()]:
        output, origin = args.output.resolve(), source.resolve()
        if output.is_relative_to(origin) or origin.is_relative_to(output):
            raise ValueError("The output must not overlap any input")
    actual = inventory(args.bundle)
    original = actual.pop("bundle.json", None)
    if original is None or original["bytes"] > 1_048_576 or original["sha256"] != args.manifest_sha256:
        raise ValueError("The raw bundle differs from its pinned manifest")
    raw = (args.bundle / "bundle.json").read_bytes()
    if {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} != original:
        raise ValueError("The pinned manifest changed while reading")
    manifest = json.loads(raw)
    validate_manifest(manifest)
    if (actual != manifest["assets"] or manifest.get("compiled_assets")
            or any(manifest.get(key) is not True for key in ("verified", "numeric_verified", "quality_verified"))):
        raise ValueError("AOT comparison requires a complete accepted raw bundle")
    planned = [(args.bundle / name, name, record) for name, record in actual.items()]
    compiled = {}
    for kind, source in sources.items():
        if kind not in manifest["profile"]["components"]:
            raise ValueError("Unknown compiled component")
        records = inventory(source)
        metadata_file = records.get("metadata.json")
        if metadata_file is None or metadata_file["bytes"] > 1_048_576:
            raise ValueError("Compiled metadata is missing or oversized")
        component = manifest["profile"]["components"][kind]
        raw_model = actual.get(component["asset"] + "/main.mlirb")
        raw_metadata = (source / "metadata.json").read_bytes()
        if {"bytes": len(raw_metadata), "sha256": hashlib.sha256(raw_metadata).hexdigest()} != metadata_file:
            raise ValueError("Compiled metadata changed while reading")
        metadata = json.loads(raw_metadata)
        source_hash = metadata.get("sourceHash") if isinstance(metadata, dict) else None
        if raw_model is None or not isinstance(source_hash, str) or source_hash.lower() != raw_model["sha256"]:
            raise ValueError("The compiled component belongs to another raw model")
        path = "compiled/" + kind + "." + args.architecture + ".aimodelc"
        compiled[kind] = {"asset": path, "architecture": args.architecture, "source_sha256": raw_model["sha256"]}
        planned += [(source / name, path + "/" + name, record) for name, record in records.items()]
    if len({name for _, name, _ in planned}) != len(planned):
        raise ValueError("Compiled paths overlap the raw inventory")
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = args.output / "bundle"
    bundle.mkdir()
    marker = bundle / ".packing"
    marker.write_text("AOT candidate creation in progress\n")
    for source, name, expected in planned:
        copy_file(source, bundle / name, expected)
    manifest.update(assets={name: record for _, name, record in planned}, compiled_assets=compiled,
                    raw_bundle_manifest=original, compiled_verification="pending", device_verification="pending")
    validate_manifest(manifest)
    marker.unlink()
    write_json(bundle / "bundle.json", manifest)
    saved = file_record(bundle / "bundle.json")
    if inventory(bundle) != {**manifest["assets"], "bundle.json": saved} or file_record(args.bundle / "bundle.json") != original:
        raise ValueError("The source or packed inventory changed")
    write_json(args.output / "receipt.json", {"format": 1, "base_manifest": original, "manifest": saved,
               "bundle_bytes": sum(record["bytes"] for record in manifest["assets"].values()) + saved["bytes"],
               "file_count": len(manifest["assets"]) + 1, "compiled_assets": compiled,
               "architecture_declared": args.architecture, "device_verified": False,
               "scope": "Pinned raw-source and file integrity only; compiled execution and architecture compatibility require device validation"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--compiled", nargs=2, action="append", required=True, metavar=("COMPONENT", "DIRECTORY"))
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--output", type=Path, required=True)
    pack(parser.parse_args())


if __name__ == "__main__":
    main()
