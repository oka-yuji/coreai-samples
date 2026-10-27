import argparse
import gc
import json
from pathlib import Path
import time

from .contracts import digest, require_heavy_job
from .evaluation import canonical_hash
from .fixtures import file_record
from .mobile_artifacts import converter_identity, inventory, read_profile, validate_environment, validate_source
from .mobile_profile import CANDIDATES, make_profile


def new_output(directory, names):
    if directory.is_symlink():
        raise FileExistsError("The output directory must not be a symbolic link")
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() or (directory / name).is_symlink() for name in names):
        raise FileExistsError("Conversion outputs must not overwrite earlier files or links")


def write_new_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def export_component(args):
    import torch
    from coreai.authoring import OSVersion
    from coreai.runtime import AIModelAssetMetadata
    from coreai_torch import TorchConverter, get_decomp_table
    from .mobile_models import compress_original, export_boundary, load_original

    require_heavy_job()
    versions = validate_environment()
    profile = read_profile(args.profile, args.dependency_lock)
    if args.kind not in profile["components"]:
        raise ValueError("Unknown component")
    new_output(args.output, ("component.json", "compression.json", args.kind + ".aimodel"))
    source = validate_source(args.snapshot, args.kind)
    torch.set_num_threads(8)
    original = load_original(args.snapshot, args.kind, profile)
    original, compression = compress_original(original, args.kind, profile)
    write_new_json(args.output / "compression.json", {"format": 1, "kind": args.kind,
              "profile_sha256": canonical_hash(profile), "compression": compression})
    rotary = None
    if args.kind == "denoiser" and profile["components"][args.kind]["compute_dtype"] == "float32":
        from .metal_rotary import Float32Rotary
        rotary = Float32Rotary()
    wrapper = export_boundary(original, args.kind, profile, rotary=rotary)
    specification = profile["components"][args.kind]
    dtypes = {"float32": torch.float32, "int32": torch.int32, "bool": torch.bool}
    example = tuple(torch.ones(item["shape"], dtype=dtypes[item["dtype"]])
                    if item["dtype"] == "bool" else torch.zeros(item["shape"], dtype=dtypes[item["dtype"]])
                    for item in specification["inputs"])
    started = time.perf_counter()
    with torch.no_grad():
        program = torch.export.export(wrapper, example, strict=False).run_decompositions(get_decomp_table())
    converter = TorchConverter(mode=TorchConverter.Mode.RELEASE)
    if rotary is not None:
        converter.register_custom_kernels([rotary.kernel])
    converter.add_exported_program(program, input_names=[item["name"] for item in specification["inputs"]],
                                   output_names=[item["name"] for item in specification["outputs"]],
                                   state_names=[], entrypoint_name="main")
    asset = converter.to_coreai()
    metadata = AIModelAssetMetadata()
    metadata.author = "Qwen Team, Alibaba Group; conversion by Yuji Oka"
    metadata.license = "Qwen Research License"
    metadata.model_description = f"Qwen Image 2.1 Turbo mobile candidate: {args.kind}; numerical and quality validation required"
    metadata.set_custom("source_revision", profile["source"]["revision"])
    metadata.set_custom("profile_sha256", canonical_hash(profile))
    if rotary is not None:
        metadata.set_custom("rotary_arithmetic", "float32-explicit-metal-fma")
    target = args.output / (args.kind + ".aimodel")
    if target.exists() or target.is_symlink():
        raise FileExistsError("The asset destination changed during conversion")
    asset.save_asset(target, metadata, minimum_os=OSVersion.v27)
    elapsed = time.perf_counter() - started
    del asset, converter, program, wrapper, original
    gc.collect()
    if profile["source"]["converter_sha256"] != converter_identity():
        raise ValueError("Converter code changed during export")
    files = inventory(args.output / (args.kind + ".aimodel"))
    write_new_json(args.output / "component.json", {"format": 2, "kind": args.kind, "profile": profile,
              "profile_sha256": canonical_hash(profile), "asset": args.kind + ".aimodel",
              "inputs": specification["inputs"], "outputs": specification["outputs"], "state_names": [],
              "conversion_complete": True, "numeric_verification": "pending", "device_verification": "pending",
              "compression_file": file_record(args.output / "compression.json"), "asset_inventory": files,
              "asset_bytes": sum(item["bytes"] for item in files.values()), "source_files": source,
              "versions": versions, "target_minimum_os": "27.0", "export_seconds": elapsed})


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser("profile")
    profile.add_argument("--candidate", choices=CANDIDATES, required=True)
    profile.add_argument("--dependency-lock", type=Path, required=True)
    profile.add_argument("--output", type=Path, required=True)
    export = commands.add_parser("export")
    export.add_argument("--snapshot", type=Path, required=True)
    export.add_argument("--profile", type=Path, required=True)
    export.add_argument("--dependency-lock", type=Path, required=True)
    export.add_argument("--kind", choices=("vision", "text", "denoiser", "vae_encoder", "vae_decoder"), required=True)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "profile":
        validate_environment()
        new_output(args.output, ("profile.json",))
        target = args.output / "profile.json"
        write_new_json(target, make_profile(args.candidate, digest(args.dependency_lock), converter_identity()))
    else:
        export_component(args)


if __name__ == "__main__":
    main()
