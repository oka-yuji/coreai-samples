import argparse
import math
from pathlib import Path
import shutil
import stat
import subprocess
import sys

from .contracts import PROFILE, KINDS, read_json, write_json, digest, check_profile, component_inventory


ATTRIBUTION = ("Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 "
               "Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.")
README = """Image Research 512
Built with Qwen. Converted from Qwen-Image-2.1-Turbo.

Local macOS research bundle: FP32, 512-square, batch one, eight denoising steps,
one optional reference image, CFG one, and no prefix KV cache. This is not an
iOS release or a production-quality certification. See bundle.json for the
fixed profile and complete file hashes. Verification covers the recorded
generation/editing cases, not arbitrary image quality.

Model weights and derived assets remain under the Qwen Research License:
research and evaluation only. Commercial use requires a separate license.
Read LICENSE and NOTICE.txt. Conversion does not remove these restrictions.
The independently authored sample code is MIT-licensed in coreai-samples;
that license does not relicense these model assets. Diffusers and Transformers
are Apache-2.0 dependencies and retain their own license terms.
"""


def regular(path, follow=False):
    path = Path(path)
    if path.is_symlink() and follow:
        path = path.resolve(strict=True)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"A regular file is required: {path}")
    return path


def directory(path):
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"A real directory is required: {path}")


def files(root, extensions=None):
    directory(root)
    result = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if extensions is not None and ".cache" in relative.parts:
            continue
        if path.is_dir():
            directory(path)
        elif extensions is None or path.suffix in extensions:
            result.append((regular(path, follow=extensions is not None), relative))
        elif not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError(f"Unsupported configuration entry: {path}")
    if not result:
        raise ValueError(f"No permitted files found: {root}")
    return result


def verification_summary(cases):
    rows = []
    for case in cases:
        row = {"case": case["case"], "passed": True}
        for name in ("latent", "rgba"):
            value = case[name]
            if value.get("finite") is not True or value.get("passed") is not True:
                raise ValueError("Both tensor comparisons must be finite and passing")
            shape = value.get("shape")
            if not isinstance(shape, list) or not shape or any(type(size) is not int or size < 1 for size in shape):
                raise ValueError("Verification shapes must contain positive integer dimensions")
            result = {"finite": True, "shape": shape, "passed": True}
            for key in ("absolute_rmse", "normalized_max", "normalized_rmse"):
                number = value[key]
                if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                    raise ValueError("Verification metrics must be finite nonnegative numbers")
                result[key] = number
            row[name] = result
        rows.append(row)
    return {"passed": True, "cases": rows}


def pack(args):
    output = args.output
    if output.exists() or output.is_symlink():
        directory(output)
        if any(output.iterdir()):
            raise FileExistsError("The bundle output must be empty")
    directory(args.snapshot)
    directory(args.verification)
    directory(args.source_record)
    provenance = read_json(regular(args.verification / "provenance.json"))
    criteria = read_json(regular(args.verification / "criteria.json"))
    source_record = read_json(regular(args.source_record / "source.json"))
    for record in (provenance, criteria, source_record):
        check_profile(record["profile"])
    source_files = source_record.get("files")
    if source_record.get("verified") is not True or not isinstance(source_files, dict) or not source_files:
        raise ValueError("A verified source file inventory is required")
    for relative in source_files:
        if (not isinstance(relative, str) or not relative or "\\" in relative
                or Path(relative).is_absolute() or ".." in Path(relative).parts
                or not Path(relative).parts or Path(relative).as_posix() != relative
                or ":" in Path(relative).parts[0]):
            raise ValueError("Source inventory paths must be safe relative file names")
    if not isinstance(provenance.get("components"), dict) or set(provenance["components"]) != set(KINDS):
        raise ValueError("Verification provenance must identify all five components")
    verification = read_json(regular(args.verification / "comparison.json"))
    cases = verification.get("cases", [])
    if (verification.get("passed") is not True or not isinstance(cases, list) or len(cases) != 2
            or any(not isinstance(case, dict) or case.get("passed") is not True for case in cases)
            or {case.get("case") for case in cases} != {"generate", "edit"}):
        raise ValueError("Both generation and editing verification must pass")
    summary = verification_summary(cases)
    if criteria.get("latent_normalized_rmse") != 5e-3 or criteria.get("rgba_absolute_rmse") != 1e-2:
        raise ValueError("Verification criteria differ from the fixed research profile")
    if any(case["latent"]["normalized_rmse"] > 5e-3 or case["rgba"]["absolute_rmse"] > 1e-2
           for case in summary["cases"]):
        raise ValueError("Verification metrics do not satisfy the fixed research profile")
    planned, kinds = [], set()
    for source in args.components:
        directory(source)
        metadata = read_json(regular(source / "component.json"))
        check_profile(metadata["profile"])
        kind = metadata.get("kind")
        if kind not in KINDS or kind in kinds or metadata.get("passed") is not True:
            raise ValueError("Components must be distinct, recognized, and verified")
        if metadata.get("asset") != f"{kind}.aimodel":
            raise ValueError("The component asset path does not match its kind")
        comparison = read_json(regular(source / "comparison.json"))
        if comparison.get("passed") is not True:
            raise ValueError("Every component comparison must pass")
        if component_inventory(source) != provenance["components"][kind]:
            raise ValueError(f"Component contents differ from the verified run: {kind}")
        kinds.add(kind)
        for name in ("component.json", "comparison.json"):
            planned.append((regular(source / name), Path(kind) / name, False))
        planned.extend((path, Path(kind) / metadata["asset"] / relative, False)
                       for path, relative in files(source / metadata["asset"]))
    if kinds != set(KINDS):
        raise ValueError("All five components are required")
    processor = args.snapshot / "processor"
    planned.extend((path, Path("config/processor") / relative, True)
                   for path, relative in files(processor, {".json", ".txt", ".jinja"}))
    for name in ("text_encoder/config.json", "transformer/config.json", "vae/config.json",
                 "scheduler/scheduler_config.json"):
        directory((args.snapshot / name).parent)
        planned.append((regular(args.snapshot / name, follow=True), Path("config") / name, True))
    license_file = regular(args.snapshot / "LICENSE", follow=True)
    if ATTRIBUTION not in license_file.read_text():
        raise ValueError("The source must include the expected Qwen Research License")
    for source, relative, _ in [*[(source, relative.relative_to("config"), force)
                                 for source, relative, force in planned if force],
                                (license_file, Path("LICENSE"), True)]:
        expected = source_files.get(relative.as_posix())
        if (not isinstance(expected, dict) or source.stat().st_size != expected.get("bytes")
                or digest(source) != expected.get("sha256")):
            raise ValueError(f"Configuration differs from the verified source: {relative}")
    planned.extend(((license_file, Path("LICENSE"), True), (regular(args.lock), Path("uv.lock"), True)))
    planned.extend((regular(args.verification / name), Path(name), True)
                   for name in ("provenance.json", "criteria.json"))
    planned.append((regular(args.source_record / "source.json"), Path("source.json"), True))
    resolved = output.resolve()
    if any(root.resolve() == resolved or root.resolve() in resolved.parents
           for root in [args.snapshot, args.verification, args.source_record, *args.components]):
        raise ValueError("The bundle cannot be written inside a source directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    destination_volume = output if output.exists() else output.parent
    volume = destination_volume.stat().st_dev
    copies = [(source, target, not force and sys.platform == "darwin" and source.stat().st_dev == volume)
              for source, target, force in planned]
    required = sum(source.stat().st_size for source, _, clone in copies if not clone) + 10 * 1024 ** 3
    if shutil.disk_usage(destination_volume).free < required:
        raise OSError("Insufficient free space for file copies plus the 10 GiB reserve")
    output.mkdir(exist_ok=True)
    marker = output / ".packing"
    with marker.open("x") as stream:
        stream.write("Bundle creation in progress\n")
    for source, relative, clone in copies:
        source = regular(source)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to replace a bundle file: {relative}")
        if clone:
            subprocess.run(["/bin/cp", "-c", "-n", str(source), str(target)], check=True)
        else:
            shutil.copy2(source, target, follow_symlinks=True)
        regular(target)
    changes = "\n".join(f"{kind}/{kind}.aimodel: converted and modified for the fixed Core AI profile."
                        for kind in KINDS)
    (output / "NOTICE.txt").write_text(f"Image Research 512\nBuilt with Qwen.\n\n{ATTRIBUTION}\n\n"
                                     f"Modified model assets:\n{changes}\n\n"
                                     "Original configuration and LICENSE files are copied unchanged.\n")
    (output / "README.txt").write_text(README)
    write_json(output / "verification.json", summary)
    marker.unlink()
    inventory = {relative.as_posix(): {"sha256": digest(path), "bytes": path.stat().st_size}
                 for path, relative in files(output)}
    write_json(output / "bundle.json", {"profile": PROFILE, "component_verified": True,
               "end_to_end_verified": True, "components": list(KINDS), "assets": inventory})


def main():
    parser = argparse.ArgumentParser()
    for name in ("snapshot", "output", "verification", "source-record", "lock"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--components", type=Path, nargs="+", required=True)
    pack(parser.parse_args())


if __name__ == "__main__":
    main()
