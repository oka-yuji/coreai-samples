import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from string import Template

from ...storage import canonical, file_hash, write_json
from . import bundle, strip
from .source import load_contract

PROFILE = "reference_fp32"
BASE = "fp32"
ASSET = "refenc"
PRECISION = "fp32"
FORMAT = 2
ARTIFACT = bundle.asset_path(ASSET, PRECISION)
RAW = Path(ARTIFACT).name
FILES = ("main.hash", "main.mlirb", "metadata.json")
ROOT_FILES = ("LICENSE", "NOTICE", "manifest.json", "tokenizer.json")
GENERATED = ("NOTICE", "manifest.json")
BUNDLE_FILES = 25
CASES = 4
LIMIT = 0.001
DESCRIPTOR = {"inputs": [["pcm", "float32", [1, 1, 161280]]], "outputs": [["mu", "float32", [1, 64, 252]]], "states": []}
REFERENCE = {"sample_rate": 16000, "minimum_samples": 80000, "maximum_samples": 160000, "hop_samples": 640,
             "start_token": 103, "end_token": 104, "padding": "right_zero"}
REPOSITORY = Path(__file__).resolve().parents[4]
RECIPE = REPOSITORY / "Tools/Conversion/recipes/voxcpm2"
INVENTORY = REPOSITORY / "CoreAISamplesKit/Sources/SpeechAssets/Resources/voxcpm2-fp32.inventory.json"
SHA256 = re.compile(r"[0-9a-f]{64}")


def parse(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def regular(path):
    if not stat.S_ISREG(os.lstat(path).st_mode):
        raise ValueError(f"Expected a regular file, not a link or a folder: {path.name}")
    return path


def folder(path):
    if not stat.S_ISDIR(os.lstat(path).st_mode):
        raise ValueError(f"Expected a folder, not a link or a file: {path.name}")
    return path


def document(path):
    data = regular(path).read_bytes()
    return parse(data), {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def exact(value, names, label):
    if not isinstance(value, dict) or sorted(value) != sorted(names):
        raise ValueError(f"{label} must have exactly the fields {sorted(names)}")
    return value


def pin(value, label):
    exact(value, ("bytes", "sha256"), label)
    if type(value["bytes"]) is not int or value["bytes"] < 1 or type(value["sha256"]) is not str \
            or not SHA256.fullmatch(value["sha256"]):
        raise ValueError(f"{label} needs a positive integer size and a lowercase hexadecimal SHA-256")
    return value


def selection(path):
    value, identity = document(path)
    exact(value, ("format", "profile", "report", "files"), "The selection")
    if type(value["format"]) is not int or value["format"] != 1 or value["profile"] != PROFILE:
        raise ValueError(f"The selection must be format 1 of the {PROFILE} profile")
    pin(value["report"], "The selected report")
    for name in exact(value["files"], FILES, "The selected files"):
        pin(value["files"][name], f"The selected {name}")
    return value, identity


def lookup(value, keys):
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def eligible(report, files, contract):
    sizes = {name: files[name]["bytes"] for name in FILES}
    expected = {("version",): 1, ("stage",): "complete", ("complete",): True, ("passed",): True,
                ("precision",): PRECISION, ("descriptor",): DESCRIPTOR, ("source_encoder", "unchanged"): True,
                ("contract",): {**{key: contract[key] for key in ("model_id", "revision", "upstream_revision", "files")},
                                "assets": {ASSET: {"functions": {"main": DESCRIPTOR}}}},
                ("export", "file"): RAW, ("export", "files"): sizes, ("export", "bytes"): sum(sizes.values()),
                ("export", "main_mlirb_sha256"): files["main.mlirb"]["sha256"]}
    expected |= {("summary", key): CASES for key in ("cases", "required_cases", "passed_cases", "original_inputs_matched")}
    for kind in ("fixed", "coreai"):
        expected |= {("summary", kind, "count"): CASES, ("summary", kind, "passed"): CASES, ("summary", kind, "nonfinite"): 0}
    wrong = [".".join(keys) for keys, value in expected.items() if canonical(lookup(report, keys)) != canonical(value)]
    for kind in ("fixed", "coreai"):
        value = lookup(report, ("summary", kind, "normalized_max"))
        if type(value) not in (int, float) or not math.isfinite(value) or value > LIMIT:
            wrong.append(f"summary.{kind}.normalized_max")
    if wrong:
        raise ValueError(f"The selected report is not an eligible FP32 encoder run: {wrong}")


def encoder(run, chosen, contract):
    report, identity = document(folder(run) / "report.json")
    if identity != chosen["report"]:
        raise ValueError("report.json differs from the selected size or SHA-256")
    source = folder(run / RAW)
    if sorted(path.name for path in source.iterdir()) != sorted(FILES):
        raise ValueError(f"The encoder folder must contain exactly {list(FILES)}")
    files = {name: file_hash(regular(source / name)) for name in FILES}
    if files != chosen["files"]:
        raise ValueError("The encoder files differ from the selected sizes or SHA-256 values")
    eligible(report, files, contract)
    return source, files


def trusted(contract):
    listing, identity = document(INVENTORY)
    exact(listing, ("format", "count", "bytes", "files"), "The trusted inventory")
    files = {}
    for row in listing["files"] if isinstance(listing["files"], list) else []:
        exact(row, ("path", "bytes", "sha256"), "A trusted inventory row")
        files[row["path"]] = pin({"bytes": row["bytes"], "sha256": row["sha256"]}, "A trusted inventory row")
    names = {*ROOT_FILES, *(f"{bundle.asset_path(asset, PRECISION)}/{name}" for asset in contract["assets"] for name in FILES)}
    if set(files) != names or canonical(listing) != canonical(bundle.listing(files)):
        raise ValueError("The trusted inventory does not list the files of the FP32 bundle")
    return listing, files, identity


def base(path, contract):
    listing, files, identity = trusted(contract)
    bundle.verify_inventory(folder(path), listing)
    manifest = document(path / "manifest.json")[0]
    if canonical(manifest) != canonical(bundle.manifest(contract, BASE, {asset: PRECISION for asset in contract["assets"]})):
        raise ValueError("The base manifest is not the format 1 FP32 manifest of model.json")
    return files, manifest, identity


def reference_manifest(base_manifest):
    content = copy.deepcopy(base_manifest)
    content["format"] = FORMAT
    content["configuration"] = {"name": PROFILE,
                                "precisions": {**base_manifest["configuration"]["precisions"], ASSET: PRECISION}}
    content["assets"][ASSET] = {"path": ARTIFACT, "precision": PRECISION, "functions": {"main": copy.deepcopy(DESCRIPTOR)}}
    content["reference"] = dict(REFERENCE)
    return content


def destination(output, inputs):
    output = Path(os.path.abspath(output))
    if os.path.lexists(output):
        raise FileExistsError("The output already exists; use a new folder")
    resolved = output.parent.resolve() / output.name
    if any(resolved.is_relative_to(Path(path).resolve()) for path in inputs):
        raise ValueError("The output must be outside the base bundle and the encoder run")
    return output


def package(root, source, files, base_manifest, row, contract, template):
    expected, copies = {}, {"cloned": 0, "copied": 0}
    for name, entry in files.items():
        if name not in GENERATED:
            copies["cloned" if bundle.clone(source / name, root / name) else "copied"] += 1
            expected[name] = entry
    expected |= {f"{ARTIFACT}/{name}": entry for name, entry in row["files_after"].items()}
    content = reference_manifest(base_manifest)
    notice = bundle.notice(template, contract, PROFILE, content["configuration"]["precisions"])
    for name, text in (("manifest.json", json.dumps(content, indent=2) + "\n"), ("NOTICE", notice)):
        data = text.encode()
        with (root / name).open("xb") as handle:
            handle.write(data)
        expected[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    listing = bundle.listing(expected)
    bundle.verify_inventory(root, listing)
    return listing, copies


def verdict(check):
    scan = check["path_scan"]
    return (strip.passed([check["strip"]], 1) and check["files"] == scan["files"] == BUNDLE_FILES and scan["hits"] == 0
            and all(check["canaries"]["detected"].values()) and not check["canaries"]["clean_detected"])


def assemble(args):
    contract = load_contract(RECIPE / "model.json")
    template = Template((RECIPE / "bundle-notice.txt").read_text(encoding="utf-8"))
    output = destination(args.output, (args.base_bundle, args.encoder_run))
    chosen, selected = selection(args.selection)
    source, raw = encoder(args.encoder_run, chosen, contract)
    files, base_manifest, listed = base(args.base_bundle, contract)
    output.mkdir(parents=True)
    tokens = bundle.patterns()
    check = {"complete": False, "passed": False, "profile": PROFILE, "selection_sha256": selected["sha256"],
             "base_inventory_sha256": listed["sha256"], "patterns": len(tokens), "encodings": list(bundle.ENCODINGS),
             "separators": list(bundle.SEPARATORS)}
    write_json(output / "check.json", check)
    root = output / PROFILE
    (root / "artifacts").mkdir(parents=True)
    row = check["strip"] = strip.strip_one(ASSET, PRECISION, source, root / ARTIFACT)
    write_json(output / "check.json", check)
    if not strip.passed([row], 1) or row["files_before"] != raw or sorted(row["files_after"]) != sorted(FILES):
        raise RuntimeError("Removing debug information changed the selected encoder")
    listing, copies = package(root, args.base_bundle, files, base_manifest, row, contract, template)
    check.update(files=listing["count"], bytes=listing["bytes"], retained_files=copies,
                 manifest_sha256=file_hash(root / "manifest.json")["sha256"], before_removal=bundle.scan(source, tokens),
                 path_scan=bundle.scan(root, tokens), canaries=bundle.canaries(tokens))
    check["passed"] = verdict(check)
    if check["passed"]:
        write_json(output / f"{PROFILE}.inventory.json", listing)
        check["inventory_sha256"] = file_hash(output / f"{PROFILE}.inventory.json")["sha256"]
    check["complete"] = True
    write_json(output / "check.json", check)
    if not check["passed"]:
        raise RuntimeError("The reference bundle failed its checks")
    return check


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--encoder-run", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    check = assemble(parser.parse_args(argv))
    print(json.dumps({key: check[key] for key in ("profile", "passed", "files", "bytes", "selection_sha256",
                                                   "manifest_sha256", "inventory_sha256")}), flush=True)


if __name__ == "__main__":
    main()
