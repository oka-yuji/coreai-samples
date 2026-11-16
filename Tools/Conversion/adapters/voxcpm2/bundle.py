import argparse
import ctypes
import getpass
import hashlib
import json
from pathlib import Path
import shutil
from string import Template
import subprocess
import sys
import tempfile

from ...storage import file_hash, read_json, write_json
from .runtime import entries
from .source import load_contract

FORMAT = 1
CHUNK = 64 * 1024 * 1024
ROOTS = ("Users", "home", "Volumes", "var/folders", "private/var", "private/tmp", "tmp")
PREFIXES = (*(f"/{root}/" for root in ROOTS), "-Users-")


def absolute(*parts):
    return "/".join(["", *parts])


def asset_path(asset, precision):
    return f"artifacts/{asset}-{precision}.aimodel"


def manifest(contract, name, precisions):
    architecture = contract["architecture"]
    if sorted(precisions) != sorted(contract["assets"]) or not set(precisions.values()) <= {"fp32", "fp16"}:
        raise ValueError("Every contract asset needs FP32 or FP16")
    return {"format": FORMAT,
            "model": {key: contract[key] for key in ("model_id", "revision", "upstream_revision", "license")},
            "configuration": {"name": name, "precisions": dict(precisions)},
            "assets": {asset: {"path": asset_path(asset, precisions[asset]), "precision": precisions[asset],
                               "functions": {function: {kind: entries(contract, asset, function, kind, precisions[asset])
                                                        for kind in ("inputs", "outputs", "states")}
                                             for function in contract["assets"][asset]["functions"]}}
                       for asset in contract["assets"]},
            "execution": contract["execution"], "generation": contract["generation"],
            "structure": {"patch": [1, architecture["feat_dim"], architecture["patch_size"]],
                          "sample_rate": architecture["sample_rate"],
                          "samples_per_patch": contract["generation"]["samples_per_patch"]},
            "tokenizer": "tokenizer.json"}


def inventory(root):
    files = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise ValueError(f"The bundle contains a symbolic link: {path.relative_to(root).as_posix()}")
        if path.is_file():
            files.append({"path": path.relative_to(root).as_posix(), **file_hash(path)})
    return {"format": FORMAT, "count": len(files), "bytes": sum(entry["bytes"] for entry in files), "files": files}


def verify_inventory(root, expected):
    actual = {entry["path"]: entry for entry in inventory(root)["files"]}
    wanted = {entry["path"]: entry for entry in expected["files"]}
    problems = {"missing": sorted(set(wanted) - set(actual)), "extra": sorted(set(actual) - set(wanted)),
                "changed": sorted(path for path in set(actual) & set(wanted) if actual[path] != wanted[path])}
    if any(problems.values()):
        raise ValueError(f"The bundle does not match its inventory: {problems}")


def patterns(user=None):
    names = {*PREFIXES, user or getpass.getuser()}
    return sorted({name.encode(encoding) for name in names for encoding in ("utf-8", "utf-16-le")})


def path_hits(path, tokens, chunk=CHUNK):
    width, count, tail = max(map(len, tokens)) - 1, 0, b""
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            buffer = tail + block
            for token in tokens:
                start = buffer.find(token)
                while start != -1:
                    count += start + len(token) > len(tail)
                    start = buffer.find(token, start + 1)
            tail = buffer[-width:]
    return count


def scan(root, tokens):
    files = [path for path in sorted(Path(root).rglob("*")) if path.is_file()]
    hits = {path.relative_to(root).as_posix(): path_hits(path, tokens) for path in files}
    return {"files": len(files), "bytes": sum(path.stat().st_size for path in files),
            "hits": sum(hits.values()), "files_with_hits": sorted(name for name, count in hits.items() if count)}


def canaries(tokens, user=None):
    home = absolute("Users", user or getpass.getuser(), "work")
    samples = {"utf8": home.encode() + b"\0", "utf16": home.encode("utf-16-le"),
               "volume": ("x" + absolute("Volumes", "disk", "y")).encode(),
               "temporary": absolute("private", "var", "folders", "ab", "T", "z").encode(),
               "boundary": b"a" * 13 + absolute("Volumes", "").encode() + b"b" * 20,
               "clean": b"artifacts/tslm-fp32.aimodel/main.mlirb"}
    with tempfile.TemporaryDirectory() as directory:
        found = {}
        for name, data in samples.items():
            path = Path(directory) / name
            path.write_bytes(data)
            found[name] = path_hits(path, tokens, chunk=16) > 0
    return {"detected": {name: value for name, value in found.items() if name != "clean"}, "clean_detected": found["clean"]}


def clone(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    clonefile = getattr(ctypes.CDLL(None, use_errno=True), "clonefile", None)
    if clonefile is None or clonefile(str(source).encode(), str(target).encode(), 0) != 0:
        shutil.copyfile(source, target)


def clone_tree(source, target):
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Source contains a symbolic link: {path.name}")
        if path.is_file():
            clone(path, target / path.relative_to(source))


class DigestWriter:
    def __init__(self):
        self.digest, self.size = hashlib.sha256(), 0

    def write(self, value):
        data = value if isinstance(value, bytes) else value.encode()
        self.digest.update(data)
        self.size += len(data)
        return len(value)


def semantic(program):
    sink = DigestWriter()
    operation = program._module._mlir_module.operation
    operation.verify()
    operation.print(file=sink, binary=True, enable_debug_info=False, print_generic_op_form=True,
                    large_elements_limit=None, large_resource_limit=None)
    return {"sha256": sink.digest.hexdigest(), "bytes": sink.size}


def digest(path):
    from coreai.authoring import AIModelAsset

    print(json.dumps(semantic(AIModelAsset.load(Path(path)).program)), flush=True)


def folder_files(folder):
    return {path.name: file_hash(path) for path in sorted(folder.iterdir()) if path.is_file()}


def strip(args):
    from coreai.authoring import AIModelAsset
    from coreai_torch.debugging.debug_info import strip_debug_info

    contract = load_contract(args.contract)
    args.output.mkdir(parents=True)
    report = {"complete": False, "passed": False, "method": "AIModelAsset.load, strip_debug_info, save_asset",
              "comparison": "generic MLIR without locations, with every constant and resource, after loading, after stripping and "
                            "after reloading the saved asset in a new process", "assets": []}
    for directory in args.export:
        export = read_json(directory / "export.json")
        if not export["complete"] or export["contract"] != contract:
            raise ValueError("Unexpected or incomplete export")
        for precision, entry in sorted(export["precisions"].items()):
            source, target = directory / entry["file"], args.output / Path(asset_path(export["asset"], precision)).name
            loaded = AIModelAsset.load(source)
            program = loaded.program
            before = semantic(program)
            strip_debug_info(program)
            after = semantic(program)
            program.save_asset(target, loaded.metadata)
            del loaded, program
            reloaded = json.loads(subprocess.run([sys.executable, "-m", __spec__.name, "digest", str(target)], check=True,
                                                 capture_output=True, text=True).stdout)
            files_before, files_after = folder_files(source), folder_files(target)
            report["assets"].append({"asset": export["asset"], "precision": precision, "semantic": before,
                                     "semantic_equal": before == after == reloaded,
                                     "bytecode_changed": files_before["main.mlirb"] != files_after["main.mlirb"],
                                     "files_before": files_before, "files_after": files_after})
            write_json(args.output / "strip.json", report)
    report["passed"] = (len(report["assets"]) == 2 * len(contract["assets"])
                        and all(row["semantic_equal"] and row["bytecode_changed"] for row in report["assets"]))
    report["complete"] = True
    write_json(args.output / "strip.json", report)
    if not report["passed"]:
        raise RuntimeError("Removing debug information changed an asset")


def assemble(args):
    contract = load_contract(args.contract)
    configurations = read_json(args.settings)["compare"]["configurations"]
    stripped = read_json(args.stripped / "strip.json")
    if not stripped["complete"] or not stripped["passed"]:
        raise ValueError("The stripped assets were not verified")
    license_text = args.license.read_text(encoding="utf-8")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise ValueError("The upstream license is not Apache-2.0")
    template = Template(args.notice.read_text(encoding="utf-8"))
    tokens = patterns()
    args.output.mkdir(parents=True)
    report = {"complete": False, "passed": False, "bundles": {}, "patterns": len(tokens), "encodings": ["utf-8", "utf-16-le"]}
    for name, precisions in configurations.items():
        root = args.output / name
        for asset, precision in precisions.items():
            clone_tree(args.stripped / Path(asset_path(asset, precision)).name, root / asset_path(asset, precision))
        content = manifest(contract, name, precisions)
        (root / "manifest.json").write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
        shutil.copyfile(args.tokenizer, root / "tokenizer.json")
        shutil.copyfile(args.license, root / "LICENSE")
        converted = sorted(asset for asset, precision in precisions.items() if precision == "fp16")
        (root / "NOTICE").write_text(template.substitute(
            model_id=contract["model_id"], revision=contract["revision"], upstream_revision=contract["upstream_revision"],
            configuration=name, half_precision=", ".join(converted) if converted else "none"), encoding="utf-8")
        listing = inventory(root)
        expected = {"manifest.json", "tokenizer.json", "LICENSE", "NOTICE"} | {
            f"{asset_path(asset, precision)}/{file}" for asset, precision in precisions.items()
            for file in folder_files(args.stripped / Path(asset_path(asset, precision)).name)}
        if {entry["path"] for entry in listing["files"]} != expected:
            raise ValueError(f"Unexpected bundle files in {name}")
        write_json(args.output / f"{name}.inventory.json", listing)
        verify_inventory(root, listing)
        report["bundles"][name] = {"files": listing["count"], "bytes": listing["bytes"],
                                   "manifest_sha256": file_hash(root / "manifest.json")["sha256"],
                                   "inventory_sha256": file_hash(args.output / f"{name}.inventory.json")["sha256"],
                                   "path_scan": scan(root, tokens)}
        write_json(args.output / "check.json", report)
    report["before_removal"] = {read_json(directory / "export.json")["asset"]: scan(directory, tokens) for directory in args.export}
    report["canaries"] = canaries(tokens)
    report["passed"] = (all(bundle["path_scan"]["hits"] == 0 for bundle in report["bundles"].values())
                        and all(report["canaries"]["detected"].values()) and not report["canaries"]["clean_detected"]
                        and sorted(report["bundles"]) == sorted(configurations))
    report["complete"] = True
    write_json(args.output / "check.json", report)
    if not report["passed"]:
        raise RuntimeError("A bundle failed its checks")


def main():
    if sys.argv[1:2] == ["digest"]:
        digest(sys.argv[2])
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["strip", "assemble"])
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--export", type=Path, action="append")
    parser.add_argument("--stripped", type=Path)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--license", type=Path)
    parser.add_argument("--notice", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    {"strip": strip, "assemble": assemble}[args.mode](args)


if __name__ == "__main__":
    main()
