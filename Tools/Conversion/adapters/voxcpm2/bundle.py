import argparse
import ctypes
import getpass
import hashlib
import json
from pathlib import Path
import shutil
from string import Template
import tempfile

from ...storage import file_hash, read_json, write_json
from .runtime import entries
from .source import load_contract

FORMAT = 1
CHUNK = 64 * 1024 * 1024
ROOTS = ("Users", "home", "Volumes", "var/folders", "private/var", "private/tmp", "tmp")
SEPARATORS = ("/", "\\/", "\\u002f", "\\u002F", "%2f", "%2F")
ENCODINGS = ("utf-8", "utf-16-le", "utf-16-be")
UPSTREAM_LICENSE = {"bytes": 11298, "sha256": "4f10acc209addacfad28293315c74c4cd648f771ee1263a748f1781d1e0265e4"}


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
                          "sample_rate": architecture["sample_rate"]},
            "tokenizer": "tokenizer.json"}


def notice(template, contract, name, precisions):
    converted = sorted(asset for asset, precision in precisions.items() if precision == "fp16")
    precision = (f"- Assets converted to 16-bit floating point: {', '.join(converted)}. In those assets,\n"
                 "  RMSNorm and the sinusoidal time embedding still compute in 32-bit floating point.\n"
                 if converted else "- All assets keep 32-bit floating point.\n")
    return template.substitute(model_id=contract["model_id"], revision=contract["revision"],
                               upstream_revision=contract["upstream_revision"], configuration=name, precision=precision)


def listing(files):
    rows = [{"path": path, **files[path]} for path in sorted(files)]
    return {"format": FORMAT, "count": len(rows), "bytes": sum(row["bytes"] for row in rows), "files": rows}


def inventory(root):
    files = {}
    for path in Path(root).rglob("*"):
        if path.is_symlink():
            raise ValueError(f"The bundle contains a symbolic link: {path.relative_to(root).as_posix()}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = file_hash(path)
    return listing(files)


def verify_inventory(root, expected):
    actual = {entry["path"]: entry for entry in inventory(root)["files"]}
    wanted = {entry["path"]: entry for entry in expected["files"]}
    problems = {"missing": sorted(set(wanted) - set(actual)), "extra": sorted(set(actual) - set(wanted)),
                "changed": sorted(path for path in set(actual) & set(wanted) if actual[path] != wanted[path])}
    if any(problems.values()):
        raise ValueError(f"The bundle does not match its inventory: {problems}")


def patterns(user=None):
    names = {*(separator + root.replace("/", separator) + separator for root in ROOTS for separator in SEPARATORS),
             "-Users-", user or getpass.getuser()}
    return sorted({name.encode(encoding) for name in names for encoding in ENCODINGS})


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
    home = absolute("Users", "someone", "work")
    samples = {"utf8": home.encode() + b"\0", "utf16le": home.encode("utf-16-le"),
               "utf16be": absolute("Users", "").encode("utf-16-be"), "json": home.replace("/", "\\/").encode(),
               "unicode": home.replace("/", "\\u002F").encode(), "percent": home.replace("/", "%2f").encode(),
               "volume": ("x" + absolute("Volumes", "disk", "y")).encode(),
               "temporary": absolute("private", "var", "folders", "ab", "T", "z").encode(),
               "boundary": b"a" * 13 + absolute("Volumes", "").encode() + b"b" * 20,
               "user": ("owner " + (user or getpass.getuser())).encode(),
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
    if clonefile is not None and clonefile(str(source).encode(), str(target).encode(), 0) == 0:
        return True
    shutil.copyfile(source, target)
    return False


def clone_tree(source, target):
    counts = {"cloned": 0, "copied": 0}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Source contains a symbolic link: {path.name}")
        if path.is_file():
            counts["cloned" if clone(path, target / path.relative_to(source)) else "copied"] += 1
    return counts


def build(root, name, precisions, contract, stripped, records, tokenizer, license, template, license_pin=UPSTREAM_LICENSE):
    expected, copies = {}, {"cloned": 0, "copied": 0}
    for asset, precision in precisions.items():
        folder = asset_path(asset, precision)
        for kind, count in clone_tree(stripped / Path(folder).name, root / folder).items():
            copies[kind] += count
        expected |= {f"{folder}/{file}": entry for file, entry in records[(asset, precision)].items()}
    for file, text in (("manifest.json", json.dumps(manifest(contract, name, precisions), indent=2) + "\n"),
                       ("NOTICE", notice(template, contract, name, precisions))):
        data = text.encode()
        (root / file).write_bytes(data)
        expected[file] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    for file, source, pin in (("tokenizer.json", tokenizer, contract["files"]["tokenizer.json"]), ("LICENSE", license, license_pin)):
        shutil.copyfile(source, root / file)
        expected[file] = pin
    result = listing(expected)
    verify_inventory(root, result)
    return result, copies


def verdict(report, names):
    return (sorted(report["bundles"]) == sorted(names)
            and all(bundle["path_scan"]["hits"] == 0 for bundle in report["bundles"].values())
            and all(report["canaries"]["detected"].values()) and not report["canaries"]["clean_detected"])


def assemble(args):
    contract = load_contract(args.contract)
    configurations = read_json(args.settings)["compare"]["configurations"]
    stripped = read_json(args.stripped / "strip.json")
    if not stripped["complete"] or not stripped["passed"]:
        raise ValueError("The stripped assets were not verified")
    records = {(row["asset"], row["precision"]): row["files_after"] for row in stripped["assets"]}
    template = Template(args.notice.read_text(encoding="utf-8"))
    tokens = patterns()
    args.output.mkdir(parents=True)
    report = {"complete": False, "passed": False, "bundles": {}, "patterns": len(tokens), "encodings": list(ENCODINGS),
              "separators": list(SEPARATORS)}
    for name, precisions in configurations.items():
        root = args.output / name
        result, copies = build(root, name, precisions, contract, args.stripped, records, args.tokenizer, args.license, template)
        write_json(args.output / f"{name}.inventory.json", result)
        report["bundles"][name] = {"files": result["count"], "bytes": result["bytes"], "asset_files": copies,
                                   "manifest_sha256": file_hash(root / "manifest.json")["sha256"],
                                   "inventory_sha256": file_hash(args.output / f"{name}.inventory.json")["sha256"],
                                   "path_scan": scan(root, tokens)}
        write_json(args.output / "check.json", report)
    report["before_removal"] = {read_json(directory / "export.json")["asset"]: scan(directory, tokens) for directory in args.export}
    report["canaries"] = canaries(tokens)
    report["passed"] = verdict(report, configurations)
    report["complete"] = True
    write_json(args.output / "check.json", report)
    if not report["passed"]:
        raise RuntimeError("A bundle failed its checks")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--export", type=Path, action="append", required=True)
    parser.add_argument("--stripped", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--notice", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assemble(args)


if __name__ == "__main__":
    main()
