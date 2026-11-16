import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from ...storage import file_hash, read_json, write_json
from .source import load_contract


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


def record(asset, precision, before, after, reloaded, files_before, files_after):
    return {"asset": asset, "precision": precision, "semantic": before, "semantic_equal": before == after == reloaded,
            "bytecode_changed": files_before["main.mlirb"] != files_after["main.mlirb"],
            "files_before": files_before, "files_after": files_after}


def passed(rows, count):
    return len(rows) == count and all(row["semantic_equal"] and row["bytecode_changed"] for row in rows)


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
            source, target = directory / entry["file"], args.output / f"{export['asset']}-{precision}.aimodel"
            loaded = AIModelAsset.load(source)
            program = loaded.program
            before = semantic(program)
            strip_debug_info(program)
            after = semantic(program)
            program.save_asset(target, loaded.metadata)
            del loaded, program
            reloaded = json.loads(subprocess.run([sys.executable, "-m", __spec__.name, "digest", str(target)], check=True,
                                                 capture_output=True, text=True).stdout)
            report["assets"].append(record(export["asset"], precision, before, after, reloaded, folder_files(source),
                                           folder_files(target)))
            write_json(args.output / "strip.json", report)
    report["passed"] = passed(report["assets"], 2 * len(contract["assets"]))
    report["complete"] = True
    write_json(args.output / "strip.json", report)
    if not report["passed"]:
        raise RuntimeError("Removing debug information changed an asset")


def main():
    if sys.argv[1:2] == ["digest"]:
        digest(sys.argv[2])
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--export", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    strip(args)


if __name__ == "__main__":
    main()
