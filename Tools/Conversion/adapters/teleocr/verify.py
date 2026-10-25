import argparse
import json
import subprocess
from pathlib import Path

from .contracts import digest, inspected_functions, read_json, write_json


def inspect(path):
    command = ["xcrun", "coreai-build", "inspect", str(path), "--io", "--metadata", "--storage", "--compute", "--json"]
    return json.loads(subprocess.check_output(command, text=True, timeout=600))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--component", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = read_json(args.bundle / "metadata.json")
    files = {str(path.relative_to(args.bundle)): digest(path)
             for path in sorted(args.bundle.rglob("*")) if path.is_file() and path != args.bundle / "metadata.json"}
    result = {"files_match": files == metadata["files"], "components": {}, "assets": {}}
    for component in args.component:
        comparison = read_json(component / "comparison.json")
        result["components"][comparison["kind"]] = comparison["passed"]
    for kind, name in metadata["assets"].items():
        report = inspect(args.bundle / name)
        observed = inspected_functions(report)
        result["assets"][kind] = {"matches_contract": observed == metadata["functions"][kind], "functions": observed,
                                  "storage_types": report["summary"].get("storageTypes"),
                                  "compute_types": report["summary"].get("computeTypes"),
                                  "metadata": report.get("metadata"),
                                  "bytes": sum(path.stat().st_size for path in (args.bundle / name).rglob("*") if path.is_file())}
    result["passed"] = (result["files_match"] and set(result["components"]) == {"vision", "embed", "decoder"}
                        and all(result["components"].values())
                        and all(asset["matches_contract"] for asset in result["assets"].values()))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "comparison.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "assets"}), flush=True)
    if not result["passed"]:
        raise SystemExit("The bundle does not match its contract")


if __name__ == "__main__":
    main()
