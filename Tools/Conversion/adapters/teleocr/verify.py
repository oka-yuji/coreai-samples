import argparse
import json
import subprocess
from pathlib import Path

from .contracts import digest, inspected_functions, read_json, write_json


def inspect(path):
    command = ["xcrun", "coreai-build", "inspect", str(path), "--io", "--metadata", "--storage", "--compute", "--json"]
    return json.loads(subprocess.check_output(command, text=True, timeout=600))


def compressed_storage_matches(record, report):
    bits = record.get("bits")
    layers = record.get("layers", [])
    if bits not in (4, 8) or not layers or len({row["name"] for row in layers}) != len(layers):
        return False
    functions = record.get("functions", {})
    expected_functions = {"prefill", "main"} if record.get("kind") == "decoder" else {"main"}
    prefix = "decoder." if record.get("kind") == "decoder" else ""
    modules = sorted(prefix + row["name"] for row in layers)
    if (set(functions) != expected_functions
            or {row["name"] for row in report["summary"].get("functions", [])} != expected_functions
            or any(row != {"modules": modules, "parameters_match": True} for row in functions.values())):
        return False
    expected = sum(row["integer"]["shape"][0] * row["integer"]["shape"][1] for row in layers)
    stored = {row["name"]: row["count"] for row in report["summary"].get("storageTypes", [])}
    return expected > 0 and stored.get(f"Int{bits}", 0) >= expected


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
    compression = {}
    for component in args.component:
        comparison = read_json(component / "comparison.json")
        result["components"][comparison["kind"]] = comparison["passed"]
        if "compression" in comparison:
            compression[comparison["kind"]] = comparison["compression"]
    result["compression_matches"] = compression == metadata.get("compression", {})
    for kind, name in metadata["assets"].items():
        report = inspect(args.bundle / name)
        observed = inspected_functions(report)
        result["assets"][kind] = {"matches_contract": observed == metadata["functions"][kind], "functions": observed,
                                  "storage_types": report["summary"].get("storageTypes"),
                                  "compute_types": report["summary"].get("computeTypes"),
                                  "metadata": report.get("metadata"),
                                  "bytes": sum(path.stat().st_size for path in (args.bundle / name).rglob("*") if path.is_file())}
        profile = compression.get("decoder" if kind == "main" else kind)
        result["assets"][kind]["compression_matches"] = profile is None or compressed_storage_matches(profile, report)
    result["passed"] = (result["files_match"] and set(result["components"]) == {"vision", "embed", "decoder"}
                        and all(result["components"].values())
                        and result["compression_matches"]
                        and all(asset["matches_contract"] and asset["compression_matches"]
                                for asset in result["assets"].values()))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "comparison.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "assets"}), flush=True)
    if not result["passed"]:
        raise SystemExit("The bundle does not match its contract")


if __name__ == "__main__":
    main()
