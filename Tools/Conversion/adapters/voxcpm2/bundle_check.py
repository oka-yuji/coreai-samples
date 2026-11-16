import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from ...storage import read_json, write_json
from .checks import Checks, aggregate, fsq_summary, verdict
from .runtime import Components
from .source import load_contract


def shim(root, scratch, contract):
    content = read_json(root / "manifest.json")
    groups = {}
    for asset, entry in content["assets"].items():
        directory = scratch / content["configuration"]["name"] / asset
        directory.mkdir(parents=True)
        write_json(directory / "export.json", {"asset": asset, "complete": True, "contract": contract,
                                               "precisions": {entry["precision"]: {"file": str((root / entry["path"]).resolve())}}})
        groups.setdefault(entry["precision"], []).append(directory)
    return groups


def judge(summary, fsq, criterion_met, baseline):
    def counts(values):
        return {key: {name: value[name] for name in ("count", "passed", "gate")} for key, value in values.items()}

    def classes(values):
        return {key: values[key] for key in ("calls", "match", "boundary", "fail")}

    result = {"criterion_met": criterion_met, "counts_equal_before_removal": counts(summary) == counts(baseline["summary"]),
              "fsq_equal_before_removal": classes(fsq) == classes(baseline["fsq"]),
              "errors_equal_before_removal": {key: value.get("normalized_max") for key, value in summary.items()}
              == {key: value.get("normalized_max") for key, value in baseline["summary"].items()}}
    result["passed"] = bool(criterion_met and result["counts_equal_before_removal"] and result["fsq_equal_before_removal"])
    return result


def recheck(args):
    contract = load_contract(args.contract)
    reference = read_json(args.reference / "reference.json")
    baseline = read_json(args.baseline / "comparison.json")
    if not reference["complete"] or reference["contract"] != contract or not baseline["complete"] or not baseline["passed"]:
        raise ValueError("The reference or the comparison before removal is not usable")
    configurations = sorted(path.name for path in args.bundles.iterdir() if path.is_dir())
    if "fp32" not in configurations:
        raise ValueError("The FP32 bundle is required")
    args.output.mkdir(parents=True)
    report = {"complete": False, "passed": False, "descriptors": {}}
    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        for name in configurations:
            for precision, directories in shim(args.bundles / name, Path(scratch), contract).items():
                components = Components(directories, precision, contract)
                report["descriptors"].setdefault(name, {})[precision] = sorted(components.exports)
                if name == "fp32":
                    rows = []
                    for record in reference["records"]:
                        rows.extend(Checks(components, contract, args.reference, record, np.float32).run(list(contract["assets"])))
                components.close()
    summary, fsq = aggregate(rows), fsq_summary(rows, contract["criteria"])
    gated = [row for row in rows if row["gate"]]
    report |= {"summary": summary, "fsq": fsq, "gated_rows": len(gated), "gated_failed": sum(not row["passed"] for row in gated)}
    report |= judge(summary, fsq, verdict(rows, contract["criteria"]), baseline)
    write_json(args.output / "rows.json", rows)
    report["complete"] = True
    write_json(args.output / "comparison.json", report)
    print(json.dumps({key: report[key] for key in ("passed", "gated_rows", "gated_failed", "counts_equal_before_removal",
                                                   "fsq_equal_before_removal", "errors_equal_before_removal")}), flush=True)
    if not report["passed"]:
        raise RuntimeError("The bundle assets do not reproduce the comparison before removal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    recheck(args)


if __name__ == "__main__":
    main()
