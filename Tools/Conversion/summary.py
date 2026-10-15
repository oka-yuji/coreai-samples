import argparse
import json
import math
from pathlib import Path
import sys

from .storage import PipelineError, read_json


def choice(value, allowed):
    if value not in allowed:
        raise PipelineError("Unexpected report status or role")
    return value


def number(value, *, integer=False):
    if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value) or value < 0:
        raise PipelineError("Expected a finite nonnegative report metric")
    return value


def project(report, comparison=None):
    if not isinstance(report, dict) or (comparison is not None and not isinstance(comparison, dict)):
        raise PipelineError("Expected report objects")
    if type(report.get("version")) is not int or report["version"] != 1:
        raise PipelineError("Expected a version 1 pipeline report")
    result = {
        "schema": "coreai-conversion-summary/1",
        "status": choice(report["status"], {"running", "completed", "failed", "cancelled"}),
        "verification": choice(report["verification"], {"not_run", "not_configured", "incomplete", "passed", "failed"}),
        "scope": "Fixed-input conversion check; not model production qualification.",
        "stages": [],
    }
    if "elapsed_s" in report:
        result["elapsed_s"] = number(report["elapsed_s"])
    if not isinstance(report["stages"], list):
        raise PipelineError("Expected an array of stages")
    for index, stage in enumerate(report["stages"]):
        row = {
            "index": index,
            "role": choice(stage["role"], {"prepare", "reference", "export", "verify", "benchmark"}),
            "status": choice(stage["status"], {"running", "executed", "reused", "failed", "cancelled"}),
        }
        for key in ("prepare_s", "execute_s", "total_s"):
            if key in stage:
                row[key] = number(stage[key])
        result["stages"].append(row)
    if comparison is not None:
        if type(comparison.get("passed")) is not bool:
            raise PipelineError("Expected a comparison result")
        cases = comparison["cases"]
        if not isinstance(cases, list) or not cases:
            raise PipelineError("Expected comparison cases")
        totals = {key: 0 for key in ("positions", "top1_matches", "near_tie_mismatches", "significant_mismatches")}
        maximum = 0.0
        for case in cases:
            for key in totals:
                totals[key] += number(case[key], integer=True)
            maximum = max(maximum, number(case["max_probability_error"]))
        result["comparison"] = {"passed": comparison["passed"], "cases": len(cases), **totals,
                                "max_probability_error": maximum}
    return result


def main():
    parser = argparse.ArgumentParser(description="Print only shareable status and numeric fields from a local report.")
    parser.add_argument("report", type=Path)
    parser.add_argument("--comparison", type=Path)
    args = parser.parse_args()
    try:
        comparison = read_json(args.comparison) if args.comparison else None
        print(json.dumps(project(read_json(args.report), comparison), indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"Cannot summarize local report ({type(error).__name__}).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
