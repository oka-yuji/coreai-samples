import argparse
import json
from pathlib import Path
import sys

from ... import summary
from ...storage import PipelineError, read_json

SCHEMA = "typed-decision-quantization/1"
CHECKS = ("hashes_match", "rows_complete", "finite")
ROWS = ("F", "Q", "F_forwarded", "Q_forwarded", "rejected")
SIDES = ("stage2_vs_1m", "stage0q_vs_0")
TEMPERATURES = ("temperature_1", "shipping_temperature")


def side(comparison):
    result = {"rows": summary.number(comparison["rows"], integer=True)}
    for name in TEMPERATURES:
        values = comparison[name]
        result[name] = {"max_probability_error": summary.number(values["max_probability_error"]),
                        "mean_probability_error": summary.number(values["mean_probability_error"]),
                        "top1_changed": summary.number(values["top1_changed"]["rows"], integer=True),
                        "margin_within_2e": summary.number(values["margin_within_2e"]["rows"], integer=True)}
    return result


def quantization(comparison):
    checks = comparison["checks"]
    if type(comparison.get("passed")) is not bool or any(type(checks.get(key)) is not bool for key in CHECKS):
        raise PipelineError("Expected quantization check results")
    result = {"passed": comparison["passed"], **{key: checks[key] for key in CHECKS},
              "hashed_tensors": summary.number(comparison["hashes"]["items"], integer=True),
              "matching_tensors": summary.number(comparison["hashes"]["matching"], integer=True),
              "rows": {key: summary.number(comparison["rows"][key], integer=True) for key in ROWS}}
    result.update({name: side(comparison[name]) for name in SIDES if name in comparison})
    if "adaptation" in comparison:
        result["adaptation_mean"] = {f"{group}: {label}": summary.number(value["mean"])
                                     for group, values in comparison["adaptation"].items()
                                     for label, value in values.items() if value["rows"]}
    return result


def project(report, comparison):
    if not isinstance(comparison, dict) or comparison.get("schema") != SCHEMA:
        raise PipelineError(f"Expected a {SCHEMA} comparison")
    return {**summary.project(report), "comparison": quantization(comparison)}


def main():
    parser = argparse.ArgumentParser(description="Print only shareable numbers from a JEV-27B quantization check.")
    parser.add_argument("report", type=Path)
    parser.add_argument("--comparison", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(project(read_json(args.report), read_json(args.comparison)), indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"Cannot summarize local report ({type(error).__name__}).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
