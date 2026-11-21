import argparse
import json
from pathlib import Path
import re
import sys

from ... import summary
from ...storage import PipelineError, read_json
from .compare import SCHEMA as RERUN_SCHEMA
from .reference import SCHEMA as REFERENCE_SCHEMA

SCHEMA = "gemma4-31b-reference-summary/1"
WORD = re.compile(r"[A-Za-z0-9_.+:-]{1,64}")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
HEX = {40: re.compile(r"[0-9a-f]{40}"), 64: re.compile(r"[0-9a-f]{64}")}
SHARD = re.compile(r"model-[0-9]{5}-of-[0-9]{5}\.safetensors")


def matched(value, pattern):
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise PipelineError("Unexpected text in a shareable field")
    return value


def word(value):
    return matched(value, WORD)


def flag(value):
    if type(value) is not bool:
        raise PipelineError("Expected a boolean result")
    return value


def count(value):
    return summary.number(value, integer=True)


def optional(value, integer=False):
    return None if value is None else summary.number(value, integer=integer)


def case(record):
    rerun = record["rerun"]
    logits = record["files"]["logits"]
    return {"id": word(record["id"]), "kind": word(record["kind"]), "bos": word(record["bos"]),
            **{key: count(record[key]) for key in ("input_tokens", "output_tokens", "max_new_tokens", "saved_positions",
                                                    "first_scoring_row", "scoring_positions", "forced_argmax_matches",
                                                    "forced_top_ties", "step_top_ties")},
            "stop_reason": word(record["stop_reason"]), "stop_token": optional(record["stop_token"], True),
            "non_finite": {word(key): count(value) for key, value in record["non_finite"].items()},
            "greedy_chosen_is_max": flag(record["greedy_chosen_is_max"]),
            "forced_mismatches": [{"position": count(item["position"]),
                                   "probability_difference": optional(item.get("probability_difference"))}
                                  for item in record["forced_mismatches"]],
            "step_vs_forced_max_abs_logit": optional(record["step_vs_forced_max_abs_logit"]),
            "timing": {word(key): optional(value) for key, value in record["timing"].items()},
            "rerun_tokens_match": flag(rerun["tokens_match"]), "rerun_greedy_s": summary.number(rerun["greedy_s"]),
            "rerun_first_difference": None if rerun["first_difference"] is None
            else count(rerun["first_difference"]["position"]),
            "logits": {"bytes": count(logits["bytes"]), "sha256": matched(logits["sha256"], HEX[64]),
                       "shape": [count(value) for value in logits["shape"]], "dtype": word(logits["dtype"])}}


def peak(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return count(max(values)) if values else None


def reference(record):
    if record.get("schema") != REFERENCE_SCHEMA:
        raise PipelineError(f"Expected a {REFERENCE_SCHEMA} record")
    model, environment, loading, resources = record["model"], record["environment"], record["loading"], record["resources"]
    rows = [loading["after_load"], *resources["cases"], resources["final"]]
    swap, monitor = resources["swap"], resources["monitor"]
    return {"passed": flag(record["passed"]), "checks": {word(key): flag(value) for key, value in record["checks"].items()},
            "model": {"id": matched(model["id"], REPOSITORY), "revision": matched(model["revision"], HEX[40]),
                      "contract_sha256": matched(model["contract_sha256"], HEX[64]),
                      "weights_sha256": {matched(name, SHARD): matched(value, HEX[64])
                                         for name, value in model["weights_sha256"].items()}},
            "environment": {"python": word(environment["python"]), "os_build": word(environment["os_build"]),
                            "packages": {word(name): word(version) for name, version in environment["packages"].items()}},
            "dtype": {word(key): word(value) for key, value in record["dtype"].items()}, "device": word(record["device"]),
            "loading": {"class": word(loading["class"]), "attn_implementation": word(loading["attn_implementation"]),
                        "parameters": count(loading["parameters"]),
                        "parameter_dtypes": [word(value) for value in loading["parameter_dtypes"]],
                        "parameter_devices": [word(value) for value in loading["parameter_devices"]],
                        "loading_info_entries": {word(key): len(value) for key, value in loading["info"].items()},
                        "tied_embeddings": flag(loading["tied_embeddings"])},
            "fixtures_sha256": matched(record["fixtures_sha256"], HEX[64]), "cases": [case(item) for item in record["cases"]],
            "resources": {"max_rss_bytes": peak(rows, "max_rss_bytes"),
                          "lifetime_peak_footprint_bytes": peak(rows, "lifetime_peak_footprint_bytes"),
                          "mps_driver_allocated_peak_bytes": peak(rows, "mps_driver_allocated_bytes"),
                          "mps_current_allocated_peak_bytes": peak(rows, "mps_current_allocated_bytes"),
                          "mps_recommended_max_bytes": optional(resources["mps_recommended_max_bytes"], True),
                          "swap": {word(key): optional(value, True) for key, value in swap.items()},
                          "monitor": {"samples": count(monitor["samples"]), "interval_s": summary.number(monitor["interval_s"]),
                                      "aborts": count(monitor["aborts"]), "failed": monitor["failure"] is not None,
                                      "unreadable_samples": count(monitor["unreadable_samples"]),
                                      "peak_rss_bytes": count(monitor["peak"]["rss_bytes"]),
                                      "lowest_free_percent": optional(monitor["lowest"]["free_percent"], True)}},
            "timing": {word(key): summary.number(value) for key, value in record["timing"].items()}}


def rerun(comparison):
    if comparison.get("schema") != RERUN_SCHEMA:
        raise PipelineError(f"Expected a {RERUN_SCHEMA} comparison")
    return {"passed": flag(comparison["passed"]),
            "max_abs_logit_difference": summary.number(comparison["max_abs_logit_difference"]),
            "cases": [{"id": word(item["id"]), "input_tokens_match": flag(item["input_tokens_match"]),
                       "output_tokens_match": flag(item["output_tokens_match"]),
                       "first_difference": optional(item["first_difference"], True),
                       "compared_positions": count(item["compared_positions"]),
                       "max_abs_logit_difference": summary.number(item["max_abs_logit_difference"]),
                       "argmax_changes": count(item["argmax_changes"])} for item in comparison["cases"]]}


def project(reports, references, comparison=None):
    result = {"schema": SCHEMA,
              "scope": "BF16 MPS reference outputs of the diagnostic fixtures; not a quality evaluation.",
              "runs": [{"pipeline": summary.project(report), "reference": reference(record)}
                       for report, record in zip(reports, references, strict=True)]}
    if comparison is not None:
        result["rerun"] = rerun(comparison)
    return result


def main():
    parser = argparse.ArgumentParser(description="Print only shareable numbers from Gemma 4 31B reference runs.")
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--comparison", type=Path)
    args = parser.parse_args()
    try:
        comparison = read_json(args.comparison) if args.comparison else None
        result = project([read_json(path) for path in args.report], [read_json(path) for path in args.reference],
                         comparison)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"Cannot summarize local records ({type(error).__name__}).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
