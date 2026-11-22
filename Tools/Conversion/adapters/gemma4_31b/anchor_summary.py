import argparse
import json
import math
from pathlib import Path
import sys

from ... import summary as pipeline
from ...storage import PipelineError
from .anchor import LAYERS_SCHEMA, read
from .anchor_compare import RECIPE, SCHEMA as NUMERICS_SCHEMA, conforms, definition, expectations, identity, provenance, shaped
from .summary import HEX, count, flag, matched, optional, peak, word

SUMMARY_SCHEMA = "gemma4-31b-anchor-summary/1"
STAGES = {"layers": "layers.json", "anchor": "reference.json", "anchor_w16": "reference.json",
          "anchor_scale_fp32": "reference.json", "anchor_scale_fp16": "reference.json", "half": "reference.json"}
MODULE = "Tools.Conversion.adapters.gemma4_31b.anchor"
METRICS = ("max_probability_error_max", "max_probability_error_mean", "max_abs_logit_difference")
SIGNED = ("kl_mean", "kl_p99", "kl_max")
COUNTS = ("rows", "top1_matches", "near_tie_mismatches", "other_mismatches")
NUMBER = (int, float)
ANY = (str, int, float, bool, None, dict, list)
REPORT = {"version": int, "status": str, "verification": str, "run_id": str,
          "stages": [{"id": str, "role": str, "status": str}]}
STAGE = {"status": str, "origin": str, "artifacts": str}
MANIFEST = {"version": int, "status": str, "run_id": str, "stage": str, "outputs": {"artifacts": {"tree": {str: dict}}}}
ROW = {str: (int, float, None)}
COMMON = {"checks": {str: bool}, "timing": {str: NUMBER},
          "loading": {"class": str, "attn_implementation": str, "parameters": int, "parameter_dtypes": [str],
                      "parameter_devices": [str], "info": {str: list}, "embed_scale": NUMBER, "after_load": ROW},
          "resources": {"cases": [ROW], "final": ROW, "swap": {str: (int, None)},
                        "monitor": {"samples": int, "aborts": int, "failure": ANY, "unreadable_samples": int,
                                    "peak": {"rss_bytes": int}, "lowest": {"free_percent": (int, None)}}}}
LOGITS = {"cases": [{"non_finite": int, "seconds": NUMBER}]}
RANGES = {"ranges": [{"point": str, "layer": (int, None), "max_abs": (int, float, None), "case": (str, None),
                      "above_float16_max": int, "above_half_float16_max": int}]}
CAST = {"cast": {**{key: int for key in ("tensors", "elements", "changed", "underflow", "non_finite", "above_float16_max")},
                 "max_abs_error": NUMBER}}
LAYERS = {"layers": int, "layer_types": [str], "limit": NUMBER, "attention_rows": int,
          "cases": [{"positions": int, "max_abs_difference": (int, float, None), "non_finite": {str: int}}]}
METRIC = {**{key: int for key in COUNTS}, **{key: NUMBER for key in METRICS + SIGNED}}
COMPARISON = {"schema": str, "records_sha256": [str], "passed": bool, "problems": list, "vocabulary": int, "window": int,
              "near_tie_margin": NUMBER,
              "cases": [{"id": str, "problems": list, "non_finite": {str: int}, "all": (METRIC, None), "scoring": (METRIC, None),
                         "mismatches": [{"row": int, "scoring": bool, "near_tie": bool, "probability_gap": NUMBER}]}],
              "groups": [{"kind": str, "band": str, "scope": str, "cases": [str], **METRIC}]}
WINDOW = {"first_row": int, "last_row": int}


def checked(value, *patterns):
    if not all(conforms(value, pattern) for pattern in patterns):
        raise PipelineError("A local record does not have the expected shape")
    return value


def signed(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise PipelineError("Expected a finite number")
    return value


def definitions(expected):
    steps = {step["id"]: step["command"] for step in read(RECIPE / "recipe.json")[0]["steps"]}
    result = {"reference": definition("reference", expected)}
    for name in STAGES.keys() - {"layers"}:
        command = steps[name]
        options = {key: value for key, value in zip(command, command[1:]) if key.startswith("--")}
        result[name] = definition(command[command.index(MODULE) + 1], expected, options.get("--weights"),
                                  options.get("--embed-scale"))
    return result


def records(report):
    stages = {stage["id"]: stage for stage in checked(report, REPORT)["stages"]}
    expected = expectations()
    wanted, loaded = definitions(expected), {}
    for name, file in (("reference", "reference.json"), *STAGES.items()):
        stage = stages.get(name)
        if not conforms(stage, STAGE) or stage["status"] not in ("executed", "reused") or (
                name != "reference" and (stage["status"] != "executed" or stage["origin"] != report["run_id"])):
            raise PipelineError("Each diagnostic stage must have been executed by the given run")
        folder = Path(stage["artifacts"])
        record, digest, size = read(folder / file)
        manifest = checked(read(folder.parent / "manifest.json")[0], MANIFEST)
        if (manifest["version"], manifest["status"], manifest["run_id"],
                manifest["stage"]) != (1, "completed", stage["origin"], name) \
                or manifest["outputs"]["artifacts"]["tree"].get(file) != {"bytes": size, "sha256": digest}:
            raise PipelineError(f"The {name} record is not the output that the run of its stage recorded")
        if not shaped(record):
            raise PipelineError(f"The {name} record does not have the shape of a known record")
        if name == "layers":
            matches = record["schema"] == LAYERS_SCHEMA and record["stage"] == "layers" and not provenance(record, expected)
        else:
            matches = identity(record, expected) == (wanted[name], [])
        if not matches:
            raise PipelineError(f"The {name} record does not match its definition, the contract or the fixtures")
        if name != "reference" and record["source"] != {"reference_sha256": loaded["reference"]["sha256"]}:
            raise PipelineError(f"The {name} record was not computed from the reference record of the report")
        loaded[name] = {"stage": stage, "record": record, "sha256": digest}
    return loaded


def usage(record):
    resources, monitor = record["resources"], record["resources"]["monitor"]
    rows = [record["loading"]["after_load"], *resources["cases"], resources["final"]]
    return {"max_rss_bytes": peak(rows, "max_rss_bytes"),
            "lifetime_peak_footprint_bytes": peak(rows, "lifetime_peak_footprint_bytes"),
            "mps_driver_allocated_peak_bytes": peak(rows, "mps_driver_allocated_bytes"),
            "swap": {word(key): optional(value, True) for key, value in resources["swap"].items()},
            "monitor": {"samples": count(monitor["samples"]), "aborts": count(monitor["aborts"]),
                        "failed": monitor["failure"] is not None,
                        "unreadable_samples": count(monitor["unreadable_samples"]),
                        "peak_rss_bytes": count(monitor["peak"]["rss_bytes"]),
                        "lowest_free_percent": optional(monitor["lowest"]["free_percent"], True)}}


def loading(record):
    data = record["loading"]
    return {"class": word(data["class"]), "attn_implementation": word(data["attn_implementation"]),
            "parameters": count(data["parameters"]), "parameter_dtypes": [word(value) for value in data["parameter_dtypes"]],
            "parameter_devices": [word(value) for value in data["parameter_devices"]],
            "loading_info_entries": {word(key): len(value) for key, value in data["info"].items()},
            "embed_scale": signed(data["embed_scale"])}


def common(entry, name):
    record = checked(entry["record"], COMMON)
    return {"stage": word(name), "record_sha256": matched(entry["sha256"], HEX[64]), "passed": flag(record["passed"]),
            "checks": {word(key): flag(value) for key, value in record["checks"].items()},
            "source_reference_sha256": matched(record["source"]["reference_sha256"], HEX[64]),
            "loading": loading(record), "resources": usage(record),
            "timing": {word(key): pipeline.number(value) for key, value in record["timing"].items()}}


def logits_record(entry, name):
    record = entry["record"]
    compute = record["compute"]
    checked(record, LOGITS, *([RANGES] if record["stage"] == "anchor" else []),
            *([CAST] if record["stage"] == "anchor" and compute["weights"] == "float16" else []))
    result = {**common(entry, name),
              "compute": {"weights": word(compute["weights"]), "compute": word(compute["compute"]),
                          "embed_scale": signed(compute["embed_scale"]),
                          "attention_rows": optional(compute["attention_rows"], True)},
              "cases": [{"id": word(case["id"]), "kind": word(case["kind"]),
                         **{key: count(case[key]) for key in ("input_tokens", "output_tokens", "saved_positions",
                                                               "first_scoring_row", "scoring_positions", "non_finite")},
                         "seconds": pipeline.number(case["seconds"]),
                         "logits_sha256": matched(case["files"]["logits"]["sha256"], HEX[64])}
                        for case in record["cases"]]}
    if record["stage"] == "anchor":
        result["ranges"] = [{"point": word(row["point"]), "layer": optional(row["layer"], True),
                             "max_abs": optional(row["max_abs"]), "case": None if row["case"] is None else word(row["case"]),
                             "above_float16_max": count(row["above_float16_max"]),
                             "above_half_float16_max": count(row["above_half_float16_max"])} for row in record["ranges"]]
    if record["stage"] == "anchor" and compute["weights"] == "float16":
        cast = record["cast"]
        result["cast"] = {**{key: count(cast[key]) for key in ("tensors", "elements", "changed", "underflow", "non_finite",
                                                                "above_float16_max")},
                          "max_abs_error": pipeline.number(cast["max_abs_error"])}
    return result


def layers_record(entry):
    record = checked(entry["record"], LAYERS)
    return {**common(entry, "layers"), "layers": count(record["layers"]),
            "layer_types": [word(value) for value in record["layer_types"]], "limit": pipeline.number(record["limit"]),
            "attention_rows": count(record["attention_rows"]),
            "cases": [{"id": word(case["id"]), "positions": count(case["positions"]),
                       "max_abs_difference": optional(case["max_abs_difference"]),
                       "non_finite": {word(key): count(value) for key, value in case["non_finite"].items()}}
                      for case in record["cases"]]}


def metrics(values):
    if values is None:
        return None
    return {**{key: count(values[key]) for key in COUNTS}, **{key: pipeline.number(values[key]) for key in METRICS},
            **{key: signed(values[key]) for key in SIGNED}}


def comparison(data, names):
    hashes = checked(data, COMPARISON)["records_sha256"]
    if data["schema"] != NUMERICS_SCHEMA or len(hashes) != 2 or not all(digest in names for digest in hashes):
        raise PipelineError(f"Expected a {NUMERICS_SCHEMA} comparison of two records of the report")
    for group in data["groups"]:
        if group["scope"] == "window":
            checked(group, WINDOW)
    return {"baseline": names[hashes[0]], "candidate": names[hashes[1]], "passed": flag(data["passed"]),
            "problems": len(data["problems"]), "vocabulary": count(data["vocabulary"]), "window": count(data["window"]),
            "near_tie_margin": pipeline.number(data["near_tie_margin"]),
            "cases": [{"id": word(case["id"]), "problems": len(case["problems"]),
                       "non_finite": {word(key): count(value) for key, value in case["non_finite"].items()},
                       "all": metrics(case["all"]), "scoring": metrics(case["scoring"]),
                       "mismatches": [{"row": count(item["row"]), "scoring": flag(item["scoring"]),
                                       "near_tie": flag(item["near_tie"]),
                                       "probability_gap": pipeline.number(item["probability_gap"])}
                                      for item in case["mismatches"]]} for case in data["cases"]],
            "groups": [{"kind": word(group["kind"]), "band": word(group["band"]), "scope": word(group["scope"]),
                        "cases": [word(value) for value in group["cases"]],
                        **{key: count(group[key]) for key in ("first_row", "last_row") if group["scope"] == "window"},
                        **metrics(group)} for group in data["groups"]]}


def project(report, loaded, comparisons):
    names = {entry["sha256"]: name for name, entry in loaded.items()}
    if len(names) != len(loaded):
        raise PipelineError("Two stages hold the same record")
    reference = loaded["reference"]
    return {"schema": SUMMARY_SCHEMA,
            "scope": "FP32 anchor and FP16 numerics of the diagnostic fixtures; not a quality evaluation.",
            "pipeline": pipeline.project(report), "run_id": word(report["run_id"]),
            "reference": {"origin": word(reference["stage"]["origin"]), "status": word(reference["stage"]["status"]),
                          "record_sha256": matched(reference["sha256"], HEX[64])},
            "layers": layers_record(loaded["layers"]),
            "records": [logits_record(loaded[name], name) for name in STAGES if name != "layers"],
            "comparisons": [comparison(data, names) for data in comparisons]}


def main():
    parser = argparse.ArgumentParser(description="Print only shareable numbers from the Gemma 4 31B FP32 anchor run.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, action="append", required=True)
    args = parser.parse_args()
    try:
        comparisons = [read(path) for path in args.comparison]
        if len({digest for _, digest, _ in comparisons}) != len(comparisons):
            raise PipelineError("A comparison is given twice")
        report = read(args.report)[0]
        result = project(report, records(report), [data for data, _, _ in comparisons])
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"Cannot summarize local records ({type(error).__name__}).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
