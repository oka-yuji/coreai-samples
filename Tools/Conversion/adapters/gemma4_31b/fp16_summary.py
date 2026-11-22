import argparse
import json
from pathlib import Path
import re
import sys

from ... import summary as pipeline
from ...storage import PipelineError
from .anchor import read
from .anchor_summary import signed
from .compare import CONTRACT
from .effect import SCHEMA as EFFECT_SCHEMA
from .export import QUANTIZATION, SCHEMA as EXPORT_SCHEMA
from .layers import CONTEXT_SCHEMA, RESTORED, compare as compare_layers
from .native import PUBLIC, SCHEMA as COMPARISON_SCHEMA, VERIFIER
from .summary import HEX, REPOSITORY, SHARD, count, flag, matched, optional, word

SCHEMA = "gemma4-31b-fp16-summary/2"
QUANTIZED_SCHEMA = "gemma4-31b-8bit-summary/1"
SCOPES = {SCHEMA: "FP16 export of the first layers with the outputs for speculation, the comparison of its logits with "
                  "the FP32 reference and of the model class with the original implementation; not a quality "
                  "evaluation.",
          QUANTIZED_SCHEMA: "8-bit export of the first layers with the outputs for speculation, the comparison of its "
                            "logits with the FP32 reference restored from the same quantized weights, and the effect of "
                            "the quantization against the FP32 reference with float16-rounded weights; not a quality "
                            "evaluation."}
RELATIVE = re.compile(r"[A-Za-z0-9_.-]{1,64}(/[A-Za-z0-9_.-]{1,64}){0,3}")


def number(value):
    return pipeline.number(value)


def relative(value):
    if not isinstance(value, str) or not RELATIVE.fullmatch(value) or ".." in value.split("/"):
        raise PipelineError("Unexpected text in a shareable path")
    return value


def model(entry):
    return {"id": matched(entry["id"], REPOSITORY), "revision": matched(entry["revision"], HEX[40]),
            "contract_sha256": matched(entry["contract_sha256"], HEX[64]),
            "weights_sha256": {matched(name, SHARD): matched(value["sha256"], HEX[64])
                               for name, value in entry["verified_before_loading"].items() if name.endswith(".safetensors")}}


def kinds(values):
    return {word(kind): values.count(kind) for kind in sorted(set(values))}


def numbers(values, integer=False):
    return {word(key): optional(value, integer) for key, value in values.items()}


def logits(entry):
    return {"bytes": count(entry["bytes"]), "sha256": matched(entry["sha256"], HEX[64])}


def reference(record):
    if record.get("schema") != CONTEXT_SCHEMA:
        raise PipelineError(f"Expected a {CONTEXT_SCHEMA} record")
    restored = {} if record.get("weights") != RESTORED else {
        "weights": "8bit", "quantized": {"export_sha256": matched(record["quantized"]["export_sha256"], HEX[64]),
                                         "file": logits(record["quantized"]["file"]),
                                         "tensors": count(record["quantized"]["tensors"])}}
    return {**restored, "implementation": word(record["implementation"]), "layers": count(record["layers"]),
            "layer_types": kinds(record["layer_types"]), "model": model(record["model"]),
            "embed_scale": number(record["embed_scale"]), "chunk": count(record["chunk"]),
            "versions": {word(name): word(version) for name, version in record["versions"].items()},
            "inputs": {word(name): {"positions": count(entry["positions"]),
                                    "tokens_sha256": matched(entry["tokens_sha256"], HEX[64]), "rows": len(entry["rows"]),
                                    "calls": len(entry["calls"]), "seconds": number(entry["seconds"]),
                                    "logits": {**logits(entry["logits"]), "dtype": word(entry["logits"]["dtype"]),
                                               "shape": [count(value) for value in entry["logits"]["shape"]],
                                               "non_finite": count(entry["logits"]["non_finite"])}}
                       for name, entry in record["inputs"].items()},
            "timing": numbers(record["timing"]), "memory": numbers(record["memory"], True)}


def layout(functions):
    return {word(name): {word(kind): {word(item): {"dtype": word(entry["dtype"]),
                                                   "shape": [None if size is None else count(size) for size in entry["shape"]]}
                                      for item, entry in items.items()} for kind, items in function.items()}
            for name, function in functions.items()}


def quantization(record):
    quantized = record["quantized"]
    if not isinstance(quantized["layers"], list) or not isinstance(quantized["tensors"], dict):
        raise PipelineError("Expected the quantized weights as a list and their tensors as a mapping")
    return {"quantized": {"layers": len(quantized["layers"]), "skipped": count(quantized["skipped"]),
                          "warnings_enabled": flag(quantized["warnings_enabled"]), "file": logits(quantized["file"]),
                          "tensors": len(quantized["tensors"])}}


def exported(record):
    eight_bit = record.get("precision") == "8bit"
    if record.get("schema") != EXPORT_SCHEMA or record["compression_config"] != (QUANTIZATION if eight_bit else None):
        raise PipelineError(f"Expected an uncompressed {EXPORT_SCHEMA} record or one of the fixed 8-bit config")
    metadata, bundle, resources = record["metadata"], record["bundle"], record["resources"]
    monitor = resources["monitor"]
    differences = {"uint8_minus_weights": signed(record["asset"]["uint8_minus_weights"]),
                   "dequantizations": count(record["asset"]["dequantizations"])} if eight_bit else {}
    return {**(quantization(record) if eight_bit else {}),
            "passed": flag(record["passed"]), "checks": {word(key): flag(value) for key, value in record["checks"].items()},
            "problems": len(record["problems"]), "model": model(record["model"]), "layers": count(record["layers"]),
            "layer_types": kinds(record["layer_types"]), "functions": [word(name) for name in record["functions"]],
            "context": count(record["context"]), "precision": word(record["precision"]),
            "upstream_revision": matched(record["upstream_revision"], HEX[40]),
            "versions": {word(name): word(version) for name, version in record["versions"].items()},
            "weights": {"tensors": count(record["weights"]["tensors"]), "elements": count(record["weights"]["elements"]),
                        "elements_by_dtype": numbers(record["weights"]["elements_by_dtype"], True),
                        "quantized_layouts": count(record["weights"]["quantized_layouts"])},
            "asset": {"functions": layout(record["asset"]["functions"]),
                      "output_order": {word(name): [word(item) for item in items]
                                       for name, items in record["asset"]["output_order"].items()},
                      "storage_types": numbers(record["asset"]["storage_types"], True),
                      "float16_minus_weights": signed(record["asset"]["float16_minus_weights"]), **differences},
            "tokenizer": {word(name): matched(value, HEX[64]) for name, value in record["tokenizer"].items()},
            "metadata": {"name": word(metadata["name"]), "vocab_size": count(metadata["language"]["vocab_size"]),
                         "max_context_length": count(metadata["language"]["max_context_length"]),
                         "hf_model_id": matched(metadata["source"]["hf_model_id"], REPOSITORY),
                         "revision": matched(metadata["source"]["revision"], HEX[40]),
                         "compression": None if metadata["compression"] is None else word(metadata["compression"]),
                         "tap_layers": [count(tap) for tap in metadata["tap_layers"]],
                         "tap_convention": word(metadata["tap_convention"])},
            "bundle": {"name": word(bundle["name"]), "main_hash": matched(bundle["main_hash"], HEX[64]),
                       "bytes": count(bundle["bytes"]),
                       "files": {relative(path): logits(entry) for path, entry in bundle["files"].items()}},
            "disk": numbers(record["disk"], True), "timing": numbers(record["timing"]),
            "resources": {"max_rss_bytes": count(resources["max_rss_bytes"]),
                          "lifetime_peak_footprint_bytes": optional(resources["lifetime_peak_footprint_bytes"], True),
                          "swap": numbers(resources["swap"], True),
                          "monitor": {"samples": count(monitor["samples"]), "interval_s": number(monitor["interval_s"]),
                                      "aborts": count(monitor["aborts"]), "failed": monitor["failure"] is not None,
                                      "unreadable_samples": count(monitor["unreadable_samples"]),
                                      "peak_rss_bytes": count(monitor["peak"]["rss_bytes"]),
                                      "lowest_free_percent": optional(monitor["lowest"]["free_percent"], True)}}}


SHARED = {"count": count, "seconds": number, "flag": flag, "text": word}


def condition(entry):
    verifier = entry["verifier"]
    prefix = verifier.get("prefix_function")
    if (prefix is None) != (entry["start"] == 0):
        raise PipelineError("Expected a prefix function exactly for the conditions that start after position 0")
    return {"name": word(entry["name"]), "input": word(entry["input"]),
            **{key: count(entry[key]) for key in ("tokens", "start", "chunk", "rows", "non_finite", "reference_non_finite",
                                                  "constant_rows")},
            "passed": flag(entry["passed"]), "problems": len(entry["problems"]),
            "checks": {word(key): flag(value) for key, value in entry["checks"].items()},
            "max_abs_logit": optional(entry["max_abs_logit"]), "top1_mismatches": numbers(entry["top1_mismatches"], True),
            "known": {"rows": count(entry["known"]["rows"]), "max_abs_logit": optional(entry["known"]["max_abs_logit"])},
            "seconds": number(entry["seconds"]), "output": logits(entry["output"]),
            "verifier": {key: None if key == "prefix_function" and prefix is None else SHARED[VERIFIER[key]](verifier[key])
                         for key in PUBLIC}}


def compared(record):
    if record.get("schema") != COMPARISON_SCHEMA:
        raise PipelineError(f"Expected a {COMPARISON_SCHEMA} comparison")
    return {"passed": flag(record["passed"]), "problems": len(record["problems"]),
            "criteria": {word(key): number(value) for key, value in record["criteria"].items()},
            "known_call": {"ends": [count(value) for value in record["known_call"]["ends"]],
                           "shortest": count(record["known_call"]["shortest"])},
            "verifier_sha256": matched(record["verifier"]["sha256"], HEX[64]),
            "main_hash": matched(record["export"]["main_hash"], HEX[64]),
            "conditions": [condition(entry) for entry in record["conditions"]]}


def ratio(entry):
    return {"largest_ratio": optional(entry["largest_ratio"]), "passed": flag(entry["passed"])}


def model_class(record, exports):
    if any((export["model"]["contract_sha256"], export["layers"], export["metadata"]["tap_layers"])
           != (record["contract_sha256"], record["layers"], record["tap_layers"]) for export in exports):
        raise PipelineError("The runs of the original implementation and the model class must be of the contract, the "
                            "layers and the tap layers of every export")
    return {"passed": flag(record["passed"]), "problems": len(record["problems"]), "layers": count(record["layers"]),
            "tap_layers": [count(tap) for tap in record["tap_layers"]], "ratio_limit": number(record["ratio_limit"]),
            "records_sha256": [matched(value, HEX[64]) for value in record["records_sha256"]],
            "inputs": {word(name): {"criteria": {word(key): number(value) for key, value in entry["criteria"].items()},
                                    "rows": count(entry["rows"]), "max_abs_logit": optional(entry["max_abs_logit"]),
                                    "max_probability_error": optional(entry["max_probability_error"]),
                                    "top1_mismatches": count(entry["top1_mismatches"]),
                                    "hidden_states": ratio(entry["outputs"]["hidden_states"]),
                                    "tap_hidden_states": {word(tap): ratio(value) for tap, value
                                                          in entry["outputs"]["tap_hidden_states"].items()},
                                    "checks": {word(key): flag(value) for key, value in entry["checks"].items()}}
                       for name, entry in record["inputs"].items()}}


def paired(reports, exports, comparisons, reference_hash):
    files = [Path(path).resolve() for path, _, _, _ in exports]
    if not len(reports) == len(exports) == len(comparisons) or len(set(files)) != len(files) \
            or len({report.get("run_id") for report in reports}) != len(reports):
        raise PipelineError("Give each export once, with its report and its comparison")
    for report, file, (_, _, digest, size), comparison in zip(reports, files, exports, comparisons):
        stage = next((stage for stage in report.get("stages", []) if stage.get("role") == "export"), {})
        if stage.get("status") != "executed" or stage.get("origin") != report.get("run_id") or not stage.get("artifacts") \
                or Path(stage["artifacts"]).resolve() / "export" / "export.json" != file:
            raise PipelineError("Each report must be the run that wrote the export record given with it")
        manifest = read(Path(stage["artifacts"]).parent / "manifest.json")[0]
        if (manifest.get("version"), manifest.get("status"), manifest.get("run_id"), manifest.get("stage")) \
                != (1, "completed", report["run_id"], "export") \
                or manifest["outputs"]["artifacts"]["tree"].get("export/export.json") != {"bytes": size, "sha256": digest}:
            raise PipelineError("Each export record must be the output that the run of its stage recorded")
        if (comparison.get("export") or {}).get("sha256") != digest \
                or (comparison.get("reference") or {}).get("sha256") != reference_hash:
            raise PipelineError("Each comparison must be made from the export record given with it and the reference")


def effected(record, reference_hash):
    restored = record.get("restored") if isinstance(record, dict) else None
    if not isinstance(restored, dict) or record.get("schema") != EFFECT_SCHEMA or record.get("problems") != [] \
            or restored.get("sha256") != reference_hash or not isinstance(record.get("rounded"), dict):
        raise PipelineError("Expected the effect of the quantization on the restored reference")
    return {"rounded_sha256": matched(record["rounded"]["sha256"], HEX[64]), "layers": count(record["restored"]["layers"]),
            "criteria": {word(key): number(value) for key, value in record["criteria"].items()},
            "conditions": [{"name": word(entry["name"]), "input": word(entry["input"]),
                            **{key: count(entry[key]) for key in ("tokens", "start", "chunk", "rows", "non_finite",
                                                                  "reference_non_finite")},
                            "max_abs_logit": optional(entry["max_abs_logit"]),
                            "top1_mismatches": numbers(entry["top1_mismatches"], True)}
                           for entry in record["conditions"]]}


def project(reports, exports, comparisons, context, layered, effect):
    if any(export["model"]["contract_sha256"] != context["model"]["contract_sha256"] for export in exports):
        raise PipelineError("The FP32 reference must be of the contract of every export")
    schema = SCHEMA if effect is None else QUANTIZED_SCHEMA
    classes = {} if effect is not None else {"model_class": model_class(layered, exports)}
    return {"schema": schema, "scope": SCOPES[schema], "reference": reference(context), **classes,
            "assets": [{"pipeline": pipeline.project(report), "export": exported(record), "comparison": compared(result)}
                       for report, record, result in zip(reports, exports, comparisons, strict=True)],
            **({} if effect is None else {"effect": effect})}


def main():
    parser = argparse.ArgumentParser(description="Print only shareable numbers from Gemma 4 31B FP16 or 8-bit export "
                                                 "records.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--original", type=Path,
                        help="Folder that layers.py run wrote for the original implementation (FP16 exports only).")
    parser.add_argument("--model-class", type=Path,
                        help="Folder that layers.py run wrote for the model class with the same layers and tap layers "
                             "(FP16 exports only).")
    parser.add_argument("--effect", type=Path, help="Output of effect (8-bit exports only).")
    for name in ("report", "export", "comparison"):
        parser.add_argument(f"--{name}", type=Path, action="append", required=True)
    args = parser.parse_args()
    try:
        reports = [read(path)[0] for path in args.report]
        exports = [(path, *read(path)) for path in args.export]
        comparisons = [read(path)[0] for path in args.comparison]
        context, reference_hash, _ = read(args.reference)
        paired(reports, exports, comparisons, reference_hash)
        precisions = {record.get("precision") for _, record, _, _ in exports}
        given = tuple(value is not None for value in (args.original, args.model_class, args.effect))
        if precisions == {"float16"} and given == (True, True, False):
            layered, effect = compare_layers(args.original, args.model_class, CONTRACT), None
        elif precisions == {"8bit"} and given == (False, False, True):
            layered, effect = None, effected(read(args.effect)[0], reference_hash)
        else:
            raise PipelineError("Give FP16 exports with --original and --model-class, or 8-bit exports with --effect")
        print(json.dumps(project(reports, [record for _, record, _, _ in exports], comparisons, context, layered, effect),
                         indent=2, allow_nan=False))
        return 0
    except Exception as error:
        print(f"Cannot summarize local records ({type(error).__name__}).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
