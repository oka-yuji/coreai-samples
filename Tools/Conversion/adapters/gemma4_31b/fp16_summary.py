import argparse
import json
from pathlib import Path
import re
import sys

from ... import summary as pipeline
from ...storage import PipelineError, canonical
from .anchor_summary import signed
from .compare import CONTRACT
from .effect import effect as quantization_effect
from .export import QUANTIZATION, SCHEMA as EXPORT_SCHEMA
from .layers import CONDITIONS, CONTEXT_SCHEMA, RESTORED, WEIGHTS, compare as compare_layers
from .native import (CRITERIA, KNOWN_CALL, PUBLIC, SCHEMA as COMPARISON_SCHEMA, VERIFIER, comparison_passed,
                     condition_problems, correspondence, load_export, load_reference, measurement_problems,
                     read_record, reference_record_problems, regular_file, texts, valid)
from .prepare import load_contract
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
CONDITION_KEYS = ("name", "input", "tokens", "start", "chunk", "rows")
FIXED = [[name, source, length, start, chunk, length - start] for name, source, length, start, chunk in CONDITIONS]
ROLES = ["prepare", "export"]
TIMES = ("prepare_s", "execute_s", "total_s")
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
    if record.get("schema") != EXPORT_SCHEMA \
            or canonical(record["compression_config"]) != canonical(QUANTIZATION if eight_bit else None):
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


SHOWN = {"name": word, "input": word,
         **dict.fromkeys(("tokens", "start", "chunk", "rows", "non_finite", "reference_non_finite", "constant_rows"), count),
         "passed": flag, "problems": len, "checks": lambda value: {word(key): flag(item) for key, item in value.items()},
         "max_abs_logit": optional, "top1_mismatches": lambda value: numbers(value, True),
         "known": lambda value: {"rows": count(value["rows"]), "max_abs_logit": optional(value["max_abs_logit"])},
         "seconds": number, "output": logits,
         "verifier": lambda value: {key: SHARED[VERIFIER[key]](value[key]) if key in value else None for key in PUBLIC}}


def condition(entry):
    return {key: show(entry[key]) for key, show in SHOWN.items() if key in entry}


def fixed(entries):
    return canonical([[entry[key] for key in CONDITION_KEYS] for entry in entries])


def compared(record, folder, exported_record, config, arrays):
    if record.get("schema") != COMPARISON_SCHEMA:
        raise PipelineError(f"Expected a {COMPARISON_SCHEMA} comparison")
    if not texts(record["problems"]):
        raise PipelineError("Each comparison must record its problems as a list of texts")
    if canonical(record["criteria"]) != canonical(CRITERIA):
        raise PipelineError("Each comparison must record the fixed criteria")
    if canonical(record["known_call"]) != canonical(KNOWN_CALL):
        raise PipelineError("Each comparison must record the fixed call shape that the engine avoids")
    if fixed(record["conditions"]) != canonical([] if record["problems"] else FIXED):
        raise PipelineError("Each comparison must hold the twelve fixed conditions in order, or none with its problems")
    if (record["verifier"] is None) != bool(record["problems"]):
        raise PipelineError("Each comparison must record its verifier exactly when it has no problems")
    for entry in record["conditions"]:
        if condition_problems(entry, config, exported_record["bundle"]["main_hash"], "prefill" in exported_record["functions"]):
            raise PipelineError("Each condition must hold the fields that the comparison writes on its paths")
        if "output" in entry:
            found, data = regular_file(folder / f"{entry['name']}.f16", entry["output"]["bytes"], keep="known" in entry)
            if canonical(entry["output"]) != canonical(found):
                raise PipelineError("Each condition must record the size and SHA-256 of its logits file")
            if "known" in entry and measurement_problems(entry, config["vocab_size"], data, arrays):
                raise PipelineError("Each condition must hold the numbers that its logits file and the reference give")
    if canonical(record["passed"]) != canonical(comparison_passed(record["problems"], record["conditions"])):
        raise PipelineError("Each comparison must record the verdict that its conditions and problems give")
    return {"passed": flag(record["passed"]), "problems": len(record["problems"]),
            "criteria": {word(key): number(value) for key, value in record["criteria"].items()},
            "known_call": {"ends": [count(value) for value in record["known_call"]["ends"]],
                           "shortest": count(record["known_call"]["shortest"])},
            "verifier_sha256": None if record["verifier"] is None else matched(record["verifier"]["sha256"], HEX[64]),
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


def timed(report, manifest, stage, record):
    stages = [[entry[key] for key in TIMES] for entry in report["stages"]]
    spent = (record.get("timing") or {}).get("total_s")
    return all(prepare + execute <= total for prepare, execute, total in stages) \
        and sum(total for *_, total in stages) <= report["elapsed_s"] \
        and canonical(manifest.get("execute_s")) == canonical(stage.get("execute_s")) \
        and (not valid("seconds", spent) or spent <= stage["execute_s"])


def paired(reports, exports, comparisons, context, reference_hash):
    files = [Path(path).resolve() for path, _, _, _ in exports]
    if not len(reports) == len(exports) == len(comparisons) or len(set(files)) != len(files) \
            or len({report.get("run_id") for report in reports}) != len(reports):
        raise PipelineError("Give each export once, with its report and its comparison")
    for report, file, (_, record, digest, size), comparison in zip(reports, files, exports, comparisons):
        stage = next((stage for stage in report.get("stages", []) if stage.get("role") == "export"), {})
        if stage.get("status") != "executed" or stage.get("origin") != report.get("run_id") or not stage.get("artifacts") \
                or Path(stage["artifacts"]).resolve() / "export" / "export.json" != file:
            raise PipelineError("Each report must be the run that wrote the export record given with it")
        if (report.get("status"), report.get("verification"), [entry.get("role") for entry in report["stages"]],
                [entry.get("status") for entry in report["stages"]]) \
                != ("completed", "not_configured", ROLES, ["executed"] * len(ROLES)):
            raise PipelineError("Each report must be a completed run of the fixed stages, which has no verify stage")
        manifest = read_record(Path(stage["artifacts"]).parent / "manifest.json")[0]
        if (manifest.get("version"), manifest.get("status"), manifest.get("run_id"), manifest.get("stage")) \
                != (1, "completed", report["run_id"], "export") \
                or manifest["outputs"]["artifacts"]["tree"].get("export/export.json") != {"bytes": size, "sha256": digest}:
            raise PipelineError("Each export record must be the output that the run of its stage recorded")
        if not timed(report, manifest, stage, record):
            raise PipelineError("Each report must record the times of its stages within its run and the time of its export "
                                "that the manifest recorded and that the export record fits in")
        if (comparison.get("export") or {}).get("sha256") != digest \
                or (comparison.get("reference") or {}).get("sha256") != reference_hash:
            raise PipelineError("Each comparison must be made from the export record given with it and the reference")
        tree = manifest["outputs"]["artifacts"]["tree"]
        if canonical({path.removeprefix("export/bundle/"): entry for path, entry in tree.items()
                      if path.startswith("export/bundle/") and entry.get("directory") is not True}) \
                != canonical(record["bundle"]["files"]):
            raise PipelineError("Each export record must list the bundle files that the manifest of its stage recorded")
        if record.get("precision") == "8bit" \
                and canonical(tree.get("export/quantized.safetensors")) != canonical(record["quantized"]["file"]):
            raise PipelineError("Each 8-bit export record must list the quantized file that the manifest recorded")
        if canonical(record.get("layers")) != canonical(context.get("layers")):
            raise PipelineError("Each export must be of the layers of the reference")
        if correspondence(record, digest, context):
            raise PipelineError("The reference must use the weights of each export, restored from its quantized tensors "
                                "for 8 bits")
        if canonical(comparison["export"].get("layers")) != canonical(record.get("layers")):
            raise PipelineError("Each comparison must record the layers of its export")
        if canonical(comparison["export"].get("functions")) != canonical(record.get("functions")):
            raise PipelineError("Each comparison must record the functions of its export")
        if canonical(comparison["export"].get("main_hash")) != canonical(record["bundle"]["main_hash"]):
            raise PipelineError("Each comparison must record the asset of its export")
        if canonical(comparison["reference"].get("layers")) != canonical(context.get("layers")):
            raise PipelineError("Each comparison must record the layers of the reference")


def effected(record, reference_hash):
    if record["problems"] or record["restored"]["sha256"] != reference_hash:
        raise PipelineError("The effect must be computed from the restored reference of the comparison and a rounded "
                            "reference of its layers")
    return {"rounded_sha256": matched(record["rounded"]["sha256"], HEX[64]), "layers": count(record["restored"]["layers"]),
            "criteria": {word(key): number(value) for key, value in record["criteria"].items()},
            "conditions": [{"name": word(entry["name"]), "input": word(entry["input"]),
                            **{key: count(entry[key]) for key in ("tokens", "start", "chunk", "rows", "non_finite",
                                                                  "reference_non_finite")},
                            "max_abs_logit": optional(entry["max_abs_logit"]),
                            "top1_mismatches": numbers(entry["top1_mismatches"], True)}
                           for entry in record["conditions"]]}


def project(reports, exports, comparisons, folders, context, layered, effect, config, arrays):
    schema = SCHEMA if effect is None else QUANTIZED_SCHEMA
    classes = {} if effect is not None else {"model_class": model_class(layered, exports)}
    return {"schema": schema, "scope": SCOPES[schema], "reference": reference(context), **classes,
            "assets": [{"pipeline": pipeline.project(report), "export": exported(record),
                        "comparison": compared(result, folder, record, config, arrays)}
                       for report, record, result, folder in zip(reports, exports, comparisons, folders, strict=True)],
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
    parser.add_argument("--rounded", type=Path, help="Folder that layers.py context --weights float16 wrote (8-bit "
                                                     "exports only).")
    for name in ("report", "export", "comparison"):
        parser.add_argument(f"--{name}", type=Path, action="append", required=True)
    args = parser.parse_args()
    try:
        reports = [read_record(path)[0] for path in args.report]
        exports = [(path, *read_record(path)) for path in args.export]
        comparisons = [read_record(path)[0] for path in args.comparison]
        context, reference_hash, _ = read_record(args.reference)
        contract = load_contract(CONTRACT)
        paired(reports, exports, comparisons, context, reference_hash)
        _, loaded_hash, arrays, found = load_reference(args.reference.parent, contract, CONTRACT)
        if found or loaded_hash != reference_hash:
            raise PipelineError("The reference must be a complete reference of the contract with its logits files")
        if reference_record_problems(context, contract, arrays):
            raise PipelineError("The reference must record the fixed chunks, calls, embedding scale and logits")
        if any(load_export(Path(path).parent, contract, CONTRACT)[3] for path, _, _, _ in exports):
            raise PipelineError("Each export must be a passed export of the contract")
        precisions = {record.get("precision") for _, record, _, _ in exports}
        given = tuple(value is not None for value in (args.original, args.model_class, args.rounded))
        if precisions == {"float16"} and given == (True, True, False):
            if any(read_record(Path(folder) / "record.json")[0].get("weights") != WEIGHTS
                   for folder in (args.original, args.model_class)):
                raise PipelineError("The runs of the original implementation and the model class must use the weights "
                                    "rounded to float16")
            layered, effect = compare_layers(args.original, args.model_class, CONTRACT), None
        elif precisions == {"8bit"} and given == (False, False, True):
            layered, effect = None, effected(quantization_effect(args.reference.parent, args.rounded, CONTRACT),
                                             reference_hash)
        else:
            raise PipelineError("Give FP16 exports with --original and --model-class, or 8-bit exports with --rounded")
        print(json.dumps(project(reports, [record for _, record, _, _ in exports], comparisons,
                                 [Path(path).parent for path in args.comparison], context, layered, effect,
                                 contract["config"], arrays),
                         indent=2, allow_nan=False))
        return 0
    except Exception as error:
        print(f"Cannot summarize local records ({type(error).__name__}).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
