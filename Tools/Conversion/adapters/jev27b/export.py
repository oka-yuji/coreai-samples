import argparse
import contextlib
import io
import json
from pathlib import Path
import re
import resource
import runpy
import shutil
import sys
import tempfile
import time

from ...storage import read_json, write_json
from ..pinned import apple_toolchain, check_upstream, pinned_hub, sha256
from .decision import DECISION, document, listing, validate, write_sums
from .monitor import Monitor
from .prepare import TOKENIZER_FILES, load_contract

REFERENCE_BUNDLE = {"model": "Qwen/Qwen3.8-27B 4bit", "bundle_bytes": 16859646207, "main_mlirb_bytes": 16839645784}


def arguments(description, *names):
    parser = argparse.ArgumentParser(description=description)
    for name in ("model", "merge", "upstream", "recipe", "output", *names):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--bundle", required=True)
    return parser.parse_args(sys.argv[2:])


def merged_identity(args):
    contract = load_contract(args.model)
    revision = read_json(args.merge / "manifest.json")["merged_revision"]
    return contract, revision, {"model_id": contract["merged_model_id"], "revision": revision}


def dry_run():
    args = arguments("Resolve and load the merged JEV-27B backbone without exporting it.", "scratch")
    if (args.output / "dry-run.json").exists():
        raise FileExistsError("dry-run.json must be written to a new directory")
    contract, revision, identity = merged_identity(args)
    check_upstream(args.upstream, contract)
    recipe = args.recipe.resolve()
    command = ["coreai.llm.export", contract["merged_model_id"], "--platform", "macOS", "--compute-precision",
               contract["precision"], "--compression-config", str(recipe), "--max-context-length",
               str(contract["context_length"]), "--output-name", args.bundle, "--output-dir", str(args.scratch / "export"),
               "--experimental", "--dry-run"]
    printed = io.StringIO()
    with pinned_hub(args.merge / "checkpoint", identity):
        import torch

        from . import backbone

        backbone.check_location(args.upstream, recipe)
        entry = backbone.install(backbone.pipeline, backbone.COMPOSITE)
        with contextlib.redirect_stdout(printed):
            sys.argv = command
            try:
                runpy.run_module("coreai_models.llm.export", run_name="__main__")
                exit_code = 0
            except SystemExit as error:
                exit_code = error.code
        config = backbone.export_config(contract, recipe, contract["merged_model_id"], args.scratch / "export", args.bundle)
        resolved = backbone.pipeline.resolve_model(config)
        started = time.perf_counter()
        model = backbone.TypedDecisionBackbone.from_hf_memory_efficient(
            contract["merged_model_id"], max_context_length=contract["context_length"], target_dtype=torch.bfloat16,
            mmap_path=str(args.scratch / "layers"), hf_config_attr=entry.hf_config_attr,
            hf_state_dict_prefix=entry.hf_state_dict_prefix)
        load_s = time.perf_counter() - started
        load = backbone.load_check(model, resolved.hf_config, torch.bfloat16)
    resolution = {"model_type": resolved.model_type, "model_class": resolved.model_class.__name__,
                  "hf_config_attr": resolved.entry.hf_config_attr, "hf_state_dict_prefix": resolved.entry.hf_state_dict_prefix,
                  "config_class": type(resolved.hf_config).__name__, "max_context_length": resolved.max_context_length,
                  "compression": config.compression, "quantization_config": sorted(resolved.quantization_config or {})}
    checks = {"cli": exit_code in (0, None),
              "resolved_qwen3_5": resolution["model_type"] == backbone.COMPOSITE
              and resolution["model_class"] == backbone.TypedDecisionBackbone.__name__
              and resolution["hf_config_attr"] == "text_config"
              and resolution["hf_state_dict_prefix"] == "model.language_model.",
              "load": load["passed"]}
    lines = [printed.getvalue().rstrip(), f"CLI exit: {exit_code}",
             "Resolved: " + ", ".join(f"{key}={value}" for key, value in resolution.items()),
             "Fused: " + ", ".join(f"{name} {load['fused'][name]} of {load['fused_expected'][name]}" for name in load["fused"]),
             f"Parameters not loaded: {len(load['meta_parameters'])}", f"lm_head present: {load['lm_head']}",
             f"Parameters: {load['parameters']} tensors, {load['elements']} elements",
             f"Export contract: {'valid' if load['contract_error'] is None else load['contract_error']}",
             f"Load: {load_s:.1f} s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} B",
             f"Passed: {all(checks.values())}"]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dry-run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(args.output / "dry-run.json", {
        "passed": all(checks.values()), "checks": checks, "command": command, "merged_revision": revision,
        "resolution": resolution, "load": {**load, "load_s": round(load_s, 1),
                                           "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}})
    return 0 if all(checks.values()) else 1


def instrument(pipeline, phases, on_quantized):
    quantize, convert = pipeline.quantize_for_export, pipeline.export_macos_model

    def quantized(*args, **kwargs):
        phases["quantize_start"] = time.perf_counter()
        model = quantize(*args, **kwargs)
        phases["quantize_end"] = time.perf_counter()
        on_quantized(model)
        phases["hash_end"] = time.perf_counter()
        return model

    def converted(*args, **kwargs):
        phases["convert_start"] = time.perf_counter()
        program = convert(*args, **kwargs)
        phases["convert_end"] = time.perf_counter()
        return program

    pipeline.quantize_for_export, pipeline.export_macos_model = quantized, converted


def durations(phases):
    spans = {"load_s": ("start", "quantize_start"), "quantize_s": ("quantize_start", "quantize_end"),
             "hash_s": ("quantize_end", "hash_end"), "convert_s": ("convert_start", "convert_end"),
             "save_s": ("convert_end", "export_end"), "finish_s": ("export_end", "end"), "total_s": ("start", "end")}
    return {name: round(phases[last] - phases[first], 1) for name, (first, last) in spans.items()
            if first in phases and last in phases}


def copy_checked(source, destination, *expected):
    shutil.copyfile(source, destination)
    digest = sha256(destination)
    if any(digest != sha256(path) for path in (source, *expected)):
        raise ValueError(f"Copy differs from its source: {destination.name}")
    return digest


def notice(contract, revision):
    base, lora = contract["base_model"], contract["lora"]
    return "\n".join([
        "# Notice", "",
        "This bundle is derived from two works licensed under the Apache License, Version 2.0 (see `LICENSE`):", "",
        f"- `{base['id']}` by Qwen, revision `{base['revision']}`.",
        f"- `{contract['model_id']}` by AutoTrust AI, revision `{contract['revision']}`. Its System 1 adapter and decision head "
        "were trained on `SargeDev/jev-distill-corpus-v3` (Apache-2.0), most of whose labels are output distributions "
        "of TypeSafe Jev 1.13.", "",
        "Changes made to these works (Apache License 2.0, section 4(b)):", "",
        f"- The System 1 LoRA adapter (r {lora['r']}, alpha {lora['alpha']}) was merged into the text backbone weights as "
        f"`bf16(fp32(W) + {lora['scale']} * B @ A)` (merged revision `{revision}`).",
        f"- The merged text backbone was quantized to uint4 with the `{contract['compression']}` recipe (per-block 32, minval) "
        f"and converted with `apple/coreai-models` `{contract['upstream_revision']}` to a Core AI asset whose `main` "
        "function returns the final hidden states instead of logits.",
        "- `head.safetensors` and the files in `tokenizer/` are unmodified copies of the files distributed with "
        f"`{contract['model_id']}`.", "",
        "The sets used to check this conversion were built from `SargeDev/jev-distill-corpus-v3` and JGLUE "
        "(CC BY-SA 4.0). They are not redistributed with this bundle.", ""])


def finish_bundle(destination, args, contract, revision, hidden_size):
    checkpoint = args.merge / "checkpoint"
    tokenizer = destination / "tokenizer"
    if tokenizer.is_symlink() or not tokenizer.resolve().is_relative_to(destination.resolve()):
        raise ValueError("Unsafe tokenizer destination")
    shutil.rmtree(tokenizer)
    tokenizer.mkdir()
    tokenizer_sha256 = {name: copy_checked(checkpoint / name, tokenizer / name, args.snapshot / name)
                        for name in TOKENIZER_FILES}
    metadata_path = destination / "metadata.json"
    metadata = read_json(metadata_path)
    if metadata["source"]["hf_model_id"] != contract["merged_model_id"]:
        raise ValueError("Exported source identity changed")
    metadata["source"]["revision"] = revision
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    head_sha256 = copy_checked(args.snapshot / contract["head"], destination / contract["head"])
    license_sha256 = copy_checked(args.base / "LICENSE", destination / "LICENSE")
    (destination / "NOTICE.md").write_text(notice(contract, revision), encoding="utf-8")
    digests = {"head": head_sha256, "calibration": sha256(args.snapshot / "calibration.json"),
               "adapter": sha256(args.snapshot / "adapter/adapter_model.safetensors"),
               "adapter_config": sha256(args.snapshot / "adapter/adapter_config.json")}
    decision = document(contract, read_json(args.snapshot / "judge_config.json"), read_json(args.snapshot / "calibration.json"),
                        digests, revision, f"{args.bundle}.aimodel", hidden_size, listing(destination))
    write_json(destination / DECISION, decision)
    sums = write_sums(destination)
    errors = validate(read_json(destination / DECISION), destination)
    return {"tokenizer_sha256": tokenizer_sha256, "head_sha256": head_sha256, "license_sha256": license_sha256,
            "license_source": "base", "files": len(sums), "decision_errors": errors}


TIMES = " \u00d7 "
TYPE = re.compile(r"NDArray \((\w+), ([0-9?]+(?:" + TIMES + r"[0-9?]+)*)\)")


def described(text):
    match = TYPE.fullmatch(text)
    if match is None:
        raise ValueError(f"Unreadable asset type: {text}")
    return {"dtype": match[1].lower(), "shape": [None if size == "?" else int(size) for size in match[2].split(TIMES)]}


def inspect_asset(path):
    from coreai.runtime import _AIModelAsset

    summary = _AIModelAsset(path).summary(include_statistics=False)
    return {name: {kind: {item: described(text) for item, text in getter(name)} for kind, getter in (
        ("inputs", summary.function_inputs), ("outputs", summary.function_outputs), ("states", summary.function_states))}
        for name in summary.function_names}


def expected_functions(backbone, config):
    import torch

    states = backbone.zero_states(config, torch.bfloat16, 1, device="meta")
    shapes = [list(state.shape) for state in states]
    for shape in shapes[:2]:
        shape[3] = None
    return {"main": {"inputs": {name: {"dtype": "int32", "shape": [1, None]} for name in backbone.INPUTS},
                     "outputs": {backbone.OUTPUT: {"dtype": "float16", "shape": [1, None, config.hidden_size]}},
                     "states": {name: {"dtype": "bfloat16", "shape": shape} for name, shape in zip(backbone.STATES, shapes)}}}


def asset_bytes(path):
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def export():
    args = arguments("Export the merged JEV-27B backbone to a typed-decision Core AI bundle.", "snapshot", "base")
    destination = args.output / args.bundle
    if destination.exists() or (args.output / "provenance.json").exists():
        raise FileExistsError("The bundle must be written to a new directory")
    contract, revision, identity = merged_identity(args)
    upstream_revision = check_upstream(args.upstream, contract)
    recipe = args.recipe.resolve()
    temporary = Path(tempfile.gettempdir())
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {"merged_revision": revision, "upstream_revision": upstream_revision, "bundle": args.bundle,
                  "export_config": {"hf_model_id": contract["merged_model_id"], "variant": "macOS",
                                    "compression": contract["compression"], "recipe_sha256": sha256(recipe),
                                    "compute_precision": contract["precision"],
                                    "max_context_length": contract["context_length"]}}

    def aborted(reason):
        write_json(args.output / "provenance.json", {**provenance, "aborted": reason})

    phases = {"start": time.perf_counter()}
    monitor = Monitor(args.output / "monitor.tsv", [args.output, temporary], [str(temporary / "coreai_export_*")], aborted)
    monitor.start()
    try:
        with pinned_hub(args.merge / "checkpoint", identity):
            from . import backbone

            backbone.check_location(args.upstream, recipe)
            backbone.install(backbone.pipeline, backbone.COMPOSITE)
            settings = {**identity, "model_type": backbone.COMPOSITE, "recipe_sha256": sha256(recipe),
                        "compute_precision": contract["precision"], "compression": contract["compression"],
                        "context_length": contract["context_length"]}

            def record(model):
                devices = sorted({tensor.device.type for tensor in [*model.parameters(), *model.buffers()]})
                write_json(args.output / "quantized-hashes.json",
                           backbone.hash_record(model, {**settings, "device": ",".join(devices)}))

            instrument(backbone.pipeline, phases, record)
            config = backbone.export_config(contract, recipe, contract["merged_model_id"], args.output, args.bundle)
            resolved = backbone.pipeline.resolve_model(config)
            if resolved.model_type != backbone.COMPOSITE or resolved.model_class is not backbone.TypedDecisionBackbone:
                raise ValueError(f"Unexpected resolution: {resolved.model_type}")
            bundle = Path(backbone.pipeline.export_model(config))
            phases["export_end"] = time.perf_counter()
            if bundle.resolve() != destination.resolve():
                raise ValueError("The exporter wrote another bundle")
            hidden_size, vocab_size = resolved.hf_config.hidden_size, resolved.hf_config.vocab_size
            expected = expected_functions(backbone, resolved.hf_config)
        bundled = finish_bundle(destination, args, contract, revision, hidden_size)
        phases["end"] = time.perf_counter()
    except BaseException as error:
        phases["end"] = time.perf_counter()
        write_json(args.output / "provenance.json", {**provenance, "failed": f"{type(error).__name__}: {error}",
                                                     "timing": durations(phases), "monitor": monitor.finish()})
        raise
    monitor_record = monitor.finish()
    aimodel = destination / f"{args.bundle}.aimodel"
    inspect_started = time.perf_counter()
    functions = inspect_asset(aimodel)
    inspect_s = round(time.perf_counter() - inspect_started, 1)
    matches = functions == expected
    sizes = {"aimodel_bytes": asset_bytes(aimodel), "main_mlirb_bytes": (aimodel / "main.mlirb").stat().st_size,
             "bundle_bytes": asset_bytes(destination)}
    lm_head = {"rows": vocab_size, "hidden": hidden_size, "uint4_bytes": vocab_size * hidden_size // 2}
    comparison = {**REFERENCE_BUNDLE, "lm_head": lm_head,
                  "aimodel_smaller_than_reference_bundle_by": REFERENCE_BUNDLE["bundle_bytes"] - sizes["aimodel_bytes"],
                  "main_mlirb_smaller_by": REFERENCE_BUNDLE["main_mlirb_bytes"] - sizes["main_mlirb_bytes"]}
    comparison["smaller_by_at_least_lm_head"] = comparison["aimodel_smaller_than_reference_bundle_by"] >= lm_head["uint4_bytes"]
    comparison["main_mlirb_smaller_by_at_least_lm_head"] = comparison["main_mlirb_smaller_by"] >= lm_head["uint4_bytes"]
    write_json(args.output / "inspect.json", {
        "asset": aimodel.name, "load_s": inspect_s, "functions": functions, "expected": expected, "matches_expected": matches,
        "only_main": list(functions) == ["main"],
        "no_logits_output": all("logits" not in function["outputs"] for function in functions.values()),
        "sizes": sizes, "reference_comparison": comparison, "decision_errors": bundled["decision_errors"],
        "passed": matches and not bundled["decision_errors"]})
    write_json(args.output / "provenance.json", {
        **provenance, "timing": durations(phases), "monitor": monitor_record, "sizes": sizes,
        "peak_rss_bytes": monitor_record["peak"]["max_rss_bytes"], "apple_toolchain": apple_toolchain(),
        "versions": read_json(args.output / "quantized-hashes.json")["versions"], "bundle_files": bundled})
    return 0 if matches and not bundled["decision_errors"] and monitor_record["aborts"] == 0 else 1


def main():
    stages = {"dry-run": dry_run, "export": export}
    if len(sys.argv) < 2 or sys.argv[1] not in stages:
        raise SystemExit(f"usage: export.py {{{','.join(stages)}}} ...")
    return stages[sys.argv[1]]()


if __name__ == "__main__":
    sys.exit(main())
