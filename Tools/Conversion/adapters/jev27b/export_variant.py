from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from ...storage import read_json, write_json
from ..pinned import apple_toolchain, check_upstream, pinned_hub, sha256
from .decision import DECISION, compare_hashes, listing, validate, write_sums
from .export import (arguments, asset_bytes, durations, expected_functions, finish_bundle, inspect_asset, instrument,
                     merged_identity)
from .monitor import Monitor

VARIANT = "gdn-fp32"
ANCHOR = "- The merged text backbone was quantized"
NOTE = ("- In every linear-attention layer, the six inputs of the gated delta update are cast to float32 before the "
        "composite operation and its two outputs are cast back, so that the update runs in float32.")
DECLARATION = re.compile(r'^\s*coreai\.graph private noinline @(\S+)\((.*?)\) -> .*?composite_declaration<"(\w+)"', re.M)
CALL = re.compile(r"coreai\.invoke @(\S+?)\(.*?\)\s*:\s*\((.*?)\) ->")
TENSOR = re.compile(r"tensor<[^>]*>")
CAST = re.compile(r"\bcoreai\.cast\b")
KEY = re.compile(r"[0-9a-f]{64}")


def ir_counts(text):
    kinds = {name: kind for name, _, kind in DECLARATION.findall(text)}
    calls = CALL.findall(text)
    operands = Counter(types for callee, types in calls if kinds.get(callee) == "gated_delta_update")
    float32 = sum(count for types, count in operands.items()
                  if len(TENSOR.findall(types)) == 6 and all(item.endswith("xf32>") for item in TENSOR.findall(types)))
    return {"composite_declarations": dict(Counter(kinds.values())),
            "invoke_sites_by_kind": dict(Counter(kinds.get(callee, "?") for callee, _ in calls)),
            "gated_delta_update_call_operand_types": dict(operands),
            "gated_delta_update_sites": sum(operands.values()), "gated_delta_update_float32_sites": float32,
            "cast_operations": len(CAST.findall(text))}


def ir_of(asset):
    from coreai.authoring import AIModelAsset

    module = AIModelAsset.load(str(asset)).program._module._mlir_module
    return ir_counts(module.operation.get_asm(large_elements_limit=16, large_resource_limit=64))


def ir_summary(asset):
    started = time.perf_counter()
    completed = subprocess.run([sys.executable, "-m", "Tools.Conversion.adapters.jev27b.export_variant", "ir", str(asset)],
                               capture_output=True, text=True, check=True)
    return {**json.loads(completed.stdout.strip().splitlines()[-1]), "seconds": round(time.perf_counter() - started, 1)}


def add_note(destination):
    path = destination / "NOTICE.md"
    lines = path.read_text(encoding="utf-8").split("\n")
    index = next((number for number, line in enumerate(lines) if line.startswith(ANCHOR)), None)
    if index is None:
        raise ValueError("NOTICE.md has no conversion line to extend")
    lines.insert(index + 1, NOTE)
    path.write_text("\n".join(lines), encoding="utf-8")


def finish_variant(destination, args, contract, revision, hidden_size):
    bundled = finish_bundle(destination, args, contract, revision, hidden_size)
    add_note(destination)
    decision = read_json(destination / DECISION)
    decision["files"] = listing(destination)
    write_json(destination / DECISION, decision)
    sums = write_sums(destination)
    return {**bundled, "files": len(sums), "decision_errors": validate(read_json(destination / DECISION), destination)}


def baseline_identity(baseline):
    key = Path(baseline).parent.name[:64]
    return {"export_key": key if KEY.fullmatch(key) else None,
            "quantized_hashes_sha256": sha256(Path(baseline) / "quantized-hashes.json"),
            "bundle": read_json(Path(baseline) / "provenance.json")["bundle"]}


def verdict(hashes, wrapped, layers, ir, inspected, monitor_aborts):
    checks = {"hashes_match": hashes["hashes_match"], "wrapped_every_linear_attention_layer": wrapped == layers > 0,
              "float32_gated_delta_updates": ir["gated_delta_update_sites"] == ir["gated_delta_update_float32_sites"]
              == layers,
              "inspect": inspected, "monitor": monitor_aborts == 0}
    return {"passed": all(checks.values()), "checks": checks}


def export():
    args = arguments("Export the merged JEV-27B backbone with a float32 gated delta update.", "snapshot", "base",
                     "baseline")
    destination = args.output / args.bundle
    if destination.exists() or (args.output / "provenance.json").exists():
        raise FileExistsError("The bundle must be written to a new directory")
    contract, revision, identity = merged_identity(args)
    upstream_revision = check_upstream(args.upstream, contract)
    recipe = args.recipe.resolve()
    baseline = baseline_identity(args.baseline)
    temporary = Path(tempfile.gettempdir())
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {"merged_revision": revision, "upstream_revision": upstream_revision, "bundle": args.bundle,
                  "variant": VARIANT, "baseline": baseline,
                  "export_config": {"hf_model_id": contract["merged_model_id"], "variant": "macOS",
                                    "compression": contract["compression"], "recipe_sha256": sha256(recipe),
                                    "compute_precision": contract["precision"],
                                    "max_context_length": contract["context_length"]}}

    def aborted(reason):
        write_json(args.output / "provenance.json", {**provenance, "aborted": reason})

    phases = {"start": time.perf_counter()}
    wrapped = {}
    monitor = Monitor(args.output / "monitor.tsv", [args.output, temporary], [str(temporary / "coreai_export_*")], aborted)
    monitor.start()
    try:
        with pinned_hub(args.merge / "checkpoint", identity):
            from . import backbone, backbone_gdnfp32

            backbone.check_location(args.upstream, recipe)
            backbone_gdnfp32.install(backbone.pipeline)
            settings = {**identity, "model_type": backbone.COMPOSITE, "recipe_sha256": sha256(recipe),
                        "compute_precision": contract["precision"], "compression": contract["compression"],
                        "context_length": contract["context_length"]}

            def record(model):
                wrapped["layers"] = backbone_gdnfp32.wrapped_count(model)
                devices = sorted({tensor.device.type for tensor in [*model.parameters(), *model.buffers()]})
                write_json(args.output / "quantized-hashes.json",
                           backbone.hash_record(model, {**settings, "device": ",".join(devices)}))

            instrument(backbone.pipeline, phases, record)
            config = backbone.export_config(contract, recipe, contract["merged_model_id"], args.output, args.bundle)
            resolved = backbone.pipeline.resolve_model(config)
            if resolved.model_type != backbone.COMPOSITE \
                    or resolved.model_class is not backbone_gdnfp32.TypedDecisionBackboneGdnFp32:
                raise ValueError(f"Unexpected resolution: {resolved.model_type}")
            bundle = Path(backbone.pipeline.export_model(config))
            phases["export_end"] = time.perf_counter()
            if bundle.resolve() != destination.resolve():
                raise ValueError("The exporter wrote another bundle")
            hidden_size = resolved.hf_config.hidden_size
            layers = backbone.layer_count(resolved.hf_config, "linear_attention")
            expected = expected_functions(backbone, resolved.hf_config)
        bundled = finish_variant(destination, args, contract, revision, hidden_size)
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
    inspected = matches and not bundled["decision_errors"]
    write_json(args.output / "inspect.json", {
        "asset": aimodel.name, "load_s": inspect_s, "functions": functions, "expected": expected, "matches_expected": matches,
        "only_main": list(functions) == ["main"],
        "no_logits_output": all("logits" not in function["outputs"] for function in functions.values()),
        "sizes": sizes, "decision_errors": bundled["decision_errors"], "passed": inspected})
    hashes = compare_hashes(read_json(args.baseline / "quantized-hashes.json"),
                            read_json(args.output / "quantized-hashes.json"))
    ir = {"variant": ir_summary(aimodel),
          "baseline": ir_summary(args.baseline / baseline["bundle"] / f"{baseline['bundle']}.aimodel")}
    result = verdict(hashes, wrapped.get("layers", 0), layers, ir["variant"], inspected, monitor_record["aborts"])
    write_json(args.output / "variant.json", {
        **result, "variant": VARIANT, "linear_attention_layers": layers, "wrapped_layers": wrapped.get("layers", 0),
        "hash_comparison": hashes, "ir": ir, "baseline": baseline, "timing": durations(phases),
        "peak_rss_bytes": monitor_record["peak"]["max_rss_bytes"], "sizes": sizes})
    write_json(args.output / "provenance.json", {
        **provenance, "timing": durations(phases), "monitor": monitor_record, "sizes": sizes,
        "peak_rss_bytes": monitor_record["peak"]["max_rss_bytes"], "apple_toolchain": apple_toolchain(),
        "versions": read_json(args.output / "quantized-hashes.json")["versions"], "bundle_files": bundled})
    return 0 if result["passed"] else 1


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "ir":
        print(json.dumps(ir_of(Path(sys.argv[2]))))
        return 0
    if len(sys.argv) < 2 or sys.argv[1] != "export":
        raise SystemExit("usage: export_variant.py {export ...,ir <asset>}")
    return export()


if __name__ == "__main__":
    sys.exit(main())
