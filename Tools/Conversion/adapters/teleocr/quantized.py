import argparse
import copy
import gc
import importlib.metadata
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .contracts import (
    VISION_CASES, VISION_FP16_COSINE_MIN, VISION_FP16_NRMSE_LIMIT, WINDOW_LIMIT, WINDOW_PATCHES, digest, logit_metrics,
    metrics, restore_order, write_json,
)
from .convert import (
    DECODER_INPUTS, VISION_INPUTS, Runtime, against_eager, export, half, load_model, save, save_decoder, tensors,
)
from .quantization import audit_program, quantize, save_reference
from .wrappers import Decoder, StatelessPrefill, Vision

DECODER_NRMSE_LIMIT = 0.05
DECODER_COSINE_MIN = 0.999


def plain_reference(module, weights):
    result = copy.deepcopy(module)
    for name, value in weights.items():
        target = result.get_submodule(name.removesuffix(".weight"))
        dense = torch.nn.Linear(target.in_features, target.out_features, bias=target.bias is not None, dtype=value.dtype)
        with torch.no_grad():
            dense.weight.copy_(value)
            if target.bias is not None:
                dense.bias.copy_(target.bias)
        result.set_submodule(name.removesuffix(".weight"), dense)
    if any(torch.nn.utils.parametrize.is_parametrized(child) for child in result.modules()):
        raise ValueError("The independent reference cannot retain compression operators")
    return result.eval().requires_grad_(False)


def compressed(module, args):
    module, record, weights = quantize(module, args.kind, args.bits)
    path = args.output / "reference.safetensors"
    save_reference(path, weights, record)
    record["reference_sha256"] = digest(path)
    reference = plain_reference(module, weights)
    del weights
    return module, reference, record


def conversion_passes(check):
    return (check["finite"] and check["nrmse"] <= DECODER_NRMSE_LIMIT
            and (check["cosine"] or 0) >= DECODER_COSINE_MIN)


def compile_vision(path, scratch):
    architecture = subprocess.check_output(
        ["xcrun", "swift", "-e", "import CoreAI; print(AIModel.deviceArchitectureName)"], text=True, timeout=120).strip()
    if not architecture or not architecture.isascii() or not architecture.isalnum():
        raise ValueError("The SDK did not return a valid host architecture")
    output = scratch / "specialized"
    started = time.perf_counter()
    subprocess.run(["xcrun", "coreai-build", "compile", str(path), "--output", str(output),
                    "--architecture", architecture, "--preferred-compute", "gpu", "--expect-frequent-reshapes"],
                   check=True, timeout=1200)
    compiled = output / f"{path.stem}.{architecture}.aimodelc"
    if not compiled.is_dir():
        raise ValueError("The host specialization did not produce its declared asset")
    return compiled, {"architecture": architecture, "preferred_compute": "gpu", "expect_frequent_reshapes": True,
                       "seconds": time.perf_counter() - started}


def decoder(args):
    data = dict(np.load(args.reference / "prefill-f1-448.npz"))
    names = ["inputs_embeds", *DECODER_INPUTS]
    core, reference, record = compressed(Decoder(load_model(args.snapshot)).eval(), args)
    with torch.no_grad():
        expected = StatelessPrefill(reference)(*tensors(data, names)).numpy()
    del reference
    gc.collect()
    result = {"quality": logit_metrics(data["logits"], expected), "compression": record}
    program = export(StatelessPrefill(core).eval(), tensors(data, names), half=True)
    with torch.no_grad():
        eager = program.module(check_guards=False)(*tensors(half(data), names)).float().numpy()
    result["float16_difference"] = logit_metrics(expected, eager)
    path = args.scratch / "prefill-stateless.aimodel"
    save([("main", program, names, ["logits"], [])], path, f"TeleOCR decoder INT{args.bits} stateless conversion check")
    del program
    gc.collect()
    runtime = Runtime(path)
    actual, result["coreai_s"] = runtime(half({name: data[name] for name in names}), "logits")
    runtime.close()
    result["conversion"] = logit_metrics(expected, actual)
    result["conversion_vs_eager_fp16"] = against_eager(logit_metrics, eager, actual)
    check = result["conversion"]
    check["numeric_divergence"] = bool(conversion_passes(check) and not check["argmax_match"]
                                       and check["reference_top2_gap"] <= 2 * check["max_abs"])
    result["gate"] = {"nrmse_max": DECODER_NRMSE_LIMIT, "cosine_min": DECODER_COSINE_MIN,
                       "reference": "independently dequantized weights, float32 PyTorch"}
    passed = result["quality"]["finite"] and conversion_passes(check)
    save_decoder(core, args.output / "main.aimodel",
                 f"TeleOCR decoder with INT{args.bits} group64 attention/MLP weights and FP16 activations and KV", record)
    return passed, result, "main.aimodel"


def vision(args):
    cases = {name: dict(np.load(args.reference / f"vision-{name}.npz")) for name in VISION_CASES}
    module, reference, record = compressed(Vision(load_model(args.snapshot).visual).eval(), args)
    expected, rows = {}, {}
    with torch.no_grad():
        for name, case in cases.items():
            expected[name] = reference(*tensors(case, VISION_INPUTS)).numpy()
            rows[name] = {"quality": metrics(case["reference"], restore_order(expected[name], case["order"]))}
    del reference
    gc.collect()
    windows = torch.export.Dim("windows", min=1, max=WINDOW_LIMIT)
    program = export(module, tensors(cases["square448"], VISION_INPUTS),
                     {name: {0: WINDOW_PATCHES * windows} for name in VISION_INPUTS}, half=True)
    record["functions"] = {"main": audit_program(program, record)}
    path = args.output / "vision.aimodel"
    save([("main", program, VISION_INPUTS, ["image_features"], [])], path,
         "TeleOCR vision with INT8 per-output-channel Linear weights and FP16 activations")
    del module, program
    gc.collect()
    compiled, specialization = compile_vision(path, args.scratch)
    runtime = Runtime(compiled)
    for name, case in cases.items():
        actual, rows[name]["coreai_s"] = runtime(half({key: case[key] for key in VISION_INPUTS}), "image_features")
        rows[name]["conversion"] = metrics(restore_order(expected[name], case["order"]), restore_order(actual, case["order"]))
    runtime.close()
    passed = all(row["quality"]["finite"] and row["conversion"]["finite"]
                 and row["conversion"]["nrmse"] <= VISION_FP16_NRMSE_LIMIT
                 and (row["conversion"]["cosine"] or 0) >= VISION_FP16_COSINE_MIN for row in rows.values())
    return passed, {"cases": rows, "compression": record, "specialization": specialization}, "vision.aimodel"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--kind", choices=["decoder", "vision"], required=True)
    parser.add_argument("--bits", type=int, choices=[4, 8], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    if args.kind == "vision" and args.bits != 8:
        parser.error("Only vision INT8 is declared")
    args.output.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    started = time.perf_counter()
    passed, result, asset = {"decoder": decoder, "vision": vision}[args.kind](args)
    write_json(args.output / "compression.json", result["compression"])
    write_json(args.output / "comparison.json", {
        "kind": args.kind, "precision": f"int{args.bits}", "passed": bool(passed), **result,
        "seconds": time.perf_counter() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "coreai-torch", "coreai-core", "coreai-opt")},
        "code_sha256": {name: digest(Path(__file__).with_name(name))
                        for name in ("quantized.py", "quantization.py", "convert.py", "wrappers.py", "contracts.py")},
        "asset": asset,
    })
    if not passed:
        raise SystemExit(f"The INT{args.bits} {args.kind} conversion comparison failed")


if __name__ == "__main__":
    main()
