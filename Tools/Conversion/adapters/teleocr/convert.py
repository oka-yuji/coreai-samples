import argparse
import asyncio
import gc
import importlib.metadata
import resource
import time
from pathlib import Path

import numpy as np
import torch

from .contracts import (
    REQUIRED_VISION_CASES, REVISION, TOKEN_LIMIT, VISION_CASES, VISION_FP16_COSINE_MIN, VISION_FP16_NRMSE_LIMIT,
    VISION_NRMSE_LIMIT, WINDOW_LIMIT, WINDOW_PATCHES, digest, logit_metrics, metrics, restore_order, write_json,
)
from .wrappers import Decoder, Embed, Main, Prefill, StatelessPrefill, Vision

VISION_INPUTS = ["pixel_values", "key_valid", "cos", "sin"]
DECODER_INPUTS = ["cos", "sin", "mask"]
STATES = ["keyCache", "valueCache"]


def load_model(snapshot):
    from transformers import AutoModel
    return AutoModel.from_pretrained(str(snapshot), trust_remote_code=True, torch_dtype=torch.float32,
                                     local_files_only=True).eval().requires_grad_(False)


def export(module, example, dynamic_shapes=None, half=False):
    from coreai_opt.casting import cast_to_16_bit_precision
    from coreai_torch import get_decomp_table
    with torch.no_grad():
        program = torch.export.export(module, example, dynamic_shapes=dynamic_shapes, strict=False)
    program = program.run_decompositions(get_decomp_table())
    return cast_to_16_bit_precision(program) if half else program


def save(programs, path, description):
    from coreai.runtime import AIModelAssetMetadata
    from coreai_torch import TorchConverter
    converter = TorchConverter(mode=TorchConverter.Mode.RELEASE)
    for entry, program, inputs, outputs, states in programs:
        converter.add_exported_program(program, input_names=inputs, output_names=outputs, state_names=states,
                                       entrypoint_name=entry)
    metadata = AIModelAssetMetadata()
    metadata.author = "XingChen-AGI (TeleOCR); converted by Yuji Oka"
    metadata.license = "Apache-2.0"
    metadata.model_description = description
    metadata.set_custom("source_revision", REVISION)
    if path.exists():
        raise FileExistsError(path)
    converter.to_coreai().save_asset(path, metadata)


class Runtime:
    def __init__(self, path):
        from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions
        self.runner = asyncio.Runner()
        options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
        started = time.perf_counter()
        self.model = self.runner.run(AIModel.load(path, specialization_options=options))
        self.function = self.model.load_function("main")
        self.load_seconds = time.perf_counter() - started

    def __call__(self, feeds, output):
        from coreai.runtime import NDArray
        started = time.perf_counter()
        result = self.runner.run(self.function({key: NDArray(np.ascontiguousarray(value)) for key, value in feeds.items()}))
        return np.asarray(result[output].numpy()).astype(np.float32), time.perf_counter() - started

    def close(self):
        self.function = self.model = None
        self.runner.close()
        gc.collect()


def against_eager(measure, eager, actual):
    if not np.isfinite(eager).all():
        return {"finite": False, "shape": list(eager.shape), "skipped": "non-finite PyTorch fp16 output"}
    return measure(eager, actual)


def half(values):
    return {key: value.astype(np.float16) if value.dtype == np.float32 else value for key, value in values.items()}


def tensors(values, names):
    return tuple(torch.from_numpy(np.ascontiguousarray(values[name])) for name in names)


def vision(args):
    cases = {name: dict(np.load(args.reference / f"vision-{name}.npz")) for name in VISION_CASES}
    module = Vision(load_model(args.snapshot).visual).eval()
    rows = {name: {"windows": len(case["key_valid"]) // WINDOW_PATCHES, "patches": len(case["key_valid"])}
            for name, case in cases.items()}

    def compare(name, key, output):
        rows[name][key] = metrics(cases[name]["reference"], restore_order(output, cases[name]["order"]))

    with torch.no_grad():
        for name, case in cases.items():
            compare(name, "wrapper_fp32", module(*tensors(case, VISION_INPUTS)).numpy())
    windows = torch.export.Dim("windows", min=1, max=WINDOW_LIMIT)
    shapes = {name: {0: WINDOW_PATCHES * windows} for name in VISION_INPUTS}
    example = tensors(cases["square448"], VISION_INPUTS)
    timing = {}
    for precision in ("fp32", "fp16"):
        started = time.perf_counter()
        program = export(module, example, shapes, half=precision == "fp16")
        if precision == "fp16":
            graph = program.module(check_guards=False)
            with torch.no_grad():
                for name, case in cases.items():
                    eager = graph(*tensors(half(case), VISION_INPUTS)).float().numpy()
                    rows[name]["eager_fp16"] = eager
                    compare(name, "eager_fp16_vs_reference", eager)
            del graph
        path = args.scratch / "vision-fp32.aimodel" if precision == "fp32" else args.output / "vision.aimodel"
        save([("main", program, VISION_INPUTS, ["image_features"], [])], path,
             f"TeleOCR vision encoder and merger, dynamic windows, {precision}")
        timing[f"export_{precision}_s"] = time.perf_counter() - started
        del program
        gc.collect()
        runtime = Runtime(path)
        timing[f"load_{precision}_s"] = runtime.load_seconds
        for name, case in cases.items():
            feeds = {key: case[key] for key in VISION_INPUTS}
            output, seconds = runtime(half(feeds) if precision == "fp16" else feeds, "image_features")
            rows[name][f"coreai_{precision}_s"] = seconds
            compare(name, f"coreai_{precision}_vs_reference", output)
            if precision == "fp16":
                eager = rows[name].pop("eager_fp16")
                rows[name]["coreai_fp16_vs_eager_fp16"] = against_eager(metrics, restore_order(eager, case["order"]),
                                                                        restore_order(output, case["order"]))
            print(name, precision, {key: value for key, value in rows[name].items() if key.startswith("coreai")}, flush=True)
        runtime.close()

    def gated(name):
        exact, half_precision = rows[name]["coreai_fp32_vs_reference"], rows[name]["coreai_fp16_vs_reference"]
        return (exact["finite"] and exact["nrmse"] <= VISION_NRMSE_LIMIT and half_precision["finite"]
                and (half_precision["cosine"] or 0.0) >= VISION_FP16_COSINE_MIN
                and half_precision["nrmse"] <= VISION_FP16_NRMSE_LIMIT)

    gate = {"fp32_nrmse_max": VISION_NRMSE_LIMIT, "fp16_cosine_min": VISION_FP16_COSINE_MIN,
            "fp16_nrmse_max": VISION_FP16_NRMSE_LIMIT, "required": REQUIRED_VISION_CASES}
    return all(gated(name) for name in REQUIRED_VISION_CASES), {"cases": rows, "gate": gate, **timing}, "vision.aimodel"


def embed(args):
    from safetensors import safe_open
    prefill = dict(np.load(args.reference / "prefill-f1-448.npz"))
    with safe_open(str(args.snapshot / "model.safetensors"), "pt") as handle:
        weight = handle.get_tensor("model.embed_tokens.weight")
    module = Embed(weight).eval()
    tokens = torch.export.Dim("tokens", min=1, max=TOKEN_LIMIT)
    started = time.perf_counter()
    program = export(module, (torch.from_numpy(prefill["input_ids"]),), {"input_ids": {1: tokens}})
    path = args.output / "embed.aimodel"
    save([("main", program, ["input_ids"], ["inputs_embeds"], [])], path,
         "TeleOCR token embedding table (original bfloat16 values, float32 output)")
    result = {"export_s": time.perf_counter() - started, "table_dtype": str(weight.dtype)}
    runtime = Runtime(path)
    output, result["coreai_s"] = runtime({"input_ids": prefill["input_ids"]}, "inputs_embeds")
    result["f1_prompt"] = metrics(prefill["embeddings"], output)
    single, _ = runtime({"input_ids": prefill["input_ids"][:, -1:]}, "inputs_embeds")
    result["single_token"] = metrics(prefill["embeddings"][:, -1:], single)
    runtime.close()
    passed = all(result[key]["finite"] and result[key]["max_abs"] == 0 for key in ("f1_prompt", "single_token"))
    return passed, result, "embed.aimodel"


def decoder(args):
    prefill = dict(np.load(args.reference / "prefill-f1-448.npz"))
    core = Decoder(load_model(args.snapshot)).eval()
    names = ["inputs_embeds", *DECODER_INPUTS]
    reference = prefill["logits"]
    stateless = StatelessPrefill(core).eval()
    with torch.no_grad():
        result = {"wrapper_fp32": logit_metrics(reference, stateless(*tensors(prefill, names)).numpy())}
    started = time.perf_counter()
    program = export(stateless, tensors(prefill, names), half=True)
    with torch.no_grad():
        eager = program.module(check_guards=False)(*tensors(half(prefill), names)).float().numpy()
    result["eager_fp16_vs_reference"] = logit_metrics(reference, eager)
    path = args.scratch / "prefill-stateless-fp16.aimodel"
    save([("main", program, names, ["logits"], [])], path, "TeleOCR decoder stateless check, fp16")
    del program
    gc.collect()
    runtime = Runtime(path)
    output, result["coreai_s"] = runtime(half({name: prefill[name] for name in names}), "logits")
    runtime.close()
    result["coreai_fp16_vs_reference"] = logit_metrics(reference, output)
    result["coreai_fp16_vs_eager_fp16"] = against_eager(logit_metrics, eager, output)
    result["stateless_s"] = time.perf_counter() - started
    print({key: value for key, value in result.items() if isinstance(value, dict)}, flush=True)
    started = time.perf_counter()
    count, past, capacity = 8, 5, 64
    cache = torch.zeros(len(core.layers), 1, core.kv_heads, capacity, core.head_dim)
    context = (torch.zeros(1, count, core.head_dim), torch.zeros(1, count, core.head_dim),
               torch.zeros(1, 1, count, past + count))
    tokens = torch.export.Dim("tokens", min=1, max=TOKEN_LIMIT)
    total = torch.export.Dim("total", min=1, max=TOKEN_LIMIT)
    length = torch.export.Dim("capacity", min=1, max=TOKEN_LIMIT)
    shared = ({1: tokens}, {1: tokens}, {2: tokens, 3: total}, {3: length}, {3: length})
    programs = []
    for entry, module, first, name in (
        ("prefill", Prefill(core), torch.zeros(1, count, core.hidden), "inputs_embeds"),
        ("main", Main(core), torch.zeros(1, count, dtype=torch.int32), "input_ids"),
    ):
        program = export(module.eval(), (first, *context, cache.clone(), cache.clone()), ({1: tokens}, *shared), half=True)
        programs.append((entry, program, [name, *DECODER_INPUTS], ["logits"], STATES))
    save(programs, args.output / "main.aimodel", "TeleOCR decoder with prefill and main functions and FP16 KV states")
    result["stateful_export_s"] = time.perf_counter() - started
    passed = result["coreai_fp16_vs_reference"]["finite"] and result["coreai_fp16_vs_reference"]["argmax_match"]
    return passed, result, "main.aimodel"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--kind", choices=["vision", "embed", "decoder"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    started = time.perf_counter()
    passed, result, asset = {"vision": vision, "embed": embed, "decoder": decoder}[args.kind](args)
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "coreai-torch", "coreai-core", "coreai-opt")}
    write_json(args.output / "comparison.json", {"kind": args.kind, "passed": bool(passed), **result,
                                                 "seconds": time.perf_counter() - started,
                                                 "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                                 "versions": versions,
                                                 "code_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                                                 ("convert.py", "wrappers.py", "contracts.py")},
                                                 "asset": asset})
    if not passed:
        raise SystemExit(f"The {args.kind} comparison failed")


if __name__ == "__main__":
    main()
