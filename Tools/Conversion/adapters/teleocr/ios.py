import argparse
import gc
import importlib.metadata
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .contracts import (
    IOS_CONTEXT, IOS_MASKED, IOS_PREFILL, IOS_WINDOWS, PALETTE_BITS, PALETTE_GROUP, VISION_CASES,
    VISION_FP16_COSINE_MIN, VISION_FP16_NRMSE_LIMIT, WINDOW_PATCHES, digest, logit_metrics, metrics, restore_order,
    write_json,
)
from .convert import STATES, VISION_INPUTS, Runtime, export, half, load_model, save, tensors
from .quantized import DECODER_COSINE_MIN, DECODER_NRMSE_LIMIT, conversion_passes
from .wrappers import Decoder, Embed, StaticMain, StaticPrefill, Vision

PREFILL_INPUTS = ["inputs_embeds", "cos", "sin", "mask", "logit_row"]
MAIN_INPUTS = ["input_ids", "cos", "sin", "mask", "write_mask"]


def bucket(sizes, count):
    return min(size for size in sizes if size >= count)


def vision_variants(cases):
    for name, case in cases.items():
        yield name, case, bucket(IOS_WINDOWS, len(case["key_valid"]) // WINDOW_PATCHES)
        if name == "square448":
            yield "square448_windows32", case, 32


def decoder_comparison_passes(check):
    bounded = conversion_passes(check)
    check["numeric_divergence"] = bool(
        bounded and not check["argmax_match"]
        and set(check["top5"][:2]) == set(check["reference_top5"][:2])
        and check["reference_top2_gap"] <= 2 * check["max_abs"])
    return bounded and (check["argmax_match"] or check["numeric_divergence"])


def padded_window_rows(case, windows):
    rows = WINDOW_PATCHES * windows - len(case["key_valid"])
    fill = {"pixel_values": 0.0, "key_valid": 0.0, "cos": 1.0, "sin": 0.0}
    return {name: np.concatenate([case[name], np.full((rows, *case[name].shape[1:]), fill[name], case[name].dtype)])
            for name in VISION_INPUTS}


def prefill_inputs(embeddings, cos, sin, size):
    count, width = embeddings.shape[1], cos.shape[-1]
    mask = np.full((1, 1, size, size), IOS_MASKED, np.float32)
    mask[0, 0, :count, :count] = np.triu(np.full((count, count), IOS_MASKED, np.float32), 1)
    return {"inputs_embeds": np.pad(embeddings, ((0, 0), (0, size - count), (0, 0))).astype(np.float32),
            "cos": np.concatenate([cos, np.ones((1, size - count, width), np.float32)], 1),
            "sin": np.concatenate([sin, np.zeros((1, size - count, width), np.float32)], 1),
            "mask": mask, "logit_row": np.array([count - 1], np.int32)}


def step_inputs(token, position, rope_position, theta, width):
    inverse = 1.0 / theta ** (np.arange(0, width, 2, dtype=np.float64) / width)
    angles = np.concatenate([rope_position * inverse, rope_position * inverse]).astype(np.float32)
    mask = np.full((1, 1, 1, IOS_CONTEXT), IOS_MASKED, np.float32)
    mask[..., :position + 1] = 0
    write = np.zeros(IOS_CONTEXT, np.float32)
    write[position] = 1
    return {"input_ids": np.array([[token]], np.int32), "cos": np.cos(angles)[None, None],
            "sin": np.sin(angles)[None, None], "mask": mask, "write_mask": write}


def vision(args):
    cases = {name: dict(np.load(args.reference / f"vision-{name}.npz")) for name in VISION_CASES}
    module = Vision(load_model(args.snapshot).visual, masked=IOS_MASKED).eval()
    windows = torch.export.Dim("windows", min=1, max=max(IOS_WINDOWS))
    started = time.perf_counter()
    program = export(module, tensors(cases["square448"], VISION_INPUTS),
                     {name: {0: WINDOW_PATCHES * windows} for name in VISION_INPUTS}, half=True)
    del module
    gc.collect()
    shapes = {f'"{count}"': {"pixel_values": (WINDOW_PATCHES * count, 1176), "key_valid": (WINDOW_PATCHES * count,),
                             "cos": (WINDOW_PATCHES * count, 80), "sin": (WINDOW_PATCHES * count, 80)}
              for count in IOS_WINDOWS}
    save([("windows", program, VISION_INPUTS, ["image_features"], [])], args.output / "vision.aimodel",
         "TeleOCR vision encoder and merger, static window buckets for iOS, fp16", static={"windows": shapes})
    result = {"export_s": time.perf_counter() - started, "cases": {}}
    del program
    gc.collect()
    static = Runtime(args.output / "vision.aimodel", [f"windows_{count}" for count in IOS_WINDOWS])
    dynamic = Runtime(args.dynamic / "vision.aimodel")
    result["load_s"] = static.load_seconds
    for name, case, size in vision_variants(cases):
        count = len(case["key_valid"]) // WINDOW_PATCHES
        output, seconds = static(half(padded_window_rows(case, size)), "image_features", f"windows_{size}")
        actual = restore_order(output[:count * 16], case["order"])
        baseline, _ = dynamic(half({key: case[key] for key in VISION_INPUTS}), "image_features")
        result["cases"][name] = {"windows": count, "bucket": size, "static_s": seconds,
                                 "static_vs_dynamic": metrics(restore_order(baseline, case["order"]), actual),
                                 "static_vs_reference": metrics(case["reference"], actual)}
        print(name, result["cases"][name], flush=True)
    static.close()
    dynamic.close()

    def gated(row):
        reference, baseline = row["static_vs_reference"], row["static_vs_dynamic"]
        return (reference["finite"] and baseline["finite"] and (reference["cosine"] or 0.0) >= VISION_FP16_COSINE_MIN
                and max(reference["nrmse"], baseline["nrmse"]) <= VISION_FP16_NRMSE_LIMIT)

    result["gate"] = {"cosine_vs_reference_min": VISION_FP16_COSINE_MIN, "nrmse_max": VISION_FP16_NRMSE_LIMIT}
    return all(gated(row) for row in result["cases"].values()), result, "vision.aimodel"


def embed(args):
    from safetensors import safe_open
    prefill = dict(np.load(args.reference / "prefill-f1-448.npz"))
    with safe_open(str(args.snapshot / "model.safetensors"), "pt") as handle:
        weight = handle.get_tensor("model.embed_tokens.weight")
    count = prefill["input_ids"].shape[1]
    ids = np.zeros((1, IOS_PREFILL[-1]), np.int32)
    ids[:, :count] = prefill["input_ids"]
    started = time.perf_counter()
    program = export(Embed(weight).eval(), (torch.from_numpy(ids),))
    save([("main", program, ["input_ids"], ["inputs_embeds"], [])], args.output / "embed.aimodel",
         f"TeleOCR token embedding table for iOS (original bfloat16 values, float32 output, {IOS_PREFILL[-1]} tokens)")
    result = {"export_s": time.perf_counter() - started, "table_dtype": str(weight.dtype)}
    runtime = Runtime(args.output / "embed.aimodel")
    output, result["coreai_s"] = runtime({"input_ids": ids}, "inputs_embeds")
    runtime.close()
    result["f1_prompt"] = metrics(prefill["embeddings"], output[:, :count])
    return result["f1_prompt"]["finite"] and result["f1_prompt"]["max_abs"] == 0, result, "embed.aimodel"


def palettize(core, example):
    os.environ["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])
    os.environ["TORCH_EXTENSIONS_DIR"] = str(Path(tempfile.gettempdir()) / "torch_extensions")
    from coreai_opt import ExportBackend
    from coreai_opt.palettization import (
        KMeansPalettizer, KMeansPalettizerConfig, ModuleKMeansPalettizerConfig, PalettizationSpec,
    )
    from coreai_opt.palettization.spec import PerGroupedChannelGranularity
    spec = PalettizationSpec(n_bits=PALETTE_BITS, granularity=PerGroupedChannelGranularity(axis=0, group_size=PALETTE_GROUP))
    config = KMeansPalettizerConfig(
        global_config=None,
        module_type_configs={"torch.nn.modules.linear.Linear": ModuleKMeansPalettizerConfig(op_state_spec={"weight": spec})},
        module_name_configs={r"decoder\.head": None})
    wrapper = StaticPrefill(core).eval()
    palettizer = KMeansPalettizer(wrapper, config)
    wrapper = palettizer.prepare(example_inputs=example)
    names = sorted(name for name, module in wrapper.named_modules() if hasattr(module, "parametrizations"))
    return palettizer.finalize(backend=ExportBackend.CoreAI).decoder, names


def decoder(args):
    from coreai.runtime import NDArray
    prefill = dict(np.load(args.reference / "prefill-f1-448.npz"))
    model = load_model(args.snapshot)
    theta = model.config.text_config.rope_theta
    core = Decoder(model).eval()
    del model
    count = prefill["input_ids"].shape[1]
    size = bucket(IOS_PREFILL, count)
    feeds = prefill_inputs(prefill["inputs_embeds"], prefill["cos"], prefill["sin"], size)
    token = int(prefill["logits"].argmax())
    step = step_inputs(token, count, count + int(prefill["rope_deltas"].reshape(-1)[0]), theta, core.head_dim)
    cache = torch.zeros(len(core.layers), 1, core.kv_heads, IOS_CONTEXT, core.head_dim)
    with torch.no_grad():
        keys, values = torch.zeros(len(core.layers), 1, core.kv_heads, count + 1, core.head_dim), \
            torch.zeros(len(core.layers), 1, core.kv_heads, count + 1, core.head_dim)
        causal = torch.full((count, count), float("-inf")).triu(1)[None, None]
        first = core.decode(*tensors(prefill, ["inputs_embeds", "cos", "sin"]), causal, keys, values).numpy()
        second = core.decode(core.embed(torch.from_numpy(step["input_ids"])), *tensors(step, ["cos", "sin"]),
                             torch.zeros(1, 1, 1, count + 1), keys, values).numpy()
    result = {"precision": args.precision, "prompt_tokens": count, "prefill_bucket": size, "step_token": token,
              "wrapper_fp32_prefill_vs_reference": logit_metrics(prefill["logits"], first)}
    started = time.perf_counter()
    example = (*tensors(feeds, PREFILL_INPUTS), cache.clone(), cache.clone())
    if args.precision == "pal4":
        core, result["palettized_modules"] = palettize(core, example)
        result["palettize_s"] = time.perf_counter() - started
    tokens = torch.export.Dim("tokens", min=2, max=IOS_PREFILL[-1])
    prefill_program = export(StaticPrefill(core).eval(), example,
                             ({1: tokens}, {1: tokens}, {1: tokens}, {2: tokens, 3: tokens}, None, None, None), half=True)
    main_program = export(StaticMain(core).eval(), (*tensors(step, MAIN_INPUTS), cache.clone(), cache.clone()), half=True)
    shapes = {f'"{size}"': {"inputs_embeds": (1, size, core.hidden), "cos": (1, size, core.head_dim),
                            "sin": (1, size, core.head_dim), "mask": (1, 1, size, size)} for size in IOS_PREFILL}
    path = args.output / "main.aimodel"
    label = {"fp16": "fp16", "pal4": f"{PALETTE_BITS}-bit k-means palettized linear weights (group {PALETTE_GROUP})"}
    save([("prefill", prefill_program, PREFILL_INPUTS, ["logits"], STATES),
          ("main", main_program, MAIN_INPUTS, ["logits"], STATES)], path,
         f"TeleOCR decoder for iOS, static prefill buckets and decode with a fixed {IOS_CONTEXT}-token FP16 KV state, "
         f"{label[args.precision]}", static={"prefill": shapes})
    result["export_s"] = time.perf_counter() - started
    del prefill_program, main_program, core
    gc.collect()
    runtime = Runtime(path, [f"prefill_{size}", "main"])
    result["load_s"] = runtime.load_seconds
    state = {name: NDArray(np.zeros(cache.shape, np.float16)) for name in STATES}
    logits, result["prefill_s"] = runtime(half(feeds), "logits", f"prefill_{size}", state)
    result["prefill_vs_reference"] = logit_metrics(prefill["logits"], logits)
    result["prefill_vs_wrapper_fp32"] = logit_metrics(first, logits)
    logits, result["step_s"] = runtime(half(step), "logits", "main", state)
    result["step_vs_wrapper_fp32"] = logit_metrics(second, logits)
    runtime.close()
    checks = [result[key] for key in ("prefill_vs_reference", "prefill_vs_wrapper_fp32", "step_vs_wrapper_fp32")]
    comparisons = [decoder_comparison_passes(check) for check in checks]
    result["gate"] = {"fp16_nrmse_max": DECODER_NRMSE_LIMIT, "fp16_cosine_min": DECODER_COSINE_MIN,
                      "fp16_argmax": "match or the same top-two candidates within the bounded maximum error",
                      "pal4": "finite outputs; differences from the uncompressed FP32 reference are recorded only"}
    passed = all(check["finite"] for check in checks)
    if args.precision == "fp16":
        passed = passed and all(comparisons)
    return passed, result, "main.aimodel"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--kind", choices=["vision", "embed", "decoder"], required=True)
    parser.add_argument("--precision", choices=["fp16", "pal4"], default="fp16")
    parser.add_argument("--dynamic", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.kind == "vision" and args.dynamic is None:
        parser.error("--dynamic is required for vision")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    started = time.perf_counter()
    passed, result, asset = {"vision": vision, "embed": embed, "decoder": decoder}[args.kind](args)
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "coreai-torch", "coreai-core", "coreai-opt")}
    write_json(args.output / "comparison.json", {"kind": args.kind, "platform": "ios", "passed": bool(passed), **result,
                                                 "seconds": time.perf_counter() - started,
                                                 "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                                 "versions": versions,
                                                 "code_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                                                 ("ios.py", "convert.py", "wrappers.py", "contracts.py",
                                                                  "quantized.py", "quantization.py")},
                                                 "asset": asset})
    if not passed:
        raise SystemExit(f"The iOS {args.kind} comparison failed")


if __name__ == "__main__":
    main()
