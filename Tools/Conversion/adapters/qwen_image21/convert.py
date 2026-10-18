import argparse
import asyncio
import gc
import importlib.metadata
import os
from pathlib import Path
import time

import numpy as np
import torch

from .contracts import KINDS, PROFILE, digest, require_heavy_job, tensor_metrics, write_json
from .examples import comparable, load_examples


def load_wrapper(snapshot, kind):
    from diffusers import AutoencoderKLQwenImage21, QwenImage21Transformer2DModel
    from .denoiser import Denoiser, VAEPart
    from .encoders import TextEncoder, VisionEncoder, load_encoder

    if kind in ("text", "vision"):
        original = load_encoder(snapshot, kind)
        wrapper = TextEncoder(original) if kind == "text" else VisionEncoder(original)
    elif kind == "denoiser":
        original = QwenImage21Transformer2DModel.from_pretrained(snapshot / "transformer", torch_dtype=torch.float32, local_files_only=True)
        wrapper = Denoiser(original)
    else:
        original = AutoencoderKLQwenImage21.from_pretrained(snapshot / "vae", torch_dtype=torch.float32, local_files_only=True)
        wrapper = VAEPart(original, kind)
    return wrapper.eval().requires_grad_(False)


async def convert(args):
    require_heavy_job()
    from coreai.runtime import AIModel, AIModelAssetMetadata, ComputeUnitKind, NDArray, SpecializationOptions
    from coreai_torch import TorchConverter, get_decomp_table

    if os.environ.get("USE_LOCAL_COREAI") or not SpecializationOptions.is_supported():
        raise RuntimeError("Validation requires the macOS Core AI runtime")
    args.output.mkdir(parents=True, exist_ok=True)
    package = args.output / f"{args.kind}.aimodel"
    if package.exists() or (args.output / "component.json").exists():
        raise FileExistsError("Component outputs must be written to a new directory")
    torch.set_num_threads(10)
    examples = load_examples(args.kind, args.reference, args.snapshot)
    module = load_wrapper(args.snapshot, args.kind)
    rows = []
    for example in examples:
        values = tuple(torch.from_numpy(np.ascontiguousarray(value)) for value in example["values"].values())
        with torch.no_grad():
            result = module(*values)
        result = result if isinstance(result, tuple) else (result,)
        comparisons = [tensor_metrics(expected, comparable(args.kind, actual.detach().numpy(), expected))
                       for expected, actual in zip(example["expected"], result, strict=True)]
        rows.append({"case": example["case"], "wrapper": comparisons})
        print(rows[-1], flush=True)
    write_json(args.output / "wrapper-comparison.json", rows)
    if not all(value["passed"] for row in rows for value in row["wrapper"]):
        raise RuntimeError("The wrapper does not satisfy the fixed fidelity gate")
    example = examples[0]
    values = tuple(torch.from_numpy(np.ascontiguousarray(value)) for value in example["values"].values())
    started = time.perf_counter()
    with torch.no_grad():
        program = torch.export.export(module, values, strict=False).run_decompositions(get_decomp_table())
    converter = TorchConverter(mode=TorchConverter.Mode.RELEASE)
    converter.add_exported_program(program, input_names=list(example["values"]),
                                   output_names=list(example["output_names"]), state_names=[], entrypoint_name="main")
    asset = converter.to_coreai()
    metadata = AIModelAssetMetadata()
    metadata.author = "Qwen Team, Alibaba Group; conversion by Yuji Oka"
    metadata.license = "Qwen Research License"
    metadata.model_description = f"Image Research 512: {args.kind}; converted from Qwen-Image-2.1-Turbo for non-commercial research and evaluation"
    metadata.set_custom("source_revision", PROFILE["revision"])
    asset.save_asset(package, metadata)
    export_seconds = time.perf_counter() - started
    del module, program, converter, asset, result
    gc.collect()
    started = time.perf_counter()
    options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
    model = await AIModel.load(package, specialization_options=options)
    function = model.load_function("main")
    load_seconds = time.perf_counter() - started
    if function.desc.state_names:
        raise RuntimeError("The research profile requires stateless components")
    for example, row in zip(examples, rows, strict=True):
        inputs = {name: NDArray(np.ascontiguousarray(value)) for name, value in example["values"].items()}
        started = time.perf_counter()
        result = await function(inputs=inputs, state={})
        row["seconds"] = time.perf_counter() - started
        row["coreai"] = [tensor_metrics(expected, comparable(args.kind, result[name].numpy(), expected))
                         for name, expected in zip(example["output_names"], example["expected"], strict=True)]
        print(row, flush=True)
    passed = all(value["passed"] for row in rows for value in row["wrapper"] + row["coreai"])
    write_json(args.output / "comparison.json", {"passed": passed, "rows": rows})
    if not passed:
        raise RuntimeError("The Core AI component does not satisfy the fixed fidelity gate")
    write_json(args.output / "component.json", {"kind": args.kind, "profile": PROFILE, "asset": package.name,
               "inputs": [{"name": name, "shape": list(value.shape), "dtype": value.dtype.name}
                          for name, value in examples[0]["values"].items()],
               "output_names": list(examples[0]["output_names"]), "passed": True,
               "export_code_sha256": {name: digest(Path(__file__).with_name(name))
                                      for name in ("convert.py", "contracts.py", "examples.py", "preprocessing.py", "denoiser.py", "encoders.py")},
               "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "diffusers", "coreai-torch", "coreai-core")},
               "export_seconds": export_seconds, "load_seconds": load_seconds,
               "comparison_conditions": "FP32; fixed source inputs; diagnostic comparison enabled; first call includes specialization costs not paid by load"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=KINDS, required=True)
    args = parser.parse_args()
    asyncio.run(convert(args))


if __name__ == "__main__":
    main()
