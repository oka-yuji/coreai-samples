import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoConfig, EmbeddingGemma2Model

from .contracts import compare_vectors
from .model_wrappers import TextTower, VisionTower, use_scaled_norms, use_materialized_key_values


def load_inputs(root, kind, dtype=torch.float32):
    names = ["inputs_embeds", "attention_mask"] if kind == "text" else ["pixel_values", "image_position_ids", "pooling_weights"]
    result = []
    for name in names:
        value = torch.from_numpy(np.load(root / f"{name}.npy", allow_pickle=False))
        result.append(value.to(torch.int32 if name == "image_position_ids" else dtype))
    return names, tuple(result)


def tensor_metrics(expected, actual):
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(actual)):
        return {"finite": False, "max_abs": None, "cosine": None}
    denominator = np.linalg.norm(expected) * np.linalg.norm(actual)
    return {"finite": True, "max_abs": float(np.abs(expected - actual).max()),
            "cosine": float(np.dot(expected, actual) / denominator) if denominator else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=["text", "vision"], required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16_weights", "fp16_all", "fp16_safe"], required=True)
    parser.add_argument("--kv-layout", choices=["native", "materialized"], default="materialized")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    if args.kv_layout == "materialized":
        use_materialized_key_values()
    specification = json.loads((args.reference / "fixtures.json").read_text())
    if not json.loads((args.reference / "reference-summary.json").read_text())["complete"]:
        raise ValueError("The reference run is incomplete")
    config = AutoConfig.from_pretrained(args.source, local_files_only=True)
    config.audio_config = None
    if args.kind == "text":
        config.vision_config = None
    model = EmbeddingGemma2Model.from_pretrained(args.source, config=config, dtype=torch.float32,
                                                attn_implementation="eager", local_files_only=True).eval()
    module = TextTower(model, safe_normalize=args.precision == "fp16_safe") if args.kind == "text" else VisionTower(model)
    module.eval().requires_grad_(False)
    if args.precision == "fp16_safe":
        print("Scaled RMSNorms", len(use_scaled_norms(module)), flush=True)
    dtype = torch.float32
    if args.precision == "fp16_weights":
        module.half()
        dtype = torch.float16
    fixtures = [item for item in specification["fixtures"] if args.kind == "text" or "image" in item]
    records = []
    with torch.inference_mode():
        for item in fixtures:
            root = args.reference / item["id"]
            names, inputs = load_inputs(root, args.kind, dtype)
            actual = module(*inputs).float().numpy()
            np.save(args.output / f"{item['id']}-wrapper.npy", actual)
            if args.kind == "text":
                expected = np.load(root / "reference.npy", allow_pickle=False)
                gate = specification["low_precision_gate"] if args.precision == "fp16_weights" else specification["fp32_gate"]
                result = compare_vectors(expected, actual, **gate)
                records.append({"id": item["id"], "wrapper": result})
            else:
                expected = np.load(root / "soft_tokens.npy", allow_pickle=False)
                valid = np.load(root / "valid_soft_tokens.npy", allow_pickle=False)
                result = tensor_metrics(expected, actual[0, valid])
                records.append({"id": item["id"], "wrapper": result})
            print(json.dumps(records[-1], allow_nan=False), flush=True)
    (args.output / "wrapper-comparison.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    if args.precision in ("fp32", "fp16_all", "fp16_safe") and args.kind == "text":
        if not all(record["wrapper"]["pass"] for record in records):
            raise RuntimeError("The FP32 wrapper failed the pre-conversion fidelity gate")
    import coreai_torch
    from coreai_opt.casting import cast_fp32_to_fp16
    from coreai.runtime import AIModelAssetMetadata
    names, example = load_inputs(args.reference / fixtures[0]["id"], args.kind, dtype)
    started = time.perf_counter()
    with torch.no_grad():
        exported = torch.export.export(module, example, strict=False)
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    if args.precision in ("fp16_all", "fp16_safe"):
        cast_fp32_to_fp16(exported)
        example = tuple(value.to(torch.float16) if value.is_floating_point() else value for value in example)
    print("Exported operations", len(list(exported.graph.nodes)), flush=True)
    with torch.no_grad():
        sample = exported.module()(*example)
    np.save(args.output / "exported-program-sample.npy", sample.detach().float().numpy())
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(exported, input_names=names,
                                   output_names=["embedding" if args.kind == "text" else "soft_tokens"],
                                   state_names=[], entrypoint_name="main")
    program = converter.to_coreai()
    metadata = AIModelAssetMetadata()
    metadata.author = "Google DeepMind; converted by Yuji Oka"
    metadata.license = "Apache-2.0"
    metadata.model_description = f"EmbeddingGemma 2 {args.kind}, {args.precision}, fixed research profile"
    metadata.set_custom("source_revision", "914f7f89142e33e77833254d9c9b90c3cef7303b")
    metadata.set_custom("precision_profile", args.precision)
    package = args.output / f"{args.kind}.aimodel"
    if package.exists():
        raise FileExistsError(package)
    program.save_asset(package, metadata)
    contract = {"kind": args.kind, "precision": args.precision, "entry": "main", "kv_layout": args.kv_layout,
                "inputs": [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch.")}
                           for name, value in zip(names, example, strict=True)],
                "output": "embedding" if args.kind == "text" else "soft_tokens",
                "reference": str(args.reference), "seconds": time.perf_counter() - started,
                "fixtures_sha256": hashlib.sha256((args.reference / "fixtures.json").read_bytes()).hexdigest(),
                "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ["convert.py", "model_wrappers.py"]},
                "model_revision": args.source.name, "complete": True}
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps(contract), flush=True)


if __name__ == "__main__":
    main()
