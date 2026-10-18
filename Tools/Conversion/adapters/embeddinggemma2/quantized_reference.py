import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from transformers import AutoConfig, EmbeddingGemma2Model

from .contracts import compare_vectors
from .convert import load_inputs
from .model_wrappers import TextTower, VisionTower, use_materialized_key_values, use_scaled_norms
from .quantization import TABLE_SHAPE, dequantize_table, prepare_quantization, quantize_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    specification = json.loads((args.reference / "fixtures.json").read_text())
    if not json.loads((args.reference / "reference-summary.json").read_text())["complete"]:
        raise ValueError("Incomplete original reference")
    torch.set_num_threads(4)
    use_materialized_key_values()
    table = np.load(args.reference / "token_embeddings.npy", mmap_mode="r", allow_pickle=False)
    if table.shape != TABLE_SHAPE or table.dtype != np.float32 or np.any(table.view(np.uint32) & 0xFFFF):
        raise ValueError("The source table must be the exact BF16 expansion")
    packed, scales = quantize_table(table)
    vocabulary = dequantize_table(packed, scales)
    shutil.copytree(args.reference, args.output, ignore=shutil.ignore_patterns("token_embeddings.npy"))
    packed.tofile(args.output / "token_embeddings.int4")
    scales.tofile(args.output / "token_embedding_scales.f32")
    config = AutoConfig.from_pretrained(args.source, local_files_only=True)
    config.audio_config = None
    model = EmbeddingGemma2Model.from_pretrained(args.source, config=config, dtype=torch.float32,
                                                attn_implementation="eager", local_files_only=True).eval()
    text = TextTower(model, safe_normalize=True).eval().requires_grad_(False)
    vision = VisionTower(model).eval().requires_grad_(False)
    use_scaled_norms(text)
    use_scaled_norms(vision)
    first = args.reference / specification["fixtures"][0]["id"]
    first_image = args.reference / next(f["id"] for f in specification["fixtures"] if "image" in f)
    _, text, text_layers = prepare_quantization(text, "text", load_inputs(first, "text")[1])
    _, vision, vision_layers = prepare_quantization(vision, "vision", load_inputs(first_image, "vision")[1])
    records = []
    vectors = []
    originals = []
    with torch.inference_mode():
        for fixture in specification["fixtures"]:
            folder = args.output / fixture["id"]
            ids = np.load(folder / "input_ids.npy", allow_pickle=False)
            mask = torch.from_numpy(np.load(folder / "attention_mask.npy", allow_pickle=False)).float()
            inputs = torch.from_numpy(vocabulary[ids]) * np.sqrt(np.float32(512))
            if "image" in fixture:
                soft = vision(*load_inputs(folder, "vision")[1])
                valid = np.load(folder / "valid_soft_tokens.npy", allow_pickle=False)
                features = soft[0, valid]
                inputs[torch.from_numpy(ids == config.image_token_id)] = features
                np.save(folder / "soft_tokens.npy", features.numpy())
            vector = text(inputs, mask).numpy()
            original = np.load(args.reference / fixture["id"] / "reference.npy", allow_pickle=False)
            result = compare_vectors(original, vector, **specification["low_precision_gate"])
            if not np.isfinite(vector).all() or abs(float(np.linalg.norm(vector)) - 1) > 0.001:
                raise ValueError("Quantization produced an invalid embedding")
            records.append({"id": fixture["id"], **result})
            vectors.append(vector.reshape(-1))
            originals.append(original.reshape(-1))
            np.save(folder / "inputs_embeds.npy", inputs.numpy())
            np.save(folder / "reference.npy", vector)
    specification["precision"] = "int4_int8"
    specification["original_fixtures_sha256"] = hashlib.sha256((args.reference / "fixtures.json").read_bytes()).hexdigest()
    (args.output / "fixtures.json").write_text(json.dumps(specification, ensure_ascii=False, indent=2) + "\n")
    images = [i for i, f in enumerate(specification["fixtures"]) if "image" in f]
    texts = [i for i, f in enumerate(specification["fixtures"]) if "image" not in f]
    rankings = []
    for index in texts:
        before = np.asarray(originals)[images] @ originals[index]
        after = np.asarray(vectors)[images] @ vectors[index]
        rankings.append({"id": specification["fixtures"][index]["id"],
                         "image_ids": [specification["fixtures"][i]["id"] for i in images],
                         "original_scores": before.tolist(), "quantized_scores": after.tolist(),
                         "top1_agrees": int(before.argmax()) == int(after.argmax())})
    quality = {"scope": "PTQ degradation against original precision, not Core AI conversion fidelity or QAT equivalence.",
               "cases": records, "rankings": rankings, "layers": {"text": text_layers, "vision": vision_layers},
               "weight_scheme": "symmetric_with_clipping; linear/conv axis 0; vision position table axis 2",
               "retained_parameters": {
                   kind: [{"name": name, "shape": list(parameter.shape)} for name, parameter in tower.named_parameters()
                          if ".parametrizations." not in name]
                   for kind, tower in (("text", text), ("vision", vision))},
               "table_scheme": "signed INT4 low-nibble-first per vocabulary row, FP32 scale, range [-7, 7]",
               "seconds": time.perf_counter() - started}
    (args.output / "quantization-quality.json").write_text(json.dumps(quality, indent=2, allow_nan=False) + "\n")
    (args.output / "reference-summary.json").write_text(json.dumps({"records": records, "complete": True, "precision": "int4_int8"}, indent=2) + "\n")
    print(json.dumps({"complete": True, "seconds": quality["seconds"], "top1_agreement": sum(r["top1_agrees"] for r in rankings)}))


if __name__ == "__main__":
    main()
