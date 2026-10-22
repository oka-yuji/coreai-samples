import argparse
import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

from .contracts import REVISION, SIGMAS, check_profile
from .fixtures import file_record, save_json, validate_fixture_suite
from .preprocessing import denoiser_layout, prompt_inputs, text_inputs


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def run(args):
    torch.set_num_threads(2)
    if args.snapshot.name != REVISION:
        raise ValueError("Unexpected source snapshot")
    reference = json.loads((args.reference / "reference.json").read_text())
    if reference.get("complete") is not True or reference.get("precision") != "fp32" or reference.get("cache") != "off":
        raise ValueError("Saved prediction replay requires the independent FP32 cache-off reference")
    check_profile(reference.get("target_profile"))
    definition = json.loads((args.fixtures / "fixtures.json").read_text())
    validate_fixture_suite(args.fixtures, definition)
    args.output.mkdir(parents=True, exist_ok=False)
    config = Qwen3VLConfig.from_pretrained(args.snapshot / "text_encoder", local_files_only=True)
    helper = SimpleNamespace(config=config)
    helper.get_vision_position_ids = MethodType(Qwen3VLModel.get_vision_position_ids, helper)
    cases = []
    from PIL import Image
    case_ids = sorted({row["fixture_id"] for row in definition["runs"]})
    for case_id in case_ids:
        metadata = json.loads((args.fixtures / case_id / "input.json").read_text())
        image = Image.open(args.fixtures / metadata["reference"]).convert("RGBA") if metadata["reference"] else None
        cases.append((case_id, metadata["prompt"], image))
    cases.append(("wide-token", " " * 50_000 + "red", None))
    for reference in (False, True):
        image = Image.new("RGBA", (512, 512), "white") if reference else None
        first, _, _ = prompt_inputs(args.snapshot, "red", image)
        prompt = "red" + " red" * (512 - first["input_ids"].shape[1])
        encoded, _, _ = prompt_inputs(args.snapshot, prompt, image)
        if encoded["input_ids"].shape[1] != 512:
            raise ValueError("Could not construct the exact source tokenizer boundary")
        cases.append(("limit-edit" if reference else "limit-generate", prompt, image))
        cases.append(("over-edit" if reference else "over-generate", prompt + " red", image))
    records = []
    for name, prompt, image in cases:
        if name.startswith("over-"):
            try:
                prompt_inputs(args.snapshot, prompt, image)
            except ValueError:
                records.append({"id": name, "prompt": prompt, "hasReference": image is not None, "overflow": True})
                continue
            raise ValueError("The overflow fixture did not overflow")
        encoded, drop, image_id = prompt_inputs(args.snapshot, prompt, image)
        ids = encoded["input_ids"]
        positions, _ = Qwen3VLModel.get_rope_index(
            helper, torch.from_numpy(ids), torch.from_numpy(encoded.get("mm_token_type_ids", np.zeros_like(ids))),
            image_grid_thw=torch.from_numpy(encoded["image_grid_thw"]) if image else None,
            attention_mask=torch.from_numpy(encoded["attention_mask"]))
        visual = (ids[0] == image_id)[drop:]
        structure = denoiser_layout(visual)
        text = text_inputs(args.snapshot, encoded, [np.zeros((256, 4096), np.float32)] * 4 if image else None)
        record = {"id": name, "prompt": prompt, "hasReference": image is not None, "overflow": False,
                  "tokens": ids[0].tolist(), "drop": drop, "positions": positions[:, 0].tolist(),
                  "expanded": structure["length"], "textMask": array_hash(text["mask"]),
                  "layoutHashes": {key: array_hash(structure[key]) for key in
                                   ("text_index", "image_index", "image_mask", "mask", "target")}}
        if name == "e06":
            features = [np.asarray((np.arange(256 * 4096).reshape(256, 4096) % 1024) + (tap + 1) * 1000,
                                   dtype=np.float32) for tap in range(4)]
            conditioning = text_inputs(args.snapshot, encoded, features)
            hidden = (np.arange(512 * 4096, dtype=np.float32) % 997).reshape(1, 512, 4096)
            compact = hidden[:, drop:ids.shape[1]]
            padded = np.zeros((1, 512, 4096), np.float32)
            padded[:, :compact.shape[1]] = compact
            record["conditioning"] = {"vision": array_hash(conditioning["vision"]),
                                      "deepstack": array_hash(conditioning["deepstack"]),
                                      "text": array_hash(padded)}
        if name in ("g02", "e06"):
            record["rotary"] = {}
            for label, values in (("textCosine", text["cosine"]), ("textSine", text["sine"]),
                                  ("denoiserCosine", structure["cosine"]), ("denoiserSine", structure["sine"])):
                path = args.output / f"{name}-{label}.f32"
                values.astype("<f4").tofile(path)
                record["rotary"][label] = path.name
        records.append(record)
    initial = np.array([0.12345, -1.23456, 3.14159, -0.999, 0, 100, -60, 0.0003] * 2, dtype=np.float32)
    prediction = np.array([1.111, -2.222, 0.3333, -0.4444, 0.5555, -0.6666, 7.777, -8.888] * 2, dtype=np.float32)
    steps = []
    for name, dtype in (("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)):
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.snapshot / "scheduler", local_files_only=True)
        scheduler.set_timesteps(sigmas=list(SIGMAS))
        scheduler.set_begin_index(0)
        latent = torch.from_numpy(initial).to(dtype)
        predicted = torch.from_numpy(prediction).to(dtype)
        outputs = []
        for timestep in scheduler.timesteps:
            latent = scheduler.step(predicted, timestep, latent, return_dict=False)[0]
            outputs.append(latent.float().tolist())
        steps.append({"precision": name, "initial": torch.from_numpy(initial).to(dtype).float().tolist(),
                      "prediction": predicted.float().tolist(), "outputs": outputs,
                      "modelTimesteps": [(value.to(dtype) / 1000).float().item() for value in scheduler.timesteps]})
    vae = json.loads((args.snapshot / "vae/config.json").read_text())
    source_noise = np.load(args.fixtures / "noise/20261018.npy", allow_pickle=False)
    if not np.array_equal(source_noise, np.load(args.reference / "generate/initial-noise.npy", allow_pickle=False)):
        raise ValueError("The source reference and frozen suite use different noise")
    source_noise.astype("<f4").tofile(args.output / "initial-noise.f32")
    predictions = []
    for index in range(8):
        prediction = np.load(args.reference / "generate" / f"noise-{index}.npy", allow_pickle=False)
        if prediction.ndim != 3 or prediction.shape[0] != 1 or prediction.shape[2] != 64 \
                or not 1024 <= prediction.shape[1] <= 2304 or prediction.dtype != np.float32 or not np.isfinite(prediction).all():
            raise ValueError("Unexpected source prediction")
        prediction = np.ascontiguousarray(prediction[:, -1024:])
        name = f"prediction-{index}.f32"
        prediction.astype("<f4").tofile(args.output / name)
        predictions.append((name, prediction))
    replay = []
    for name, dtype in (("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)):
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.snapshot / "scheduler", local_files_only=True)
        scheduler.set_timesteps(sigmas=list(SIGMAS))
        scheduler.set_begin_index(0)
        latent = torch.from_numpy(source_noise).to(dtype)
        hashes = []
        for timestep, (_, prediction) in zip(scheduler.timesteps, predictions, strict=True):
            latent = scheduler.step(torch.from_numpy(prediction).to(dtype), timestep, latent, return_dict=False)[0]
            hashes.append(array_hash(latent.float().numpy()))
        replay.append({"precision": name, "hashes": hashes})
    save_json(args.output / "native-golden.json", {
        "format": 1, "sourceRevision": definition["source_revision"],
        "fixtureManifestSHA256": file_record(args.fixtures / "fixtures.json")["sha256"],
        "cases": records, "scheduler": steps, "rotaryAbsoluteTolerance": 0.00005,
        "scaling": {"mean": vae["latents_mean"], "standardDeviation": vae["latents_std"]},
        "savedNoise": {"file": "initial-noise.f32", "sourceNPYSHA256": file_record(args.fixtures / "noise/20261018.npy")["sha256"],
                       "predictions": [name for name, _ in predictions], "replay": replay,
                       "scope": "Replay saved FP32 source predictions at each host dtype; not half-precision network quality"},
        "scope": "Original CPU tokenizer/positions/rotary/scheduler; no learned network execution"})
    print(json.dumps({"cases": len(records), "scheduler_precisions": len(steps), "output": str(args.output)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
