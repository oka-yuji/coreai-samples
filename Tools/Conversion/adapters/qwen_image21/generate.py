import argparse
import os
from pathlib import Path
import signal
import time

import numpy as np
from PIL import Image
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps

from .contracts import KINDS, PROFILE, SIGMAS, read_json, validate_bundle, write_json
from .preprocessing import denoiser_inputs, denoiser_layout, initial_noise, prompt_inputs, text_inputs
from .runtime import Component


def generate(components, config, output, prompt, image=None, noise=None, seed=20261018):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Generation requires a new output directory")
    if set(components) != set(KINDS):
        raise ValueError("All five independently verified components are required")
    encoded, drop, image_id = prompt_inputs(config, prompt, image)
    noise = initial_noise(seed) if noise is None else np.asarray(noise)
    if noise.shape != (1, 1024, 64) or noise.dtype != np.float32 or not np.isfinite(noise).all():
        raise ValueError("Initial noise must be finite float32 [1, 1024, 64]")
    np.save(output / "initial-noise.npy", noise)
    (output / "prompt.txt").write_text(prompt + "\n")
    metrics = {"profile": PROFILE, "seed": seed, "diagnostics": "tensor capture enabled", "components": {}}

    def record(kind, component):
        metrics["components"][kind] = {"load_seconds": component.load_seconds, "call_seconds": component.calls,
                                       "first_call_after_load": 0, "subsequent_calls": list(range(1, len(component.calls)))}
        write_json(output / "metrics.json", metrics)

    features = None
    if image is not None:
        with Component(components["vision"]) as component:
            features = component.call({"pixel_values": encoded["pixel_values"].astype(np.float32)})
            record("vision", component)
        np.savez(output / "vision.npz", **{f"value_{index}": value for index, value in enumerate(features)})
    values = text_inputs(config, encoded, features)
    with Component(components["text"]) as component:
        hidden, = component.call(values)
        record("text", component)
    length = encoded["input_ids"].shape[1]
    text = hidden[:, drop:length].copy()
    slots = (encoded["input_ids"][0] == image_id)[drop:]
    structure = denoiser_layout(slots)
    np.save(output / "prompt-embeddings.npy", text)
    vae_config = read_json(Path(config) / "vae/config.json")
    mean = np.asarray(vae_config["latents_mean"], dtype=np.float32).reshape(1, 64, 1, 1, 1)
    std = np.asarray(vae_config["latents_std"], dtype=np.float32).reshape(1, 64, 1, 1, 1)
    condition = None
    if image is not None:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / np.float32(255) * np.float32(2) - np.float32(1)
        value = np.ascontiguousarray(rgba.transpose(2, 0, 1)[None, :, None])
        with Component(components["vae_encoder"]) as component:
            encoded_image, = component.call({"value": value})
            record("vae_encoder", component)
        normalized = (encoded_image - mean) / std
        condition = normalized.reshape(1, 64, 1024).transpose(0, 2, 1).copy()
        np.save(output / "condition-latents.npy", condition)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(Path(config) / "scheduler", local_files_only=True)
    shift = calculate_shift(1024, scheduler.config.get("base_image_seq_len", 256),
                            scheduler.config.get("max_image_seq_len", 4096),
                            scheduler.config.get("base_shift", 0.5), scheduler.config.get("max_shift", 1.15))
    timesteps, steps = retrieve_timesteps(scheduler, 8, torch.device("cpu"), sigmas=list(SIGMAS), mu=shift)
    if steps != 8:
        raise ValueError("The fixed sampling schedule must contain eight steps")
    scheduler.set_begin_index(0)
    latents = torch.from_numpy(noise.copy())
    with Component(components["denoiser"]) as component:
        for index, timestep in enumerate(timesteps):
            packed = latents.numpy()
            if condition is not None:
                packed = np.concatenate((condition, packed), axis=1)
            values = denoiser_inputs(packed, text, float(timestep / 1000), structure)
            result, = component.call(values)
            predicted = np.ascontiguousarray(result[:, structure["target_indices"]])
            np.save(output / f"noise-{index}.npy", predicted)
            latents = scheduler.step(torch.from_numpy(predicted), timestep, latents, return_dict=False)[0]
            if not torch.isfinite(latents).all():
                raise ValueError("The scheduler returned non-finite latents")
        record("denoiser", component)
    np.save(output / "final-latents.npy", latents.numpy())
    decoder_input = latents.numpy().transpose(0, 2, 1).reshape(1, 64, 1, 32, 32) * std + mean
    with Component(components["vae_decoder"]) as component:
        rgba, = component.call({"value": np.ascontiguousarray(decoder_input)})
        record("vae_decoder", component)
    if rgba.shape != (1, 4, 1, 512, 512):
        raise ValueError("The decoder output does not match the RGBA profile")
    np.save(output / "rgba.npy", rgba)
    pixels = np.clip(rgba[0, :, 0].transpose(1, 2, 0) / 2 + 0.5, 0, 1)
    Image.fromarray((pixels * 255).round().astype(np.uint8), "RGBA").save(output / "output.png")
    metrics.update(seconds=time.perf_counter() - started, complete=True, steps=8)
    write_json(output / "metrics.json", metrics)
    return latents.numpy(), rgba


def main():
    from Tools.Conversion.storage import directory_lock
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--noise", type=Path)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--lock-dir", type=Path, default=Path(os.environ.get("COREAI_HEAVY_JOB_LOCK", Path.home() / "Library/Caches/coreai-samples/heavy-job")))
    args = parser.parse_args()
    validate_bundle(args.bundle)
    image = Image.open(args.image).convert("RGBA") if args.image else None
    noise = np.load(args.noise, allow_pickle=False) if args.noise else None
    def stop(signum, frame):
        raise KeyboardInterrupt

    previous_signal = signal.signal(signal.SIGTERM, stop)
    previous_lock = os.environ.get("COREAI_HEAVY_JOB_LOCK")
    os.environ["COREAI_HEAVY_JOB_LOCK"] = str(args.lock_dir)
    try:
        with directory_lock(args.lock_dir, {"session": "image-research", "job": "generate", "expected_minutes": 10}):
            generate({kind: args.bundle / kind for kind in KINDS}, args.bundle / "config", args.output,
                     args.prompt, image, noise, args.seed)
    finally:
        signal.signal(signal.SIGTERM, previous_signal)
        if previous_lock is None:
            os.environ.pop("COREAI_HEAVY_JOB_LOCK", None)
        else:
            os.environ["COREAI_HEAVY_JOB_LOCK"] = previous_lock


if __name__ == "__main__":
    main()
