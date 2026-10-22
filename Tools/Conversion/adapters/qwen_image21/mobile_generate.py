from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps

from .evaluation import SIGMAS, canonical_hash
from .measurement import PhaseRecorder, write_metrics
from .mobile_models import DTYPES
from .mobile_profile import validate_mobile_profile
from .mobile_runtime import CandidateComponent
from .preprocessing import denoiser_inputs, denoiser_layout, text_inputs


def array(value):
    return value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value, dtype=np.float32)


def vae_image(image):
    if image.size != (512, 512):
        raise ValueError("Mobile evaluation requires a frozen 512-square reference image")
    pixels = np.asarray(image.convert("RGBA"), dtype=np.float32) / np.float32(255) * np.float32(2) - np.float32(1)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None, :, None])


def scheduler(snapshot):
    result = FlowMatchEulerDiscreteScheduler.from_pretrained(Path(snapshot) / "scheduler", local_files_only=True)
    shift = calculate_shift(1024, result.config.get("base_image_seq_len", 256), result.config.get("max_image_seq_len", 4096),
                            result.config.get("base_shift", 0.5), result.config.get("max_shift", 1.15))
    timesteps, count = retrieve_timesteps(result, 8, torch.device("cpu"), sigmas=list(SIGMAS), mu=shift)
    if count != 8:
        raise ValueError("The source scheduler must retain all eight steps")
    result.set_begin_index(0)
    return result, timesteps


def generate(paths, profile, snapshot, output, encoded, metadata, noise, image=None, *, component_type=CandidateComponent):
    validate_mobile_profile(profile)
    if set(paths) != set(profile["components"]):
        raise ValueError("A mobile rollout requires all five component bindings")
    if noise.shape != (1, 1024, 64) or noise.dtype != np.float32 or not np.isfinite(noise).all():
        raise ValueError("Mobile initial noise must be the finite frozen Float32 tensor")
    if (image is not None) != (metadata["operation"] == "edit"):
        raise ValueError("The reference image does not match the frozen operation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    recorder = PhaseRecorder()
    metrics = {"profile_sha256": canonical_hash(profile), "complete": False, "components": {},
               "scope": "Captured mobile candidate rollout; excludes clean latency and device acceptance"}
    failure = None
    started = time.perf_counter()

    def save(name, value):
        values = array(value)
        np.save(output / (name + ".npy"), values, allow_pickle=False)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite Core AI rollout value: " + name)

    @contextmanager
    def opened(kind):
        model = None
        try:
            with recorder.phase(kind + ".load-including-specialization"):
                model = component_type(paths[kind], profile, kind)
            yield model
        finally:
            if model is not None:
                metrics["components"][kind] = {"load_seconds": model.load_seconds, "call_seconds": list(model.calls)}
                with recorder.phase(kind + ".release"):
                    model.close()

    def call(kind, model, values):
        with recorder.phase(kind):
            return model.call(values)

    try:
        dit_dtype = DTYPES[profile["components"]["denoiser"]["compute_dtype"]]
        vae_dtype = DTYPES[profile["components"]["vae_encoder"]["compute_dtype"]]
        features = None
        if image is not None:
            with opened("vision") as model:
                features = call("vision", model, {"pixel_values": encoded["pixel_values"].astype(np.float32)})
            for name, value in zip(("merged", "deepstack_0", "deepstack_1", "deepstack_2"), features, strict=True):
                save("vision-" + name, value)
        with opened("text") as model:
            hidden, = call("text", model, text_inputs(snapshot, encoded, features))
        length = encoded["input_ids"].shape[1]
        save("language-reference", hidden[:, :length])
        text = hidden[:, metadata["drop"]:length].copy()
        slots = (encoded["input_ids"][0] == metadata["image_token_id"])[metadata["drop"]:]
        structure = denoiser_layout(slots)
        configuration = json.loads((Path(snapshot) / "vae/config.json").read_text())
        mean = torch.tensor(configuration["latents_mean"]).reshape(1, 64, 1, 1, 1).to(vae_dtype)
        std = torch.tensor(configuration["latents_std"]).reshape(1, 64, 1, 1, 1).to(vae_dtype)
        condition = None
        if image is not None:
            value = vae_image(image)
            save("vae-encode-input", torch.from_numpy(value).to(vae_dtype))
            with opened("vae_encoder") as model:
                encoded_image, = call("vae_encoder", model, {"value": value})
            save("vae-encode-output", encoded_image)
            normalized = (torch.from_numpy(encoded_image).to(vae_dtype) - mean) / std
            condition = normalized.reshape(1, 64, 1024).transpose(1, 2).contiguous().to(dit_dtype)
            save("condition-latents", condition)
        euler, timesteps = scheduler(snapshot)
        latents = torch.from_numpy(noise.copy()).to(dit_dtype)
        save("initial-latents", latents)
        with opened("denoiser") as model:
            for index, timestep in enumerate(timesteps):
                packed = latents if condition is None else torch.cat((condition, latents), dim=1)
                model_time = timestep.expand(1).to(dit_dtype) / 1000
                values = denoiser_inputs(array(packed), text, model_time.float().item(), structure)
                np.savez(output / f"denoiser-{index}-inputs.npz", **values)
                predicted, = call(f"denoiser.step-{index}", model, values)
                save("joint-noise-" + str(index), predicted)
                noise_tensor = torch.from_numpy(predicted[:, structure["target_indices"]].copy()).to(dit_dtype)
                save("step-latent-input-" + str(index), latents)
                save("step-noise-input-" + str(index), noise_tensor)
                latents = euler.step(noise_tensor, timestep, latents, return_dict=False)[0].to(dit_dtype)
                save("latent-" + str(index), latents)
        decoder_input = latents.transpose(1, 2).reshape(1, 64, 1, 32, 32).to(vae_dtype) * std + mean
        save("vae-decode-input", decoder_input)
        with opened("vae_decoder") as model:
            rgba, = call("vae_decoder", model, {"value": np.ascontiguousarray(array(decoder_input))})
        save("rgba", rgba)
        pixels = np.clip(rgba[0, :, 0].transpose(1, 2, 0) / 2 + 0.5, 0, 1)
        Image.fromarray((pixels * 255).round().astype(np.uint8), "RGBA").save(output / "output.png")
        metrics.update(complete=True, status="completed", steps=8, seconds=time.perf_counter() - started)
        return array(latents), rgba
    except BaseException as error:
        failure = error
        metrics.update(status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed", error_type=type(error).__name__,
                       error=str(error))
        raise
    finally:
        metrics["measurement"] = recorder.report()
        write_metrics(output / "metrics.json", metrics, preserving=failure)
