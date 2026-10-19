import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import psutil
import torch
from diffusers import QwenImage21Pipeline



from .contracts import PROFILE, require_heavy_job, write_json


def run_reference(args):
    require_heavy_job()
    out = args.output
    snapshot = args.snapshot
    out.mkdir(parents=True, exist_ok=True)
    if (out / "identity.json").exists():
        raise FileExistsError("Reference outputs must use a new directory")
    device = args.device
    dtype = torch.float32 if args.precision == "fp32" else torch.bfloat16
    cache_enabled = args.cache == "on"
    torch.set_num_threads(10)
    records = []
    def record(event, **kwargs):
        row = {"event": event, "time": time.time(), "rss_bytes": psutil.Process().memory_info().rss, **kwargs}
        if device == "mps":
            row.update(mps_allocated_bytes=torch.mps.current_allocated_memory(), mps_driver_allocated_bytes=torch.mps.driver_allocated_memory())
        records.append(row)
        (out / "events.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    identity = {"source": "Qwen/Qwen-Image-2.1-Turbo", "revision": snapshot.name, "diffusers_commit": "b800f828f3951c4d206e154411bcbfa693cd322e", "device": device, "dtype": str(dtype), "system": platform.platform(), "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "diffusers", "coreai-torch", "coreai-core", "safetensors")}, "diagnostics": "input/output capture enabled; not a clean benchmark", "resolution": 512, "output_resolution": 512, "seed": 20261018, "use_kv_cache": cache_enabled, "cfg": 1.0}
    (out / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    record("load-start")
    pipeline = QwenImage21Pipeline.from_pretrained(snapshot, dtype=dtype, local_files_only=True).to(device)
    record("loaded", sigmas=list(pipeline.config.sample_sigmas))
    pipeline.set_progress_bar_config(disable=True)
    fixture = Image.new("RGBA", (512, 512), (255, 255, 255, 255))
    ImageDraw.Draw(fixture).ellipse((128, 128, 384, 384), fill=(220, 35, 35, 255))
    fixture.save(out / "edit-input.png")
    cases = [("generate", "A blue ceramic teapot on a white table, soft daylight, simple clean product photograph.", None), ("edit", "Change the red circle to blue. Keep its shape and the white background unchanged.", fixture)]
    original_forward = pipeline.transformer.forward
    original_encode = pipeline.vae.encode
    original_decode = pipeline.vae.decode
    active = {"case": "", "step": 0}
    def dump_inputs(name, args, kwargs):
        arrays = {}
        for index, value in enumerate(args):
            if isinstance(value, torch.Tensor):
                arrays["arg_" + str(index)] = value.detach().float().cpu().numpy() if value.is_floating_point() else value.detach().cpu().numpy()
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                arrays[key] = value.detach().float().cpu().numpy() if value.is_floating_point() else value.detach().cpu().numpy()
            elif isinstance(value, (tuple, list)) and all(isinstance(x, torch.Tensor) for x in value):
                for index, item in enumerate(value):
                    arrays[key + "_" + str(index)] = item.detach().float().cpu().numpy()
        np.savez(out / active["case"] / (name + "-inputs.npz"), **arrays)

    pipeline.text_encoder.register_forward_pre_hook(lambda module, args, kwargs: dump_inputs("encoder", args, kwargs), with_kwargs=True)
    pipeline.text_encoder.model.language_model.register_forward_pre_hook(lambda module, args, kwargs: dump_inputs("language", args, kwargs), with_kwargs=True)
    pipeline.text_encoder.model.language_model.norm.register_forward_pre_hook(lambda module, args: np.save(out / active["case"] / "language-reference.npy", args[0].detach().float().cpu().numpy()))
    pipeline.text_encoder.model.visual.register_forward_pre_hook(lambda module, args, kwargs: dump_inputs("vision", args, kwargs), with_kwargs=True)
    def capture_vision_output(module, args, result):
        np.savez(out / active["case"] / "vision-reference.npz", merged=result.pooler_output.detach().float().cpu().numpy(), **{"deepstack_" + str(index): value.detach().float().cpu().numpy() for index, value in enumerate(result.deepstack_features)})
    pipeline.text_encoder.model.visual.register_forward_hook(capture_vision_output)
    def capture_forward(*args, **kwargs):
        step = active["step"]
        case_dir = out / active["case"]
        if step == 0:
            tensors = {key: value.detach().float().cpu().numpy() if value.is_floating_point() else value.detach().cpu().numpy() for key, value in kwargs.items() if isinstance(value, torch.Tensor)}
            np.savez(case_dir / "transformer-inputs.npz", **tensors)
            (case_dir / "transformer-layout.json").write_text(json.dumps({"img_shapes": kwargs["img_shapes"], "kv_cache_mode": kwargs["kv_cache_mode"]}, indent=2))
        started = time.perf_counter()
        result = original_forward(*args, **kwargs)
        if not torch.isfinite(result[0]).all():
            raise ValueError("The source denoiser returned non-finite values")
        torch.mps.synchronize() if device == "mps" else None
        np.save(case_dir / f"noise-{step}.npy", result[0].detach().float().cpu().numpy())
        record("denoise", case=active["case"], step=step, seconds=time.perf_counter() - started, shape=list(result[0].shape), finite=bool(torch.isfinite(result[0]).all()))
        active["step"] += 1
        return result

    def capture_encode(value, *args, **kwargs):
        case_dir = out / active["case"]
        np.save(case_dir / "vae-encode-input.npy", value.detach().float().cpu().numpy())
        result = original_encode(value, *args, **kwargs)
        np.save(case_dir / "vae-encode-output.npy", result.latent_dist.mode().detach().float().cpu().numpy())
        return result

    def capture_decode(value, *args, **kwargs):
        case_dir = out / active["case"]
        np.save(case_dir / "vae-decode-input.npy", value.detach().float().cpu().numpy())
        result = original_decode(value, *args, **kwargs)
        if not torch.isfinite(result[0]).all():
            raise ValueError("The source VAE returned non-finite values")
        np.save(case_dir / "vae-decode-output.npy", result[0].detach().float().cpu().numpy())
        return result

    def capture_step(pipeline, index, timestep, values):
        np.save(out / active["case"] / ("latents-" + str(index) + ".npy"), values["latents"].detach().float().cpu().numpy())
        if index == 7:
            np.save(out / active["case"] / "final-latents.npy", values["latents"].detach().float().cpu().numpy())
        return values

    pipeline.transformer.forward = capture_forward
    pipeline.vae.encode = capture_encode
    pipeline.vae.decode = capture_decode
    for name, prompt, image in cases:
        (out / name).mkdir()
        active.update(case=name, step=0)
        (out / name / "prompt.txt").write_text(prompt + "\n")
        noise = torch.randn(1, 1, 64, 32, 32, generator=torch.Generator("cpu").manual_seed(20261018)).reshape(1, 64, 1024).transpose(1, 2).contiguous()
        np.save(out / name / "initial-noise.npy", noise.numpy())
        record("case-start", case=name)
        started = time.perf_counter()
        output = pipeline(prompt=prompt, image=image, height=512, width=512, output_resolution=512, true_cfg_scale=1.0, generator=torch.Generator("cpu").manual_seed(20261018), use_kv_cache=cache_enabled, latents=noise.to(dtype=dtype), callback_on_step_end=capture_step)
        output.images[0].save(out / name / "output.png")
        record("case-complete", case=name, seconds=time.perf_counter() - started, denoising_steps=active["step"])
    record("complete")
    write_json(out / "reference.json", {"complete": True, "target_profile": PROFILE, "precision": args.precision, "cache": args.cache, "cases": ["generate", "edit"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--cache", choices=("on", "off"), default="off")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    run_reference(parser.parse_args())


if __name__ == "__main__":
    main()
