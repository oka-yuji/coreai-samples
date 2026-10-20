import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import time

from .evaluation import IMPLEMENTATION, REVISION, SIGMAS, require
from .fixtures import file_record, save_json, validate_fixture_suite


def execute(args):
    import numpy as np
    from PIL import Image
    import psutil
    import torch
    from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline
    from .contracts import require_heavy_job, tensor_metrics
    from .preprocessing import prompt_inputs

    require_heavy_job()
    require(args.snapshot.name == REVISION, "Unexpected source snapshot")
    manifest = json.loads((args.fixtures / "fixtures.json").read_text())
    validate_fixture_suite(args.fixtures, manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    expected = {run["run_id"]: run for run in manifest["runs"]}
    selected = list(expected) if args.run_ids is None else args.run_ids.split(",")
    require(set(selected) <= set(expected) and len(selected) == len(set(selected)), "Unknown or duplicate run id")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    vae_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.vae_precision or args.precision]
    torch.set_num_threads(10)
    identity = {"format": 1, "route": "independent-diffusers", "precision": args.precision,
                "source_revision": REVISION, "implementation_revision": IMPLEMENTATION,
                "fixtures_sha256": file_record(args.fixtures / "fixtures.json")["sha256"],
                "runner_sha256": file_record(Path(__file__))["sha256"],
                "sigmas": SIGMAS, "prefix_kv_cache": False, "device": args.device,
                "host": platform.platform(), "diagnostics": "trace", "scope": "reference fidelity; not clean latency",
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers")}}
    identity["component_dtypes"] = {"vision": str(dtype), "text": str(dtype), "denoiser": str(dtype),
                                   "vae_encoder": str(vae_dtype), "vae_decoder": str(vae_dtype)}
    identity["mixed_vae_boundary"] = vae_dtype != dtype
    if vae_dtype != dtype:
        identity["route"] = "independent-diffusers-with-explicit-vae-boundary"
        identity["boundary_rule"] = "Canonical image to VAE dtype; source encode/normalize/pack; cast condition to prompt dtype"
    save_json(args.output / "identity.json", identity)
    outcomes = {key: {"run_id": key, "status": "not_run", "total_seconds": None} for key in expected}
    repeats = {key: {"run_id": key, "status": "not_run"} for key in ("g01-20261018", "e01-20261018")}
    def publish():
        save_json(args.output / "summary.json", {"identity": identity, "runs": list(outcomes.values()),
                                               "repeatability": list(repeats.values()),
                                               "complete": all(row["status"] == "completed" for row in outcomes.values())
                                               and all(row["status"] == "completed" for row in repeats.values())})
    publish()
    started = time.monotonic()
    try:
        overrides = {}
        if vae_dtype != dtype:
            overrides["vae"] = AutoencoderKLQwenImage21.from_pretrained(
                args.snapshot / "vae", torch_dtype=vae_dtype, local_files_only=True)
        pipeline = QwenImage21Pipeline.from_pretrained(
            args.snapshot, dtype=dtype, local_files_only=True, **overrides).to(args.device)
        del overrides
    except BaseException as error:
        identity.update(load_status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                        load_error_type=type(error).__name__, load_error=str(error))
        publish()
        raise
    identity["load_status"] = "completed"
    pipeline.set_progress_bar_config(disable=True)
    require(list(pipeline.config.sample_sigmas) == list(SIGMAS), "Source sampling schedule changed")
    require(pipeline.transformer.dtype == dtype and pipeline.text_encoder.dtype == dtype and pipeline.vae.dtype == vae_dtype,
            "Actual component dtypes differ from the reference identity")
    original_prepare = pipeline.prepare_latents
    if vae_dtype != dtype:
        def prepare(images, batch_size, channels, height, width, requested_dtype, device, generator, latents=None):
            require(batch_size == 1 and channels == 64 and (images is None or len(images) == 1),
                    "The explicit mixed-precision boundary supports the fixed one-image profile")
            target, _ = original_prepare(None, batch_size, channels, height, width, requested_dtype, device, generator, latents)
            condition = None
            if images is not None:
                encoded = pipeline._encode_vae_image(images[0].to(device=device, dtype=vae_dtype), generator)
                condition = pipeline._pack_latents(encoded, batch_size, channels, encoded.shape[3], encoded.shape[4])
                condition = condition.to(requested_dtype)
            return target, condition
        pipeline.prepare_latents = prepare
    identity["load_seconds"] = time.monotonic() - started
    save_json(args.output / "identity.json", identity)
    active = {}
    original_decode = pipeline.vae.decode
    def decode(value, *values, **kwargs):
        result = original_decode(value, *values, **kwargs)
        rgba = result[0].detach().float().cpu().numpy()
        require(bool(np.isfinite(rgba).all()), "The source decoder returned non-finite values")
        np.save(active["directory"] / "rgba.npy", rgba)
        return result
    pipeline.vae.decode = decode
    def step(pipeline, index, timestep, values):
        latent = values["latents"].detach().float().cpu().numpy()
        require(bool(np.isfinite(latent).all()), "The source scheduler returned non-finite values")
        np.save(active["directory"] / f"latent-{index}.npy", latent)
        active["steps"] += 1
        return values
    def run(record, directory):
        directory.mkdir()
        metadata = json.loads((args.fixtures / record["fixture_id"] / "input.json").read_text())
        image = Image.open(args.fixtures / metadata["reference"]).convert("RGBA") if metadata["reference"] else None
        encoded, drop, _ = prompt_inputs(args.snapshot, metadata["prompt"], image)
        with np.load(args.fixtures / record["fixture_id"] / "encoded.npz") as saved:
            require(np.array_equal(encoded["input_ids"], saved["input_ids"]), "Reference tokenizer input changed")
        require(drop == metadata["drop"], "Template removal changed")
        noise = np.load(args.fixtures / record["noise"], allow_pickle=False)
        active.update(directory=directory, steps=0)
        started = time.monotonic()
        result = pipeline(prompt=metadata["prompt"], image=image, height=512, width=512, output_resolution=512,
                          true_cfg_scale=1.0, generator=torch.Generator("cpu").manual_seed(record["seed"]),
                          use_kv_cache=False, latents=torch.from_numpy(noise).to(dtype=dtype),
                          callback_on_step_end=step)
        if args.device == "mps":
            torch.mps.synchronize()
        require(active["steps"] == 8, "The source pipeline did not complete all eight steps")
        result.images[0].save(directory / "output.png")
        observed = {"run_id": record["run_id"], "status": "completed", "total_seconds": time.monotonic() - started,
                    "input": record, "steps": active["steps"], "rss_bytes": psutil.Process().memory_info().rss,
                    "files": {p.name: file_record(p) for p in sorted(directory.iterdir()) if p.is_file()}}
        if args.device == "mps":
            observed["mps_allocated_bytes"] = torch.mps.current_allocated_memory()
            observed["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
        save_json(directory / "result.json", observed)
        return observed
    try:
        for key in selected:
            record = expected[key]
            try:
                outcomes[key] = run(record, args.output / key)
            except KeyboardInterrupt as error:
                outcomes[key] = {"run_id": key, "status": "cancelled", "total_seconds": None,
                                 "error_type": type(error).__name__}
                publish()
                raise
            except Exception as error:
                outcomes[key] = {"run_id": key, "status": "failed", "total_seconds": None,
                                 "error_type": type(error).__name__, "error": str(error)}
                publish()
                raise
            publish()
            print(json.dumps({"run_id": key, "status": outcomes[key]["status"],
                              "seconds": outcomes[key]["total_seconds"]}), flush=True)
        for key in ("g01-20261018", "e01-20261018"):
            if key not in selected:
                continue
            repeat = args.output / (key + "-repeat")
            try:
                run(expected[key], repeat)
                rows = {name: tensor_metrics(np.load(args.output / key / name), np.load(repeat / name))
                        for name in ["rgba.npy"] + [f"latent-{index}.npy" for index in range(8)]}
                repeats[key] = {"run_id": key, "comparisons": rows,
                                "status": "completed" if all(row["passed"] for row in rows.values()) else "failed"}
            except BaseException as error:
                repeats[key] = {"run_id": key, "status": "cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                                "error_type": type(error).__name__, "error": str(error)}
                publish()
                raise
            publish()
    finally:
        pipeline.vae.decode = original_decode
        pipeline.prepare_latents = original_prepare
        del original_decode
        del original_prepare
        del pipeline
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
        publish()
    quality = []
    for record in manifest["runs"]:
        case = json.loads((args.fixtures / record["fixture_id"] / "input.json").read_text())
        quality.append({"run_id": record["run_id"], "status": "unreviewed",
                        "reference_status": outcomes[record["run_id"]]["status"],
                        "axes": {axis: None for axis in case["axes"]}, "notes": ""})
    save_json(args.output / "quality-ratings.json", {"format": 1, "ratings": quality})


def main():
    from Tools.Conversion.storage import directory_lock
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), required=True)
    parser.add_argument("--vae-precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--run-ids")
    args = parser.parse_args()
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    lock = Path(os.environ.get("COREAI_HEAVY_JOB_LOCK", Path.home() / "Library/Caches/coreai-samples/heavy-job"))
    with directory_lock(lock, {"session": "qmi-215", "job": "reference-" + args.precision, "expected_minutes": 40}):
        execute(args)


if __name__ == "__main__":
    main()
