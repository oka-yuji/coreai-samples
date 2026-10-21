import argparse
import gc
import json
from pathlib import Path
import platform
import signal
import time

from Tools.Conversion.storage import write_json
from .contracts import require_heavy_job, tensor_metrics
from .evaluation import SIGMAS, canonical_hash
from .fixtures import file_record, quality_axes
from .mobile_artifacts import read_component, read_profile, validate_environment, validate_source
from .mobile_evidence import mobile_fixtures


def component_paths(values):
    result = {}
    for value in values:
        kind, separator, path = value.partition("=")
        if not separator or not path or kind in result or kind not in ("vision", "text", "denoiser", "vae_encoder", "vae_decoder"):
            raise ValueError("Expected one kind=directory binding per component")
        result[kind] = Path(path)
    return result


def reference_preflight(profile, paths):
    quantized = any(value["weight_encoding"]["kind"] == "linear-integer" for value in profile["components"].values())
    if (quantized or paths) and set(paths) != set(profile["components"]):
        raise ValueError("A matched reference requires all five exported component records")
    compression, records = {}, {}
    for kind, path in paths.items():
        _, compression[kind] = read_component(path, profile, kind)
        records[kind] = file_record(path / "component.json")
    return compression, records


def selected_runs(manifest, requested):
    expected = {run["run_id"]: run for run in manifest["runs"]}
    selected = list(expected) if requested is None else requested.split(",")
    if not selected or not set(selected) <= set(expected) or len(selected) != len(set(selected)):
        raise ValueError("Unknown, empty or duplicate source run selection")
    return expected, selected


def execute(args):
    import numpy as np
    from PIL import Image
    import psutil
    import torch
    from .mobile_capture import NonFiniteSourceError, SourceTrace
    from .mobile_models import DTYPES
    from .mobile_source import source_pipeline

    require_heavy_job()
    versions = validate_environment()
    profile = read_profile(args.profile, args.dependency_lock)
    manifest = json.loads((args.fixtures / "fixtures.json").read_text())
    mobile_fixtures(args.fixtures, manifest)
    expected, selected = selected_runs(manifest, args.run_ids)
    compression, component_records = reference_preflight(profile, component_paths(args.component))
    source = validate_source(args.snapshot)
    if args.output.is_symlink() or (args.output.exists() and any(args.output.iterdir())):
        raise FileExistsError("A source suite requires a new empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    code = ("mobile_reference.py", "mobile_source.py", "mobile_capture.py", "mobile_evidence.py", "fixtures.py", "numeric_policy.py")
    identity = {"format": 2, "route": "original-diffusers-with-declared-precision-boundaries",
                "profile": profile, "profile_sha256": canonical_hash(profile), "source_files": source,
                "exported_components": component_records, "fixtures_file": file_record(args.fixtures / "fixtures.json"),
                "reference_code": {name: file_record(Path(__file__).with_name(name)) for name in code},
                "device": args.device, "versions": versions, "host": platform.platform(),
                "sigmas": list(SIGMAS), "prefix_kv_cache": False, "quality_status": "unreviewed",
                "scope": "Captured source fidelity; excludes clean latency and device acceptance"}
    outcomes = {key: {"run_id": key, "status": "not_run"} for key in expected}
    repeats = {key: {"run_id": key, "status": "not_run"} for key in ("g01-20261018", "e01-20261018")}
    axes = quality_axes()

    def publish():
        write_json(args.output / "identity.json", identity)
        write_json(args.output / "summary.json", {"identity": identity, "runs": list(outcomes.values()),
                   "repeatability": list(repeats.values()),
                   "complete": all(row["status"] == "completed" for row in (*outcomes.values(), *repeats.values()))})
        write_json(args.output / "quality-ratings.json", {"format": 1, "ratings": [
            {"run_id": key, "status": "unreviewed", "reference_status": outcomes[key]["status"],
             "axes": {axis: None for axis in axes[key]}, "notes": ""} for key in expected]})

    def publish_failure(error):
        try:
            publish()
        except Exception as recording_error:
            error.add_note("Recording the failed source evaluation also failed: " + repr(recording_error))
            raise error from recording_error

    publish()
    torch.set_num_threads(10)
    pipeline = None
    started = time.monotonic()
    try:
        pipeline, observed = source_pipeline(args.snapshot, profile, compression, args.device)
        identity.update(load_status="completed", load_seconds=time.monotonic() - started, effective_compression=observed)
        publish()
    except BaseException as error:
        identity.update(load_status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                        error_type=type(error).__name__, error=str(error))
        publish_failure(error)
        raise

    def run(record, directory, trace):
        metadata = json.loads((args.fixtures / record["fixture_id"] / "input.json").read_text())
        if pipeline._drop_idx != metadata["drop"]:
            raise ValueError("The original prompt-template removal differs from the frozen input")
        with np.load(args.fixtures / record["fixture_id"] / "encoded.npz", allow_pickle=False) as values:
            encoded = dict(values)
        trace.begin(directory, record["operation"], expected_inputs=encoded)
        image = None
        if metadata["reference"]:
            with Image.open(args.fixtures / metadata["reference"]) as original:
                image = original.convert("RGBA")
        noise = np.load(args.fixtures / record["noise"], allow_pickle=False)
        started = time.monotonic()
        with torch.no_grad():
            result = pipeline(prompt=metadata["prompt"], image=image, height=512, width=512, output_resolution=512,
                              true_cfg_scale=1.0, generator=torch.Generator("cpu").manual_seed(record["seed"]),
                              use_kv_cache=False,
                              latents=torch.from_numpy(noise).to(DTYPES[profile["components"]["denoiser"]["compute_dtype"]]),
                              callback_on_step_end=trace.callback)
        if args.device == "mps":
            torch.mps.synchronize()
        trace.finish()
        result.images[0].save(directory / "output.png")
        observed = {"run_id": record["run_id"], "input": record, "status": "completed",
                    "total_seconds": time.monotonic() - started, "rss_bytes": psutil.Process().memory_info().rss,
                    "files": {path.name: file_record(path) for path in sorted(directory.iterdir()) if path.is_file()}}
        if args.device == "mps":
            observed.update(mps_allocated_bytes=torch.mps.current_allocated_memory(),
                            mps_driver_allocated_bytes=torch.mps.driver_allocated_memory())
        write_json(directory / "result.json", observed)
        return observed

    def failure(key, error, trace):
        status = "cancelled" if isinstance(error, KeyboardInterrupt) else "nonfinite" if isinstance(error, NonFiniteSourceError) else "failed"
        record = {"run_id": key, "status": status, "error_type": type(error).__name__, "error": str(error)}
        if trace.directory is not None:
            try:
                trace.snapshot()
                record["files"] = {path.name: file_record(path) for path in trace.directory.iterdir() if path.is_file()}
                write_json(trace.directory / "result.json", record)
            except Exception as capture_error:
                record["capture_error"] = str(capture_error)
        return record

    try:
        with SourceTrace(pipeline, profile) as trace:
            for key in selected:
                trace.directory = None
                try:
                    outcomes[key] = run(expected[key], args.output / key, trace)
                except BaseException as error:
                    outcomes[key] = failure(key, error, trace)
                    publish_failure(error)
                    if not isinstance(error, NonFiniteSourceError):
                        raise
                publish()
                print(json.dumps({"run_id": key, "status": outcomes[key]["status"]}), flush=True)
            for key in repeats:
                if outcomes[key]["status"] != "completed":
                    continue
                trace.directory = None
                try:
                    directory = args.output / (key + "-repeat")
                    run(expected[key], directory, trace)
                    names = ["rgba.npy", *[f"latent-{index}.npy" for index in range(8)]]
                    rows = {name: tensor_metrics(np.load(args.output / key / name), np.load(directory / name)) for name in names}
                    repeats[key] = {"run_id": key, "comparisons": rows,
                                    "status": "completed" if all(row["passed"] for row in rows.values()) else "failed"}
                except BaseException as error:
                    repeats[key] = failure(key, error, trace)
                    publish_failure(error)
                    if not isinstance(error, NonFiniteSourceError):
                        raise
                publish()
    finally:
        del pipeline
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
        import sys
        active_error = sys.exception()
        if active_error is None:
            publish()
        else:
            publish_failure(active_error)
    if any(row["status"] not in ("completed", "not_run") for row in (*outcomes.values(), *repeats.values())):
        raise ValueError("The source candidate contains failed or non-finite evaluations; inspect the retained records")


def main():
    parser = argparse.ArgumentParser()
    for name in ("snapshot", "fixtures", "profile", "dependency-lock", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--component", action="append", default=[])
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--run-ids")
    args = parser.parse_args()
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    execute(args)


if __name__ == "__main__":
    main()
