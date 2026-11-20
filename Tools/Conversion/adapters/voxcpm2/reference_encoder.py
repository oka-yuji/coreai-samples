import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time
import wave

import numpy as np
import torch
from torch import nn
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.weight_norm import WeightNorm

from ...storage import contained, file_hash, write_json
from .convert import export
from .runtime import Components, describe, entries
from .source import add_source_arguments, compare, load_contract, load_model, summarize

ASSET = "refenc"
FUNCTION = "main"
SOURCE_RATE = 24000
SAMPLE_RATE = 16000
HOP_SAMPLES = 640
PATCH_FRAMES = 4
CHANNELS = 64
PATCH_SAMPLES = HOP_SAMPLES * PATCH_FRAMES
MAXIMUM_PATCHES = 63
MAXIMUM_FRAMES = MAXIMUM_PATCHES * PATCH_FRAMES
MAXIMUM_SAMPLES = MAXIMUM_PATCHES * PATCH_SAMPLES
SOURCE_SAMPLES = 240000
CROPS = {"5": 120000, "10": 240000}
EXPECTED_PATCHES = {"5": 32, "10": 63}
REQUIRED_CASES = 4
LIMITS = {"fp32": 0.001, "fp16": 0.01}
CRITERIA = {**LIMITS, "nonfinite": 0}
INTERFACE = {"fp32": np.float32, "fp16": np.float16}
REPOSITORY = Path(__file__).resolve().parents[4]
MANIFEST_NAME = "Tools/Conversion/recipes/voxcpm2/reference-voices.json"
MANIFEST = REPOSITORY / MANIFEST_NAME
MANIFEST_SHA256 = "20f0127777ed9abd9277f7abb2f2acba534c292fd2cae2dcd44a32bb9068b7ab"
CONTRACT_CANONICAL_SHA256 = "3fd3c882d49795d865cb3bc763e0c8e85724d02be3c10deca3da7a1edadee203"
AUDIO_VAE = "src/voxcpm/modules/audiovae/audio_vae_v2.py"
AUDIO_VAE_SHA256 = "2efdff1708d8ec1471624aae6f232d0f933de26b788b6901c434246847e2d3a8"
VOXCPM2 = "src/voxcpm/model/voxcpm2.py"
ADAPTERS = ("reference_encoder.py", "source.py", "convert.py", "runtime.py", "wrappers.py")
PACKAGES = ("numpy", "torch", "soxr", "soundfile", "librosa", "transformers", "safetensors", "voxcpm",
            "coreai-core", "coreai-torch", "coreai-opt")


def descriptor():
    return {"functions": {FUNCTION: {"inputs": [["pcm", "float32", [1, 1, MAXIMUM_SAMPLES]]],
                                     "outputs": [["mu", "float32", [1, CHANNELS, MAXIMUM_FRAMES]]],
                                     "states": []}}}


def extend(contract):
    if ASSET in contract["assets"]:
        raise ValueError(f"The contract already defines the {ASSET} asset")
    extended = copy.deepcopy(contract)
    extended["assets"][ASSET] = descriptor()
    return extended


def prepare_pcm(samples):
    if not isinstance(samples, np.ndarray) or isinstance(samples, np.ma.MaskedArray):
        raise ValueError("The PCM must be a NumPy array")
    if samples.dtype != np.float32 or samples.ndim != 1:
        raise ValueError("The PCM must be one-dimensional float32")
    count = samples.shape[0]
    if not 1 <= count <= MAXIMUM_SAMPLES:
        raise ValueError(f"The PCM must have 1 to {MAXIMUM_SAMPLES} samples")
    if not np.isfinite(samples).all():
        raise ValueError("The PCM must be finite")
    padded = np.zeros((1, 1, MAXIMUM_SAMPLES), dtype=np.float32)
    padded[0, 0, :count] = samples
    return padded, (count + PATCH_SAMPLES - 1) // PATCH_SAMPLES


def patches_from_mu(mu, patch_count):
    if type(patch_count) is not int or not 1 <= patch_count <= MAXIMUM_PATCHES:
        raise ValueError(f"The patch count must be an int from 1 to {MAXIMUM_PATCHES}")
    if not isinstance(mu, np.ndarray) or isinstance(mu, np.ma.MaskedArray) or mu.dtype not in (np.float32, np.float16):
        raise ValueError("mu must be a float32 or float16 NumPy array")
    if mu.shape != (1, CHANNELS, MAXIMUM_FRAMES):
        raise ValueError(f"mu must have the shape [1, {CHANNELS}, {MAXIMUM_FRAMES}]")
    if not np.isfinite(mu).all():
        raise ValueError("mu must be finite")
    frames = mu[0, :, :patch_count * PATCH_FRAMES].T
    return np.array(frames.reshape(patch_count, PATCH_FRAMES, CHANNELS), dtype=np.float32, order="C")


def plain_tensors(module):
    owned = {id(value) for value in (*module.parameters(), *module.buffers())}
    return [value for item in module.modules() for value in vars(item).values()
            if isinstance(value, torch.Tensor) and id(value) not in owned]


class ReferenceEncoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        source = vae.encoder
        if not isinstance(source, nn.Module):
            raise ValueError("The AudioVAE encoder must be a torch module")
        encoder = copy.deepcopy(source, {id(value): value.detach().clone() for value in plain_tensors(source)})
        with torch.no_grad():
            for item in encoder.modules():
                for hook in [hook for hook in item._forward_pre_hooks.values() if isinstance(hook, WeightNorm)]:
                    remove_weight_norm(item, hook.name)
        hooked = [name or "encoder" for name, item in encoder.named_modules()
                  if item._forward_pre_hooks or item._forward_hooks or item._backward_pre_hooks or item._backward_hooks]
        if hooked:
            raise ValueError(f"The encoder copy has unexpected hooks: {hooked}")
        self.encoder = encoder.eval().requires_grad_(False)

    def forward(self, pcm):
        return self.encoder(pcm)["mu"]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fixed_contract(path):
    contract = load_contract(path)
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    if sha256(canonical) != CONTRACT_CANONICAL_SHA256:
        raise ValueError("Use the pinned VoxCPM2 conversion contract")
    return contract


def pcm_sha256(samples):
    return sha256(np.ascontiguousarray(samples, dtype="<f4").tobytes())


def canonical_pcm(pcm16):
    import soxr

    if not isinstance(pcm16, np.ndarray) or pcm16.dtype != np.int16 or pcm16.ndim != 1 or not pcm16.size \
            or pcm16.size * SAMPLE_RATE % SOURCE_RATE:
        raise ValueError("The source PCM must be one-dimensional int16 with a whole number of 16 kHz samples")
    samples = np.ascontiguousarray(pcm16, dtype=np.float32) / np.float32(32768)
    resampled = soxr.resample(samples, SOURCE_RATE, SAMPLE_RATE, quality="HQ")
    if resampled.dtype != np.float32 or resampled.shape != (pcm16.size * SAMPLE_RATE // SOURCE_RATE,):
        raise ValueError("The resampled PCM has an unexpected type or length")
    return np.ascontiguousarray(resampled)


def read_pcm16(raw, count):
    try:
        with wave.open(io.BytesIO(raw), "rb") as handle:
            shape = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes(),
                     handle.getcomptype())
            data = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError) as error:
        raise ValueError(f"Unreadable WAV: {error}") from error
    if shape != (1, 2, SOURCE_RATE, count, "NONE") or len(data) != 2 * count:
        raise ValueError(f"The WAV must be PCM16 mono {SOURCE_RATE} Hz with {count} samples, not {list(shape)}")
    return np.frombuffer(data, dtype="<i2")


def write_pcm16(path, pcm):
    with path.open("xb") as stream, wave.open(stream, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SOURCE_RATE)
        writer.setnframes(pcm.size)
        writer.writeframes(np.ascontiguousarray(pcm, dtype="<i2").tobytes())
    raw = path.read_bytes()
    if not np.array_equal(read_pcm16(raw, pcm.size), pcm):
        raise ValueError(f"The written WAV differs from its source samples: {path.name}")
    return {"bytes": len(raw), "sha256": sha256(raw)}


def load_manifest():
    raw = Path(MANIFEST).read_bytes()
    digest = sha256(raw)
    if digest != MANIFEST_SHA256:
        raise ValueError("The voice manifest differs from the frozen resource")
    manifest = json.loads(raw)
    fixed = {"version": 1, "sample_rate": SAMPLE_RATE, "hop_samples": HOP_SAMPLES, "patch_frames": PATCH_FRAMES,
             "feature_channels": CHANNELS, "maximum_samples": MAXIMUM_SAMPLES, "maximum_patches": MAXIMUM_PATCHES,
             "criteria": CRITERIA}
    mismatched = sorted(key for key, value in fixed.items() if manifest.get(key) != value)
    if mismatched:
        raise ValueError(f"The voice manifest disagrees with the fixed geometry or criteria: {mismatched}")
    voices = manifest.get("voices")
    if not isinstance(voices, list) or len(voices) * len(CROPS) != REQUIRED_CASES \
            or len({str(voice.get("id")) for voice in voices}) != len(voices) \
            or any(not re.fullmatch(r"[0-9a-z]+", str(voice.get("id"))) or set(voice.get("pcm_f32le_sha256", {})) != set(CROPS)
                   for voice in voices):
        raise ValueError("The voice manifest must list two voices with 5 and 10 second identities")
    return manifest, {"file": MANIFEST_NAME, "bytes": len(raw), "sha256": digest}


def prepare(manifest, voices, output):
    readme = contained(voices, "README.md").read_bytes()
    if sha256(readme) != manifest["readme_sha256"]:
        raise ValueError("The voice README differs from the manifest")
    folder = output / "voices"
    folder.mkdir()
    cases = []
    for voice in manifest["voices"]:
        raw = contained(voices, voice["path"]).read_bytes()
        if len(raw) != voice["bytes"] or sha256(raw) != voice["sha256"]:
            raise ValueError(f"Voice identity mismatch: {voice['id']}")
        pcm = read_pcm16(raw, SOURCE_SAMPLES)
        for seconds, count in CROPS.items():
            name = f"{voice['id']}-{seconds}s"
            samples = canonical_pcm(pcm[:count])
            digest = pcm_sha256(samples)
            if digest != voice["pcm_f32le_sha256"][seconds]:
                raise ValueError(f"Canonical PCM identity mismatch: {name}")
            padded, patches = prepare_pcm(samples)
            if patches != EXPECTED_PATCHES[seconds]:
                raise ValueError(f"Unexpected patch count for {name}: {patches}")
            crop = folder / f"{name}.wav"
            identity = {"voice": {"path": voice["path"], "bytes": len(raw), "sha256": voice["sha256"]},
                        "crop": {"file": f"voices/{crop.name}", "source_samples": count, "source_rate": SOURCE_RATE,
                                 **write_pcm16(crop, pcm[:count])},
                        "pcm": {"samples": int(samples.size), "sample_rate": SAMPLE_RATE, "f32le_sha256": digest},
                        "padded": {"shape": list(padded.shape), "f32le_sha256": pcm_sha256(padded)}}
            row = {"case": name, "voice": voice["id"], "audio_seconds": int(seconds), "patches": patches,
                   "frames": patches * PATCH_FRAMES, "identity": identity}
            cases.append({"name": name, "crop": crop, "samples": samples, "padded": padded, "patches": patches, "row": row})
    return cases, {"readme": {"path": "README.md", "bytes": len(readme), "sha256": sha256(readme)}}


def runtime_check():
    if "USE_LOCAL_COREAI" in os.environ:
        raise ValueError("Unset USE_LOCAL_COREAI; this check uses the OS Core AI runtime")
    from coreai.runtime import SpecializationOptions

    if not SpecializationOptions.is_supported():
        raise RuntimeError("Use the macOS Core AI runtime")
    return {"specialization": "gpu_preferred", "use_os_coreai": os.environ.get("USE_OS_COREAI")}


def model_geometry(model):
    vae = model.audio_vae
    observed = {"encode_sample_rate": int(model._encode_sample_rate), "hop_samples": int(vae.hop_length),
                "chunk_samples": int(model.chunk_size), "patch_frames": int(model.patch_size),
                "latent_channels": int(vae.latent_dim), "feature_channels": int(model.feat_dim)}
    expected = {"encode_sample_rate": SAMPLE_RATE, "hop_samples": HOP_SAMPLES, "chunk_samples": HOP_SAMPLES,
                "patch_frames": PATCH_FRAMES, "latent_channels": CHANNELS, "feature_channels": CHANNELS}
    if observed != expected:
        raise ValueError(f"The model configuration differs from the fixed geometry: {observed}")
    return {**observed, "encoder_rates": [int(rate) for rate in vae.encoder_rates]}


def upstream_identity(upstream, model):
    from voxcpm.modules.audiovae import audio_vae_v2

    root = Path(upstream).resolve(strict=True)
    files = {name: file_hash(root / name) for name in (AUDIO_VAE, VOXCPM2)}
    if files[AUDIO_VAE]["sha256"] != AUDIO_VAE_SHA256:
        raise ValueError("The upstream AudioVAE differs from the reviewed source")
    if Path(audio_vae_v2.__file__).resolve() != root / AUDIO_VAE or type(model.audio_vae.encoder) is not audio_vae_v2.CausalEncoder:
        raise ValueError("The model encoder is not the pinned upstream CausalEncoder")
    return files


def adapter_identity():
    folder = Path(__file__).resolve().parent
    files = {f"Tools/Conversion/adapters/voxcpm2/{name}": file_hash(folder / name) for name in ADAPTERS}
    files["Tools/Conversion/storage.py"] = file_hash(folder.parents[1] / "storage.py")
    return files


def versions():
    result = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def environment():
    def command(arguments):
        try:
            return subprocess.run(arguments, capture_output=True, text=True, timeout=30).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    return {"machine": command(["/usr/sbin/sysctl", "-n", "hw.model"]),
            "os_build": command(["/usr/bin/sw_vers", "-buildVersion"]), "os_version": platform.mac_ver()[0] or None,
            "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
            "load_average": list(os.getloadavg()),
            "offline": {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}}


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous().reshape(-1)
        digest.update(f"{name}|{tensor.dtype}|{list(value.shape)}|".encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def weight_norm_hooks(module):
    return sum(isinstance(hook, WeightNorm) for item in module.modules() for hook in item._forward_pre_hooks.values())


def storages(module):
    tensors = [*module.parameters(), *module.buffers(), *plain_tensors(module)]
    return {tensor.untyped_storage().data_ptr() for tensor in tensors if tensor.numel()}


def original_feature(model, crop):
    captured = []
    handle = model.audio_vae.encoder.register_forward_pre_hook(
        lambda module, arguments: captured.append(arguments[0].detach().clone()))
    try:
        started = time.perf_counter()
        with torch.no_grad():
            feature = model._encode_wav(str(crop), padding_mode="right", trim_silence_vad=False)
        seconds = time.perf_counter() - started
    finally:
        handle.remove()
    if feature.dtype != torch.float32:
        raise ValueError(f"The original reference must be float32, not {feature.dtype}")
    return np.array(feature.detach().cpu().numpy(), dtype=np.float32, order="C"), captured, seconds


def input_matches(captured, padded, patches):
    count = patches * PATCH_SAMPLES
    return (len(captured) == 1 and captured[0].dtype == torch.float32 and tuple(captured[0].shape) == (1, 1, count)
            and bool(np.array_equal(captured[0].numpy(), padded[:, :, :count])))


def measure(reference, mu, patches, limit):
    if mu.shape != (1, CHANNELS, MAXIMUM_FRAMES) or mu.dtype not in (np.float32, np.float16):
        raise ValueError(f"Unexpected mu: {list(mu.shape)} {mu.dtype}")
    nonfinite = int(mu.size - np.count_nonzero(np.isfinite(mu)))
    if nonfinite:
        return None, {"passed": False, "nonfinite": nonfinite, "limit": limit}
    prefix = patches_from_mu(mu, patches)
    return prefix, {**compare(reference, prefix, limit), "limit": limit}


def record(case, kind, mu, limit):
    mu = np.asarray(mu)
    relative = f"cases/{case['name']}"
    np.save(case["folder"] / f"{kind}-mu.npy", mu, allow_pickle=False)
    case["row"]["arrays"][f"{kind}_mu"] = f"{relative}/{kind}-mu.npy"
    prefix, case["row"][kind] = measure(case["original"], mu, case["patches"], limit)
    if prefix is not None:
        np.save(case["folder"] / f"{kind}.npy", prefix, allow_pickle=False)
        case["row"]["arrays"][kind] = f"{relative}/{kind}.npy"


def advance(report, output, stage):
    report["stage"] = stage
    write_json(output / "report.json", report)


def scrub(text, paths):
    forms = [(form, token) for token, path in paths for form in {os.path.abspath(path), os.path.realpath(path)}
             if len(form) > 1]
    home = str(Path.home())
    if len(home) > 1:
        forms.append((home, "~"))
    for form, token in sorted(forms, key=lambda item: len(item[0]), reverse=True):
        text = text.replace(form, token)
    return re.sub(r"/(?:Users|Volumes|private|var|tmp|home)/[^\s'\"]*", "<path>", text)


def failure(error, args, stage):
    paths = [("$output", args.output), ("$snapshot", args.snapshot), ("$upstream", args.upstream),
             ("$voices", args.voices), ("$contract", args.contract), ("$repo", REPOSITORY)]
    return {"stage": stage, "type": type(error).__name__, "message": scrub(str(error), paths)}


def run(args, report):
    output = args.output
    started = time.perf_counter()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    report["versions"] = versions()
    report["runtime"] = runtime_check()
    base = fixed_contract(args.contract)
    contract = extend(base)
    manifest, report["manifest"] = load_manifest()
    report["criteria"] = {"wrapper": LIMITS["fp32"], "coreai": LIMITS[args.precision], "nonfinite": 0,
                          "metric": "max(abs(actual - reference)) / max(max(abs(reference)), 1e-8)"}
    report["contract"] = {"model_id": base["model_id"], "revision": base["revision"],
                          "upstream_revision": base["upstream_revision"], "files": base["files"],
                          "assets": {ASSET: contract["assets"][ASSET]}}
    advance(report, output, "voices")
    cases, report["voices"] = prepare(manifest, Path(args.voices).resolve(strict=True), output)
    rows = report["rows"] = [case["row"] for case in cases]
    advance(report, output, "model")
    loading = time.perf_counter()
    model = load_model(args.snapshot, args.upstream, base)
    report["seconds"]["load_model"] = time.perf_counter() - loading
    report["model"] = model_geometry(model)
    report["sources"] = {"adapters": adapter_identity(), "upstream": upstream_identity(args.upstream, model)}
    report["environment"] = environment()
    source = model.audio_vae.encoder
    report["source_encoder"] = {"before": {"state_sha256": state_digest(source), "weight_norm_hooks": weight_norm_hooks(source)}}
    advance(report, output, "original")
    for case in cases:
        row, case["folder"] = case["row"], output / "cases" / case["name"]
        case["folder"].mkdir(parents=True)
        np.save(case["folder"] / "pcm.npy", case["samples"], allow_pickle=False)
        original, captured, seconds = original_feature(model, case["crop"])
        np.save(case["folder"] / "original.npy", original, allow_pickle=False)
        row["arrays"] = {"pcm": f"cases/{case['name']}/pcm.npy", "original": f"cases/{case['name']}/original.npy"}
        row["seconds"] = {"original": seconds}
        shape = [case["patches"], PATCH_FRAMES, CHANNELS]
        if list(original.shape) != shape or not np.isfinite(original).all():
            raise ValueError(f"The original reference of {case['name']} must be finite {shape}, not {list(original.shape)}")
        row["original"] = {"shape": shape, "max_abs": float(np.max(np.abs(original)))}
        row["original_input_matches"] = input_matches(captured, case["padded"], case["patches"])
        case["original"] = original
        write_json(output / "report.json", report)
    advance(report, output, "fixed")
    building = time.perf_counter()
    wrapper = ReferenceEncoder(model.audio_vae)
    report["seconds"]["wrapper"] = time.perf_counter() - building
    report["source_encoder"]["shared_storage"] = bool(storages(wrapper) & storages(source))
    for case in cases:
        timer = time.perf_counter()
        with torch.no_grad():
            mu = wrapper(torch.from_numpy(case["padded"])).numpy().copy()
        case["row"]["seconds"]["fixed"] = time.perf_counter() - timer
        record(case, "fixed", mu, LIMITS["fp32"])
        write_json(output / "report.json", report)
    advance(report, output, "export")
    exporting = time.perf_counter()
    example = (torch.zeros((1, 1, MAXIMUM_SAMPLES), dtype=torch.float32),)
    report["export"] = export([(FUNCTION, wrapper, example, ["pcm"], ["mu"], [])], args.precision, contract, ASSET,
                              output / f"{ASSET}-{args.precision}.aimodel")
    report["seconds"]["export"] = time.perf_counter() - exporting
    write_json(output / "export.json", {"complete": True, "asset": ASSET, "contract": contract,
                                        "precisions": {args.precision: report["export"]}})
    advance(report, output, "coreai")
    components = Components([output], args.precision, contract)
    try:
        report["seconds"]["coreai_load"] = components.load_seconds[ASSET]
        observed = describe(components.functions[ASSET, FUNCTION])
        report["descriptor"] = observed
        expected = {kind: entries(contract, ASSET, FUNCTION, kind, args.precision) for kind in ("inputs", "outputs", "states")}
        if observed != expected:
            raise ValueError(f"The Core AI descriptor differs from the contract: {observed}")
        for case in cases:
            value = case["padded"].astype(INTERFACE[args.precision])
            timer = time.perf_counter()
            mu = components.call(ASSET, FUNCTION, {"pcm": value})["mu"]
            case["row"]["seconds"]["coreai"] = time.perf_counter() - timer
            record(case, "coreai", mu, LIMITS[args.precision])
            write_json(output / "report.json", report)
    finally:
        components.close()
    advance(report, output, "summary")
    encoder = report["source_encoder"]
    encoder["after"] = {"state_sha256": state_digest(source), "weight_norm_hooks": weight_norm_hooks(source)}
    encoder["unchanged"] = encoder["before"] == encoder["after"] and not encoder["shared_storage"]
    for row in rows:
        row["passed"] = bool(row["fixed"]["passed"] and row["coreai"]["passed"] and row["original_input_matches"])
    fixed, coreai = summarize([row["fixed"] for row in rows]), summarize([row["coreai"] for row in rows])
    names = [f"{voice['id']}-{seconds}s" for voice in manifest["voices"] for seconds in CROPS]
    report["summary"] = {"cases": len(rows), "required_cases": REQUIRED_CASES,
                         "passed_cases": sum(row["passed"] for row in rows),
                         "original_inputs_matched": sum(row["original_input_matches"] for row in rows),
                         "fixed": fixed, "coreai": coreai}
    report["seconds"]["total"] = time.perf_counter() - started
    report["passed"] = ([row["case"] for row in rows] == names and len(rows) == REQUIRED_CASES
                        and all(row["passed"] for row in rows) and encoder["unchanged"]
                        and all(item["count"] == item["passed"] == REQUIRED_CASES and item["nonfinite"] == 0
                                for item in (fixed, coreai)))
    report["complete"] = True
    report["stage"] = "complete"


def main(argv=None):
    parser = argparse.ArgumentParser()
    add_source_arguments(parser)
    parser.add_argument("--voices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=list(LIMITS), default="fp32")
    args = parser.parse_args(argv)
    if os.path.lexists(args.output):
        raise FileExistsError("The output already exists; use a new directory")
    args.output.mkdir(parents=True)
    report = {"version": 1, "complete": False, "passed": False, "precision": args.precision, "stage": "reserved",
              "seconds": {}}
    write_json(args.output / "report.json", report)
    try:
        run(args, report)
    except BaseException as error:
        report.update(complete=False, passed=False, error=failure(error, args, report.get("stage")))
        try:
            write_json(args.output / "report.json", report)
        except Exception:
            write_json(args.output / "report.json",
                       {key: report[key] for key in ("version", "complete", "passed", "precision", "stage", "error")})
        raise
    write_json(args.output / "report.json", report)
    summary = report["summary"]
    print(json.dumps({"precision": args.precision, "passed": report["passed"], "cases": summary["cases"],
                      "fixed_normalized_max": summary["fixed"]["normalized_max"],
                      "coreai_normalized_max": summary["coreai"]["normalized_max"]}), flush=True)
    if not report["passed"]:
        raise RuntimeError("The fixed-shape reference encoder did not meet the fixed criteria")


if __name__ == "__main__":
    main()
