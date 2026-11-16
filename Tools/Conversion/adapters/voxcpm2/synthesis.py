import argparse
import json
import os
from pathlib import Path
import random
import re
import shutil
import string
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np

from ...storage import file_hash, read_json, write_json
from . import host
from .checks import Checks, classify_fuse
from .runtime import Components
from .source import compare, load_contract, load_model

FLOATS = {"fp32": np.float32, "fp16": np.float16}


class GenerationError(Exception):
    def __init__(self, details):
        super().__init__(json.dumps(details))
        self.details = details


def normalize(text):
    return re.sub(r"\s+", " ", text.replace("\n", " "))


class Tokenizer:
    def __init__(self, path, contract):
        from tokenizers import Tokenizer as Backend

        path = Path(path)
        if file_hash(path) != contract["files"][path.name]:
            raise ValueError(f"{path.name} is not the pinned tokenizer")
        self.backend = Backend.from_file(str(path))
        self.vocabulary = {identifier: token for token, identifier in self.backend.get_vocab(with_added_tokens=True).items()}
        self.audio_start = contract["execution"]["audio_start_token"]

    def __call__(self, text):
        ids = self.backend.encode(normalize(text), add_special_tokens=False).ids
        return host.prompt_ids(ids, self.vocabulary, self.audio_start)


def finite(values, call, kind):
    broken = sorted(name for name, value in values.items() if value.dtype.kind == "f" and not np.isfinite(value).all())
    if broken:
        raise GenerationError({"reason": f"nonfinite_{kind}", "call": call, "names": broken})


class Assets:
    def __init__(self, directories, precisions, contract):
        exports = {read_json(Path(directory) / "export.json")["asset"]: Path(directory) for directory in directories}
        if (len(exports) != len(directories) or sorted(exports) != sorted(contract["assets"])
                or sorted(precisions) != sorted(contract["assets"])):
            raise ValueError("Every contract asset needs one export and one precision")
        self.precisions, self.loaded = dict(precisions), {}
        try:
            for precision in sorted(set(self.precisions.values())):
                chosen = [exports[asset] for asset in contract["assets"] if self.precisions[asset] == precision]
                self.loaded[precision] = Components(chosen, precision, contract)
        except BaseException:
            self.close()
            raise

    def state(self, asset, function):
        return self.loaded[self.precisions[asset]].state(asset, function)

    def call(self, asset, function, values, state=None):
        precision = self.precisions[asset]
        values = {name: np.asarray(value) for name, value in values.items()}
        finite(values, f"{asset}.{function}", "input")
        values = {name: value.astype(FLOATS[precision]) if value.dtype.kind == "f" else value for name, value in values.items()}
        outputs = self.loaded[precision].call(asset, function, values, state)
        finite(outputs, f"{asset}.{function}", "output")
        return {name: value.astype(np.float32) for name, value in outputs.items()}

    def load_seconds(self):
        return {asset: seconds for components in self.loaded.values() for asset, seconds in components.load_seconds.items()}

    def close(self):
        for components in self.loaded.values():
            components.close()
        self.loaded.clear()


class Result:
    def __init__(self, ids, limit):
        self.ids, self.limit = list(ids), limit
        self.noise, self.latents, self.stop_logits, self.audio = [], [], [], []
        self.finish, self.error, self.seconds = None, None, None

    @property
    def patches(self):
        return len(self.latents)

    @property
    def succeeded(self):
        return self.finish == "stop"

    @property
    def stopped_patch(self):
        return self.patches - 1 if self.succeeded else None

    def waveform(self):
        if not self.succeeded:
            raise ValueError(f"A generation that ended with {self.finish} has no audio")
        return np.concatenate(self.audio, axis=-1).reshape(-1)


def generate(assets, contract, ids, seed, limit=None, align=None):
    import torch

    execution, generation = contract["execution"], contract["generation"]
    size, length = execution["prefill_length"], len(ids)
    if not 1 < length <= size:
        raise ValueError(f"{length - 1} text tokens do not fit the prefill length {size}")
    allowed = host.patch_limit(length - 1, execution["capacity"], generation)
    limit = allowed if limit is None else limit
    if not 1 <= limit <= allowed:
        raise ValueError(f"The patch limit must be within 1..{allowed}")
    result = Result(ids, limit)
    noise = torch.Generator().manual_seed(seed)
    started = time.perf_counter()
    try:
        tslm, ralm, vae = (assets.state(asset, function) for asset, function in (("tslm", "prefill"), ("ralm", "prefill"), ("vae", "main")))
        padded = np.full((1, size), execution["pad_token"], np.int32)
        padded[0, :length] = ids
        count = np.array([length], np.int32)
        embeds = assets.call("tslm", "embed", {"ids": padded})["embeds"]
        hidden = assets.call("tslm", "prefill", {"embeds": embeds, "length": count}, tslm)["hidden"]
        residual = assets.call("proj", "prefill_fuse", {"hidden": hidden})["residual_inputs"]
        lm_hidden = hidden[:, length - 1]
        residual_hidden = assets.call("ralm", "prefill", {"embeds": residual, "length": count}, ralm)["hidden"][:, length - 1]
        cond = np.zeros(host.PATCH, np.float32)
        for index in range(limit):
            heads = assets.call("proj", "heads", {"lm_hidden": lm_hidden, "residual_hidden": residual_hidden})
            result.noise.append(torch.randn(host.PATCH, generator=noise, dtype=torch.float32).numpy())
            solver = host.Solver(result.noise[-1], heads["dit_hidden"], cond, generation)
            while (inputs := solver.inputs()) is not None:
                solver.advance(assets.call("dit", "main", inputs)["velocity"])
            patch = solver.latent
            result.latents.append(patch)
            result.stop_logits.append(heads["stop_logits"])
            if align is not None:
                patch = align.latent(index, patch, result.noise[-1], heads["dit_hidden"], cond)
            curr_embed = assets.call("locenc", "main", {"patch": patch})["curr_embed"]
            result.audio.append(assets.call("vae", "main", {"latent": patch}, vae)["audio"])
            outcome = host.patch_step(index, limit, heads["stop_logits"], generation)
            if outcome != "next":
                result.finish = "stop" if outcome == "stopped" else "limit"
                break
            position = np.array([length + index], np.int32)
            lm_raw = assets.call("tslm", "step", {"embed": curr_embed, "position": position}, tslm)["hidden"]
            fused = assets.call("proj", "fuse", {"lm_raw": lm_raw, "curr_embed": curr_embed})
            lm_hidden, residual_input = fused["lm_hidden"], fused["residual_input"]
            if align is not None:
                lm_hidden, residual_input = align.fuse(index, lm_hidden, residual_input)
            residual_hidden = assets.call("ralm", "step", {"embed": residual_input, "position": position}, ralm)["hidden"]
            cond = patch
    except GenerationError as error:
        result.finish, result.error = "error", error.details
    result.seconds = time.perf_counter() - started
    return result


class Alignment:
    def __init__(self, reference, candidates, limit, solve, per_sentence, schedule):
        self.reference, self.candidates, self.limit, self.solve = reference, candidates, limit, solve
        self.per_sentence, self.schedule = per_sentence, schedule
        self.positions, self.replaced, self.arrays, self.calls = [], set(), {}, 0

    def replace(self, patch, record):
        if patch not in self.replaced and len(self.replaced) >= self.per_sentence:
            raise GenerationError({"reason": "replacement_limit", "patch": patch, "replaced": sorted(self.replaced), **record})
        self.replaced.add(patch)
        self.positions.append({"patch": patch, **record})

    def fuse(self, step, lm_hidden, residual_input):
        self.calls += 1
        if step >= len(self.reference["fsq_out"]):
            raise GenerationError({"reason": "past_reference_stop", "step": step})
        expected = (self.reference["fsq_out"][step], self.reference["fuse_out"][step])
        outcome = classify_fuse(expected, (lm_hidden, residual_input), self.candidates.get(step, []), self.limit)
        if outcome["category"] == "match":
            return lm_hidden, residual_input
        if outcome["category"] == "fail":
            raise GenerationError({"reason": "fuse_mismatch", "step": step, **outcome})
        self.replace(step + 1, {"kind": "fuse", "step": step, **outcome["boundary"]})
        return expected[0].copy(), expected[1].copy()

    def latent(self, index, latent, noise, dit_hidden, cond):
        reference = self.reference
        if index >= len(reference["patch"]):
            raise GenerationError({"reason": "past_reference_stop", "patch": index})
        expected = reference["patch"][index]
        error = compare(expected, latent, self.limit)
        if error["passed"]:
            return latent
        values = (noise, dit_hidden, cond)
        names = ("noise", "mu", "cond")
        inputs = {"noise_bitwise": bool(np.array_equal(noise, reference["noise"][index])),
                  "dit_hidden": compare(reference["mu"][index], dit_hidden, self.limit),
                  "cond": compare(reference["cond"][index], cond, self.limit),
                  "valid": all(value.dtype == np.float32 and value.shape == reference[name][index].shape and np.isfinite(value).all()
                               for value, name in zip(values, names, strict=True)),
                  "schedule": self.schedule}
        original = np.asarray(self.solve(noise, dit_hidden, cond), dtype=np.float32)
        if original.shape != expected.shape or not np.isfinite(original).all():
            raise GenerationError({"reason": "invalid_original_solution", "patch": index})
        scale, reference_scale = (max(float(np.max(np.abs(value))), 1e-8) for value in (original, expected))
        a, b = (float(np.max(np.abs(original.astype(np.float64) - value))) for value in (latent, expected))
        record = {"kind": "latent", "error": error, "inputs": inputs, "a": a / scale, "a_reference_scale": a / reference_scale,
                  "b": b / scale, "b_reference_scale": b / reference_scale}
        self.arrays[f"patch_{index}"] = np.stack([latent, original, expected])
        if not (inputs["noise_bitwise"] and inputs["dit_hidden"]["passed"] and inputs["cond"]["passed"] and inputs["valid"]
                and inputs["schedule"] and record["a"] <= self.limit):
            raise GenerationError({"reason": "latent_mismatch", "patch": index, **record})
        self.replace(index, record)
        return expected.copy()


def replacement_summary(positions, total_patches, settings):
    patches = sum(len({row["patch"] for row in rows}) for rows in positions)
    kinds = {kind: sum(row["kind"] == kind for rows in positions for row in rows) for kind in ("fuse", "latent")}
    limit = settings["total_fraction"] * total_patches
    return {"replaced_patches": patches, "kinds": kinds, "reference_patches": total_patches, "limit": limit,
            "per_sentence": settings["per_sentence"], "within_limit": patches <= limit}


class Trace:
    def __init__(self, assets):
        self.assets, self.calls = assets, []

    def state(self, asset, function):
        return self.assets.state(asset, function)

    def call(self, asset, function, values, state=None):
        outputs = self.assets.call(asset, function, values, state)
        self.calls.append((asset, function, {name: np.array(value) for name, value in values.items()},
                           {name: np.array(value) for name, value in outputs.items()}))
        return outputs

    def arrays(self):
        grouped = {}
        for asset, function, inputs, outputs in self.calls:
            for side, values in (("in", inputs), ("out", outputs)):
                for name, value in values.items():
                    grouped.setdefault(f"{asset}.{function}.{side}.{name}", []).append(value)
        return {key: np.stack(values) for key, values in grouped.items()}

    def order(self):
        counts, rows = {}, []
        for asset, function, _, _ in self.calls:
            key = f"{asset}.{function}"
            rows.append([key, counts.get(key, 0)])
            counts[key] = counts.get(key, 0) + 1
        return rows


class OriginalSolver:
    def __init__(self, job, directory):
        self.job, self.directory, self.count = job, directory, 0

    def __call__(self, noise, dit_hidden, cond):
        folder = self.directory / f"original-{self.count}"
        self.count += 1
        folder.mkdir()
        np.savez(folder / "inputs.npz", noise=noise, dit_hidden=dit_hidden, cond=cond)
        write_json(folder / "job.json", {key: self.job[key] for key in ("contract", "snapshot", "upstream")})
        subprocess.run([sys.executable, "-m", __spec__.name, "solve", str(folder / "job.json")], check=True)
        return np.load(folder / "latent.npy")


def solve_original(path):
    import torch

    job = read_json(path)
    contract = load_contract(Path(job["contract"]))
    generation = contract["generation"]
    model = load_model(Path(job["snapshot"]), Path(job["upstream"]), contract)
    with np.load(path.parent / "inputs.npz") as arrays:
        noise, dit_hidden, cond = (torch.from_numpy(arrays[name].copy()) for name in ("noise", "dit_hidden", "cond"))
    with torch.inference_mode():
        span = torch.linspace(1, 0, generation["inference_timesteps"] + 1, dtype=torch.float32)
        span = span + generation["sway_sampling_coef"] * (torch.cos(torch.pi / 2 * span) - 1 + span)
        latent = model.feat_decoder.solve_euler(noise, span, dit_hidden, cond, cfg_value=generation["cfg_value"],
                                                use_cfg_zero_star=generation["cfg_zero_star"])
    np.save(path.parent / "latent.npy", latent.numpy())


def pcm16(audio):
    scaled = np.round(np.asarray(audio, dtype=np.float64) * 32767)
    clipped = int(np.count_nonzero((scaled > 32767) | (scaled < -32768)))
    return np.clip(scaled, -32768, 32767).astype("<i2"), clipped


def write_wav(path, result, sample_rate):
    samples, clipped = pcm16(result.waveform())
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return clipped


def decibels(value):
    return float(20 * np.log10(value)) if value > 0 else None


def environment():
    return {"load_average": list(os.getloadavg())}


def levels(audio, rate):
    return {"audio_seconds": audio.size / rate, "peak_dbfs": decibels(float(np.max(np.abs(audio)))),
            "rms_dbfs": decibels(float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))))}


def describe(result, reference, limit, rate, replaced=()):
    expected, stops = reference.load("patch"), reference.load("stop_logits")
    count = min(result.patches, len(expected))
    rows = [compare(expected[index], result.latents[index], limit) for index in range(count)]
    failed = [index for index, row in enumerate(rows) if not row["passed"]]
    complete = result.succeeded and result.patches == len(expected)
    audio = compare(np.concatenate(list(reference.load("stream_audio")), axis=-1).reshape(-1), result.waveform(), limit) if complete else None
    return {"finish": result.finish, "error": result.error, "patches": result.patches, "stopped_patch": result.stopped_patch,
            "reference_patches": len(expected), "reference_stopped_patch": len(expected) - 1,
            "latent_errors": [row.get("normalized_max") for row in rows],
            "latent_max": max((row["normalized_max"] for row in rows if "normalized_max" in row), default=None),
            "first_latent_out_of_criterion": failed[0] if failed else None,
            "stop_logits_max_abs": float(np.max(np.abs(np.stack(result.stop_logits[:count]) - stops[:count]))) if count else None,
            "audio": audio, "seconds": result.seconds, **(levels(result.waveform(), rate) if result.succeeded else {}),
            "passed": bool(complete and not [index for index in failed if index not in replaced] and audio["passed"])}


def save_result(result, directory, extra):
    arrays = {name: np.stack(values) for name, values in (("noise", result.noise), ("latents", result.latents),
                                                          ("stop_logits", result.stop_logits)) if values}
    if result.succeeded:
        arrays["audio"] = result.waveform()
    np.savez(directory / "result.npz", **arrays)
    write_json(directory / "result.json", {"ids": result.ids, "limit": result.limit, "finish": result.finish, "error": result.error,
                                           "seconds": result.seconds, **extra})


def load_result(directory):
    info = read_json(directory / "result.json")
    result = Result(info["ids"], info["limit"])
    with np.load(directory / "result.npz") as arrays:
        result.noise, result.latents, result.stop_logits = (list(arrays[name]) if name in arrays.files else []
                                                            for name in ("noise", "latents", "stop_logits"))
        result.audio = [arrays["audio"]] if "audio" in arrays.files else []
    result.finish, result.error, result.seconds = info["finish"], info["error"], info["seconds"]
    return result, info


def run_job(path):
    job = read_json(path)
    contract = load_contract(Path(job["contract"]))
    assets = Assets(job["export"], job["precisions"], contract)
    try:
        warmup = job.get("warmup")
        warmup_seconds = generate(assets, contract, warmup["ids"], warmup["seed"]).seconds if warmup else None
        target = Trace(assets) if job.get("trace") else assets
        alignment = None
        if job.get("fixture"):
            reference = Checks(None, contract, job["reference"], {"id": job["fixture"], "directory": job["fixture"]}, np.float32)
            arrays = {name: reference.load(name) for name in ("fsq_out", "fuse_out", "patch", "mu", "cond")}
            arrays["noise"] = reference.load("dit_x")[:, 0, :1]
            schedule = bool(np.array_equal(host.time_span(contract["generation"]), reference.load("t_span")))
            alignment = Alignment(arrays, reference.candidates(), contract["criteria"]["normalized_max_limit"],
                                  OriginalSolver(job, path.parent), job["replacement"]["per_sentence"], schedule)
        result = generate(target, contract, job["ids"], job["seed"], limit=job.get("limit"), align=alignment)
        load_seconds = assets.load_seconds()
    finally:
        assets.close()
    if job.get("trace"):
        np.savez(path.parent / "calls.npz", **target.arrays())
        write_json(path.parent / "calls.json", target.order())
    if alignment is not None and alignment.arrays:
        np.savez(path.parent / "replacements.npz", **alignment.arrays)
    save_result(result, path.parent, {"warmup_seconds": warmup_seconds, "load_seconds": load_seconds,
                                      "positions": alignment.positions if alignment else [],
                                      "replaced": sorted(alignment.replaced) if alignment else [],
                                      "fuse_calls": alignment.calls if alignment else None})


def isolated(job, directory):
    directory.mkdir(parents=True)
    write_json(directory / "job.json", job)
    completed = subprocess.run([sys.executable, "-m", __spec__.name, "worker", str(directory / "job.json")])
    if completed.returncode == 0:
        return load_result(directory)
    result = Result(job["ids"], job.get("limit"))
    result.finish, result.error = "error", {"reason": "worker_exit", "code": completed.returncode}
    return result, {}


def save_reference(result, folder, directory, seed):
    directory.mkdir(parents=True)
    np.save(directory / "ids.npy", np.array(result.ids, dtype=np.int32))
    if result.latents and (folder / "calls.npz").exists():
        latents = np.stack(result.latents)
        with np.load(folder / "calls.npz") as calls:
            dit_hidden = calls["proj.heads.out.dit_hidden"][:result.patches]
        np.savez(directory / "patches.npz", noise=np.stack(result.noise), dit_hidden=dit_hidden,
                 cond=np.concatenate([np.zeros_like(latents[:1]), latents[:-1]]), latent=latents, stop_logits=np.stack(result.stop_logits))
        for name in ("calls.npz", "calls.json"):
            shutil.copyfile(folder / name, directory / name)
    if result.succeeded:
        np.save(directory / "audio.npy", result.waveform())
    return {"seed": seed, "finish": result.finish, "patches": result.patches, "stopped_patch": result.stopped_patch,
            "files": sorted(path.name for path in directory.iterdir())}


def check_fixture(base, contract, tokenizer, text, seed, reference, root, scratch, output):
    limit, rate = contract["criteria"]["normalized_max_limit"], contract["architecture"]["sample_rate"]
    name, ids = reference.record["id"], tokenizer(text)
    job = {**base, "ids": ids, "seed": seed, "trace": True}
    folder = scratch / f"{name}-replacing"
    replacing, info = isolated({**job, "reference": root, "fixture": name}, folder)
    positions = info.get("positions", [])
    replaced = [row["patch"] for row in positions if row["kind"] == "latent"]
    noise = reference.load("dit_x")[:, 0, :1]
    count = min(replacing.patches, len(noise))
    entry = {"text_tokens": len(ids) - 1, "limit": replacing.limit,
             "ids_match": ids == reference.load("ids").reshape(-1).tolist(),
             "noise_match": bool(count) and all(np.array_equal(replacing.noise[index], noise[index]) for index in range(count)),
             "replacing": describe(replacing, reference, limit, rate, replaced), "fuse_calls": info.get("fuse_calls"),
             "positions": [{"fixture": name, **position} for position in positions], "replaced_patches": info.get("replaced", []),
             "load_seconds": info.get("load_seconds")}
    if (folder / "replacements.npz").exists():
        (output / "replacements").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / "replacements.npz", output / "replacements" / f"{name}.npz")
    free = replacing
    if positions or not replacing.succeeded:
        folder = scratch / f"{name}-free"
        free, _ = isolated(job, folder)
    entry["without_replacement"] = describe(free, reference, limit, rate)
    entry["passed"] = bool(entry["ids_match"] and entry["noise_match"] and entry["replacing"]["passed"])
    entry["swift_reference"] = save_reference(free, folder, output / "swift_reference" / name, seed)
    return entry, positions


def limit_check(base, contract, tokenizer, text, seed, settings, record, output, scratch):
    result, _ = isolated({**base, "ids": tokenizer(text), "seed": seed, "limit": settings["patches"]}, scratch / "limit")
    path = output / "limit-check.wav"
    try:
        write_wav(path, result, contract["architecture"]["sample_rate"])
    except ValueError:
        pass
    entry = {"fixture": settings["fixture"], "limit": settings["patches"], "reference_patches": record["patches"],
             "finish": result.finish, "patches": result.patches, "succeeded": result.succeeded, "audio_written": path.exists()}
    entry["passed"] = bool(settings["patches"] < record["patches"] and result.finish == "limit"
                           and result.patches == settings["patches"] and not result.succeeded and not entry["audio_written"])
    return entry


def check(args):
    contract = load_contract(args.contract)
    reference = read_json(args.reference / "reference.json")
    if not reference["complete"] or reference["contract"] != contract:
        raise ValueError("The reference is incomplete or was captured for another contract")
    settings = read_json(args.settings)
    tokenizer = Tokenizer(args.tokenizer, contract)
    texts = {fixture["id"]: fixture["text"] for fixture in reference["specification"]["fixtures"]}
    seed = reference["specification"]["seed"]
    args.output.mkdir(parents=True)
    report = {"complete": False, "passed": False, "seed": seed, "criteria": contract["criteria"], "environment_before": environment()}
    write_json(args.output / "check.json", report)
    base = {"contract": str(args.contract), "export": [str(path) for path in args.export],
            "precisions": dict.fromkeys(contract["assets"], "fp32"), "snapshot": str(args.snapshot), "upstream": str(args.upstream),
            "replacement": settings["replacement"]}
    report["replacement_settings"] = settings["replacement"]
    fixtures, positions = {}, []
    with tempfile.TemporaryDirectory() as scratch:
        for record in reference["records"]:
            checks = Checks(None, contract, args.reference, record, np.float32)
            fixtures[record["id"]], rows = check_fixture(base, contract, tokenizer, texts[record["id"]], seed, checks, str(args.reference),
                                                         Path(scratch), args.output)
            positions.append(rows)
            report["fixtures"] = fixtures
            write_json(args.output / "check.json", report)
        report["replacement"] = replacement_summary(positions, sum(record["patches"] for record in reference["records"]),
                                                    settings["replacement"])
        target = settings["limit_check"]
        record = next(record for record in reference["records"] if record["id"] == target["fixture"])
        report["limit_check"] = limit_check(base, contract, tokenizer, texts[target["fixture"]], seed, target, record, args.output,
                                            Path(scratch))
    report["environment_after"] = environment()
    report["passed"] = bool(all(entry["passed"] for entry in fixtures.values()) and report["replacement"]["within_limit"]
                            and report["limit_check"]["passed"])
    report["complete"] = True
    write_json(args.output / "check.json", report)
    print(json.dumps({"passed": report["passed"], "replacement": report["replacement"],
                      "fixtures": {name: entry["passed"] for name, entry in fixtures.items()}}), flush=True)
    if not report["passed"]:
        raise RuntimeError("The end-to-end FP32 generation did not meet the criteria")


def latent_errors(baseline, result, limit):
    count = min(baseline.patches, result.patches)
    rows = [compare(baseline.latents[index], result.latents[index], limit) for index in range(count)]
    values = [row["normalized_max"] for row in rows if "normalized_max" in row]
    failed = [index for index, row in enumerate(rows) if not row["passed"]]
    return {"latent_max": max(values, default=None), "latent_median": median(values),
            "first_latent_out_of_criterion": failed[0] if failed else None,
            "stopped_patch_matches_fp32": result.stopped_patch is not None and result.stopped_patch == baseline.stopped_patch}


def median(values):
    return float(np.median(values)) if values else None


def summary(rows):
    finished = [row for row in rows if row["finish"] == "stop"]
    compared = [row for row in rows if "stopped_patch_matches_fp32" in row]
    result = {"generations": len(rows), "finish": {name: sum(row["finish"] == name for row in rows) for name in ("stop", "limit", "error")},
              "median_seconds_per_audio_second": median([row["seconds_per_audio_second"] for row in finished]),
              "median_ms_per_patch": median([row["ms_per_patch"] for row in rows if row["ms_per_patch"] is not None]),
              "clipped_samples": sum(row.get("clipped_samples", 0) for row in rows)}
    if compared:
        firsts = [row["first_latent_out_of_criterion"] for row in compared if row["first_latent_out_of_criterion"] is not None]
        result |= {"stopped_patch_matches_fp32": sum(row["stopped_patch_matches_fp32"] for row in compared),
                   "latent_max": max((row["latent_max"] for row in compared if row["latent_max"] is not None), default=None),
                   "latent_median_of_medians": median([row["latent_median"] for row in compared if row["latent_median"] is not None]),
                   "generations_out_of_criterion": len(firsts), "first_latent_out_of_criterion_median": median(firsts)}
    return result


def compare_runs(args):
    contract = load_contract(args.contract)
    settings = read_json(args.settings)["compare"]
    fixtures = {fixture["id"]: fixture["text"] for fixture in read_json(args.fixtures)["fixtures"]}
    sentences = {name: fixtures[name] for name in settings["fixtures"]} | {item["id"]: item["text"] for item in settings["sentences"]}
    configurations, baseline = settings["configurations"], settings["baseline"]
    order = [baseline] + [name for name in configurations if name != baseline]
    tokenizer = Tokenizer(args.tokenizer, contract)
    ids = {name: tokenizer(text) for name, text in sentences.items()}
    limit, rate = contract["criteria"]["normalized_max_limit"], contract["architecture"]["sample_rate"]
    shuffle = random.Random(settings["label_seed"])
    labels = {}
    for sentence in sentences:
        names = list(configurations)
        shuffle.shuffle(names)
        labels[sentence] = dict(zip(string.ascii_uppercase, names))
    labelled = {sentence: {name: label for label, name in mapping.items()} for sentence, mapping in labels.items()}
    args.output.mkdir(parents=True)
    report = {"complete": False, "settings": settings, "criterion": limit,
              "sentences": {name: {"text_tokens": len(value) - 1} for name, value in ids.items()},
              "configurations": {}, "rows": []}
    write_json(args.output / "comparison.json", report)
    base = {"contract": str(args.contract), "export": [str(path) for path in args.export]}
    first = next(iter(sentences))
    warmup = {"ids": ids[first], "seed": settings["seeds"][0]}
    missing, baselines = [], {}
    with tempfile.TemporaryDirectory() as scratch:
        for name in order:
            entry = {"precisions": configurations[name], "warmup": {"sentence": first, "seed": warmup["seed"]},
                     "environment_before": environment()}
            rows = []
            for sentence in sentences:
                for seed in settings["seeds"]:
                    before = os.getloadavg()
                    result, info = isolated({**base, "precisions": configurations[name], "ids": ids[sentence], "seed": seed,
                                             "warmup": warmup}, Path(scratch) / f"{name}-{sentence}-{seed}")
                    row = {"sentence": sentence, "seed": seed, "configuration": name, "finish": result.finish, "error": result.error,
                           "patches": result.patches, "stopped_patch": result.stopped_patch, "seconds": result.seconds,
                           "ms_per_patch": 1000 * result.seconds / result.patches if result.seconds and result.patches else None,
                           "warmup_seconds": info.get("warmup_seconds"), "load_seconds": sum((info.get("load_seconds") or {}).values()),
                           "load_average": [list(before), list(os.getloadavg())]}
                    if name == baseline:
                        baselines[sentence, seed] = result
                    else:
                        row |= latent_errors(baselines[sentence, seed], result, limit)
                    if result.succeeded:
                        audio = result.waveform()
                        folder = Path("listen") if seed == settings["listen_seed"] else Path("listen-extra") / str(seed)
                        row |= {**levels(audio, rate), "seconds_per_audio_second": result.seconds * rate / audio.size,
                                "clipped_samples": write_wav(args.output / folder / sentence / f"{labelled[sentence][name]}.wav", result, rate)}
                    else:
                        missing.append({"sentence": sentence, "seed": seed, "label": labelled[sentence][name], "finish": result.finish})
                    rows.append(row)
            entry["environment_after"] = environment()
            entry["summary"] = summary(rows)
            report["configurations"][name] = entry
            report["rows"].extend(rows)
            write_json(args.output / "comparison.json", report)
    write_json(args.output / "listen-key.json", {"label_seed": settings["label_seed"], "listen_seed": settings["listen_seed"],
                                                 "labels": labels, "without_audio": missing})
    report["complete"] = True
    write_json(args.output / "comparison.json", report)
    print(json.dumps({name: entry["summary"]["finish"] for name, entry in report["configurations"].items()}), flush=True)


def main():
    if sys.argv[1:2] in (["worker"], ["solve"]):
        (run_job if sys.argv[1] == "worker" else solve_original)(Path(sys.argv[2]))
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["check", "compare"])
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--export", type=Path, action="append", required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    check(args) if args.mode == "check" else compare_runs(args)


if __name__ == "__main__":
    main()
