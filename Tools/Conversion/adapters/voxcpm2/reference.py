import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import re
import time
from unittest.mock import patch

import numpy as np
import torch

from ...storage import read_json, write_json
from .checks import fuse_candidates, tie_distance
from .source import add_source_arguments, load_contract, load_model


def array(value):
    value = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    return value.astype(np.int32 if np.issubdtype(value.dtype, np.integer) else np.float32)


class Tape:
    def __init__(self, model):
        self.model = model
        self.phase = "prefill"
        self.values = {}
        self.lists = {}

    def keep(self, name, value):
        self.values[name] = array(value)

    def add(self, name, value):
        self.lists.setdefault(name, []).append(array(value))

    def install(self, stack):
        model = self.model

        def hook(module, handler, kwargs=False):
            stack.callback(module.register_forward_hook(handler, with_kwargs=kwargs).remove)

        def embed(module, args, out):
            self.keep("ids", args[0])
            self.keep("embeds", out)

        hook(model.base_lm.embed_tokens, embed)

        def prefill(name):
            def handler(module, args, kwargs, out):
                hidden, kv = out
                self.keep(f"{name}_prefill_in", kwargs["inputs_embeds"])
                self.keep(f"{name}_prefill_out", hidden)
                self.keep(f"{name}_prefill_k", torch.stack([key[0] for key, _ in kv]))
                self.keep(f"{name}_prefill_v", torch.stack([value[0] for _, value in kv]))
                if name == "ralm":
                    self.phase = "loop"
            return handler

        hook(model.base_lm, prefill("tslm"), kwargs=True)
        hook(model.residual_lm, prefill("ralm"), kwargs=True)

        def fusion(module, args, out):
            if self.phase == "prefill":
                self.keep("prefill_fuse_in", args[0])
                self.keep("prefill_fuse_out", out)
            else:
                self.add("fuse_in", args[0])
                self.add("fuse_out", out)

        hook(model.fusion_concat_proj, fusion)

        def loop(inputs=(), output=None):
            def handler(module, args, out):
                if self.phase == "loop":
                    for name, value in zip(inputs, args):
                        self.add(name, value)
                    if output:
                        self.add(output, out)
            return handler

        hook(model.lm_to_dit_proj, loop(["lm_hidden_in"]))
        hook(model.res_to_dit_proj, loop(["residual_hidden_in"]))
        hook(model.stop_head, loop(output="stop_logits"))
        hook(model.enc_to_lm_proj, loop(output="curr_embed"))
        hook(model.fsq_layer, loop(["fsq_in"], "fsq_out"))
        hook(model.fsq_layer.in_proj, loop(output="fsq_pre"))
        hook(model.feat_decoder.estimator, loop(["dit_x", "dit_mu", "dit_t", "dit_cond", "dit_dt"], "dit_velocity"))

        solve = model.feat_decoder.solve_euler

        def solve_euler(x, t_span, mu, cond, cfg_value=1.0, use_cfg_zero_star=True):
            result = solve(x, t_span, mu, cond, cfg_value=cfg_value, use_cfg_zero_star=use_cfg_zero_star)
            self.add("noise", x)
            self.add("mu", mu)
            self.add("cond", cond)
            self.add("patch", result)
            self.keep("t_span", t_span)
            self.settings = {"cfg_value": float(cfg_value), "cfg_zero_star": bool(use_cfg_zero_star)}
            return result

        stack.enter_context(patch.object(model.feat_decoder, "solve_euler", side_effect=solve_euler))
        for name, lm in (("tslm", model.base_lm), ("ralm", model.residual_lm)):
            stack.enter_context(patch.object(lm, "forward_step", side_effect=self.step(name, lm.forward_step)))
        decode = model.audio_vae.decode

        def vae_decode(z, sr_cond=None):
            result = decode(z, sr_cond)
            self.keep("latent", z)
            self.keep("full_audio", result)
            return result

        stack.enter_context(patch.object(model.audio_vae, "decode", side_effect=vae_decode))

    def step(self, name, original):
        def forward_step(inputs_embeds, position_id):
            result = original(inputs_embeds, position_id)
            self.add(f"{name}_step_in", inputs_embeds)
            self.add(f"{name}_step_position", position_id.reshape(-1))
            self.add(f"{name}_step_out", result)
            return result
        return forward_step


def host_checks(values, length):
    equal = np.array_equal
    patches = values["patch"].shape[0]
    steps = values["tslm_step_out"].shape[0]
    zero = np.zeros_like(values["tslm_prefill_out"])
    t_span = values["t_span"].astype(np.float32)
    times = []
    t, dt = t_span[0], t_span[0] - t_span[1]
    for index in range(1, len(t_span)):
        if index > max(1, int(len(t_span) * 0.04)):
            times.append(t)
        t = np.float32(t - dt)
        if index < len(t_span) - 1:
            dt = np.float32(t - t_span[index + 1])
    expected_times = np.repeat(np.array(times, np.float32)[None, :, None], 2, axis=2)
    return {
        "embeds_are_prefill_input": equal(values["embeds"], values["tslm_prefill_in"]),
        "prefill_fuse_input_is_hidden_and_zero": equal(values["prefill_fuse_in"], np.concatenate((values["tslm_prefill_out"], zero), -1)),
        "ralm_prefill_input_is_prefill_fuse": equal(values["ralm_prefill_in"], values["prefill_fuse_out"]),
        "first_lm_hidden_is_last_prefill_row": equal(values["lm_hidden_in"][0], values["tslm_prefill_out"][:, length - 1]),
        "first_residual_is_last_prefill_row": equal(values["residual_hidden_in"][0], values["ralm_prefill_out"][:, length - 1]),
        "lm_hidden_follows_fsq": equal(values["lm_hidden_in"][1:], values["fsq_out"]),
        "residual_follows_ralm_step": equal(values["residual_hidden_in"][1:], values["ralm_step_out"]),
        "dit_conditional_mu_is_solver_mu": equal(values["dit_mu"][:, :, 0], np.repeat(values["mu"], 9, axis=1)),
        "first_cond_is_zero": not np.any(values["cond"][0]),
        "cond_is_previous_patch": equal(values["cond"][1:], values["patch"][:-1]),
        "step_input_is_locenc_output": equal(values["tslm_step_in"], values["curr_embed"][:steps, :, 0]),
        "fsq_input_is_tslm_output": equal(values["fsq_in"], values["tslm_step_out"]),
        "fuse_input_is_fsq_and_curr_embed": equal(values["fuse_in"], np.concatenate((values["fsq_out"], values["tslm_step_in"]), -1)),
        "ralm_input_is_fuse_output": equal(values["ralm_step_in"], values["fuse_out"]),
        "positions_follow_prefill": equal(values["tslm_step_position"][:, 0], np.arange(length, length + steps))
                                    and equal(values["ralm_step_position"], values["tslm_step_position"]),
        "steps_are_patches_minus_one": steps == patches - 1,
        "dit_batch_repeats_x_cond_t": equal(values["dit_x"][:, :, 0], values["dit_x"][:, :, 1])
                                      and equal(values["dit_cond"][:, :, 0], values["dit_cond"][:, :, 1])
                                      and equal(values["dit_cond"][:, :, 0], np.repeat(values["cond"], 9, axis=1)),
        "dit_unconditional_mu_and_dt_are_zero": not np.any(values["dit_mu"][:, :, 1]) and not np.any(values["dit_dt"]),
        "dit_times_follow_t_span": equal(values["dit_t"], np.broadcast_to(expected_times, values["dit_t"].shape)),
        "latent_is_patches": equal(values["latent"], np.concatenate(list(values["patch"][:, 0]), -1)[None]),
    }


def quantizer(model, values, distance):
    layer, fusion, scale = model.fsq_layer, model.fusion_concat_proj, model.fsq_layer.scale
    rows, reproduced = {key: [] for key in ("step", "dim", "distance", "code", "lm_hidden", "residual_input")}, True
    scaled_steps, code_steps = [], []
    with torch.inference_mode():
        for step, pre in enumerate(values["fsq_pre"]):
            scaled = torch.tanh(torch.from_numpy(pre)) * scale
            codes = torch.round(scaled)
            reproduced &= np.array_equal(array(layer.out_proj(codes / scale)), values["fsq_out"][step])
            for candidate in fuse_candidates(scaled.numpy(), codes.numpy(), distance):
                changed = codes.clone()
                changed[0, candidate["dim"]] = candidate["candidate_code"]
                lm_hidden = layer.out_proj(changed / scale)
                residual = fusion(torch.cat((lm_hidden, torch.from_numpy(values["tslm_step_in"][step])), dim=-1))
                for key, value in (("step", step), ("dim", candidate["dim"]), ("distance", candidate["distance"]),
                                   ("code", [candidate["code"], candidate["candidate_code"]]),
                                   ("lm_hidden", array(lm_hidden)), ("residual_input", array(residual))):
                    rows[key].append(value)
            scaled_steps.append(array(scaled))
            code_steps.append(array(codes))
    values["fsq_scaled"], values["fsq_code"] = np.stack(scaled_steps), np.stack(code_steps)
    values["fsq_tie_distance"] = tie_distance(values["fsq_scaled"]).min(axis=(1, 2)).astype(np.float32)
    shapes = {"step": (0,), "dim": (0,), "distance": (0,), "code": (0, 2), "lm_hidden": (0, 1, values["fsq_out"].shape[-1]),
              "residual_input": (0, 1, values["fuse_out"].shape[-1])}
    for key, items in rows.items():
        dtype = np.int32 if key in ("step", "dim") else np.float32
        values[f"fsq_candidate_{key}"] = np.array(items, dtype=dtype) if items else np.zeros(shapes[key], dtype)
    return bool(reproduced)


def capture(model, fixture, specification, contract, root):
    execution = contract["execution"]
    text = re.sub(r"\s+", " ", fixture["text"].replace("\n", " "))
    ids = model.text_tokenizer(text)
    length = len(ids) + 1
    if length > execution["prefill_length"] or length + min(int(len(ids) * 6.0 + 10), 4096) > execution["capacity"]:
        raise ValueError(f"The fixture does not fit the execution contract: {fixture['id']}")
    tape = Tape(model)
    started = time.perf_counter()
    with torch.inference_mode(), ExitStack() as stack:
        tape.install(stack)
        audio, _, _ = model.generate_with_prompt_cache(
            target_text=text, prompt_cache=None, min_len=specification["min_len"], max_len=specification["max_len"],
            inference_timesteps=specification["inference_timesteps"], cfg_value=specification["cfg_value"],
            retry_badcase=False, seed=specification["seed"])
    seconds = time.perf_counter() - started
    values = {**tape.values, **{name: np.stack(rows) for name, rows in tape.lists.items()}}
    patches = values["patch"].shape[0]
    with torch.inference_mode(), model.audio_vae.streaming_decode() as decoder:
        chunks = [decoder.decode_chunk(torch.from_numpy(values["latent"][..., 4 * i:4 * i + 4].copy())) for i in range(patches)]
    values["stream_audio"] = np.stack([array(chunk) for chunk in chunks])
    shapes = {"dit_x": (patches, 9, 2), "dit_mu": (patches, 9, 2), "dit_t": (patches, 9, 2), "dit_cond": (patches, 9, 2),
              "dit_dt": (patches, 9, 2), "dit_velocity": (patches, 9, 2)}
    for name, prefix in shapes.items():
        values[name] = values[name].reshape(*prefix, *values[name].shape[2:])
    reproduced = quantizer(model, values, contract["criteria"]["fsq_boundary_distance"])
    if values["ids"].shape != (1, length) or values["ids"][0, -1] != execution["audio_start_token"]:
        raise ValueError("The prompt is not the text followed by the audio start token")
    checks = {**host_checks(values, length), "fsq_codes_reproduce_fsq_out": reproduced}
    if not all(checks.values()) or tape.settings != {"cfg_value": specification["cfg_value"], "cfg_zero_star": True}:
        raise ValueError(f"The original computation differs from the recorded host contract: {checks}")
    if any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError("Non-finite reference values")
    directory = root / fixture["id"]
    directory.mkdir()
    for name, value in values.items():
        np.save(directory / f"{name}.npy", value, allow_pickle=False)
    stop = values["stop_logits"][-1, 0]
    full, stream = values["full_audio"].reshape(-1), values["stream_audio"].reshape(-1)
    return {"id": fixture["id"], "directory": fixture["id"], "text_tokens": length - 1, "length": length,
            "patches": patches, "steps": patches - 1, "positions": [length, length + patches - 2],
            "stopped_by_head": bool(np.argmax(stop) == 1), "audio_seconds": float(audio.shape[-1]) / model.sample_rate,
            "stream_full_max_abs": float(np.max(np.abs(full - stream))), "seconds": seconds,
            "fsq_min_tie_distance": float(values["fsq_tie_distance"].min()),
            "fsq_candidates": len(values["fsq_candidate_step"]),
            "host_checks": checks, "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                              for name, value in sorted(values.items())}}


def main():
    parser = argparse.ArgumentParser()
    add_source_arguments(parser)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract = load_contract(args.contract)
    specification = read_json(args.fixtures)
    if specification.get("version") != 1 or specification["normalize"] or specification["retry_badcase"]:
        raise ValueError("Only unnormalized text without bad-case retries is supported")
    model = load_model(args.snapshot, args.upstream, contract)
    args.output.mkdir(parents=True)
    report = {"complete": False, "contract": contract, "specification": specification,
              "profile": "original_fp32_cpu", "torch_threads": torch.get_num_threads(), "records": []}
    write_json(args.output / "reference.json", report)
    for fixture in specification["fixtures"]:
        record = capture(model, fixture, specification, contract, args.output)
        report["records"].append(record)
        write_json(args.output / "reference.json", report)
        print(json.dumps({key: record[key] for key in ("id", "length", "patches", "stopped_by_head", "seconds")}), flush=True)
    report["complete"] = True
    write_json(args.output / "reference.json", report)


if __name__ == "__main__":
    main()
