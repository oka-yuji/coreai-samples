import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import re
import resource
import sys
import time

from ...storage import PipelineError, read_json, write_json
from ..pinned import sha256
from . import reference
from .prepare import load_contract

SCHEMA = "gemma4-31b-anchor/1"
LAYERS_SCHEMA = "gemma4-31b-layers/1"
ROWS = 512
LAYERS = 6
LIMIT = 1e-4
FLOAT16_MAX = 65504.0
CAST_BLOCK = 1 << 26
SCALE = "model.language_model.embed_tokens.embed_scale"
LAYER_KEY = re.compile(r"model\.language_model\.layers\.([0-9]+)\.")


def parse(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise PipelineError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise PipelineError(f"Non-finite JSON number: {value}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def read(path):
    data = Path(path).read_bytes()
    return parse(data), hashlib.sha256(data).hexdigest(), len(data)


def forced_inputs(folder, vocabulary):
    record, digest, _ = read(Path(folder) / "reference.json")
    if record.get("schema") != reference.SCHEMA or record.get("passed") is not True:
        raise ValueError("The BF16 reference record is missing or did not pass")
    cases = []
    for case in record["cases"]:
        name = case["id"]
        if not isinstance(name, str) or not reference.CASE_ID.fullmatch(name):
            raise ValueError("Unexpected case ID in the BF16 reference record")
        tokens, found, size = read(Path(folder) / "cases" / f"{name}.tokens.json")
        if (size, found) != (case["files"]["tokens"]["bytes"], case["files"]["tokens"]["sha256"]):
            raise ValueError(f"{name}: the tokens file differs from the size and SHA-256 in its reference.json")
        inputs, outputs = tokens["input_tokens"], tokens["output_tokens"]
        if (case["input_tokens"], case["output_tokens"]) != (len(inputs), len(outputs)) \
                or any(type(token) is not int or not 0 <= token < vocabulary for token in inputs + outputs):
            raise ValueError(f"{name}: the token IDs differ from the record or leave the vocabulary")
        cases.append({"id": name, "kind": case["kind"], "input_tokens": inputs, "output_tokens": outputs})
    return digest, cases


def resident(model, dtype):
    import torch

    tensors = {**dict(model.named_buffers(remove_duplicate=False)), **dict(model.named_parameters(remove_duplicate=False))}
    cast = getattr(torch, dtype)
    return lambda name: tensors[name].to(cast).float()


def scaled(weights, value):
    import torch

    return lambda name: torch.tensor(value) if name == SCALE else weights(name)


def embed_scale(model, dtype):
    import torch

    return float(torch.tensor(model.config.get_text_config().hidden_size ** 0.5).to(getattr(torch, dtype)))


def materializing(unit, prefix, weights):
    import torch

    forward = unit.forward
    tensors = [*unit.named_parameters(prefix=prefix, remove_duplicate=False),
               *(item for item in unit.named_buffers(prefix=prefix, remove_duplicate=False) if item[1].is_floating_point())]

    def call(*args, **kwargs):
        kept = [(tensor, tensor.data) for _, tensor in tensors]
        try:
            for name, tensor in tensors:
                tensor.data = weights(name).to(device=tensor.device, dtype=torch.float32)
            return forward(*args, **kwargs)
        finally:
            for tensor, data in kept:
                tensor.data = data

    return call


@contextlib.contextmanager
def layerwise(model, weights, rows):
    import torch
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def attention(module, query, key, value, attention_mask, **kwargs):
        parts = [sdpa_attention_forward(module, query[:, :, start:start + rows], key, value,
                                        attention_mask[:, :, start:start + rows], **kwargs)[0]
                 for start in range(0, query.shape[2], rows)]
        return torch.cat(parts, dim=1), None

    text = model.model.language_model
    names = {id(module): name for name, module in model.named_modules(remove_duplicate=False)}
    units = [text.embed_tokens, *text.layers, text.norm, model.lm_head]
    ALL_ATTENTION_FUNCTIONS["sdpa"] = attention
    try:
        for unit in units:
            unit.forward = materializing(unit, names[id(unit)], weights)
        yield
    finally:
        for unit in units:
            unit.__dict__.pop("forward", None)
        del ALL_ATTENTION_FUNCTIONS["sdpa"]


def masks(model, length, device):
    import torch
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    arguments = {"config": model.config.get_text_config(), "inputs_embeds": torch.empty(1, length, 1, device=device),
                 "attention_mask": None, "past_key_values": None, "allow_is_causal_skip": False}
    return {"full_attention": create_causal_mask(**arguments),
            "sliding_attention": create_sliding_window_causal_mask(**arguments)}


def logits(model, sequence, weights, device, rows=ROWS):
    import torch

    ids = torch.tensor([sequence], device=device)
    with layerwise(model, weights, rows):
        output = model(input_ids=ids, attention_mask=masks(model, len(sequence), device), use_cache=False).logits
    return output[0].cpu().numpy()


def agreement(model, sequences, device, rows=ROWS):
    import numpy as np

    weights = resident(model, "bfloat16")
    split = [logits(model, sequence, weights, device, rows) for sequence in sequences]
    model.float()
    results = []
    for sequence, first in zip(sequences, split):
        second, _ = reference.forced(model, sequence, [], device)
        results.append({"positions": len(sequence), "max_abs_difference": reference.finite(np.max(np.abs(first - second))),
                        "non_finite": {"layerwise": reference.non_finite(first), "plain": reference.non_finite(second)}})
    return results


class Ranges:
    def __init__(self):
        self.rows, self.case = {}, None

    def hook(self, point, layer):
        def note(module, inputs, output=None):
            value = (inputs[0] if output is None else output[0] if isinstance(output, tuple) else output).detach().abs()
            row = self.rows.setdefault((point, layer), {"point": point, "layer": layer, "max_abs": 0.0, "case": None,
                                                        "above_float16_max": 0, "above_half_float16_max": 0})
            peak = float(value.max())
            if peak > row["max_abs"]:
                row.update(max_abs=peak, case=self.case)
            row["above_float16_max"] += int((value > FLOAT16_MAX).sum())
            row["above_half_float16_max"] += int((value > FLOAT16_MAX / 2).sum())

        return note

    @contextlib.contextmanager
    def watching(self, model):
        text = model.model.language_model
        handles = []
        try:
            for index, layer in enumerate(text.layers):
                handles += [layer.register_forward_pre_hook(self.hook("residual_input", index)),
                            layer.register_forward_hook(self.hook("residual_output", index)),
                            layer.self_attn.register_forward_hook(self.hook("attention_output", index)),
                            layer.mlp.down_proj.register_forward_pre_hook(self.hook("mlp_intermediate", index))]
            handles += [text.norm.register_forward_pre_hook(self.hook("final_norm_input", None)),
                        text.norm.register_forward_hook(self.hook("final_norm_output", None)),
                        model.lm_head.register_forward_hook(self.hook("logits_before_softcap", None))]
            yield self
        finally:
            for handle in handles:
                handle.remove()


def cast_statistics(model):
    import torch

    rows, seen = [], set()
    for name, tensor in model.state_dict(keep_vars=True).items():
        if id(tensor) in seen or not tensor.is_floating_point():
            continue
        seen.add(id(tensor))
        flat = tensor.detach().reshape(-1)
        row = {"name": name, "dtype": str(tensor.dtype), "elements": flat.numel(), "changed": 0, "underflow": 0,
               "non_finite": 0, "above_float16_max": 0, "max_abs_error": 0.0}
        for start in range(0, flat.numel(), CAST_BLOCK):
            exact = flat[start:start + CAST_BLOCK].float()
            half = exact.to(torch.float16)
            back, finite = half.float(), torch.isfinite(half)
            row["changed"] += int((back != exact).sum())
            row["underflow"] += int(((half == 0) & (exact != 0)).sum())
            row["non_finite"] += int((~finite).sum())
            row["above_float16_max"] += int((exact.abs() > FLOAT16_MAX).sum())
            row["max_abs_error"] = max(row["max_abs_error"], float(torch.where(finite, (back - exact).abs(), 0).max()))
        rows.append(row)
    totals = {key: sum(row[key] for row in rows) for key in ("elements", "changed", "underflow", "non_finite",
                                                            "above_float16_max")}
    return {"tensors": len(rows), **totals, "max_abs_error": max(row["max_abs_error"] for row in rows), "rows": rows}


def load(snapshot, dtype, device, config=None):
    import torch
    import transformers

    options = {"config": config} if config is not None else {}
    return getattr(transformers, reference.MODEL_CLASS).from_pretrained(
        str(snapshot), dtype=getattr(torch, dtype), device_map=device, output_loading_info=True, **options)


def loaded(model, info, device):
    parameters = list(model.parameters())
    return {"class": type(model).__name__, "info": {key: sorted(map(str, value)) for key, value in info.items()},
            "attn_implementation": model.config._attn_implementation, "parameters": sum(p.numel() for p in parameters),
            "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
            "parameter_devices": sorted({str(p.device) for p in parameters}), "training": model.training,
            "embed_scale": float(model.model.language_model.embed_tokens.embed_scale),
            "after_load": reference.memory(device)}


def clean(loading, dtype, device, unexpected=()):
    info = loading["info"]
    return set(info.get("unexpected_keys", [])) == set(unexpected) \
        and not any(values for key, values in info.items() if key != "unexpected_keys") \
        and loading["parameter_dtypes"] == [f"torch.{dtype}"] and not loading["training"] \
        and all(name.startswith(device) for name in loading["parameter_devices"])


def save(output, case, values, seconds):
    import numpy as np

    folder = Path(output) / "cases"
    folder.mkdir(parents=True, exist_ok=True)
    tokens, array = folder / f"{case['id']}.tokens.json", folder / f"{case['id']}.logits.npy"
    write_json(tokens, {"id": case["id"], "input_tokens": case["input_tokens"], "output_tokens": case["output_tokens"]})
    np.save(array, values)
    inputs, outputs = len(case["input_tokens"]), len(case["output_tokens"])
    return {"id": case["id"], "kind": case["kind"], "input_tokens": inputs, "output_tokens": outputs,
            "saved_positions": len(values), "first_scoring_row": inputs - 1, "scoring_positions": outputs,
            "non_finite": reference.non_finite(values), "seconds": seconds,
            "files": {"tokens": reference.file_record(tokens),
                      "logits": {**reference.file_record(array), "dtype": str(values.dtype), "shape": list(values.shape)}}}


def common(contract, ready, verified, source):
    return {"model": {"id": contract["model_id"], "revision": contract["revision"],
                      "contract_sha256": ready["contract_sha256"],
                      "weights_sha256": {name: entry["sha256"] for name, entry in verified.items()
                                         if name.endswith(".safetensors")},
                      "verified_before_loading": verified},
            "environment": {"python": ready["environment"]["python"], "packages": ready["environment"]["packages"],
                            "os_build": ready["environment"]["os_build"]},
            "device": contract["reference"]["device"], "source": {"reference_sha256": source}}


def logits_stage(args, contract, ready):
    import torch

    started = time.perf_counter()
    source, cases = forced_inputs(args.reference, contract["config"]["vocab_size"])
    verified = reference.verify_inputs(args.snapshot, contract)
    device = contract["reference"]["device"]
    dtype = "bfloat16" if args.stage == "anchor" else "float16"
    hashed = time.perf_counter()
    model, info = load(args.snapshot, dtype, device)
    model.eval()
    loading = loaded(model, info, device)
    ready_at = time.perf_counter()
    record = {"schema": SCHEMA, "stage": args.stage, **common(contract, ready, verified, source), "loading": loading}
    watch, records, peaks = Ranges(), [], []
    if args.stage == "anchor":
        weights = scaled(resident(model, args.weights), embed_scale(model, args.embed_scale))
        record["compute"] = {"weights": args.weights, "compute": "float32", "embed_scale": float(weights(SCALE)),
                             "attention_rows": ROWS}
        record["cast"] = cast_statistics(model) if args.weights == "float16" else None
        run, scope = (lambda case: logits(model, case["input_tokens"] + case["output_tokens"], weights, device),
                      watch.watching(model))
    else:
        record["compute"] = {"weights": "float16", "compute": "float16", "embed_scale": loading["embed_scale"],
                             "attention_rows": None}
        run, scope = (lambda case: reference.forced(model, case["input_tokens"], case["output_tokens"], device)[0],
                      contextlib.nullcontext())
    computed = time.perf_counter()
    with torch.no_grad(), scope:
        for case in cases:
            watch.case, began = case["id"], time.perf_counter()
            values = run(case)
            records.append(save(args.output, case, values, time.perf_counter() - began))
            peaks.append(reference.memory(device))
            print(case["id"], len(values), f"{records[-1]['seconds']:.2f}s", flush=True)
    record["checks"] = {"loading_clean": clean(loading, dtype, device)}
    if args.stage == "anchor":
        record["checks"]["finite"] = all(case["non_finite"] == 0 for case in records)
        record["ranges"] = [{**row, "max_abs": reference.finite(row["max_abs"])} for row in watch.rows.values()]
    record.update(cases=records, resources={"cases": peaks},
                  timing={"input_hash_s": hashed - started, "load_s": ready_at - hashed, "cast_s": computed - ready_at,
                          "cases_s": time.perf_counter() - computed})
    return record


def layers_stage(args, contract, ready):
    import torch
    import transformers

    started = time.perf_counter()
    source, cases = forced_inputs(args.reference, contract["config"]["vocab_size"])
    verified = reference.verify_inputs(args.snapshot, contract)
    device = contract["reference"]["device"]
    config = transformers.AutoConfig.from_pretrained(str(args.snapshot))
    text = config.get_text_config()
    text.num_hidden_layers, text.layer_types = LAYERS, text.layer_types[:LAYERS]
    index, found, size = read(Path(args.snapshot) / "model.safetensors.index.json")
    if (size, found) != tuple(contract["files"]["model.safetensors.index.json"][key] for key in ("bytes", "sha256")):
        raise ValueError("model.safetensors.index.json differs from the contract")
    unexpected = [name for name in index["weight_map"] if (match := LAYER_KEY.match(name)) and int(match[1]) >= LAYERS]
    hashed = time.perf_counter()
    model, info = load(args.snapshot, "bfloat16", device, config)
    model.eval()
    loading = loaded(model, info, device)
    ready_at = time.perf_counter()
    with torch.no_grad():
        results = agreement(model, [case["input_tokens"] + case["output_tokens"] for case in cases], device)
    rows = [{"id": case["id"], **result} for case, result in zip(cases, results)]
    differences = [row["max_abs_difference"] for row in rows]
    return {"schema": LAYERS_SCHEMA, "stage": args.stage, **common(contract, ready, verified, source),
            "layers": LAYERS, "layer_types": list(text.layer_types), "limit": LIMIT, "attention_rows": ROWS,
            "checks": {"loading_clean": clean(loading, "bfloat16", device, unexpected),
                       "finite": all(not any(row["non_finite"].values()) for row in rows),
                       "agreement": None not in differences and max(differences) <= LIMIT},
            "cases": rows, "loading": loading, "resources": {"cases": [reference.memory(device)]},
            "timing": {"input_hash_s": hashed - started, "load_s": ready_at - hashed,
                       "cases_s": time.perf_counter() - ready_at}}


def monitored(output, name, work, device):
    path = Path(output) / name
    if path.exists():
        raise FileExistsError(f"{name} must be written to a new directory")
    reasons = []
    monitor = reference.ReferenceMonitor(Path(output) / "monitor.tsv", [output], [], reasons.append)
    started = time.perf_counter()

    def fail(reason):
        write_json(Path(output) / "failure.json", {"schema": SCHEMA, "passed": False, "abort": reason,
                                                   "elapsed_s": time.perf_counter() - started,
                                                   "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                                   "lifetime_peak_footprint_bytes": reference.footprint()[1],
                                                   "monitor": monitor.finish()})
        return 1

    try:
        monitor.start()
        if not monitor.ready.wait(reference.FIRST_SAMPLE_TIMEOUT_S):
            return fail(reasons[0] if reasons else f"no readable monitor sample within {reference.FIRST_SAMPLE_TIMEOUT_S} s")
        record = work()
        observed = monitor.finish()
        if reasons:
            return fail(reasons[0])
        reference.finalize(record, observed, reference.memory(device))
        record["timing"]["total_s"] = time.perf_counter() - started
        write_json(path, record)
    except KeyboardInterrupt:
        if not reasons:
            raise
        return fail(reasons[0])
    return 0 if record["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description="Record the FP32 anchor and FP16 diagnostics of Gemma 4 31B.")
    stages = parser.add_subparsers(dest="stage", required=True)
    for name in ("layers", "anchor", "half"):
        stage = stages.add_parser(name)
        for option in ("model", "snapshot", "prepare", "reference", "output"):
            stage.add_argument(f"--{option}", type=Path, required=True)
        if name == "anchor":
            stage.add_argument("--weights", choices=("bfloat16", "float16"), required=True)
            stage.add_argument("--embed-scale", choices=("bfloat16", "float32", "float16"), required=True)
    args = parser.parse_args()
    contract = load_contract(args.model)
    ready = read_json(args.prepare / "ready.json")
    if ready.get("ready") is not True or ready.get("contract_sha256") != sha256(args.model):
        raise ValueError("Run prepare with this contract first")
    work, name = (layers_stage, "layers.json") if args.stage == "layers" else (logits_stage, "reference.json")
    return monitored(args.output, name, lambda: work(args, contract, ready), contract["reference"]["device"])


if __name__ == "__main__":
    sys.exit(main())
