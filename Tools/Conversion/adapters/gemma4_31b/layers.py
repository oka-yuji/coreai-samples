import argparse
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import random
import re
import resource
import sys
import time

from ...storage import digest, read_json, write_json
from ..pinned import sha256
from .compare import BLOCK_ROWS, CONTRACT, recorded
from .prepare import load_contract
from .reference import LOADED_FILES, file_record, footprint, verify_inputs

SCHEMA = "gemma4-31b-layer-outputs/1"
COMPARISON = "gemma4-31b-layer-comparison/2"
CONTEXT_SCHEMA = "gemma4-31b-context-logits/1"
PREFIX = "model.language_model."
LAYER = re.compile(r"layers\.(\d+)\.")
TAP_LIST = re.compile(r"[1-9][0-9]*(,[1-9][0-9]*)*")
OUTPUTS = ("logits", "hidden_states", "tap_hidden_states")
RATIO = 1e-3
IMPLEMENTATIONS = ("original", "class")
INPUTS = {"short": (26, 0), "long": (1400, 1)}
CONTEXT = {"context": (16896, 2)}
REFERENCE_CHUNK = 1024
CONDITIONS = (("short-c1", "short", 26, 0, 1), ("short-c26", "short", 26, 0, 26),
              ("window-c1", "context", 1300, 0, 1), ("window-c2", "context", 1300, 0, 2),
              ("window-c4", "context", 1300, 0, 4), ("window-c8", "context", 1300, 0, 8),
              ("prefix-c1", "context", 1300, 1000, 1), ("capacity-2048", "context", 2056, 2040, 4),
              ("capacity-4096", "context", 4104, 4088, 8), ("capacity-8192", "context", 8200, 8184, 2),
              ("capacity-16384", "context", 16392, 16376, 4), ("tail", "context", 16896, 16832, 1))
CHUNKS = (1, 2, 4, 8)
CRITERIA = {"short": {"max_abs_logit": 1e-3},
            "long": {"max_abs_logit": 5e-3, "max_probability_error": 1e-6, "top1_mismatches": 0}}
WEIGHTS = "every checkpoint tensor rounded to float16, computed in float32 on the CPU"
PACKAGES = ("numpy", "safetensors", "torch", "transformers")
CONVERTER = ("coreai-core", "coreai-models", "coreai-opt", "coreai-torch")


def check_layers(config, layers):
    kinds = config["layer_pattern"] * (config["num_hidden_layers"] // len(config["layer_pattern"]))
    if type(layers) is not int or not 1 <= layers <= len(kinds) or kinds[layers - 1] != "full_attention":
        raise ValueError(f"The first {layers} layers must end with a full-attention layer")
    return kinds[:layers]


def check_taps(text, layers):
    taps = tuple(map(int, text.split(","))) if isinstance(text, str) and TAP_LIST.fullmatch(text) else ()
    if not taps or type(layers) is not int or list(taps) != sorted(set(taps)) or taps[-1] > layers:
        raise ValueError(f"The tap layers must be ascending distinct layer counts from 1 to {layers}: {text!r}")
    return taps


def joined(taps):
    return ",".join(map(str, taps)) if type(taps) is list and all(type(tap) is int for tap in taps) else None


def tokens(name, config):
    length, seed = {**INPUTS, **CONTEXT}[name]
    rng, bos = random.Random(seed), config["bos_token_id"]
    return [bos] + [rng.randrange(bos + 1, config["vocab_size"]) for _ in range(length - 1)]


def kept_rows(name):
    return sorted({row for _, source, length, start, _ in CONDITIONS if source == name for row in range(start, length)})


def calls(length, window):
    bounds, sizes = [[0, min(length, window)]], itertools.cycle(CHUNKS)
    while bounds[-1][1] < length:
        start = bounds[-1][1]
        bounds.append([start, min(start + next(sizes), length)])
    return bounds


def rounded_scale(hidden_size):
    import torch

    return float(torch.tensor(hidden_size ** 0.5, dtype=torch.float32).to(torch.bfloat16))


def checkpoint(snapshot, layers):
    import torch
    from safetensors import safe_open

    files = {}
    for name, file in read_json(Path(snapshot) / "model.safetensors.index.json")["weight_map"].items():
        layer = LAYER.match(name, len(PREFIX))
        if name.startswith(PREFIX) and not (layer and int(layer[1]) >= layers):
            files.setdefault(file, []).append(name)
    state = {}
    for file, names in sorted(files.items()):
        with safe_open(str(Path(snapshot) / file), framework="pt") as handle:
            for name in names:
                state["model." + name[len(PREFIX):]] = handle.get_tensor(name).to(torch.float16).to(torch.float32)
    return state


def original_model(snapshot, layers):
    import torch
    from transformers.models.gemma4 import modeling_gemma4
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    text = dict(read_json(Path(snapshot) / "config.json")["text_config"])
    text.update(num_hidden_layers=layers, layer_types=text["layer_types"][:layers])
    config = Gemma4TextConfig(**text)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = modeling_gemma4.Gemma4ForCausalLM(config)
    missing, unexpected = model.load_state_dict(checkpoint(snapshot, layers), strict=False, assign=True)
    if unexpected or missing != ["lm_head.weight"]:
        raise ValueError(f"The checkpoint does not fit the original model: missing {missing}, unexpected {unexpected}")
    model.model.rotary_emb = modeling_gemma4.Gemma4TextRotaryEmbedding(config)
    model.model.embed_tokens.embed_scale = torch.tensor(rounded_scale(config.hidden_size))
    model.lm_head.weight = model.model.embed_tokens.weight
    if any(tensor.is_meta for tensor in [*model.parameters(), *model.buffers()]):
        raise ValueError("Tensors of the original model were not loaded")
    return model.eval(), float(model.model.embed_tokens.embed_scale), list(config.layer_types)


def class_model(snapshot, layers, taps, contract):
    import coreai_models
    import torch

    from ...local import check_checkout
    from .model import Gemma4ForCausalLM

    check_checkout(Path(coreai_models.__file__).resolve().parents[3], contract["upstream_revision"])
    model = Gemma4ForCausalLM.from_hf_memory_efficient(str(snapshot), target_dtype=torch.float16, num_layers=layers,
                                                      hf_config_attr="text_config", hf_state_dict_prefix=PREFIX)
    model.tap_layers = taps
    return model.float().eval(), model.embed_scale, list(model.config.layer_types)


def outputs(model, implementation, ids, bounds, taps):
    import torch

    with torch.no_grad():
        if implementation == "original":
            last = []
            hook = model.model.layers[-1].register_forward_hook(lambda module, args, output: last.append(output))
            try:
                output = model(input_ids=torch.tensor([ids]), position_ids=torch.arange(len(ids)).unsqueeze(0),
                               use_cache=False, output_hidden_states=True)
            finally:
                hook.remove()
            columns = [*output.hidden_states[:-1], last[0]]
            found = (output.logits, output.hidden_states[-1], torch.cat([columns[tap] for tap in taps], dim=-1))
        else:
            states = model.states(len(ids))
            parts = [model(torch.tensor([ids[start:end]], dtype=torch.int32),
                           torch.arange(end, dtype=torch.int32).unsqueeze(0), *states) for start, end in bounds]
            found = [torch.cat([part[index] for part in parts], dim=1) for index in range(len(OUTPUTS))]
    return {name: tensor[0].float().numpy() for name, tensor in zip(OUTPUTS, found)}


def run(implementation, contract_path, snapshot, layers, taps, output):
    contract = load_contract(contract_path)
    config = contract["config"]
    check_layers(config, layers)
    taps = check_taps(taps, layers)
    import numpy as np

    output.mkdir(parents=True)
    started = time.perf_counter()
    verified = verify_inputs(snapshot, contract)
    hashed = time.perf_counter()
    model, scale, layer_types = (original_model(snapshot, layers) if implementation == "original"
                                 else class_model(snapshot, layers, taps, contract))
    loaded = time.perf_counter()
    inputs = {}
    for name in INPUTS:
        ids = tokens(name, config)
        bounds = [[0, len(ids)]] if implementation == "original" else calls(len(ids), config["sliding_window"])
        begin = time.perf_counter()
        arrays = outputs(model, implementation, ids, bounds, taps)
        inputs[name] = {"positions": len(ids), "tokens_sha256": digest(ids), "calls": bounds,
                        "seconds": time.perf_counter() - begin}
        for key, array in arrays.items():
            path = output / f"{name}.{key}.npy"
            np.save(path, array)
            inputs[name][key] = {**file_record(path), "dtype": str(array.dtype), "shape": list(array.shape),
                                 "non_finite": int(array.size - np.isfinite(array).sum())}
    packages = PACKAGES + (CONVERTER if implementation == "class" else ())
    record = {"schema": SCHEMA, "implementation": implementation, "layers": layers, "layer_types": layer_types,
              "tap_layers": list(taps),
              "model": {"id": contract["model_id"], "revision": contract["revision"],
                        "contract_sha256": sha256(contract_path), "verified_before_loading": verified},
              "weights": WEIGHTS, "embed_scale": scale,
              "upstream_revision": contract["upstream_revision"] if implementation == "class" else None,
              "versions": {name: importlib.metadata.version(name) for name in packages}, "inputs": inputs,
              "timing": {"input_hash_s": hashed - started, "load_s": loaded - hashed,
                         "total_s": time.perf_counter() - started},
              "memory": {"max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         "lifetime_peak_footprint_bytes": footprint()[1]}}
    write_json(output / "record.json", record)
    return record


def kept_logits(model, ids, keep, size=REFERENCE_CHUNK):
    import numpy as np
    import torch

    blocks, bounds, cache = [], [], None
    with torch.no_grad():
        for start in range(0, len(ids), size):
            end = min(start + size, len(ids))
            rows = torch.tensor([row - start for row in keep if start <= row < end], dtype=torch.long)
            output = model(input_ids=torch.tensor([ids[start:end]]), position_ids=torch.arange(start, end).unsqueeze(0),
                           past_key_values=cache, use_cache=True, logits_to_keep=rows)
            cache = output.past_key_values
            blocks.append(output.logits[0].float().numpy())
            bounds.append([start, end])
    return np.concatenate(blocks), bounds


def context(contract_path, snapshot, layers, output):
    if os.path.lexists(output):
        raise FileExistsError(f"The reference must be written to a new folder: {output}")
    contract = load_contract(contract_path)
    config = contract["config"]
    check_layers(config, layers)
    import numpy as np

    output.mkdir(parents=True)
    started = time.perf_counter()
    verified = verify_inputs(snapshot, contract)
    hashed = time.perf_counter()
    model, scale, layer_types = original_model(snapshot, layers)
    loaded = time.perf_counter()
    inputs = {}
    for name in ("short", "context"):
        ids, rows = tokens(name, config), kept_rows(name)
        begin = time.perf_counter()
        array, bounds = kept_logits(model, ids, rows)
        path = output / f"{name}.npy"
        np.save(path, array)
        inputs[name] = {"positions": len(ids), "tokens_sha256": digest(ids), "rows": rows, "calls": bounds,
                        "seconds": time.perf_counter() - begin,
                        "logits": {**file_record(path), "dtype": str(array.dtype), "shape": list(array.shape),
                                   "non_finite": int(array.size - np.isfinite(array).sum())}}
    record = {"schema": CONTEXT_SCHEMA, "implementation": "original", "layers": layers, "layer_types": layer_types,
              "model": {"id": contract["model_id"], "revision": contract["revision"],
                        "contract_sha256": sha256(contract_path), "verified_before_loading": verified},
              "weights": WEIGHTS, "embed_scale": scale, "chunk": REFERENCE_CHUNK,
              "versions": {name: importlib.metadata.version(name) for name in PACKAGES}, "inputs": inputs,
              "timing": {"input_hash_s": hashed - started, "load_s": loaded - hashed,
                         "total_s": time.perf_counter() - started},
              "memory": {"max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         "lifetime_peak_footprint_bytes": footprint()[1]}}
    write_json(output / "record.json", record)
    return record


def softmax(rows):
    import numpy as np

    weights = np.exp(rows - rows.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def differences(first, second):
    import numpy as np

    logit = probability = 0.0
    mismatches, non_finite = 0, [0, 0]
    for start in range(0, len(first), BLOCK_ROWS):
        x, y = (np.asarray(array[start:start + BLOCK_ROWS], dtype=np.float64) for array in (first, second))
        non_finite = [count + int(block.size - np.isfinite(block).sum()) for count, block in zip(non_finite, (x, y))]
        logit = float(np.maximum(logit, np.max(np.abs(x - y))))
        probability = float(np.maximum(probability, np.max(np.abs(softmax(x) - softmax(y)))))
        mismatches += int((x.argmax(-1) != y.argmax(-1)).sum())
    finite = not any(non_finite)
    return {"rows": len(first), "max_abs_logit": logit if finite else None,
            "max_probability_error": probability if finite else None, "top1_mismatches": mismatches,
            "non_finite": dict(zip(IMPLEMENTATIONS, non_finite))}


def same_file(first, second):
    return first.exists() and second.exists() and first.samefile(second)


def below(value, limit):
    return value is not None and (value < limit if isinstance(limit, float) else value <= limit)


def ratios(model_class, original):
    import numpy as np

    x, y = (np.asarray(array, dtype=np.float64) for array in (model_class, original))
    difference, scale = np.abs(x - y).max(axis=-1), np.abs(y).max(axis=-1)
    usable = bool(np.isfinite(x).all() and np.isfinite(y).all() and (scale > 0).all())
    return {"largest_ratio": float((difference / scale).max()) if usable else None,
            "passed": usable and bool((difference < RATIO * scale).all())}


def compare(original, model_class, contract_path=CONTRACT):
    import numpy as np

    contract = load_contract(contract_path)
    config = contract["config"]
    folders = [Path(original), Path(model_class)]
    records = [read_json(folder / "record.json") for folder in folders]
    model = {"id": contract["model_id"], "revision": contract["revision"], "contract_sha256": sha256(contract_path),
             "verified_before_loading": {name: entry for name, entry in contract["files"].items()
                                         if name in LOADED_FILES or name.endswith(".safetensors")}}
    problems = [f"the {kind} run holds a record of the {record.get('implementation')} implementation"
                for kind, record in zip(IMPLEMENTATIONS, records) if record.get("implementation") != kind]
    recorded_taps = []
    for kind, record in zip(IMPLEMENTATIONS, records):
        if record.get("model") != model:
            problems.append(f"the {kind} run recorded another model or contract")
        try:
            if record.get("layer_types") != check_layers(config, record.get("layers")):
                problems.append(f"the {kind} run recorded layer types that differ from the contract")
            recorded_taps.append(check_taps(joined(record.get("tap_layers")), record.get("layers")))
        except ValueError as error:
            problems.append(f"the {kind} run: {error}")
    problems += [f"the runs differ in {key}" for key in ("model", "layers", "layer_types", "tap_layers")
                 if records[0].get(key) != records[1].get(key)]
    taps = recorded_taps[0] if len(recorded_taps) == 2 and recorded_taps[0] == recorded_taps[1] else ()
    if same_file(*(folder / "record.json" for folder in folders)):
        problems.append("both runs read the same record.json")
    hidden = config["hidden_size"]
    widths = {"logits": config["vocab_size"], "hidden_states": hidden, "tap_hidden_states": hidden * len(taps)}
    results = {}
    for name, criteria in CRITERIA.items():
        ids = tokens(name, config)
        entries = [(record.get("inputs") or {}).get(name) or {} for record in records]
        found = [f"{name}: the {kind} run did not read the {name} input" for kind, entry in zip(IMPLEMENTATIONS, entries)
                 if (entry.get("positions"), entry.get("tokens_sha256")) != (len(ids), digest(ids))]
        arrays = {}
        for output, width in widths.items():
            paths = [folder / f"{name}.{output}.npy" for folder in folders]
            if same_file(*paths):
                found.append(f"{name}: both runs read the same {output} file")
            for kind, path, entry in zip(IMPLEMENTATIONS, paths, entries):
                if not recorded(path, entry.get(output)):
                    found.append(f"{name}: the {kind} {output} differ from the size and SHA-256 in their record")
                    continue
                array = np.load(path, mmap_mode="r")
                if array.dtype != np.float32 or array.shape != (len(ids), width) or not width:
                    found.append(f"{name}: the {kind} {output} are {array.dtype} {list(array.shape)}, "
                                 f"not float32 {[len(ids), width]}")
                arrays.setdefault(output, []).append(array)
        metrics, compared = ({}, {}) if found else (differences(*arrays["logits"]), {
            "hidden_states": ratios(arrays["hidden_states"][1], arrays["hidden_states"][0]),
            "tap_hidden_states": {str(tap): ratios(*(array[:, index * hidden:(index + 1) * hidden]
                                                     for array in reversed(arrays["tap_hidden_states"])))
                                  for index, tap in enumerate(taps)}})
        if any((metrics.get("non_finite") or {}).values()):
            found.append(f"{name}: non-finite logits {metrics['non_finite']}")
        checks = {key: not found and below(metrics.get(key), limit) for key, limit in criteria.items()}
        checks.update(hidden_states=not found and compared["hidden_states"]["passed"],
                      tap_hidden_states=not found and all(entry["passed"] for entry in compared["tap_hidden_states"].values()))
        results[name] = {"criteria": criteria, **metrics, "outputs": compared, "checks": checks}
        problems += found
    return {"schema": COMPARISON, "problems": problems,
            "passed": not problems and all(all(result["checks"].values()) for result in results.values()),
            "contract_sha256": model["contract_sha256"] if all(record.get("model") == model for record in records) else None,
            "layers": records[0].get("layers") if taps and records[0].get("layers") == records[1].get("layers") else None,
            "tap_layers": list(taps) or None, "ratio_limit": RATIO,
            "records_sha256": [sha256(folder / "record.json") for folder in folders], "inputs": results}


def main():
    parser = argparse.ArgumentParser(description="Logits, final hidden states and tap hidden states of the first layers "
                                                 "of Gemma 4 31B in float32 and their comparison between the original "
                                                 "implementation and the model class.")
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--implementation", choices=IMPLEMENTATIONS, required=True)
    runner.add_argument("--layers", type=int, required=True)
    runner.add_argument("--taps", required=True, help="Ascending distinct counts of layers, separated by commas.")
    for name in ("model", "snapshot", "output"):
        runner.add_argument(f"--{name}", type=Path, required=True)
    checker = commands.add_parser("compare")
    for name in ("original", "model-class"):
        checker.add_argument(f"--{name}", type=Path, required=True)
    contexts = commands.add_parser("context", help="FP32 logits of the original implementation on the rows that the "
                                                   "twelve conditions of the Core AI comparison score.")
    contexts.add_argument("--layers", type=int, required=True)
    for name in ("model", "snapshot", "output"):
        contexts.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command in ("run", "context"):
        record = (run(args.implementation, args.model, args.snapshot, args.layers, args.taps, args.output)
                  if args.command == "run" else context(args.model, args.snapshot, args.layers, args.output))
        print({name: entry["logits"]["shape"] for name, entry in record["inputs"].items()}, record["timing"],
              record["memory"], flush=True)
        return 0
    result = compare(args.original, args.model_class)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
