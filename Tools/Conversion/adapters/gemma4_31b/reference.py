import argparse
import ctypes
import os
from pathlib import Path
import re
import resource
import sys
import time

from ...storage import read_json, write_json
from ..jev27b.monitor import Monitor
from ..pinned import sha256
from .prepare import load_contract, runtime_tokenizer

SCHEMA = "gemma4-31b-reference/1"
CASE_ID = re.compile(r"[a-z][a-z0-9_]*")
ROLES = ("system", "user", "assistant")
CASE_KEYS = {"id", "kind", "max_new_tokens"}
SWAP_INCREASE_LIMIT_BYTES = 2_000_000_000
MODEL_CLASS = "Gemma4ForConditionalGeneration"


class ReferenceMonitor(Monitor):
    def verdict(self):
        first = (self.first or {}).get("swap_used_bytes")
        if first is not None and self.previous is not None and self.previous - first > SWAP_INCREASE_LIMIT_BYTES:
            return f"swap rose by {self.previous - first} B, above {SWAP_INCREASE_LIMIT_BYTES} B"
        return super().verdict()


def paragraphs(spec):
    if not 1 <= spec["ask"] <= spec["count"]:
        raise ValueError("The question must ask about one of the paragraphs")
    parts = [spec["template"].format(n=n, **{key: options[(n - 1) % len(options)] for key, options in spec["values"].items()})
             for n in range(1, spec["count"] + 1)]
    return spec["separator"].join([*parts, spec["question"].format(n=spec["ask"])])


def load_fixtures(path):
    data = read_json(path)
    cases = data.get("cases") or []
    identifiers = [case.get("id") for case in cases]
    if data.get("version") != 1 or not cases or len(set(identifiers)) != len(cases) \
            or not all(isinstance(value, str) and CASE_ID.fullmatch(value) for value in identifiers):
        raise ValueError("Fixtures need version 1 and unique lowercase case IDs")
    loaded = []
    for case in cases:
        limit = case.get("max_new_tokens")
        if type(limit) is not int or limit <= 0:
            raise ValueError(f"max_new_tokens must be a positive integer: {case['id']}")
        if case.get("kind") == "raw" and set(case) == CASE_KEYS | {"text"} and isinstance(case["text"], str) and case["text"]:
            loaded.append(dict(case))
            continue
        sources = set(case) & {"messages", "paragraphs"}
        if case.get("kind") != "chat" or set(case) - sources != CASE_KEYS or len(sources) != 1:
            raise ValueError(f"A chat case needs either messages or paragraphs: {case['id']}")
        messages = case.get("messages") or [{"role": "user", "content": paragraphs(case["paragraphs"])}]
        roles = [message.get("role") for message in messages]
        if any(set(message) != {"role", "content"} or not isinstance(message["content"], str) or not message["content"]
               for message in messages) or any(role not in ROLES for role in roles) or "system" in roles[1:] \
                or roles[-1] != "user":
            raise ValueError(f"Messages must alternate up to a final user turn: {case['id']}")
        loaded.append({"id": case["id"], "kind": "chat", "max_new_tokens": limit, "messages": messages})
    return loaded


def render(tokenizer, case, bos):
    if case["kind"] == "raw":
        ids = [bos, *tokenizer(case["text"], add_special_tokens=False)["input_ids"]]
        handling = "explicit"
    else:
        options = {"add_generation_prompt": True, "enable_thinking": False}
        ids = list(tokenizer.apply_chat_template(case["messages"], tokenize=True, return_dict=True, **options)["input_ids"])
        text = tokenizer.apply_chat_template(case["messages"], tokenize=False, **options)
        if ids != tokenizer(text, add_special_tokens=False)["input_ids"]:
            raise ValueError(f"The rendered template does not tokenize to the template IDs: {case['id']}")
        handling = "template"
    if ids[0] != bos or ids.count(bos) != 1:
        raise ValueError(f"The input must start with exactly one BOS: {case['id']}")
    return ids, handling


def scoring_slice(inputs, outputs):
    return slice(inputs - 1, inputs + outputs - 1)


def stop_reason(outputs, stops, limit):
    if outputs and outputs[-1] in stops:
        return "eos"
    return "length" if len(outputs) == limit else "other"


def finite(value):
    import math

    return float(value) if math.isfinite(value) else None


def top_two(row):
    import numpy as np

    order = np.argsort(-row, kind="stable")[:2]
    return [int(index) for index in order], [finite(row[index]) for index in order]


def probabilities(row, indices):
    import numpy as np

    values = row.astype(np.float64)
    weights = np.exp(values - values.max())
    return [float(weights[index] / weights.sum()) for index in indices]


def forced_statistics(logits, outputs):
    import numpy as np

    predicted = logits.argmax(axis=-1)
    mismatches = []
    for position in np.flatnonzero(predicted != np.asarray(outputs, dtype=np.int64)):
        row = logits[position]
        indices, values = top_two(row)
        entry = {"position": int(position), "greedy": int(outputs[position]), "top2": indices, "top2_logits": values}
        if np.isfinite(row).all():
            first, second = probabilities(row, indices)
            entry.update(top2_probabilities=[first, second], probability_difference=first - second)
        mismatches.append(entry)
    return {"positions": len(outputs), "non_finite": int(logits.size - np.isfinite(logits).sum()),
            "argmax_matches": len(outputs) - len(mismatches), "mismatches": mismatches}


def step_statistics(steps, outputs):
    import numpy as np

    chosen = steps[np.arange(len(outputs)), outputs]
    tops = [top_two(row) for row in steps]
    return {"non_finite": int(steps.size - np.isfinite(steps).sum()),
            "chosen_is_max": bool(np.all(chosen == steps.max(axis=-1))),
            "top_ties": sum(values[0] is not None and values[0] == values[1] for _, values in tops),
            "steps": [{"top2": indices, "top2_logits": values} for indices, values in tops]}


def first_difference(first, second, first_steps, second_steps):
    position = next((index for index, (a, b) in enumerate(zip(first, second)) if a != b), min(len(first), len(second)))
    if first == second:
        return None
    entry = {"position": position}
    for name, steps in (("first", first_steps), ("second", second_steps)):
        if position < len(steps):
            entry[name] = dict(zip(("top2", "top2_logits"), top_two(steps[position])))
    return entry


def footprint():
    buffer = ctypes.create_string_buffer(16 + 8 * 40)
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    if library.proc_pid_rusage(os.getpid(), 4, buffer) != 0:
        return None, None
    values = memoryview(buffer.raw)[16:].cast("Q")
    return values[7], values[28]


def memory(device):
    import torch

    current, peak = footprint()
    result = {"max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "footprint_bytes": current,
              "lifetime_peak_footprint_bytes": peak}
    if device == "mps":
        result.update(mps_driver_allocated_bytes=torch.mps.driver_allocated_memory(),
                      mps_current_allocated_bytes=torch.mps.current_allocated_memory())
    return result


def greedy(model, ids, limit, stops, device):
    import torch
    from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

    class Clock(StoppingCriteria):
        def __init__(self):
            self.times = []

        def __call__(self, input_ids, scores, **kwargs):
            if input_ids.device.type == "mps":
                torch.mps.synchronize()
            self.times.append(time.perf_counter())
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    clock = Clock()
    config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=limit, eos_token_id=list(stops),
                              return_dict_in_generate=True, output_logits=True)
    inputs = torch.tensor([ids], device=device)
    started = time.perf_counter()
    result = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), generation_config=config,
                            stopping_criteria=StoppingCriteriaList([clock]))
    elapsed = time.perf_counter() - started
    outputs = result.sequences[0, len(ids):].tolist()
    steps = torch.cat(result.logits).float().cpu().numpy()
    decode = clock.times[-1] - clock.times[0] if len(clock.times) > 1 else None
    timing = {"greedy_s": elapsed, "first_token_s": clock.times[0] - started, "tokens_per_s": len(outputs) / elapsed,
              "decode_tokens_per_s": (len(outputs) - 1) / decode if decode else None}
    return outputs, steps, timing


def forced(model, ids, outputs, device):
    import torch

    sequence = torch.tensor([[*ids, *outputs]], device=device)
    started = time.perf_counter()
    logits = model(input_ids=sequence, attention_mask=torch.ones_like(sequence), use_cache=False).logits
    selected = logits[0, scoring_slice(len(ids), len(outputs))].float().cpu().numpy()
    return selected, time.perf_counter() - started


def file_record(path):
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def run_cases(model, tokenizer, cases, stops, bos, device, output):
    import numpy as np

    folder = output / "cases"
    folder.mkdir(parents=True)
    records, peaks, kept = [], [], []
    for case in cases:
        ids, handling = render(tokenizer, case, bos)
        outputs, steps, timing = greedy(model, ids, case["max_new_tokens"], stops, device)
        logits, timing["forced_s"] = forced(model, ids, outputs, device)
        statistics = step_statistics(steps, outputs)
        tokens = folder / f"{case['id']}.tokens.json"
        write_json(tokens, {"id": case["id"], "kind": case["kind"], "bos": handling, "input_tokens": ids,
                            "output_tokens": outputs, "stop_reason": stop_reason(outputs, stops, case["max_new_tokens"]),
                            "steps": statistics.pop("steps")})
        array = folder / f"{case['id']}.logits.npy"
        np.save(array, logits)
        forced_record = forced_statistics(logits, outputs)
        records.append({"id": case["id"], "kind": case["kind"], "bos": handling, "input_tokens": len(ids),
                        "output_tokens": len(outputs), "max_new_tokens": case["max_new_tokens"],
                        "stop_reason": stop_reason(outputs, stops, case["max_new_tokens"]),
                        "stop_token": outputs[-1] if outputs and outputs[-1] in stops else None,
                        "scoring_positions": forced_record["positions"],
                        "non_finite": {"forced": forced_record["non_finite"], "steps": statistics["non_finite"]},
                        "greedy_chosen_is_max": statistics["chosen_is_max"], "step_top_ties": statistics["top_ties"],
                        "forced_argmax_matches": forced_record["argmax_matches"],
                        "forced_mismatches": forced_record["mismatches"],
                        "step_vs_forced_max_abs_logit": finite(np.max(np.abs(steps - logits))) if len(outputs) else 0.0,
                        "timing": timing, "files": {"tokens": file_record(tokens),
                                                    "logits": {**file_record(array), "dtype": str(logits.dtype),
                                                               "shape": list(logits.shape)}}})
        kept.append((ids, outputs, steps))
        peaks.append(memory(device))
        print(case["id"], len(ids), len(outputs), f"{timing['greedy_s']:.2f}s", flush=True)
    for record, (ids, first, first_steps) in zip(records, kept):
        outputs, steps, timing = greedy(model, ids, record["max_new_tokens"], stops, device)
        record["rerun"] = {"tokens_match": outputs == first, "greedy_s": timing["greedy_s"],
                           "first_difference": first_difference(first, outputs, first_steps, steps)}
        print(record["id"], "rerun", outputs == first, flush=True)
    return records, peaks


def reference(args, contract, ready):
    import torch
    import transformers

    started = time.perf_counter()
    tokenizer = runtime_tokenizer(args.snapshot)
    cases = load_fixtures(args.fixtures)
    config = contract["config"]
    device = contract["reference"]["device"]
    loading_started = time.perf_counter()
    model_class = getattr(transformers, MODEL_CLASS)
    model, info = model_class.from_pretrained(str(args.snapshot), dtype=torch.bfloat16, device_map=device,
                                              output_loading_info=True)
    model.eval()
    loaded = time.perf_counter()
    parameters = list(model.parameters())
    loading = {"class": type(model).__name__, "arguments": {"dtype": "bfloat16", "device_map": device,
                                                            "output_loading_info": True},
               "info": {key: sorted(map(str, value)) for key, value in info.items()},
               "attn_implementation": model.config._attn_implementation, "parameters": sum(p.numel() for p in parameters),
               "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
               "parameter_devices": sorted({str(p.device) for p in parameters}), "training": model.training,
               "tied_embeddings": model.get_output_embeddings().weight is model.get_input_embeddings().weight,
               "after_load": memory(device)}
    with torch.no_grad():
        records, peaks = run_cases(model, tokenizer, cases, config["eos_token_id"], config["bos_token_id"], device,
                                   args.output)
    finished = time.perf_counter()
    checks = {"loading_clean": not any(loading["info"].values()) and loading["parameter_dtypes"] == ["torch.bfloat16"]
              and all(name.startswith(device) for name in loading["parameter_devices"]) and not loading["training"],
              "finite": all(not any(record["non_finite"].values()) for record in records),
              "greedy_chosen_is_max": all(record["greedy_chosen_is_max"] for record in records),
              "rerun_tokens_match": all(record["rerun"]["tokens_match"] for record in records)}
    greedy_tokens = sum(record["output_tokens"] for record in records)
    greedy_seconds = sum(record["timing"]["greedy_s"] for record in records)
    return {"schema": SCHEMA, "checks": checks,
            "model": {"id": contract["model_id"], "revision": contract["revision"], "contract_sha256": ready["contract_sha256"],
                      "weights_sha256": {name: entry["sha256"] for name, entry in ready["files"].items()
                                         if name.endswith(".safetensors")}},
            "environment": {"python": ready["environment"]["python"], "packages": ready["environment"]["packages"],
                            "os_build": ready["environment"]["os_build"]},
            "dtype": {"weights": "bfloat16", "compute": "bfloat16", "logits_saved": "float32"}, "device": device,
            "loading": loading,
            "template": {"chat": "apply_chat_template(add_generation_prompt=True, enable_thinking=False)",
                         "thinking": "disabled; the template appends an empty thought channel",
                         "bos": {"template": "rendered by the template, tokenized with add_special_tokens=False",
                                 "explicit": "prepended to text tokenized with add_special_tokens=False"}},
            "generation": {"do_sample": False, "num_beams": 1, "eos_token_id": config["eos_token_id"],
                           "forced_forward": "one forward of input + greedy output, use_cache=False"},
            "fixtures_sha256": sha256(args.fixtures), "cases": records,
            "resources": {"cases": peaks, "mps_recommended_max_bytes": torch.mps.recommended_max_memory()
                          if device == "mps" else None},
            "timing": {"tokenizer_and_fixtures_s": loading_started - started, "load_s": loaded - loading_started,
                       "cases_and_rerun_s": finished - loaded, "greedy_tokens": greedy_tokens,
                       "greedy_s": greedy_seconds, "greedy_tokens_per_s": greedy_tokens / greedy_seconds}}


def main():
    parser = argparse.ArgumentParser(description="Record BF16 MPS reference outputs of Gemma 4 31B for the fixtures.")
    for name in ("model", "snapshot", "fixtures", "prepare", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    output = args.output / "reference.json"
    if output.exists():
        raise FileExistsError("reference.json must be written to a new directory")
    contract = load_contract(args.model)
    ready = read_json(args.prepare / "ready.json")
    if ready.get("ready") is not True or ready.get("contract_sha256") != sha256(args.model):
        raise ValueError("Run prepare with this contract first")
    reasons = []
    monitor = ReferenceMonitor(args.output / "monitor.tsv", [args.output], [], reasons.append)
    started = time.perf_counter()
    monitor.start()
    try:
        record = reference(args, contract, ready)
    except KeyboardInterrupt:
        if not reasons:
            raise
        write_json(args.output / "failure.json", {"schema": SCHEMA, "passed": False, "abort": reasons[0],
                                                  "elapsed_s": time.perf_counter() - started,
                                                  "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                                  "lifetime_peak_footprint_bytes": footprint()[1],
                                                  "monitor": monitor.finish()})
        return 1
    observed = monitor.finish()
    swap = {"first_bytes": (observed["first"] or {}).get("swap_used_bytes"), "peak_bytes": observed["peak"]["swap_used_bytes"],
            "limit_increase_bytes": SWAP_INCREASE_LIMIT_BYTES}
    swap["increase_bytes"] = swap["peak_bytes"] - swap["first_bytes"] if swap["first_bytes"] is not None else None
    record["checks"]["monitor_clean"] = observed["aborts"] == 0 and swap["increase_bytes"] is not None
    record["resources"].update(final=memory(contract["reference"]["device"]), swap=swap, monitor=observed)
    record["timing"]["total_s"] = time.perf_counter() - started
    record["passed"] = all(record["checks"].values())
    write_json(output, record)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
