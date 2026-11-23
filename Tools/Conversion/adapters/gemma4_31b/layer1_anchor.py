import argparse
import hashlib
import os
from pathlib import Path
import re
import sys
import time

from Tools.LLMEvaluation import items as evaluation, layer1
from ...storage import canonical, write_json
from ..jev27b.monitor import swap_used
from . import anchor, layers, reference
from .prepare import load_contract

SCHEMA = "gemma4-31b-layer1-anchor/1"
AGREEMENT = "gemma4-31b-layer1-agreement/1"
LOGITS = "gemma4-31b-layer1-logits/1"
ROLE, SET = "anchor", "primary"
ITEM = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
WEIGHTS = {"float16": "every checkpoint tensor rounded to float16",
           "8bit": ("the quantized Linear weights restored from the quantized tensors {file} of the export {export}, "
                    "every other checkpoint tensor rounded to float16")}
PASSING = ("anchor_cpu", "original_cpu")
RECORDED = (("anchor_run", "anchor_cpu"), ("original_eager_cpu", "original_cpu"), ("original_run", "original_cpu"),
            ("anchor_run", "original_run"))


def frozen():
    data = layer1.file_bytes(evaluation.ROOT / evaluation.FILES["contract"])
    contract = layer1.strict(data)
    return {"version": contract["version"], "sha256": hashlib.sha256(data).hexdigest()}, contract["layer1"]


def record_problems(record, vocabulary, rule):
    return layer1.inconsistencies(record, rule) or [text for text, failed in (
        ("its id is not an item id", ITEM.fullmatch(record["id"]) is None),
        ("a token is outside the vocabulary", max(record["tokens"]) >= vocabulary)) if failed]


def forced(folder, vocabulary):
    data, summary_data = (layer1.file_bytes(Path(folder) / name) for name in ("forced.jsonl", "forced.json"))
    summary, (identity, rule) = layer1.strict(summary_data), frozen()
    lines = data.split(b"\n")
    records = [layer1.strict(line) for line in (lines[:-1] if lines[-1] == b"" else lines)]
    problems = [f"forced.jsonl line {number}: {text}" for number, record in enumerate(records, 1)
                for text in record_problems(record, vocabulary, rule)]
    if not records:
        problems.append("forced.jsonl holds no record")
    elif not problems and len({record["id"] for record in records}) != len(records):
        problems.append("forced.jsonl repeats an item")
    expected = {"contract": identity, "set": SET,
                "forced": {"count": len(records), "sha256": hashlib.sha256(data).hexdigest()}}
    if not isinstance(summary, dict) or any(canonical(summary.get(key)) != canonical(value)
                                            for key, value in expected.items()):
        problems.append("forced.json is not the summary of this forced.jsonl under this contract")
    if problems:
        raise ValueError(f"The forced input is refused: {'; '.join(problems[:5])}")
    return {"sha256": expected["forced"]["sha256"], "summary_sha256": hashlib.sha256(summary_data).hexdigest()}, records


def selected(records, names):
    if names is None:
        return records
    chosen, known = names.split(","), {record["id"]: record for record in records}
    if len(set(chosen)) != len(chosen) or not set(chosen) <= set(known):
        raise ValueError(f"--items must name distinct items of the forced input: {names!r}")
    return [known[name] for name in chosen]


def model_for(snapshot, count, device):
    import transformers

    config = transformers.AutoConfig.from_pretrained(str(snapshot))
    text = config.get_text_config()
    text.num_hidden_layers, text.layer_types = count, text.layer_types[:count]
    model, info = anchor.load(snapshot, "bfloat16", device, config)
    model.eval()
    index = layer1.strict(layer1.file_bytes(Path(snapshot) / "model.safetensors.index.json"))
    unexpected = [name for name in index["weight_map"] if (match := anchor.LAYER_KEY.match(name)) and int(match[1]) >= count]
    loading = anchor.loaded(model, info, device)
    return model, loading, anchor.clean(loading, "bfloat16", device, unexpected)


def rounded(model):
    return anchor.scaled(anchor.resident(model, "float16"), anchor.embed_scale(model, "bfloat16"))


def restore_into(model, path, digests, shapes, device):
    for name, shape in sorted(shapes.items()):
        weight = layers.restore(*layers.quantized_weights(path, digests, {name: shape})[name]).to(device)
        model.get_parameter(layers.PREFIX + name.removeprefix("model.")).data = weight


def scored(model, record, weights, device, block):
    import torch

    tokens, start = record["tokens"], record["start"]
    with anchor.layerwise(model, weights, block):
        logits = model(input_ids=torch.tensor([tokens], device=device), use_cache=False,
                       attention_mask=anchor.masks(model, len(tokens), device),
                       logits_to_keep=torch.arange(start, start + record["rows"], device=device)).logits
    return logits[0].cpu().numpy()


def computed(model, record, weights, device, output):
    import numpy as np
    import torch

    began, before = time.perf_counter(), swap_used()
    values = scored(model, record, weights, device, anchor.ROWS)
    seconds = time.perf_counter() - began
    data, name = np.ascontiguousarray(values, dtype="<f4").tobytes(), f"{record['id']}.f32"
    with (output / name).open("xb") as handle:
        handle.write(data)
    memory = reference.memory(device)
    if device == "mps":
        torch.mps.empty_cache()
    return {"id": record["id"], "input_tokens": record["input_tokens"], "tokens": len(record["tokens"]),
            "start": record["start"], "rows": record["rows"], "shape": list(values.shape), "dtype": values.dtype.name,
            "non_finite": reference.non_finite(values), "seconds": seconds, "file": name, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "tokens_sha256": record["tokens_sha256"], "memory": memory,
            "swap_used_bytes": {"before": before, "after": swap_used()}}


def header(schema, args, contract, ready, verified, hashes, records, kinds):
    return {"schema": schema, "layers": args.layers, "layer_types": kinds,
            "model": {"id": contract["model_id"], "revision": contract["revision"],
                      "contract_sha256": ready["contract_sha256"], "verified_before_loading": verified},
            "environment": {key: ready["environment"][key] for key in ("python", "packages", "os_build")},
            "device": contract["reference"]["device"], "forced": {**hashes, "items": [record["id"] for record in records]},
            "items": [], "error": None, "checks": {"loading_clean": False}, "timing": {}}


def run(record, args, records):
    import torch

    device, started = record["device"], time.perf_counter()
    try:
        model, record["loading"], record["checks"]["loading_clean"] = model_for(args.snapshot, args.layers, device)
        if args.restored:
            restore_into(model, *args.restored, device)
        record["timing"]["load_s"] = time.perf_counter() - started
        weights = rounded(model)
        record["compute"]["embed_scale"] = float(weights(anchor.SCALE))
        with torch.no_grad():
            for item in records:
                record["items"].append(computed(model, item, weights, device, args.output))
                print(item["id"], len(item["tokens"]), f"{record['items'][-1]['seconds']:.2f}s", flush=True)
    except BaseException as error:
        record["error"] = type(error).__name__
    record["timing"]["items_s"] = sum(item["seconds"] for item in record["items"])
    record["checks"].update(shapes=all(item["shape"] == [item["rows"], args.vocabulary] for item in record["items"]),
                            dtypes=all(item["dtype"] == "float32" for item in record["items"]),
                            finite=all(item["non_finite"] == 0 for item in record["items"]))
    return record


def anchor_rows(args, records, device):
    import torch

    model, loading, clean = model_for(args.snapshot, args.layers, device)
    with torch.no_grad():
        weights = rounded(model)
        values = [scored(model, item, weights, device, anchor.ROWS) for item in records]
    del model, weights
    if device == "mps":
        torch.mps.empty_cache()
    return values, loading, clean


def original_rows(model, records, device, attention):
    import torch

    model.set_attn_implementation(attention)
    model.to(device)
    values = []
    with torch.no_grad():
        for item in records:
            tokens, start = item["tokens"], item["start"]
            logits = model(input_ids=torch.tensor([tokens], device=device), use_cache=False,
                           position_ids=torch.arange(len(tokens), device=device).unsqueeze(0),
                           logits_to_keep=torch.arange(start, start + item["rows"], device=device)).logits
            values.append(logits[0].float().cpu().numpy())
    return values


def agreement(record, args, records):
    import numpy as np

    device, found, started = record["device"], {}, time.perf_counter()
    variants = {"anchor_cpu": {"device": "cpu"}, "anchor_run": {"device": device},
                "original_cpu": {"device": "cpu", "attention": "sdpa"},
                "original_eager_cpu": {"device": "cpu", "attention": "eager"},
                "original_run": {"device": device, "attention": "sdpa"}}
    record.update(variants=variants, compared=list(PASSING), recorded=[list(pair) for pair in RECORDED],
                  limit=anchor.LIMIT)
    try:
        found["anchor_run"], record["loading"], clean = anchor_rows(args, records, variants["anchor_run"]["device"])
        found["anchor_cpu"], _, record["checks"]["loading_clean"] = anchor_rows(args, records,
                                                                               variants["anchor_cpu"]["device"])
        record["checks"]["loading_clean"] = clean and record["checks"]["loading_clean"]
        original, scale, _ = layers.original_model(args.snapshot, args.layers)
        record["original"] = {"weights": layers.WEIGHTS, "embed_scale": scale}
        for name in ("original_cpu", "original_eager_cpu", "original_run"):
            found[name] = original_rows(original, records, variants[name]["device"], variants[name]["attention"])
        record["timing"]["variants_s"] = time.perf_counter() - started
        for index, item in enumerate(records):
            rows = {name: values[index] for name, values in found.items()}
            difference = lambda pair: reference.finite(np.max(np.abs(rows[pair[0]] - rows[pair[1]])))
            record["items"].append({"id": item["id"], "input_tokens": item["input_tokens"], "tokens": len(item["tokens"]),
                                    "rows": item["rows"], "beyond_window": item["input_tokens"] > args.window,
                                    "shapes": {name: list(value.shape) for name, value in rows.items()},
                                    "max_abs_difference": difference(PASSING),
                                    "recorded": {" - ".join(pair): difference(pair) for pair in RECORDED},
                                    "non_finite": {name: reference.non_finite(value) for name, value in rows.items()},
                                    "memory": reference.memory(device)})
    except BaseException as error:
        record["error"] = type(error).__name__
    differences = [item["max_abs_difference"] for item in record["items"]]
    record["checks"].update(finite=not any(any(item["non_finite"].values()) for item in record["items"]),
                            agreement=None not in differences and max(differences, default=1.0) <= anchor.LIMIT,
                            beyond_window=any(item["beyond_window"] for item in record["items"]))
    return record


def finished(record, observed, final, names):
    first, peak = ((observed.get(key) or {}).get("swap_used_bytes") for key in ("first", "peak"))
    increase = peak - first if None not in (first, peak) else None
    rows = [item["memory"] for item in record["items"]] + [final]
    record["resources"] = {"final": final, "monitor": observed,
                           "swap": {"first_bytes": first, "peak_bytes": peak, "increase_bytes": increase,
                                    "limit_increase_bytes": reference.SWAP_INCREASE_LIMIT_BYTES}}
    record["checks"].update(
        items=record["error"] is None and [item["id"] for item in record["items"]] == names,
        monitor_clean=observed.get("aborts") == 0 and observed.get("failure") is None
        and observed.get("unreadable_samples") == 0 and observed.get("samples", 0) > 0,
        resources_recorded=increase is not None and all(type(row.get("max_rss_bytes")) is int
                                                        and row.get("lifetime_peak_footprint_bytes") is not None
                                                        for row in rows))
    record["passed"] = all(record["checks"].values())
    return record


def handoff(record):
    return {"schema": LOGITS, "role": ROLE, "weights": record["weights"], "forced_sha256": record["forced"]["sha256"],
            "items": [{"id": item["id"], "file": item["file"], "dtype": "float32", "shape": item["shape"],
                       "sha256": item["sha256"], "tokens_sha256": item["tokens_sha256"]} for item in record["items"]]}


def recorded(output, record):
    try:
        write_json(output / "run.json", record)
        if record["passed"] and record["schema"] == SCHEMA:
            write_json(output / "logits.json", handoff(record))
    except Exception as error:
        write_json(output / "run.json", {"schema": record["schema"], "passed": False,
                                         "error": f"the record could not be written ({type(error).__name__})"})
        return 1
    return 0 if record["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description="Write the logits of the scored rows of the layer 1 forced input that "
                                                 "the original Gemma 4 31B implementation gives in float32 with the "
                                                 "weights of a candidate, or check that computation on the first layers "
                                                 "on the CPU against the plain float32 forward of the original "
                                                 "implementation with its default sdpa attention and record the "
                                                 "differences that the device and the attention implementation make.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "agreement"):
        command = commands.add_parser(name)
        for option in ("model", "snapshot", "prepare", "forced", "output"):
            command.add_argument(f"--{option}", type=Path, required=True)
        command.add_argument("--layers", type=int, required=True)
        command.add_argument("--items", required=name == "agreement",
                             help="Distinct item ids of the forced input, separated by commas, in the order to run.")
        if name == "run":
            command.add_argument("--weights", choices=tuple(WEIGHTS), required=True)
            command.add_argument("--export", type=Path, help="The export whose quantized tensors give the 8bit weights.")
    args = parser.parse_args()
    if os.path.lexists(args.output):
        raise FileExistsError(f"The {args.command} must be written to a new folder: {args.output}")
    contract_sha256 = hashlib.sha256(layer1.file_bytes(args.model)).hexdigest()
    contract = load_contract(args.model)
    ready = layer1.strict(layer1.file_bytes(args.prepare / "ready.json"))
    if ready.get("ready") is not True or ready.get("contract_sha256") != contract_sha256:
        raise ValueError("Run prepare with this contract first")
    kinds = layers.check_layers(contract["config"], args.layers)
    args.vocabulary, args.window = contract["config"]["vocab_size"], contract["config"]["sliding_window"]
    hashes, records = forced(args.forced, args.vocabulary)
    records = selected(records, args.items)
    args.restored, quantized = (), None
    if args.command == "run" and (args.weights == "8bit") != (args.export is not None):
        raise ValueError("Give --export exactly for the weights restored from its quantized tensors")
    started = time.perf_counter()
    verified = reference.verify_inputs(args.snapshot, contract)
    hashed = time.perf_counter() - started
    if args.command == "run" and args.export is not None:
        quantized, digests = layers.quantized_export(args.export, contract, args.model, kinds)
        names = sorted({key.rsplit(".", 1)[0] for key in digests})
        shapes, path = layers.checkpoint_shapes(args.snapshot, names), args.export / layers.QUANTIZED
        for name in names:
            layers.quantized_weights(path, digests, {name: shapes[name]})
        args.restored = path, digests, shapes
        quantized["tensors"] = len(digests)
    work, schema = (run, SCHEMA) if args.command == "run" else (agreement, AGREEMENT)
    record = header(schema, args, contract, ready, verified, hashes, records, kinds)
    record["timing"]["input_hash_s"] = hashed
    if args.command == "run":
        kind = WEIGHTS[args.weights].format(**{"file": quantized["file"]["sha256"], "export": quantized["export_sha256"]}
                                            if quantized else {})
        record.update(role=ROLE, weights=(f"{contract['model_id']} at {contract['revision']}, the first {args.layers} "
                                          f"layers, the final norm and the head: {kind}, computed in float32"),
                      compute={"weights": args.weights, "compute": "float32", "attention_rows": anchor.ROWS,
                               "embed_scale": None}, quantized=quantized)
    args.output.mkdir(parents=True)

    def conclude(result, observed, began):
        finished(result, observed, reference.memory(result["device"]), [item["id"] for item in records])
        result["timing"]["total_s"] = time.perf_counter() - began
        return recorded(args.output, result)

    return reference.supervise(args.output, schema, lambda: work(record, args, records), conclude)


if __name__ == "__main__":
    sys.exit(main())
