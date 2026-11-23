import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import stat
import sys
import time

from Tools.LLMEvaluation.layer1 import file_bytes, strict, strict_records
from Tools.LLMEvaluation.scoring import generated, partition
from ...storage import write_json
from ..pinned import sha256
from .assistant import SYNTHETIC, TARGET, assistant_config, checkpoint_shapes
from .baseline import read_inputs
from .layers import CONVERTER, PACKAGES, PREFIX, RATIO, WEIGHTS, rounded_scale, same_file
from .native import valid
from .prepare import HEX40, HEX64, REPOSITORY, load_contract
from .reference import file_record, footprint

SCHEMA = "gemma4-31b-mtp-run/1"
COMPARISON = "gemma4-31b-mtp-comparison/1"
ACCEPTANCE = "gemma4-31b-mtp-acceptance/1"
EXPORT = "gemma4-31b-mtp-export/1"
IMPLEMENTATIONS = ("original", "class")
LOGIT_LIMIT = 1e-3
INDEX = "model.safetensors.index.json"
TABLE = PREFIX + "embed_tokens.weight"
ASSISTANT_FILES = ("config.json", "model.safetensors")
ITEM = {"id": "text", "category": "text", "max_new_tokens": "count", "stop": "text", "input_tokens": "count",
        "output_tokens": "count", "teacher_tokens": "count"}
ROW = {"id": "text", "category": "text", "anchor": "count", "output_index": "count", "window": "count"}
LEVEL = {"rounds": "count", "tokens": "count", "tokens_per_round": "seconds"}
ARRAYS = {"original": ("logits", "projected", "drafts"), "class": ("logits", "projected", "drafts", "repeated")}
SAME = ("steps", "stops", "window", "layers", "model", "assistant", "records", "weights", "embed_scale", "rows")


def document(path):
    return strict(file_bytes(path))


def cut(tokens, stops):
    end = next((index for index, token in enumerate(tokens) if token in stops), None)
    return list(tokens) if end is None else list(tokens[:end + 1])


def matches(tokens, teacher):
    count = 0
    while count < len(tokens) and count < len(teacher) and tokens[count] == teacher[count]:
        count += 1
    return count


def simulate(accepted, length, steps):
    rounds = tokens = position = 0
    while position < length - 1:
        taken = min(accepted[position], steps)
        rounds, tokens, position = rounds + 1, tokens + taken + 1, position + taken + 1
    return rounds, tokens


def anchors(prompt, output):
    return sorted({prompt, prompt + output // 3, prompt + 2 * output // 3, prompt + output - 1})


def within(value, limit):
    return value is not None and value <= limit


def ratio(model_class, original):
    import numpy as np

    x, y = (np.asarray(array, dtype=np.float64) for array in (model_class, original))
    scale = float(np.abs(y).max())
    return float(np.abs(x - y).max()) / scale if np.isfinite(x).all() and np.isfinite(y).all() and scale > 0 else None


def check_steps(steps):
    if type(steps) is not int or steps < 1:
        raise ValueError(f"The number of draft steps must be a whole number from 1: {steps!r}")
    return steps


def load_assistant(path):
    contract = document(path)
    files = contract.get("files") if isinstance(contract, dict) else None
    if not isinstance(files, dict) or set(contract) != {"model_id", "revision", "files"} \
            or set(files) != set(ASSISTANT_FILES) or not isinstance(contract["model_id"], str) \
            or not REPOSITORY.fullmatch(contract["model_id"]) or not isinstance(contract["revision"], str) \
            or not HEX40.fullmatch(contract["revision"]) \
            or any(not isinstance(entry, dict) or set(entry) != {"bytes", "sha256"} or type(entry["bytes"]) is not int
                   or entry["bytes"] <= 0 or not isinstance(entry["sha256"], str) or not HEX64.fullmatch(entry["sha256"])
                   for entry in files.values()):
        raise ValueError("The assistant contract needs an owner/name ID, a full revision and the size and SHA-256 of "
                         + " and ".join(ASSISTANT_FILES))
    return contract


def verified(folder, files, names):
    found = {}
    for name in names:
        path, pin = Path(folder) / name, files[name]
        status = os.lstat(path)
        if not stat.S_ISREG(status.st_mode) or status.st_size != pin["bytes"] or sha256(path) != pin["sha256"]:
            raise ValueError(f"A file to read differs from its contract: {name}")
        found[name] = dict(pin)
    return found


def layout(config):
    pattern = config["layer_pattern"]
    kinds = pattern * (config["num_hidden_layers"] // len(pattern))
    last = {kind: max(index for index, found in enumerate(kinds) if found == kind) for kind in set(kinds)}
    return {"sliding": last["sliding_attention"], "full": last["full_attention"], "window": config["sliding_window"] + 1}


def subset_record(path):
    record = document(path)
    counts, items = (record.get(key) if isinstance(record, dict) else None for key in ("counts", "items"))
    if not isinstance(counts, dict) or set(record) != {"inputs", "outputs", "rule", "counts", "items", "total"} \
            or not isinstance(items, list) \
            or not all(isinstance(record[key], dict) and set(record[key]) == {"path", "sha256"}
                       and isinstance(record[key]["sha256"], str) and HEX64.fullmatch(record[key]["sha256"])
                       for key in ("inputs", "outputs")) \
            or not all(valid("count", count) and count > 0 for count in counts.values()) \
            or not all(isinstance(entry, dict) and set(entry) == {*ITEM, "stop_token"}
                       and all(valid(kind, entry[key]) for key, kind in ITEM.items()) for entry in items):
        raise ValueError("subset.json is not the record of a subset of the quality set")
    ids = [entry["id"] for entry in items]
    if len(set(ids)) != len(ids) or len(items) != sum(counts.values()) \
            or any(sum(entry["category"] == name for entry in items) != count for name, count in counts.items()) \
            or any(entry["id"].rsplit("-", 1)[0] != entry["category"] or not entry["id"].rsplit("-", 1)[-1].isdigit()
                   for entry in items):
        raise ValueError("subset.json does not hold its counts of items per category")
    return record


def first_per_category(record):
    return [min((entry for entry in record["items"] if entry["category"] == name),
                key=lambda entry: int(entry["id"].rsplit("-", 1)[1])) for name in record["counts"]]


def header(path, names):
    from safetensors import safe_open

    if not stat.S_ISREG(os.lstat(path).st_mode):
        raise ValueError(f"{Path(path).name} is not a regular file")
    with safe_open(str(path), framework="pt") as handle:
        present = set(handle.keys())
        return {name: (handle.get_slice(name).get_dtype(), list(handle.get_slice(name).get_shape()))
                for name in names if name in present}, present


def feature_layout(path, config, place, prompt, teacher):
    end = prompt + teacher - 1
    names = {"postnorm": "postnorm", "k_sliding": f"k{place['sliding']}", "v_sliding": f"v{place['sliding']}",
             "k_full": f"k{place['full']}", "v_full": f"v{place['full']}"}
    found, _ = header(path, names.values())
    shapes = {key: found.get(name, (None, []))[1] for key, name in names.items()}
    if len(shapes["postnorm"]) != 2 or len(shapes["k_sliding"]) != 3:
        raise ValueError(f"{path.name} does not hold the hidden states and the sliding-window keys")
    first, start = end - shapes["postnorm"][0], end - shapes["k_sliding"][1]
    sliding = [config["num_key_value_heads"], end - start, config["head_dim"]]
    full = [config["num_global_key_value_heads"], end, config["global_head_dim"]]
    expected = {"postnorm": [end - first, config["hidden_size"]], "k_sliding": sliding, "v_sliding": sliding,
                "k_full": full, "v_full": full}
    if any(found.get(names[key]) != ("BF16", shape) for key, shape in expected.items()) \
            or not 0 <= first <= prompt - 1 or not 0 <= start <= max(0, prompt - place["window"]):
        raise ValueError(f"{path.name} does not hold the BF16 features of the input and the teacher")
    return {"features": path, "names": names, "first": first, "start": start}


def items_of(contract, records, inputs, outputs, choose):
    config, place = contract["config"], layout(contract["config"])
    subset = subset_record(Path(records) / "subset.json")
    vocabulary, stops = config["vocab_size"], set(config["eos_token_id"])
    data = {"inputs": file_bytes(inputs), "outputs": file_bytes(outputs)}
    if any(hashlib.sha256(value).hexdigest() != subset[key]["sha256"] for key, value in data.items()):
        raise ValueError("The input or output records differ from the SHA-256 that subset.json records")
    sources = {entry["id"]: entry for entry in read_inputs(data["inputs"], vocabulary)}
    good, broken = partition(strict_records(data["outputs"]), vocabulary)
    produced = {entry["id"]: entry for entry in good}
    if broken or len(produced) != len(good):
        raise ValueError("The output records are malformed or repeat an id")
    found = []
    for entry in choose(subset):
        source, output = sources.get(entry["id"]), produced.get(entry["id"])
        if source is None or output is None:
            raise ValueError(f"{entry['id']}: no input or output record")
        ids, teacher = source["input_ids"], generated(output)
        if (len(ids), len(output["output_ids"]), len(teacher)) != (entry["input_tokens"], entry["output_tokens"],
                                                                   entry["teacher_tokens"]) \
                or (output["input_sha256"], output["input_tokens"]) != (source["input_sha256"], len(ids)) \
                or output["non_finite"] != 0 or "error" in output or not output["output_ids"] \
                or (output["stop"], output.get("stop_token")) != (entry["stop"], entry["stop_token"]) \
                or output["stop"] == "stop_token" and output["stop_token"] not in stops \
                or not stops.isdisjoint(output["output_ids"]):
            raise ValueError(f"{entry['id']}: the input and output records differ from subset.json")
        path = Path(records) / "tf" / "features" / f"{entry['id']}.safetensors"
        found.append({"id": entry["id"], "category": entry["category"], "input_ids": ids, "teacher": teacher,
                      "outputs": len(output["output_ids"]), **feature_layout(path, config, place, len(ids),
                                                                            len(teacher))})
    return subset, found


def loaded_features(item):
    import torch
    from safetensors import safe_open

    with safe_open(str(item["features"]), framework="pt") as handle:
        return {key: handle.get_tensor(name).to(torch.float32) for key, name in item["names"].items()}


def row_inputs(item, features, anchor, window):
    import torch

    sequence, start = item["input_ids"] + item["teacher"], max(0, anchor - window)
    hidden = features["postnorm"][anchor - 1 - item["first"]].reshape(1, 1, -1)
    sliding = tuple(features[key][None, None, :, start - item["start"]:anchor - item["start"]]
                    for key in ("k_sliding", "v_sliding"))
    full = tuple(features[key][None, None, :, :anchor] for key in ("k_full", "v_full"))
    return {"token": sequence[anchor], "position": anchor, "hidden": hidden, "window": anchor - start, "full": full,
            "every_sliding": tuple(features[key][None, :, :anchor - item["start"]] for key in ("k_sliding", "v_sliding")),
            "arguments": (torch.tensor([[sequence[anchor]]], dtype=torch.int32), hidden,
                          torch.tensor([[anchor]], dtype=torch.int32), *sliding, *full)}


def expect(path, expected, every):
    found, present = header(path, expected)
    if every and present != set(expected) or found != {name: ("BF16", list(shape)) for name, shape in expected.items()}:
        raise ValueError(f"{Path(path).name} differs from the expected BF16 tensors: missing "
                         f"{sorted(set(expected) - present)}, unexpected {sorted(present - set(expected)) if every else []}")


def checkpoint(path, expected, every):
    import torch
    from safetensors import safe_open

    expect(path, expected, every)
    with safe_open(str(path), framework="pt") as handle:
        return {name: handle.get_tensor(name).to(torch.float16).to(torch.float32) for name in expected}


def table_shard(snapshot, contract):
    files = verified(snapshot, contract["files"], (INDEX,))
    places = document(Path(snapshot) / INDEX).get("weight_map")
    shard = places.get(TABLE) if isinstance(places, dict) else None
    if not isinstance(shard, str) or not shard.endswith(".safetensors") or shard not in contract["files"]:
        raise ValueError("The index of the target does not place the embedding table in a pinned shard")
    return files, shard


def table_shape(config):
    return {TABLE: [config["vocab_size"], config["hidden_size"]]}


def table(path, config):
    return checkpoint(path, table_shape(config), False)[TABLE]


def original_drafter(assistant, table, config):
    import torch
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding, Gemma4TextScaledWordEmbedding
    from transformers.models.gemma4_assistant.configuration_gemma4_assistant import Gemma4AssistantConfig
    from transformers.models.gemma4_assistant.modeling_gemma4_assistant import Gemma4AssistantForCausalLM

    settings = Gemma4AssistantConfig(**document(Path(assistant) / "config.json"))
    settings._attn_implementation = "eager"
    with torch.device("meta"):
        model = Gemma4AssistantForCausalLM(settings)
        embed = Gemma4TextScaledWordEmbedding(config["vocab_size"], config["hidden_size"], 0,
                                              embed_scale=config["hidden_size"] ** 0.5)
    expected = {name: tensor.shape for name, tensor in model.state_dict().items() if name != "lm_head.weight"}
    missing, unexpected = model.load_state_dict(checkpoint(Path(assistant) / "model.safetensors", expected, True),
                                                strict=False, assign=True)
    if unexpected or missing != ["lm_head.weight"]:
        raise ValueError(f"The checkpoint does not fit the original drafter: missing {missing}, unexpected {unexpected}")
    model.model.rotary_emb = Gemma4TextRotaryEmbedding(settings.text_config)
    model.model.embed_tokens.embed_scale = torch.tensor(rounded_scale(settings.text_config.hidden_size))
    model.lm_head.weight = model.model.embed_tokens.weight
    embed.weight = torch.nn.Parameter(table, requires_grad=False)
    embed.embed_scale = torch.tensor(rounded_scale(config["hidden_size"]))
    if any(tensor.is_meta for tensor in [*model.parameters(), *model.buffers(), *embed.parameters(), *embed.buffers()]):
        raise ValueError("Tensors of the original drafter were not loaded")
    return model.eval(), embed


def class_drafter(assistant, table, config, steps):
    import torch

    from . import drafter

    settings = assistant_config(document(Path(assistant) / "config.json"), {key: config[key] for key in TARGET})
    with torch.device("meta"):
        model = drafter.Gemma4AssistantDrafter(settings, steps)
    expected = {name: tensor.shape for name, tensor in model.state_dict().items() if name != "target_embed.weight"}
    state = checkpoint(Path(assistant) / "model.safetensors", expected, True)
    model.load_state_dict({**state, "target_embed.weight": table}, strict=True, assign=True)
    if any(tensor.is_meta for tensor in [*model.parameters(), *model.buffers()]):
        raise ValueError("Tensors of the drafter class were not loaded")
    return model.eval()


def shared_states(inputs):
    return {"sliding_attention": inputs["every_sliding"], "full_attention": tuple(tensor[0] for tensor in inputs["full"])}


def original_drafts(model, embed, inputs, steps, stops):
    import torch
    from types import SimpleNamespace
    from transformers import GenerationConfig
    from transformers.generation.candidate_generator import SinglePositionMultiTokenCandidateGenerator

    model.generation_config = GenerationConfig(num_assistant_tokens=steps, num_assistant_tokens_schedule="constant",
                                               eos_token_id=list(stops), is_assistant=True)
    length = inputs["position"] + 1
    ids = torch.zeros(1, length, dtype=torch.long)
    ids[0, -1] = inputs["token"]
    mask = torch.ones(1, length, dtype=torch.long)
    config = GenerationConfig(eos_token_id=list(stops), max_length=length + steps + 1, pad_token_id=0)
    generator = SinglePositionMultiTokenCandidateGenerator(ids, model, embed, config, {"attention_mask": mask})
    outputs = SimpleNamespace(hidden_states=(inputs["hidden"],), shared_kv_states=shared_states(inputs))
    candidates, _ = generator.get_candidates(ids, {"attention_mask": mask}, outputs, False, 0)
    return candidates[0, length:].tolist()


def original_row(model, embed, inputs, steps, stops):
    import torch

    with torch.no_grad():
        output = model(inputs_embeds=torch.cat([embed(torch.tensor([[inputs["token"]]])), inputs["hidden"]], dim=-1),
                       attention_mask=torch.ones(1, inputs["position"] + 1, dtype=torch.long),
                       position_ids=torch.tensor([[inputs["position"]]]), shared_kv_states=shared_states(inputs),
                       use_cache=False)
        drafts = original_drafts(model, embed, inputs, steps, stops)
    return output.logits[0, 0], output.last_hidden_state[0, 0], drafts + [-1] * (steps - len(drafts))


def class_row(model, inputs):
    import torch

    arguments = inputs["arguments"]
    with torch.no_grad():
        logits, projected = model.outputs(*arguments)
        unrolled = model(*arguments)[0].tolist()
        token, hidden, repeated = arguments[0], arguments[1], []
        for _ in range(model.steps):
            step, hidden = model.outputs(token, hidden, *arguments[2:])
            token = step.argmax(dim=-1).to(torch.int32)
            repeated.append(int(token))
    return logits[0, 0], projected[0, 0], unrolled, repeated


def versions(implementation):
    return {name: importlib.metadata.version(name)
            for name in PACKAGES + (CONVERTER if implementation == "class" else ())}


def prepared(contract_path, assistant_path, snapshot, assistant_snapshot, records, inputs, outputs, choose):
    contract, assistant = load_contract(contract_path), load_assistant(assistant_path)
    files, shard = table_shard(snapshot, contract)
    files.update(verified(snapshot, contract["files"], (shard,)))
    held = verified(assistant_snapshot, assistant["files"], ASSISTANT_FILES)
    subset, items = items_of(contract, records, inputs, outputs, choose)
    config = contract["config"]
    settings = assistant_config(document(Path(assistant_snapshot) / "config.json"), {key: config[key] for key in TARGET})
    expect(Path(assistant_snapshot) / "model.safetensors", checkpoint_shapes(settings), True)
    expect(Path(snapshot) / shard, table_shape(config), False)
    sources = {"model": {"id": contract["model_id"], "revision": contract["revision"],
                         "contract_sha256": sha256(contract_path), "verified_before_loading": files},
               "assistant": {"id": assistant["model_id"], "revision": assistant["revision"],
                             "contract_sha256": sha256(assistant_path), "verified_before_loading": held},
               "records": {"subset_sha256": sha256(Path(records) / "subset.json"), "inputs_sha256": sha256(inputs),
                           "outputs_sha256": sha256(outputs),
                           "features": {item["id"]: file_record(item["features"]) for item in items}}}
    return contract, subset, items, sources, Path(snapshot) / shard


def memory():
    return {"max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "lifetime_peak_footprint_bytes": footprint()[1]}


def run(implementation, contract_path, assistant_path, snapshot, assistant_snapshot, records, inputs, outputs, steps,
        output):
    if os.path.lexists(output):
        raise FileExistsError(f"The run must be written to a new folder: {output}")
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"Unknown implementation: {implementation!r}")
    check_steps(steps)
    contract, _, items, sources, shard = prepared(contract_path, assistant_path, snapshot, assistant_snapshot, records,
                                                  inputs, outputs, first_per_category)
    output.mkdir(parents=True)
    try:
        return run_rows(implementation, assistant_snapshot, contract["config"], items, sources, shard, steps, output)
    except BaseException as error:
        write_json(output / "failure.json", {"schema": SCHEMA, "implementation": implementation,
                                             "failure": type(error).__name__})
        raise


def run_rows(implementation, assistant_snapshot, config, items, sources, shard, steps, output):
    import numpy as np

    started, place = time.perf_counter(), layout(config)
    weights = table(shard, config)
    model = (original_drafter(assistant_snapshot, weights, config) if implementation == "original"
             else class_drafter(assistant_snapshot, weights, config, steps))
    loaded = time.perf_counter()
    stops, rows, arrays = list(config["eos_token_id"]), [], {name: [] for name in ARRAYS[implementation]}
    for item in items:
        features = loaded_features(item)
        for anchor in anchors(len(item["input_ids"]), item["outputs"]):
            inputs = row_inputs(item, features, anchor, place["window"])
            values = (original_row(*model, inputs, steps, stops) if implementation == "original"
                      else class_row(model, inputs))
            for name, value in zip(ARRAYS[implementation], values):
                arrays[name].append(np.asarray(value, dtype=np.int32) if isinstance(value, list) else value.numpy())
            rows.append({"id": item["id"], "category": item["category"], "anchor": anchor,
                         "output_index": anchor - len(item["input_ids"]), "window": inputs["window"]})
        del features
    computed, files = time.perf_counter(), {}
    for name, values in arrays.items():
        array = np.stack(values)
        np.save(output / f"{name}.npy", array)
        files[name] = {**file_record(output / f"{name}.npy"), "dtype": str(array.dtype), "shape": list(array.shape),
                       "non_finite": int(array.size - np.isfinite(array).sum())}
    record = {"schema": SCHEMA, "implementation": implementation, "steps": steps, "stops": stops,
              "window": place["window"], "layers": {"sliding": place["sliding"], "full": place["full"]}, **sources,
              "weights": WEIGHTS, "embed_scale": rounded_scale(config["hidden_size"]), "rows": rows, "arrays": files,
              "versions": versions(implementation),
              "timing": {"load_s": loaded - started, "rows_s": computed - loaded, "total_s": time.perf_counter() - started},
              "memory": memory()}
    write_json(output / "record.json", record)
    return record


def loaded_run(folder, kind, problems):
    import numpy as np

    try:
        record = document(Path(folder) / "record.json")
    except Exception as error:
        problems.append(f"the {kind} run has no readable record.json ({type(error).__name__})")
        return None, {}
    if not isinstance(record, dict) or record.get("schema") != SCHEMA or record.get("implementation") != kind:
        problems.append(f"the {kind} run holds another record")
        return None, {}
    files, arrays = record.get("arrays") if isinstance(record.get("arrays"), dict) else {}, {}
    for name in ARRAYS[kind]:
        path, entry = Path(folder) / f"{name}.npy", files.get(name)
        try:
            same = isinstance(entry, dict) and not os.path.islink(path) and path.is_file() \
                and file_record(path) == {key: entry.get(key) for key in ("bytes", "sha256")}
            arrays[name] = np.load(path, allow_pickle=False) if same else None
        except Exception as error:
            problems.append(f"the {kind} {name} cannot be read ({type(error).__name__})")
            continue
        if arrays[name] is None:
            problems.append(f"the {kind} {name} differ from the size and SHA-256 in their record")
    return record, arrays


def tokens_or_padding(array, vocabulary, padded):
    for row in array.tolist():
        kept = [token for token in row if token >= 0] if padded else row
        if kept != row[:len(kept)] or any(token != -1 for token in row[len(kept):]) or not kept \
                or not all(0 <= token < vocabulary for token in kept):
            return False
    return True


def compare(original, model_class, contract_path):
    import numpy as np

    config, problems = load_contract(contract_path)["config"], []
    place = layout(config)
    expected = {"stops": config["eos_token_id"], "window": place["window"], "weights": WEIGHTS,
                "layers": {"sliding": place["sliding"], "full": place["full"]},
                "embed_scale": rounded_scale(config["hidden_size"])}
    folders = {"original": Path(original), "class": Path(model_class)}
    runs = {kind: loaded_run(folder, kind, problems) for kind, folder in folders.items()}
    records, arrays = ({kind: value[index] for kind, value in runs.items()} for index in (0, 1))
    rows, steps = [], None
    if all(records.values()):
        first = records["original"]
        problems += [f"the runs differ in {key}" for key in SAME if first.get(key) != records["class"].get(key)]
        problems += [f"the runs hold another {key} than the contract" for key, value in expected.items()
                     if first.get(key) != value]
        if same_file(folders["original"] / "record.json", folders["class"] / "record.json"):
            problems.append("both runs read the same record.json")
        if not isinstance(first.get("model"), dict) or first["model"].get("contract_sha256") != sha256(contract_path):
            problems.append("the runs were made under another contract")
        rows, steps = first.get("rows"), first.get("steps")
        if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) and set(row) == set(ROW)
                                                             and all(valid(kind, row[key]) for key, kind in ROW.items())
                                                             for row in rows):
            problems.append("the runs hold no well-formed rows")
            rows = []
        if type(steps) is not int or steps < 1:
            problems.append("the runs hold no step count")
    widths = {"logits": config["vocab_size"], "projected": config["hidden_size"], "drafts": steps, "repeated": steps}
    for kind, found in arrays.items():
        for name, array in found.items():
            wanted = np.int32 if name in ("drafts", "repeated") else np.float32
            if array is not None and (array.dtype != wanted or array.shape != (len(rows), widths[name])):
                problems.append(f"the {kind} {name} are {array.dtype} {list(array.shape)}, not "
                                f"{np.dtype(wanted)} {[len(rows), widths[name]]}")
            elif array is not None and array.dtype.kind == "f" and not np.isfinite(array).all():
                problems.append(f"the {kind} {name} hold non-finite values")
            elif array is not None and array.dtype.kind == "i" and not tokens_or_padding(array, config["vocab_size"],
                                                                                          (kind, name) == ("original",
                                                                                                           "drafts")):
                problems.append(f"the {kind} {name} hold values that are not token ids")
    if problems or not rows:
        return {"schema": COMPARISON, "passed": False, "problems": problems or ["no rows to compare"]}
    stops, results = set(records["original"]["stops"]), []
    for index, row in enumerate(rows):
        logits = [np.asarray(arrays[kind]["logits"][index], dtype=np.float64) for kind in IMPLEMENTATIONS]
        projection = ratio(arrays["class"]["projected"][index], arrays["original"]["projected"][index])
        unrolled, repeated = (arrays["class"][name][index].tolist() for name in ("drafts", "repeated"))
        drafted, kept = [token for token in arrays["original"]["drafts"][index].tolist() if token >= 0], cut(unrolled, stops)
        difference = float(np.max(np.abs(logits[0] - logits[1])))
        results.append({**row, "max_abs_logit": difference, "projection_ratio": projection,
                        "argmax_equal": int(logits[0].argmax()) == int(logits[1].argmax()),
                        "unrolled_equals_repeated": unrolled == repeated, "cut_equals_original": kept == drafted,
                        "first_step_unlike_original": None if kept == drafted else matches(kept, drafted),
                        "checks": {"max_abs_logit": within(difference, LOGIT_LIMIT), "projection": within(projection, RATIO),
                                   "unrolled_equals_repeated": unrolled == repeated}})
    return {"schema": COMPARISON, "passed": all(all(result["checks"].values()) for result in results), "problems": [],
            "criteria": {"max_abs_logit": LOGIT_LIMIT, "projection_ratio": RATIO}, "steps": steps,
            "stops": records["original"]["stops"], "window": records["original"]["window"], "rows": results,
            "summary": {"rows": len(results), "items": len({result["id"] for result in results}),
                        "max_abs_logit": max(result["max_abs_logit"] for result in results),
                        "largest_projection_ratio": max(result["projection_ratio"] for result in results),
                        **{key: sum(result[key] for result in results)
                           for key in ("argmax_equal", "unrolled_equals_repeated", "cut_equals_original")}},
            "records_sha256": {kind: sha256(folder / "record.json") for kind, folder in folders.items()}}


def draft_rows(path, item):
    record = document(path)
    rows = record.get("mtp") if isinstance(record, dict) else None
    if not isinstance(rows, list) or record.get("id") != item["id"] \
            or (record.get("P"), record.get("T")) != (len(item["input_ids"]), len(item["teacher"])):
        raise ValueError(f"{Path(path).name} is not the drafter record of {item['id']}")
    found = {}
    for row in rows:
        fixed = row.get("fixed") if isinstance(row, dict) else None
        if not isinstance(fixed, dict) or not all(valid("count", row.get(key)) for key in ("a", "j")) \
                or row["a"] != row["j"] + len(item["input_ids"]) or row["j"] in found \
                or not valid("count", fixed.get("accepted")) or not isinstance(fixed.get("tokens"), list) \
                or not isinstance(fixed.get("margins"), list) or len(fixed["margins"]) != len(fixed["tokens"]) \
                or not all(valid("count", token) for token in fixed["tokens"]) \
                or not all(valid("seconds", margin) for margin in fixed["margins"]):
            raise ValueError(f"{Path(path).name} holds a malformed row")
        found[row["j"]] = fixed
    if set(found) != set(range(len(item["teacher"]) - 1)):
        raise ValueError(f"{Path(path).name} does not hold one row per anchor")
    return found


def published(path, steps, categories):
    record = document(path)
    level = (((record.get("per_round_on_teacher_path") or {}).get("mtp_fixed") or {}).get(str(steps))
             if isinstance(record, dict) else None)
    if not isinstance(level, dict) or set(level) != {*categories, "pooled"} \
            or not all(isinstance(entry, dict) and set(entry) == set(LEVEL)
                       and all(valid(kind, entry[key]) for key, kind in LEVEL.items()) for entry in level.values()):
        raise ValueError(f"trajectories.json holds no per-round values of the fixed position for {steps} steps")
    return level


def acceptance(contract_path, assistant_path, snapshot, assistant_snapshot, records, inputs, outputs, steps):
    import torch

    check_steps(steps)
    contract, subset, items, sources, shard = prepared(contract_path, assistant_path, snapshot, assistant_snapshot,
                                                       records, inputs, outputs, lambda record: record["items"])
    config, place, stops = contract["config"], layout(contract["config"]), set(contract["config"]["eos_token_id"])
    reference = published(Path(records) / "trajectories.json", steps, subset["counts"])
    drafts = {item["id"]: draft_rows(Path(records) / "drafters" / f"{item['id']}.json", item) for item in items}
    started = time.perf_counter()
    model = class_drafter(assistant_snapshot, table(shard, config), config, steps)
    loaded = time.perf_counter()
    totals = {name: {"items": 0, "anchors": 0, "rounds": 0, "tokens": 0, "reference_rounds": 0, "reference_tokens": 0,
                     "margins": []} for name in [*subset["counts"], "pooled"]}
    for item in items:
        features, prompt, length = loaded_features(item), len(item["input_ids"]), len(item["teacher"])
        ours, theirs, margins = [], [], []
        for anchor in range(prompt, prompt + length - 1):
            with torch.no_grad():
                tokens = cut(model(*row_inputs(item, features, anchor, place["window"])["arguments"])[0].tolist(), stops)
            fixed = drafts[item["id"]][anchor - prompt]
            ours.append(matches(tokens, item["teacher"][anchor - prompt + 1:anchor - prompt + 1 + steps]))
            theirs.append(min(fixed["accepted"], steps))
            if ours[-1] != theirs[-1]:
                step = matches(tokens, fixed["tokens"])
                margins.append(fixed["margins"][step] if step < len(fixed["margins"]) else None)
        del features
        counted = (simulate(ours, length, steps), simulate(theirs, length, steps))
        for name in (item["category"], "pooled"):
            entry = totals[name]
            entry.update(items=entry["items"] + 1, anchors=entry["anchors"] + length - 1,
                         rounds=entry["rounds"] + counted[0][0], tokens=entry["tokens"] + counted[0][1],
                         reference_rounds=entry["reference_rounds"] + counted[1][0],
                         reference_tokens=entry["reference_tokens"] + counted[1][1], margins=entry["margins"] + margins)
    problems = [f"{name}: the drafter records of #691 give {entry['reference_rounds']} rounds and "
                f"{entry['reference_tokens']} tokens, not the {reference[name]['rounds']} and "
                f"{reference[name]['tokens']} of trajectories.json" for name, entry in totals.items()
                if (entry["reference_rounds"], entry["reference_tokens"])
                != (reference[name]["rounds"], reference[name]["tokens"])]
    categories = {}
    for name, entry in totals.items():
        known = [margin for margin in entry["margins"] if margin is not None]
        per_round = entry["tokens"] / entry["rounds"] if entry["rounds"] else None
        categories[name] = {"items": entry["items"], "anchors": entry["anchors"], "rounds": entry["rounds"],
                            "tokens": entry["tokens"], "tokens_per_round": per_round, "reference": dict(reference[name]),
                            "anchors_with_other_acceptance": len(entry["margins"]),
                            "reference_margins_at_the_first_other_draft": {
                                "known": len(known), "largest": max(known) if known else None,
                                "at_most_0.25": sum(margin <= 0.25 for margin in known)}}
    return {"schema": ACCEPTANCE, "problems": problems, "steps": steps, "stops": sorted(stops), "position": "fixed",
            "window": place["window"], **sources, "weights": WEIGHTS, "categories": categories,
            "versions": versions("class"), "timing": {"load_s": loaded - started, "drafts_s": time.perf_counter() - loaded},
            "memory": memory()}


def expected_function(config, steps):
    sliding = {"dtype": "float16", "shape": [1, 1, config.num_key_value_heads, None, config.head_dim]}
    full = {"dtype": "float16", "shape": [1, 1, config.num_global_key_value_heads, None, config.global_head_dim]}
    return {"inputs": {"input_ids": {"dtype": "int32", "shape": [1, 1]},
                       "hidden_states": {"dtype": "float16", "shape": [1, 1, config.backbone_hidden_size]},
                       "position_ids": {"dtype": "int32", "shape": [1, 1]}, "k_sliding": sliding, "v_sliding": sliding,
                       "k_full": full, "v_full": full},
            "outputs": {"draft_tokens": {"dtype": "int32", "shape": [1, steps]}}, "states": {}}


def export_synthetic(output, steps, context, seed):
    if os.path.lexists(output):
        raise FileExistsError(f"The export must be written to a new folder: {output}")
    check_steps(steps)
    import torch
    from coreai.runtime import _AIModelAsset

    from . import drafter
    from .export import inspect_asset, weights

    if steps < 2 or type(context) is not int or type(seed) is not int \
            or context <= SYNTHETIC["target"]["sliding_window"] + 1:
        raise ValueError("The export needs at least two steps, a context above the window and a whole-number seed")
    output.mkdir(parents=True)
    try:
        record = {"schema": EXPORT, "steps": steps, "context": context, "seed": seed, "configuration": SYNTHETIC,
                  "assets": {}, "versions": versions("class")}
        for name, counts in {"one_step": (1,), "unrolled": (steps,), "both": (1, steps)}.items():
            models = [drafter.synthetic(seed, count).to(torch.float16) for count in counts]
            began = time.perf_counter()
            program = drafter.export(models, context)
            converted = time.perf_counter()
            program.save_asset(output / f"{name}.aimodel")
            asset = inspect_asset(output / f"{name}.aimodel")
            summary = _AIModelAsset(str(output / f"{name}.aimodel")).summary(include_statistics=True)
            asset["operations"] = {operation: int(count) for operation, count in summary.operation_distribution}
            expected = {f"draft_{count}": expected_function(models[0].config, count) for count in counts}
            problems = [] if set(asset["functions"]) == set(expected) else [
                f"the asset has the functions {sorted(asset['functions'])}, not {sorted(expected)}"]
            problems += [f"{function} {kind}: {json.dumps(asset['functions'][function][kind], sort_keys=True)}"
                         for function in sorted(set(expected) & set(asset["functions"]))
                         for kind in ("inputs", "outputs", "states")
                         if asset["functions"][function][kind] != expected[function][kind]]
            problems += [f"{function} returns {asset['output_order'].get(function)}" for function in expected
                         if asset["output_order"].get(function) != [drafter.OUTPUT]]
            held = [weights(model) for model in models]
            problems += [f"the PyTorch model holds {entry['elements_by_dtype']}" for entry in held
                         if set(entry["elements_by_dtype"]) != {"torch.float16"} or entry["quantized_layouts"]]
            files = [path for path in (output / f"{name}.aimodel").rglob("*") if path.is_file()]
            record["assets"][name] = {"functions": list(expected), "asset": asset, "weights": held, "problems": problems,
                                      "bytes": sum(path.stat().st_size for path in files),
                                      "files": {path.relative_to(output).as_posix(): file_record(path) for path in files},
                                      "timing": {"convert_s": converted - began, "save_s": time.perf_counter() - converted}}
        record["passed"] = not any(entry["problems"] for entry in record["assets"].values())
        write_json(output / "export.json", record)
        return record
    except BaseException as error:
        write_json(output / "failure.json", {"schema": EXPORT, "failure": type(error).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description="The MTP drafter of Gemma 4 31B: FP32 runs of the original drafter "
                                                 "and of the drafter class on recorded features, their comparison, the "
                                                 "accepted drafts on the teacher path and the export of a synthetic "
                                                 "drafter.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "acceptance"):
        command = commands.add_parser(name)
        if name == "run":
            command.add_argument("--implementation", choices=IMPLEMENTATIONS, required=True)
            command.add_argument("--output", type=Path, required=True)
        for option in ("model", "assistant", "snapshot", "assistant-snapshot", "records", "inputs", "outputs"):
            command.add_argument(f"--{option}", type=Path, required=True)
        command.add_argument("--steps", type=int, required=True)
    checker = commands.add_parser("compare")
    for option in ("original", "model-class", "model"):
        checker.add_argument(f"--{option}", type=Path, required=True)
    exporter = commands.add_parser("export")
    exporter.add_argument("--output", type=Path, required=True)
    for option in ("steps", "context", "seed"):
        exporter.add_argument(f"--{option}", type=int, required=True)
    args = parser.parse_args()
    if args.command == "run":
        record = run(args.implementation, args.model, args.assistant, args.snapshot, args.assistant_snapshot,
                     args.records, args.inputs, args.outputs, args.steps, args.output)
        print(json.dumps({"rows": len(record["rows"]), "timing": record["timing"], "memory": record["memory"]}),
              flush=True)
        return 0
    if args.command == "export":
        record = export_synthetic(args.output, args.steps, args.context, args.seed)
        print(json.dumps({name: entry["bytes"] for name, entry in record["assets"].items()}), flush=True)
        return 0 if record["passed"] else 1
    result = (compare(args.original, args.model_class, args.model) if args.command == "compare"
              else acceptance(args.model, args.assistant, args.snapshot, args.assistant_snapshot, args.records,
                              args.inputs, args.outputs, args.steps))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result.get("passed", not result["problems"]) else 1


if __name__ == "__main__":
    sys.exit(main())
