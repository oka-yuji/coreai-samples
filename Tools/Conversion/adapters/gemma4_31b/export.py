import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import resource
import shutil
import sys
import time
from types import SimpleNamespace

from ...storage import PipelineError, file_hash, write_json
from ..pinned import sha256
from .layers import PREFIX, check_layers
from .prepare import TOKENIZER_FILES, load_contract
from .reference import SWAP_INCREASE_LIMIT_BYTES, footprint, supervise, verify_inputs

SCHEMA = "gemma4-31b-export/1"
FUNCTIONS = {"main": ("main",), "main,prefill": ("main", "prefill")}
INPUTS = ("input_ids", "position_ids")
STATES = ("keyCache", "valueCache", "slidingKeyCache", "slidingValueCache")
CONSTANT_TYPES = ("Float32", "Int32", "UInt32", "UInt64")
MARGIN_BYTES = 60 * 2**30
SMALL_BYTES = 64 * 2**20
BUNDLE, INCOMPLETE, WORK = "bundle", "bundle.incomplete", "layers"
PACKAGES = ("coreai-core", "coreai-models", "coreai-opt", "coreai-torch", "safetensors", "torch", "transformers")
TIMES = " × "
TYPE = re.compile(r"NDArray \((\w+), ([0-9?]+(?:" + TIMES + r"[0-9?]+)*)\)")


def name_of(layers, functions):
    return f"gemma4_31b_l{layers}_fp16_{'_'.join(functions)}"


def metadata(contract, name, context):
    return {"metadata_version": "0.2", "kind": "llm", "name": name, "assets": {"main": f"{name}.aimodel"},
            "language": {"tokenizer": contract["model_id"], "vocab_size": contract["config"]["vocab_size"],
                         "max_context_length": context, "embedded_tokenizer": True, "function_map": {"main": ["main"]}},
            "source": {"model_definition": "torch", "hf_model_id": contract["model_id"], "revision": contract["revision"]},
            "compression": None}


def expected_functions(config, kinds, functions):
    full = [kinds.count("full_attention"), 1, config["num_global_key_value_heads"], None, config["global_head_dim"]]
    sliding = [kinds.count("sliding_attention"), 1, config["num_key_value_heads"], None, config["head_dim"]]
    shapes = dict(zip(STATES, (full, full, sliding, sliding)))
    return {name: {"inputs": {item: {"dtype": "int32", "shape": [1, None]} for item in INPUTS},
                   "outputs": {"logits": {"dtype": "float16", "shape": [1, None, config["vocab_size"]]}}
                   if name == "main" else {},
                   "states": {item: {"dtype": "float16", "shape": list(shape)} for item, shape in shapes.items()}}
            for name in functions}


def described(text):
    match = TYPE.fullmatch(text)
    if match is None:
        return {"raw": text}
    return {"dtype": match[1].lower(), "shape": [None if size == "?" else int(size) for size in match[2].split(TIMES)]}


def shown(value):
    return json.dumps(value, sort_keys=True)


def check_asset(record, contract, kinds, functions, context):
    problems = []
    if record["compression_config"] is not None:
        problems.append("the export was given a compression config")
    if record["weights"]["quantized_layouts"]:
        problems.append(f"the PyTorch model holds {record['weights']['quantized_layouts']} quantized weights")
    storage, elements = record["asset"]["storage_types"], record["weights"]["elements"]
    other = sorted(set(storage) - {"Float16", *CONSTANT_TYPES})
    if other:
        problems.append(f"the asset stores {other}; only Float16 weights and {list(CONSTANT_TYPES)} constants may remain")
    if storage.get("Float16", 0) < elements:
        problems.append(f"Float16 holds {storage.get('Float16', 0)} elements, fewer than the {elements} of the weights")
    found, expected = record["asset"]["functions"], expected_functions(contract["config"], kinds, functions)
    if set(found) != set(expected):
        problems.append(f"the asset has the functions {sorted(found)}, not {sorted(expected)}")
    for name in sorted(set(found) & set(expected)):
        problems += [f"{name} {kind}: {shown(found[name][kind])}, not {shown(expected[name][kind])}"
                     for kind in ("inputs", "outputs", "states") if found[name][kind] != expected[name][kind]]
    if record["tokenizer"] != contract["tokenizer"]["files"]:
        problems.append(f"the tokenizer files {shown(record['tokenizer'])} differ from the distributed files")
    if record["metadata"] != metadata(contract, name_of(len(kinds), functions), context):
        problems.append(f"metadata.json differs from the contract and the arguments: {shown(record['metadata'])}")
    return problems


def required_bytes(elements):
    return 4 * elements + MARGIN_BYTES


def weight_elements(snapshot, layers):
    from transformers import AutoConfig

    from .model import Gemma4ForCausalLM, text_config

    config = text_config(AutoConfig.from_pretrained(str(snapshot)).text_config, num_layers=layers)
    model = Gemma4ForCausalLM(config, model_device="meta")
    return sum(tensor.numel() for tensor in [*model.parameters(), *model.buffers()])


def weights(model):
    from torch.nn.utils import parametrize

    tensors = [*model.parameters(), *model.buffers()]
    by_dtype = {}
    for tensor in tensors:
        by_dtype[str(tensor.dtype)] = by_dtype.get(str(tensor.dtype), 0) + tensor.numel()
    return {"tensors": len(tensors), "elements": sum(by_dtype.values()), "elements_by_dtype": by_dtype,
            "quantized_layouts": sum(parametrize.is_parametrized(module) for module in model.modules())}


def inspect_asset(path):
    from coreai.runtime import _AIModelAsset

    summary = _AIModelAsset(str(path)).summary(include_statistics=True)
    getters = (("inputs", summary.function_inputs), ("outputs", summary.function_outputs),
               ("states", summary.function_states))
    return {"functions": {name: {kind: {item: described(text) for item, text in getter(name)} for kind, getter in getters}
                          for name in summary.function_names},
            "storage_types": {kind: int(count) for kind, count in summary.storage_types}}


def bundle_files(folder):
    files, contents = {}, {}
    for path in sorted(folder.rglob("*")):
        if path.is_symlink() or not path.is_dir():
            relative = path.relative_to(folder).as_posix()
            if path.is_symlink() or path.stat().st_size >= SMALL_BYTES:
                files[relative] = file_hash(path)
                continue
            contents[relative] = path.read_bytes()
            files[relative] = {"bytes": len(contents[relative]), "sha256": hashlib.sha256(contents[relative]).hexdigest()}
    return files, contents


def run(args, contract, kinds, functions, disk):
    import coreai_models
    from coreai_models.export.macos import export_macos_model
    from coreai_models.export.metadata import build_aimodel_metadata
    import torch

    from ...local import check_checkout
    from .model import Gemma4ForCausalLM

    marks = [("start", time.perf_counter())]
    verified = verify_inputs(args.snapshot, contract)
    marks.append(("input_hash_s", time.perf_counter()))
    check_checkout(args.upstream, contract["upstream_revision"])
    if Path(coreai_models.__file__).resolve().parents[3] != args.upstream.resolve():
        raise PipelineError("The converter was imported from another checkout than --upstream")
    work, name, bundle = args.output / WORK, name_of(args.layers, functions), args.output / INCOMPLETE
    try:
        model = Gemma4ForCausalLM.from_hf_memory_efficient(
            str(args.snapshot), max_context_length=args.context, target_dtype=torch.float16, mmap_path=str(work),
            num_layers=args.layers, hf_config_attr="text_config", hf_state_dict_prefix=PREFIX).eval()
        model.exports_prefill_graph = "prefill" in functions
        loaded = weights(model)
        marks.append(("load_s", time.perf_counter()))
        settings = SimpleNamespace(max_context_length=args.context, compute_precision="float16", include_debug_info=False,
                                   dynamic_max_batch_size=None)
        program = export_macos_model(model, model.config, settings)
        del model
        marks.append(("convert_s", time.perf_counter()))
        bundle.mkdir()
        program.save_asset(bundle / f"{name}.aimodel", build_aimodel_metadata(contract["model_id"]))
        del program
    finally:
        if work.exists():
            shutil.rmtree(work)
    marks.append(("save_s", time.perf_counter()))
    write_json(bundle / "metadata.json", metadata(contract, name, args.context))
    (bundle / "tokenizer").mkdir()
    for item in TOKENIZER_FILES:
        shutil.copyfile(args.snapshot / item, bundle / "tokenizer" / item)
    asset = inspect_asset(bundle / f"{name}.aimodel")
    marks.append(("inspect_s", time.perf_counter()))
    files, contents = bundle_files(bundle)
    marks.append(("bundle_hash_s", time.perf_counter()))
    record = {"schema": SCHEMA, "model": {"id": contract["model_id"], "revision": contract["revision"],
                                          "contract_sha256": sha256(args.model), "verified_before_loading": verified},
              "layers": args.layers, "layer_types": kinds, "functions": list(functions), "context": args.context,
              "precision": "float16", "compression_config": None, "upstream_revision": contract["upstream_revision"],
              "versions": {item: importlib.metadata.version(item) for item in PACKAGES}, "weights": loaded,
              "asset": asset, "metadata": json.loads(contents["metadata.json"]),
              "tokenizer": {path.split("/", 1)[1]: entry["sha256"] for path, entry in files.items()
                            if path.startswith("tokenizer/")},
              "bundle": {"folder": INCOMPLETE, "name": name, "main_hash": contents[f"{name}.aimodel/main.hash"].hex(),
                         "files": files, "bytes": sum(entry["bytes"] for entry in files.values())},
              "disk": disk, "timing": {label: end - start for (_, start), (label, end) in zip(marks, marks[1:])}}
    record["problems"] = check_asset(record, contract, kinds, functions, args.context)
    return record


def conclude(output, record, observed, started):
    lowest = observed["lowest"]["free_bytes"].get(str(output))
    first, peak = observed["first"]["swap_used_bytes"], observed["peak"]["swap_used_bytes"]
    record["resources"] = {"max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           "lifetime_peak_footprint_bytes": footprint()[1],
                           "swap": {"first_bytes": first, "peak_bytes": peak, "limit_increase_bytes": SWAP_INCREASE_LIMIT_BYTES,
                                    "increase_bytes": None if None in (first, peak) else peak - first},
                           "monitor": observed}
    record["disk"]["lowest_free_bytes"] = lowest
    record["checks"] = {"asset": not record["problems"], "free_disk": lowest is not None and lowest >= MARGIN_BYTES}
    record["passed"] = all(record["checks"].values())
    record["timing"]["total_s"] = time.perf_counter() - started
    if record["passed"]:
        record["bundle"]["folder"] = BUNDLE
    write_json(output / "export.json", record)
    if record["passed"]:
        (output / INCOMPLETE).rename(output / BUNDLE)
    return 0 if record["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description="Export the first layers of Gemma 4 31B to a dynamic-shape FP16 Core AI "
                                                 "bundle for macOS and inspect the asset.")
    for name in ("model", "snapshot", "upstream", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--functions", choices=tuple(FUNCTIONS), required=True)
    parser.add_argument("--context", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"The export must be written to a new folder: {args.output}")
    contract = load_contract(args.model)
    kinds = check_layers(contract["config"], args.layers)
    args.output.mkdir(parents=True)
    elements = weight_elements(args.snapshot, args.layers)
    free, required = shutil.disk_usage(args.output).free, required_bytes(elements)
    if free < required:
        raise PipelineError(f"The volume of the output has {free} B free, less than the {required} B for the asset, "
                            f"the loading work area and a margin of {MARGIN_BYTES} B")
    disk = {"weight_bytes": 2 * elements, "estimated_write_bytes": 4 * elements, "margin_bytes": MARGIN_BYTES,
            "required_free_bytes": required, "free_before_bytes": free}
    functions = FUNCTIONS[args.functions]
    return supervise(args.output, SCHEMA, lambda: run(args, contract, kinds, functions, disk),
                     lambda record, observed, started: conclude(args.output, record, observed, started))


if __name__ == "__main__":
    sys.exit(main())
