import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import re
import resource
import shutil
import struct
import sys
import time
from types import SimpleNamespace

from ...storage import PipelineError, file_hash, write_json
from ..pinned import sha256
from .layers import FULL, PREFIX, QUANTIZED, SLIDING, check_layers, tensor_sha256
from .prepare import TOKENIZER_FILES, load_contract
from .reference import SWAP_INCREASE_LIMIT_BYTES, footprint, supervise, verify_inputs

SCHEMA = "gemma4-31b-export/1"
FUNCTIONS = {"main": ("main",), "main,prefill": ("main", "prefill")}
PRECISIONS = {"fp16": "float16", "8bit": "8bit"}
WRITES = {"fp16": 4, "8bit": 7}
INPUTS = ("input_ids", "position_ids")
STATES = ("keyCache", "valueCache", "slidingKeyCache", "slidingValueCache")
CONSTANT_TYPES = ("Float32", "Int32", "UInt32", "UInt64")
MARGIN_BYTES = 60 * 2**30
SMALL_BYTES = 64 * 2**20
BUNDLE, INCOMPLETE, WORK, QUANTIZING = "bundle", "bundle.incomplete", "layers", "quantization"
PACKAGES = ("coreai-core", "coreai-models", "coreai-opt", "coreai-torch", "safetensors", "torch", "transformers")
TIMES = " × "
TYPE = re.compile(r"NDArray \((\w+), ([0-9?]+(?:" + TIMES + r"[0-9?]+)*)\)")
QUANTIZATION = {"execution_mode": "eager", "global_config": None, "module_type_configs": {
    "torch.nn.modules.linear.Linear": {"op_state_spec": {"weight": {
        "dtype": "torch.uint8", "qscheme": "asymmetric", "granularity": {"type": "per_block", "block_size": 32, "axis": 1},
        "qformulation": "minval"}}, "op_input_spec": None, "op_output_spec": None}}}
LAYOUT = {"data": "torch.uint8", "scale": "torch.float16", "minval": "torch.float16", "zero_point": None, "block": [1, 32]}
LINEAR = {SLIDING: ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
                    "mlp.up_proj", "mlp.down_proj"),
          FULL: ("self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                 "mlp.down_proj")}
CACHED = ("self_attn.k_proj", "self_attn.v_proj")
PARTS = ("quantized_data", "scale", "minval", "zero_point")
SAFETENSORS_TYPES = {"torch.uint8": "U8", "torch.int8": "I8", "torch.float16": "F16", "torch.bfloat16": "BF16",
                     "torch.float32": "F32"}
SKIPPED, QUANTIZER = "Skipping quantization.", "coreai_opt.quantization.spec.fake_quantize"
DEQUANTIZE = "blockwise_shift_scale"


def name_of(layers, functions, precision="fp16"):
    return f"gemma4_31b_l{layers}_{precision}_{'_'.join(functions)}"


def metadata(contract, name, context, compression=None):
    return {"metadata_version": "0.2", "kind": "llm", "name": name, "assets": {"main": f"{name}.aimodel"},
            "language": {"tokenizer": contract["model_id"], "vocab_size": contract["config"]["vocab_size"],
                         "max_context_length": context, "embedded_tokenizer": True, "function_map": {"main": ["main"]}},
            "source": {"model_definition": "torch", "hf_model_id": contract["model_id"], "revision": contract["revision"]},
            "compression": compression}


def quantized_names(kinds):
    return {f"model.layers.{index}.{part}.weight" for index, kind in enumerate(kinds) for part in LINEAR[kind]}


def dequantizations(names, kinds, functions):
    last = len(kinds) - 1
    unused = {f"model.layers.{last}.{part}.weight" for part in LINEAR[kinds[-1]] if part not in CACHED}
    return len(names) + (len(set(names) - unused) if "prefill" in functions else 0)


def differences(storage, elements):
    return {"float16_minus_weights": storage.get("Float16", 0) - elements.get("torch.float16", 0),
            "uint8_minus_weights": storage.get("UInt8", 0) - elements.get("torch.uint8", 0)}


def normalized(config):
    return json.loads(json.dumps(config, default=str))


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


def float32_elements(config):
    return config["global_head_dim"] // 2 + config["head_dim"] // 2 + 2


def shown(value):
    return json.dumps(value, sort_keys=True)


def float16_problems(record):
    problems = []
    if record["compression_config"] is not None:
        problems.append("the export was given a compression config")
    if record["weights"]["quantized_layouts"]:
        problems.append(f"the PyTorch model holds {record['weights']['quantized_layouts']} quantized weights")
    if set(record["weights"]["elements_by_dtype"]) != {"torch.float16"}:
        problems.append(f"the PyTorch model holds {shown(record['weights']['elements_by_dtype'])} elements; every "
                        "parameter and buffer must be float16")
    return problems


def quantization_problems(record, kinds, functions):
    problems, quantized = [], record["quantized"]
    if tuple(functions) != FUNCTIONS["main,prefill"]:
        problems.append(f"an 8-bit asset has the functions main and prefill, not {list(functions)}")
    if record["compression_config"] != QUANTIZATION:
        problems.append(f"the export was given {shown(record['compression_config'])}, not the fixed 8-bit config")
    names, expected = sorted(entry["name"] for entry in quantized["layers"]), sorted(quantized_names(kinds))
    if names != expected:
        problems.append(f"the PyTorch model quantized {len(names)} tensors, not the {len(expected)} Linear weights; "
                        f"missing {sorted(set(expected) - set(names))}, others {sorted(set(names) - set(expected))}")
    wrong = [entry["name"] for entry in quantized["layers"] if {key: entry.get(key) for key in LAYOUT} != LAYOUT]
    if wrong:
        problems.append(f"{wrong} are not uint8 with float16 scales and minimums in blocks of [1, 32]")
    if set(record["weights"]["elements_by_dtype"]) != {"torch.float16", "torch.uint8"}:
        problems.append(f"the PyTorch model holds {shown(record['weights']['elements_by_dtype'])} elements; every "
                        "parameter and buffer must be the uint8 quantized data or float16")
    if quantized["skipped"]:
        problems.append(f"the converter logged {SKIPPED!r} {quantized['skipped']} times")
    if not quantized["warnings_enabled"]:
        problems.append("the warnings of the quantizer were not enabled, so skipped weights could not be counted")
    expected = dequantizations(quantized_names(kinds), kinds, functions)
    if record["asset"]["dequantizations"] != expected:
        problems.append(f"the asset dequantizes {record['asset']['dequantizations']} times, not {expected}: one for each "
                        "quantized weight that each function reads")
    return problems


def check_asset(record, contract, kinds, functions, context, precision="fp16"):
    eight_bit = precision == "8bit"
    problems = quantization_problems(record, kinds, functions) if eight_bit else float16_problems(record)
    storage, expected = record["asset"]["storage_types"], float32_elements(contract["config"])
    stored = ("Float16", "UInt8") if eight_bit else ("Float16",)
    other = sorted(set(storage) - {*stored, *CONSTANT_TYPES})
    if other:
        problems.append(f"the asset stores {other}; only {' and '.join(stored)} weights and {list(CONSTANT_TYPES)} "
                        "constants may remain")
    if storage.get("Float32", 0) != expected:
        problems.append(f"Float32 holds {storage.get('Float32', 0)} elements, not the {expected} of the two RoPE tables "
                        "and the two constants")
    found, expected = record["asset"]["functions"], expected_functions(contract["config"], kinds, functions)
    if set(found) != set(expected):
        problems.append(f"the asset has the functions {sorted(found)}, not {sorted(expected)}")
    for name in sorted(set(found) & set(expected)):
        problems += [f"{name} {kind}: {shown(found[name][kind])}, not {shown(expected[name][kind])}"
                     for kind in ("inputs", "outputs", "states") if found[name][kind] != expected[name][kind]]
    if record["tokenizer"] != contract["tokenizer"]["files"]:
        problems.append(f"the tokenizer files {shown(record['tokenizer'])} differ from the distributed files")
    compression = "8bit" if eight_bit else None
    if record["metadata"] != metadata(contract, name_of(len(kinds), functions, precision), context, compression):
        problems.append(f"metadata.json differs from the contract and the arguments: {shown(record['metadata'])}")
    return problems


def required_bytes(elements, precision="fp16"):
    return WRITES[precision] * elements + MARGIN_BYTES


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


class Skips(logging.Handler):
    def __init__(self):
        super().__init__()
        self.count = 0

    def emit(self, record):
        self.count += SKIPPED in record.getMessage()


@contextlib.contextmanager
def counting(counter):
    root = logging.getLogger()
    root.addHandler(counter)
    try:
        yield logging.getLogger(QUANTIZER).isEnabledFor(logging.WARNING)
    finally:
        root.removeHandler(counter)


def quantize(model, compression, work):
    from coreai_models.export.compression import quantize_for_export, split_compression_config
    from coreai_models.llm.export import _load_compression_config_object
    import torch

    config, _ = split_compression_config(_load_compression_config_object(compression.resolve(), "macOS"))
    given = normalized(config)
    return quantize_for_export(model, model.config, torch.float16, dict(config), mmap_dir=str(work)), given


def quantized_tensors(model):
    from torch.nn.utils import parametrize

    entries, tensors = [], []
    for name, module in model.named_modules():
        if not parametrize.is_parametrized(module):
            continue
        for item, chain in module.parametrizations.items():
            for step in chain:
                parts = {part: getattr(step, part, None) for part in PARTS}
                data, scale = parts["quantized_data"], parts["scale"]
                entries.append({"name": f"{name}.{item}", "class": type(step).__name__,
                                "data": None if data is None else str(data.dtype),
                                "shape": None if data is None else list(data.shape),
                                **{part: None if parts[part] is None else str(parts[part].dtype)
                                   for part in ("scale", "minval", "zero_point")},
                                "block": None if data is None or scale is None else [
                                    size // count for size, count in zip(data.shape, scale.shape)]})
                tensors += [(f"{name}.{item}.{part}", value) for part, value in parts.items() if value is not None]
    return entries, tensors


def save_quantized(path, tensors):
    import torch

    entries, offset, ready = {}, 0, []
    for name, tensor in tensors:
        data = tensor.detach().cpu().contiguous()
        size = data.numel() * data.element_size()
        entries[name] = {"dtype": SAFETENSORS_TYPES[str(data.dtype)], "shape": list(data.shape),
                         "data_offsets": [offset, offset + size]}
        ready.append((name, data))
        offset += size
    header = json.dumps(entries, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    digests = {}
    with path.open("xb") as handle:
        handle.write(struct.pack("<Q", len(header)) + header)
        for name, data in ready:
            handle.write(data.reshape(-1).view(torch.uint8).numpy())
            digests[name] = tensor_sha256(data)
    return digests


def inspect_asset(path):
    from coreai.runtime import _AIModelAsset

    summary = _AIModelAsset(str(path)).summary(include_statistics=True)
    getters = (("inputs", summary.function_inputs), ("outputs", summary.function_outputs),
               ("states", summary.function_states))
    return {"functions": {name: {kind: {item: described(text) for item, text in getter(name)} for kind, getter in getters}
                          for name in summary.function_names},
            "storage_types": {kind: int(count) for kind, count in summary.storage_types},
            "dequantizations": int(dict(summary.operation_distribution).get(DEQUANTIZE, 0))}


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
    eight_bit = args.precision == "8bit"
    work, name, bundle = args.output / WORK, name_of(args.layers, functions, args.precision), args.output / INCOMPLETE
    quantizing, counter, config, quantized = args.output / QUANTIZING, Skips(), None, None
    try:
        with counting(counter) as enabled:
            model = Gemma4ForCausalLM.from_hf_memory_efficient(
                str(args.snapshot), max_context_length=args.context, target_dtype=torch.float16, mmap_path=str(work),
                num_layers=args.layers, hf_config_attr="text_config", hf_state_dict_prefix=PREFIX).eval()
            model.exports_prefill_graph = "prefill" in functions
            loaded = weights(model)
            marks.append(("load_s", time.perf_counter()))
            if eight_bit:
                model, config = quantize(model, args.compression, quantizing)
                loaded = weights(model)
                marks.append(("quantize_s", time.perf_counter()))
                entries, tensors = quantized_tensors(model)
                digests = save_quantized(args.output / QUANTIZED, tensors)
                del tensors
                quantized = {"layers": entries, "warnings_enabled": enabled, "file": file_hash(args.output / QUANTIZED),
                             "tensors": digests}
                marks.append(("tensors_s", time.perf_counter()))
            settings = SimpleNamespace(max_context_length=args.context, compute_precision="float16",
                                       include_debug_info=False, dynamic_max_batch_size=None)
            program = export_macos_model(model, model.config, settings)
            del model
            marks.append(("convert_s", time.perf_counter()))
        bundle.mkdir()
        program.save_asset(bundle / f"{name}.aimodel", build_aimodel_metadata(contract["model_id"]))
        del program
    finally:
        for folder in (work, quantizing):
            if folder.exists():
                shutil.rmtree(folder)
    marks.append(("save_s", time.perf_counter()))
    write_json(bundle / "metadata.json", metadata(contract, name, args.context, "8bit" if eight_bit else None))
    (bundle / "tokenizer").mkdir()
    for item in TOKENIZER_FILES:
        shutil.copyfile(args.snapshot / item, bundle / "tokenizer" / item)
    asset = inspect_asset(bundle / f"{name}.aimodel")
    if eight_bit:
        asset.update(differences(asset["storage_types"], loaded["elements_by_dtype"]))
        quantized["skipped"] = counter.count
    else:
        asset["float16_minus_weights"] = asset["storage_types"].get("Float16", 0) - loaded["elements"]
    marks.append(("inspect_s", time.perf_counter()))
    files, contents = bundle_files(bundle)
    marks.append(("bundle_hash_s", time.perf_counter()))
    record = {"schema": SCHEMA, "model": {"id": contract["model_id"], "revision": contract["revision"],
                                          "contract_sha256": sha256(args.model), "verified_before_loading": verified},
              "layers": args.layers, "layer_types": kinds, "functions": list(functions), "context": args.context,
              "precision": PRECISIONS[args.precision], "compression_config": config,
              "upstream_revision": contract["upstream_revision"],
              "versions": {item: importlib.metadata.version(item) for item in PACKAGES}, "weights": loaded,
              **({"quantized": quantized} if eight_bit else {}),
              "asset": asset, "metadata": json.loads(contents["metadata.json"]),
              "tokenizer": {path.split("/", 1)[1]: entry["sha256"] for path, entry in files.items()
                            if path.startswith("tokenizer/")},
              "bundle": {"folder": INCOMPLETE, "name": name, "main_hash": contents[f"{name}.aimodel/main.hash"].hex(),
                         "files": files, "bytes": sum(entry["bytes"] for entry in files.values())},
              "disk": disk, "timing": {label: end - start for (_, start), (label, end) in zip(marks, marks[1:])}}
    record["problems"] = check_asset(record, contract, kinds, functions, args.context, args.precision)
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
    parser = argparse.ArgumentParser(description="Export the first layers of Gemma 4 31B to a dynamic-shape FP16 or 8-bit "
                                                 "Core AI bundle for macOS and inspect the asset.")
    for name in ("model", "snapshot", "upstream", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--functions", choices=tuple(FUNCTIONS), required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--precision", choices=tuple(PRECISIONS), required=True)
    parser.add_argument("--compression", type=Path)
    args = parser.parse_args()
    if (args.precision == "8bit") != (args.compression is not None):
        parser.error("--compression is required with --precision 8bit and refused with --precision fp16")
    if args.precision == "8bit" and args.functions != "main,prefill":
        parser.error("--precision 8bit exports the functions main,prefill only")
    if os.path.lexists(args.output):
        raise FileExistsError(f"The export must be written to a new folder: {args.output}")
    contract = load_contract(args.model)
    kinds = check_layers(contract["config"], args.layers)
    args.output.mkdir(parents=True)
    try:
        return begin(args, contract, kinds)
    except Exception as error:
        write_json(args.output / "failure.json", {"schema": SCHEMA, "passed": False, "error": type(error).__name__})
        raise


def begin(args, contract, kinds):
    elements = weight_elements(args.snapshot, args.layers)
    free, required = shutil.disk_usage(args.output).free, required_bytes(elements, args.precision)
    if free < required:
        raise PipelineError(f"The volume of the output has {free} B free, less than the {required} B for the asset, "
                            f"the work areas and a margin of {MARGIN_BYTES} B")
    disk = {"weight_bytes": 2 * elements, "estimated_write_bytes": WRITES[args.precision] * elements,
            "margin_bytes": MARGIN_BYTES, "required_free_bytes": required, "free_before_bytes": free}
    functions = FUNCTIONS[args.functions]
    return supervise(args.output, SCHEMA, lambda: run(args, contract, kinds, functions, disk),
                     lambda record, observed, started: conclude(args.output, record, observed, started))


if __name__ == "__main__":
    sys.exit(main())
