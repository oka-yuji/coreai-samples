import argparse
import hashlib
from pathlib import Path
import sys
import time

from ...storage import read_json, write_json
from .fixtures import SETS, load_rows
from .load_check import lora_summary
from .merge import saved_name
from .prepare import check_integrity, distributed_tokenizer, load_contract
from .reference import decision_head, forward, header, mps_allocated
from .reference_merged import LOADING
from .typed_head import band, render, tokenize


def arguments():
    parser = argparse.ArgumentParser(description="Run fixed JEV-27B sets as stages 1 and 0 (peft) or stage 1' (merged).")
    for name in ("model", "snapshot", "corpus", "jglue", "fixtures", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--merge", type=Path)
    parser.add_argument("--stage", choices=("peft", "merged"), required=True)
    parser.add_argument("--sets", nargs="+", choices=SETS, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    if (args.stage == "merged") != (args.merge is not None) or len(set(args.sets)) != len(args.sets):
        parser.error("--merge is required with --stage merged only, and each set is named once")
    return args


def output_name(stage, sets):
    return f"reference{'-merged' if stage == 'merged' else ''}-{'-'.join(sets)}.json"


def set_rows(args, contract, tokenizer):
    pins = read_json(args.fixtures / "jglue.json")
    check_integrity(args.corpus, contract["dataset"]["id"], contract["dataset"]["revision"])
    check_integrity(args.jglue, pins["repository"], pins["revision"])
    rows = []
    for name in args.sets:
        fixture = args.fixtures / f"{name}.json"
        for entry, row in zip(read_json(fixture)["rows"], load_rows(fixture, args.corpus, args.jglue), strict=True):
            text = render(row["kind"], row["state"], row["question"], row["options"])
            ids = tokenize(tokenizer, text)
            if len(ids) != entry["tokens"] or band(len(ids)) != entry["band"]:
                raise ValueError(f"Token count differs from the fixture: {entry['id']}")
            rows.append({"set": name, "id": entry["id"], "kind": row["kind"], "options": row["options"],
                         "band": entry["band"], "row_sha256": entry["sha256"],
                         "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "token_count": len(ids),
                         "tokens": ids, "last_token": tokenizer.decode(ids[-1:]),
                         "rejected": len(ids) > contract["limits"]["input_tokens"]})
    return rows


def peft_stages(args, contract, head, rows, started, prepared):
    import torch
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(str(args.snapshot), dtype=torch.bfloat16, device_map=args.device)
    base_loaded = time.perf_counter()
    model = PeftModel.from_pretrained(base, str(args.snapshot / "adapter"))
    model.eval()
    adapter_loaded = time.perf_counter()
    lora = lora_summary(model, load_file(str(args.snapshot / "adapter/adapter_model.safetensors")))
    layers = [module for module in model.modules() if isinstance(module, LoraLayer)]
    checks = {"pairs": lora["consumed_pairs"] == lora["file_pairs"] == len(layers) == contract["lora"]["pairs"],
              "state_matches_file": not lora["missing"] and not lora["unexpected"],
              "scale": lora["scaling"] == [contract["lora"]["scale"]], "eval": not lora["training"]}
    forward1_s = forward0_s = 0.0
    allocated1 = allocated0 = 0
    if all(checks.values()):
        forward1_s, allocated1 = forward(model, rows, head, "1", args.device)
        with model.disable_adapter():
            lora["disabled_in_stage0"] = sum(module.disable_adapters for module in layers)
            forward0_s, allocated0 = forward(model, rows, head, "0", args.device)
        lora["enabled_after_stage0"] = sum(not module.disable_adapters for module in layers)
        checks.update(stage0_disables_every_layer=lora["disabled_in_stage0"] == len(layers),
                      stage1_layers_restored=lora["enabled_after_stage0"] == len(layers))
    timing = {"rows_s": round(prepared - started, 1), "base_load_s": round(base_loaded - prepared, 1),
              "adapter_load_s": round(adapter_loaded - base_loaded, 1), "forward1_s": forward1_s,
              "forward0_s": forward0_s, "total_s": round(time.perf_counter() - started, 1)}
    return {**header(args.device, head, timing, max(allocated1, allocated0), list(base.parameters())),
            "passed": all(checks.values()), "checks": checks, "lora": lora}


def merged_stage(args, contract, head, rows, started, prepared):
    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForCausalLM
    from transformers.conversion_mapping import get_model_conversion_mapping

    checkpoint = args.merge / "checkpoint"
    revision = read_json(args.merge / "manifest.json")["merged_revision"]
    config = Qwen3_5Config.from_pretrained(str(checkpoint))
    if config.model_type != "qwen3_5":
        raise ValueError(f"Expected a qwen3_5 checkpoint, found {config.model_type}")
    model, info = Qwen3_5ForCausalLM.from_pretrained(str(checkpoint), config=config.text_config, dtype=torch.bfloat16,
                                                     device_map=args.device, output_loading_info=True)
    model.eval()
    loaded = time.perf_counter()
    expected = set(read_json(checkpoint / "model.safetensors.index.json")["weight_map"])
    state = {saved_name(name) for name in model.state_dict()}
    loading = {key: sorted(str(item) for item in info[key]) for key in LOADING}
    loading.update(key_mapping=None, config_class=type(config.text_config).__name__, model_type=config.text_config.model_type,
                   conversions=[f"{type(rule).__name__}: {rule.source_patterns} -> {rule.target_patterns}"
                                for rule in get_model_conversion_mapping(model)],
                   tensors_loaded=len(state), tensors_expected=len(expected))
    checks = {"loading_info_empty": not any(loading[key] for key in LOADING), "every_tensor_loaded": state == expected,
              "eval": not model.training}
    forward_s, allocated = forward(model, rows, head, "1m", args.device) if all(checks.values()) else (0.0, 0)
    timing = {"rows_s": round(prepared - started, 1), "load_s": round(loaded - prepared, 1), "forward1m_s": forward_s,
              "total_s": round(time.perf_counter() - started, 1)}
    return {**header(args.device, head, timing, max(allocated, mps_allocated(args.device)), list(model.parameters())),
            "merged_revision": revision, "passed": all(checks.values()), "checks": checks, "loading_info": loading}


def main():
    args = arguments()
    output = args.output / output_name(args.stage, args.sets)
    if output.exists():
        raise FileExistsError(f"{output.name} must be written to a new directory")
    contract = load_contract(args.model)
    started = time.perf_counter()
    head = decision_head(args.snapshot, contract)
    source = args.merge / "checkpoint" if args.stage == "merged" else args.snapshot
    rows = set_rows(args, contract, distributed_tokenizer(source))
    prepared = time.perf_counter()
    run = merged_stage if args.stage == "merged" else peft_stages
    document = run(args, contract, head, rows, started, prepared)
    write_json(output, {**document, "sets": args.sets, "rows": rows})
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
