import json
import sys
import time

from ...storage import read_json, write_json
from .merge import saved_name
from .prepare import distributed_tokenizer, load_contract
from .reference import arguments, decision_head, fixed_rows, forward, header, mps_allocated

LOADING = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")


def main():
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3_5Config, Qwen3_5ForCausalLM
    from transformers.conversion_mapping import get_model_conversion_mapping

    args = arguments("Run the fixed set F through the merged JEV-27B checkpoint (stage 1').", "merge")
    output = args.output / "reference-merged.json"
    if output.exists():
        raise FileExistsError("reference-merged.json must be written to a new directory")
    contract = load_contract(args.model)
    checkpoint = args.merge / "checkpoint"
    revision = read_json(args.merge / "manifest.json")["merged_revision"]
    started = time.perf_counter()
    head = decision_head(args.snapshot, contract)
    rows = fixed_rows(args, contract, distributed_tokenizer(checkpoint))
    prepared = time.perf_counter()
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
    hidden = torch.zeros(len(rows), config.text_config.hidden_size, dtype=torch.float32)
    forward_s, allocated = forward(model, rows, head, "1m", args.device, hidden) if all(checks.values()) else (0.0, 0)
    mask = torch.tensor([not row["rejected"] for row in rows], dtype=torch.uint8)
    args.output.mkdir(parents=True, exist_ok=True)
    save_file({"hidden": hidden, "mask": mask}, str(args.output / "hidden.safetensors"),
              metadata={"format": "pt", "merged_revision": revision, "ids": json.dumps([row["id"] for row in rows])})
    timing = {"rows_s": round(prepared - started, 1), "load_s": round(loaded - prepared, 1), "forward1m_s": forward_s,
              "total_s": round(time.perf_counter() - started, 1)}
    write_json(output, {**header(args.device, head, timing, max(allocated, mps_allocated(args.device)),
                                 list(model.parameters())),
                        "merged_revision": revision, "passed": all(checks.values()), "checks": checks,
                        "loading_info": loading, "rows": rows})
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
