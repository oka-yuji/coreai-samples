import argparse
import hashlib
import importlib.metadata
from pathlib import Path
import resource
import sys
import time

from ...storage import read_json, write_json
from .fixtures import load_rows
from .load_check import lora_summary
from .prepare import check_integrity, distributed_tokenizer, load_contract
from .typed_head import band, decide, last_hidden, render, tokenize

PACKAGES = ("accelerate", "peft", "safetensors", "tokenizers", "torch", "transformers")


def arguments(description, *names):
    parser = argparse.ArgumentParser(description=description)
    for name in ("model", *names, "snapshot", "corpus", "jglue", "fixtures", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def fixed_rows(args, contract, tokenizer):
    pins = read_json(args.fixtures / "jglue.json")
    check_integrity(args.corpus, contract["dataset"]["id"], contract["dataset"]["revision"])
    check_integrity(args.jglue, pins["repository"], pins["revision"])
    fixture = args.fixtures / "F.json"
    rows = []
    for entry, row in zip(read_json(fixture)["rows"], load_rows(fixture, args.corpus, args.jglue), strict=True):
        text = render(row["kind"], row["state"], row["question"], row["options"])
        ids = tokenize(tokenizer, text)
        if len(ids) != entry["tokens"] or band(len(ids)) != entry["band"]:
            raise ValueError(f"Token count differs from the fixture: {entry['id']}")
        rows.append({"id": entry["id"], "kind": row["kind"], "options": row["options"], "band": entry["band"],
                     "row_sha256": entry["sha256"], "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                     "token_count": len(ids), "tokens": ids, "last_token": tokenizer.decode(ids[-1:]),
                     "rejected": len(ids) > contract["limits"]["input_tokens"]})
    return rows


def decision_head(snapshot, contract):
    from safetensors.torch import load_file

    tensors = load_file(str(snapshot / contract["head"]))
    return tensors["proj.weight"].float(), tensors["proj.bias"].float(), read_json(snapshot / "calibration.json")["per_kind"]


def mps_allocated(device):
    import torch

    return torch.mps.driver_allocated_memory() if device == "mps" else 0


def forward(model, rows, head, stage, device, hidden=None):
    weight, bias, temperatures = head
    keys = (f"logits{stage}", f"p{stage}", f"p{stage}_T", f"elapsed{stage}_s")
    allocated = 0
    started = time.perf_counter()
    for index, row in enumerate(rows):
        if row["rejected"]:
            row.update(dict.fromkeys(keys))
            continue
        begin = time.perf_counter()
        state = last_hidden(model, row["tokens"])
        elapsed = time.perf_counter() - begin
        logits = (weight @ state + bias).tolist()
        shipped, plain = decide(logits, row["kind"], len(row["options"]), temperatures[row["kind"]])
        row.update(zip(keys, (logits, plain, shipped, round(elapsed, 4))))
        if hidden is not None:
            hidden[index] = state
        allocated = max(allocated, mps_allocated(device))
        print(stage, index, row["token_count"], f"{elapsed:.2f}s", flush=True)
    return round(time.perf_counter() - started, 1), allocated


def header(device, head, timing, allocated, parameters):
    return {"device": device, "dtype": "bfloat16", "temperatures": head[2],
            "versions": {name: importlib.metadata.version(name) for name in PACKAGES}, "timing": timing,
            "memory": {"peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       "mps_driver_allocated_bytes": allocated,
                       "parameter_dtypes": sorted({str(parameter.dtype) for parameter in parameters}),
                       "parameter_devices": sorted({str(parameter.device) for parameter in parameters})}}


def main():
    import torch
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    args = arguments("Run the fixed set F through JEV-27B with its LoRA (stage 1) and without it (stage 0).")
    output = args.output / "reference.json"
    if output.exists():
        raise FileExistsError("reference.json must be written to a new directory")
    contract = load_contract(args.model)
    started = time.perf_counter()
    head = decision_head(args.snapshot, contract)
    rows = fixed_rows(args, contract, distributed_tokenizer(args.snapshot))
    prepared = time.perf_counter()
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
    finished = time.perf_counter()
    timing = {"rows_s": round(prepared - started, 1), "base_load_s": round(base_loaded - prepared, 1),
              "adapter_load_s": round(adapter_loaded - base_loaded, 1), "forward1_s": forward1_s, "forward0_s": forward0_s,
              "total_s": round(finished - started, 1)}
    write_json(output, {**header(args.device, head, timing, max(allocated1, allocated0), list(base.parameters())),
                        "passed": all(checks.values()), "checks": checks, "lora": lora, "rows": rows})
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
