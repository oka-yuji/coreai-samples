import argparse
import gc
from pathlib import Path
import resource
import sys
import time

from ...storage import read_json, write_json
from .fixtures import CARD_EXAMPLES, load_rows
from .prepare import load_contract
from .typed_head import decide, last_hidden, render, tokenize

KINDS = ("noul", "choice", "score")
CARD_TOLERANCE = 0.05


def select(entries, rows):
    chosen = [row for row in rows if row["id"] in CARD_EXAMPLES]
    for kind in KINDS:
        chosen.append(next(row for entry, row in zip(entries, rows)
                           if entry["source"] == "test_set_30k" and row["kind"] == kind))
    return chosen


def lora_summary(model, adapter_file):
    import torch
    from peft import get_peft_model_state_dict
    from peft.tuners.lora import LoraLayer

    state = get_peft_model_state_dict(model)
    layers = [module for module in model.modules() if isinstance(module, LoraLayer)]
    pairs = [key for key in adapter_file if key.endswith(".lora_A.weight")]
    consumed = sum(all(name in state and torch.equal(state[name].detach().cpu(), adapter_file[name])
                       for name in (key, key.replace(".lora_A.", ".lora_B."))) for key in pairs)
    return {"file_pairs": len(pairs), "consumed_pairs": consumed, "lora_layers": len(layers),
            "missing": sorted(set(adapter_file) - set(state)), "unexpected": sorted(set(state) - set(adapter_file)),
            "scaling": sorted({module.scaling["default"] for module in layers}),
            "active_adapters": sorted({name for module in layers for name in module.active_adapters}),
            "adapter_dtypes": sorted({str(value.dtype) for value in state.values()}), "training": model.training}


def main():
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="Load JEV-27B with its LoRA once and run the decision head on fixed rows.")
    for name in ("model", "snapshot", "corpus", "jglue", "fixtures", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    output = args.output / "load-check.json"
    if output.exists():
        raise FileExistsError("load-check.json must be written to a new directory")
    contract = load_contract(args.model)
    fixture = args.fixtures / "F.json"
    rows = select(read_json(fixture)["rows"], load_rows(fixture, args.corpus, args.jglue))
    temperatures = read_json(args.snapshot / "calibration.json")["per_kind"]
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(args.snapshot))
    base = AutoModelForCausalLM.from_pretrained(str(args.snapshot), dtype=torch.bfloat16, device_map=args.device)
    base_loaded = time.perf_counter()
    model = PeftModel.from_pretrained(base, str(args.snapshot / "adapter"))
    model.eval()
    adapter_loaded = time.perf_counter()
    lora = lora_summary(model, load_file(str(args.snapshot / "adapter/adapter_model.safetensors")))
    head = load_file(str(args.snapshot / contract["head"]))
    weight, bias = head["proj.weight"].float(), head["proj.bias"].float()
    results = []
    loop_started = time.perf_counter()
    for row in rows:
        ids = tokenize(tokenizer, render(row["kind"], row["state"], row["question"], row["options"]))
        begin = time.perf_counter()
        hidden = last_hidden(model, ids)
        elapsed = time.perf_counter() - begin
        logits = (weight @ hidden + bias).tolist()
        probabilities, probabilities_t1 = decide(logits, row["kind"], len(row["options"]), temperatures[row["kind"]])
        result = {"id": row["id"], "kind": row["kind"], "tokens": len(ids), "last_token": tokenizer.decode(ids[-1:]),
                  "forward_s": round(elapsed, 3), "logits": logits, "probabilities": probabilities,
                  "probabilities_t1": probabilities_t1, "hidden_dtype": str(hidden.dtype), "hidden_shape": list(hidden.shape)}
        if row["id"] in CARD_EXAMPLES:
            card = CARD_EXAMPLES[row["id"]]["card_probabilities"]
            result.update(card_probabilities=card,
                          card_max_difference=max(abs(value - expected) for value, expected in zip(probabilities, card)))
        results.append(result)
        print(row["id"], result["tokens"], [round(value, 4) for value in probabilities], flush=True)
    finished = time.perf_counter()
    memory = {"peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory() if args.device == "mps" else None,
              "parameter_dtypes": sorted({str(parameter.dtype) for parameter in base.parameters()}),
              "parameter_devices": sorted({str(parameter.device) for parameter in base.parameters()})}
    del model, base, hidden
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()
        memory["mps_driver_allocated_after_release_bytes"] = torch.mps.driver_allocated_memory()
    checks = {
        "pairs": lora["consumed_pairs"] == lora["file_pairs"] == lora["lora_layers"] == contract["lora"]["pairs"],
        "state_matches_file": not lora["missing"] and not lora["unexpected"],
        "scale": lora["scaling"] == [contract["lora"]["scale"]],
        "eval": not lora["training"],
        "card_examples": all(result["card_max_difference"] <= CARD_TOLERANCE
                             for result in results if "card_max_difference" in result)}
    write_json(output, {
        "passed": all(checks.values()), "checks": checks, "card_tolerance": CARD_TOLERANCE, "device": args.device,
        "dtype": "bfloat16", "lora": lora, "rows": results, "temperatures": temperatures,
        "timing_s": {"base_load": round(base_loaded - started, 1), "adapter_load": round(adapter_loaded - base_loaded, 1),
                     "forward_total": round(finished - loop_started, 1), "total": round(finished - started, 1)},
        "memory": memory})


if __name__ == "__main__":
    sys.exit(main())
