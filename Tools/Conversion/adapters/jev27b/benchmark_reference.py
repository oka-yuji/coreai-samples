import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

from ...storage import read_json, write_json
from .load_check import lora_summary
from .performance import LENGTHS, ROUNDS
from .prepare import check_integrity, distributed_tokenizer, load_contract
from .reference import decision_head, last_hidden, PACKAGES
from .typed_head import decide, render, tokenize


def token_hash(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def prepare_rows(document, manifest, tokenizer):
    expected = {row["tokens_sha256"]: row for row in manifest["rows"]}
    if {(row["tokens"], row["kind"]) for row in expected.values()} != set(LENGTHS):
        raise ValueError("The manifest must identify the four frozen P inputs")
    rows = document["rows"]
    if len(rows) != len(LENGTHS) * (ROUNDS + 1):
        raise ValueError("The schedule requires one warmup and three measured rounds")
    prepared = []
    for index, row in enumerate(rows):
        ids = tokenize(tokenizer, render(row["kind"], row["state"], row["question"], row["options"]))
        digest = token_hash(ids)
        count, kind = LENGTHS[index % len(LENGTHS)]
        found = expected.get(digest)
        if found is None or len(ids) != count or row["kind"] != kind or len(row["options"]) != found["n_options"]:
            raise ValueError(f"The frozen input differs at schedule position {index}")
        prepared.append({**row, "token_count": len(ids), "tokens_sha256": digest,
                         "warmup": index < len(LENGTHS), "round": index // len(LENGTHS)})
    return prepared


def wait_nominal(probe, limit=600, clock=time.monotonic, sleep=time.sleep):
    started = clock()
    while True:
        state = probe()
        if state not in (0, 1, 2, 3):
            raise ValueError("Unknown thermal state")
        if state == 0:
            return clock() - started
        if clock() - started >= limit:
            raise TimeoutError("The nominal thermal gate timed out")
        sleep(1)


def measure(row, tokenizer, infer, temperature, synchronize, clock=time.perf_counter):
    synchronize()
    started = clock()
    text = render(row["kind"], row["state"], row["question"], row["options"])
    ids = tokenize(tokenizer, text)
    logits = infer(ids)
    shipping, plain = decide(logits, row["kind"], len(row["options"]), temperature)
    synchronize()
    elapsed = clock() - started
    if token_hash(ids) != row["tokens_sha256"] or len(ids) != row["token_count"]:
        raise ValueError("The measured token sequence changed")
    return {key: row[key] for key in ("id", "kind", "token_count", "tokens_sha256", "warmup", "round")} | {
        "logits": logits, "probabilities": shipping, "probabilities_t1": plain,
        "elapsed_seconds": elapsed, "temperature": temperature,
    }


def validate_lora(summary, *, pairs, scale, enabled, total, merged):
    checks = {
        "pairs": summary["consumed_pairs"] == summary["file_pairs"] == enabled == total == pairs,
        "weights": not summary["missing"] and not summary["unexpected"],
        "scale": summary["scaling"] == [scale], "eval": not summary["training"], "unmerged": merged == 0,
    }
    if not all(checks.values()):
        raise ValueError("The active LoRA differs from the frozen reference contract")
    return checks


def arguments():
    parser = argparse.ArgumentParser(description="Measure the frozen JEV-27B System 1 PyTorch/MPS reference.")
    for name in ("model", "snapshot", "input", "manifest", "thermal-helper", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()


def main():
    args = arguments()
    if args.output.exists():
        raise FileExistsError("The output directory must be new")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    status = {"schema": "jev27b-reference-benchmark/1", "status": "preparing", "rows_completed": 0,
              "started_at": datetime.now(timezone.utc).isoformat(), "device": "mps", "dtype": "bfloat16",
              "adapter": "unmerged, enabled", "head": "float32 CPU, 24 slots", "batch_size": 1, "concurrency": 1}

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("Termination requested")

    signal.signal(signal.SIGTERM, interrupted)
    write_json(args.output / "status.json", status)
    try:
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
            raise RuntimeError("Disable MPS CPU fallback for this comparison")
        import torch
        from peft import PeftModel
        from peft.tuners.lora import LoraLayer
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is required; this benchmark does not fall back to CPU")
        contract = load_contract(args.model)
        check_integrity(args.snapshot, contract["model_id"], contract["revision"])
        tokenizer = distributed_tokenizer(args.snapshot)
        rows = prepare_rows(read_json(args.input), read_json(args.manifest), tokenizer)
        head = decision_head(args.snapshot, contract)
        prepared = time.perf_counter()
        base = AutoModelForCausalLM.from_pretrained(str(args.snapshot), dtype=torch.bfloat16, device_map="mps")
        base_loaded = time.perf_counter()
        model = PeftModel.from_pretrained(base, str(args.snapshot / "adapter"))
        model.eval()
        torch.mps.synchronize()
        adapter_loaded = time.perf_counter()
        summary = lora_summary(model, load_file(str(args.snapshot / "adapter/adapter_model.safetensors")))
        layers = [module for module in model.modules() if isinstance(module, LoraLayer)]
        checks = validate_lora(
            summary, pairs=contract["lora"]["pairs"], scale=contract["lora"]["scale"], total=len(layers),
            enabled=sum(not layer.disable_adapters and bool(layer.active_adapters) for layer in layers),
            merged=sum(bool(layer.merged) for layer in layers))
        if not all(parameter.device.type == "mps" for parameter in model.parameters()):
            raise RuntimeError("Every model parameter must reside on MPS")
        status.update(status="running", model_id=contract["model_id"], revision=contract["revision"], checks=checks,
                      versions={name: importlib.metadata.version(name) for name in PACKAGES},
                      input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                      manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                      load_seconds={"prepare": prepared-started, "base": base_loaded-prepared,
                                    "adapter": adapter_loaded-base_loaded, "total_before_checks": adapter_loaded-started},
                      cpu_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                      os=platform.platform(), parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                      parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                      parameter_devices=sorted({str(p.device) for p in model.parameters()}),
                      unused_generation_head_bytes=base.lm_head.weight.numel()*base.lm_head.weight.element_size(),
                      generation_head_forwarded=False, warmup_rows=len(LENGTHS), measured_rows=len(LENGTHS)*ROUNDS)
        write_json(args.output / "status.json", status)

        def infer(ids):
            hidden = last_hidden(model, ids)
            return (head[0] @ hidden + head[1]).tolist()

        def thermal():
            return int(subprocess.check_output([str(args.thermal_helper)], text=True).strip())

        results = []
        with (args.output / "rows.jsonl").open("w") as stream:
            for index, row in enumerate(rows):
                waited = wait_nominal(thermal)
                at = datetime.now(timezone.utc).isoformat()
                result = measure(row, tokenizer, infer, head[2][row["kind"]], torch.mps.synchronize)
                result.update(started=at, thermal_state_at_start="nominal", thermal_wait_seconds=waited,
                              mps_current_allocated_bytes=torch.mps.current_allocated_memory(),
                              mps_driver_allocated_bytes=torch.mps.driver_allocated_memory())
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                results.append(result)
                status["rows_completed"] = len(results)
                write_json(args.output / "status.json", status)
                print(f"[{index+1}/{len(rows)}] {result['token_count']} tokens {result['elapsed_seconds']:.6f}s", flush=True)
        write_json(args.output / "result.json", {**status, "status": "completed", "rows": results})
        status["status"] = "completed"
    except BaseException as error:
        status.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        status.update(elapsed_seconds=time.perf_counter()-started,
                      max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      finished_at=datetime.now(timezone.utc).isoformat())
        write_json(args.output / "status.json", status)


if __name__ == "__main__":
    main()
