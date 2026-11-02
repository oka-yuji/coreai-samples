import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

from ...storage import read_json, write_json
from ..pinned import check_upstream, pinned_hub, sha256
from .export import RECIPE, Monitor
from .fixtures import load_rows
from .prepare import check_integrity, distributed_tokenizer, load_contract
from .reference import decision_head
from .typed_head import band, decide, render, tokenize
from .verify_quantization import compare_hashes

SETS = ("F", "Q")
STAGES = {"merged": "2", "base": "0q"}
BUDGET_S = 6 * 3600
MPS_TOLERANCE = 1e-3
SPEED_TOKENS = (0, 512, 2048)


def arguments():
    parser = argparse.ArgumentParser(description="Re-quantize a JEV-27B backbone and run F and Q in float32 (stage 2 or 0q).")
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    for name in ("model", "snapshot", "upstream", "corpus", "jglue", "fixtures", "scratch", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--merge", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    if args.stage == "merged" and (args.merge is None or args.export is None):
        parser.error("the merged stage needs --merge and --export")
    return args


def fixed_rows(args, contract, tokenizer):
    pins = read_json(args.fixtures / "jglue.json")
    check_integrity(args.corpus, contract["dataset"]["id"], contract["dataset"]["revision"])
    check_integrity(args.jglue, pins["repository"], pins["revision"])
    rows = []
    for name in SETS:
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


def identity(args, contract):
    if args.stage == "merged":
        revision = read_json(args.merge / "manifest.json")["merged_revision"]
        return {"model_id": contract["merged_model_id"], "revision": revision}, args.merge / "checkpoint"
    return {"model_id": contract["model_id"], "revision": contract["revision"]}, args.snapshot


class Runner:
    def __init__(self, backbone, torch, model, head):
        self.backbone, self.torch, self.model, self.head = backbone, torch, model, head

    def __call__(self, row, device):
        torch = self.torch
        weight, bias, temperatures = self.head
        count = len(row["tokens"])
        ids = torch.tensor([row["tokens"]], dtype=torch.int32, device=device)
        positions = torch.arange(count, dtype=torch.int32, device=device).unsqueeze(0)
        states = self.backbone.zero_states(self.model.config, torch.float32, count, device=device)
        begin = time.perf_counter()
        with torch.no_grad():
            hidden = self.model(ids, positions, *states)[0, -1].float().cpu()
        elapsed = time.perf_counter() - begin
        logits = (weight @ hidden + bias).tolist()
        if not all(math.isfinite(value) for value in logits):
            return hidden, None, None, None, elapsed
        shipped, plain = decide(logits, row["kind"], len(row["options"]), temperatures[row["kind"]])
        return hidden, logits, plain, shipped, elapsed


def speed_rows(rows):
    chosen = []
    for target in SPEED_TOKENS:
        row = min(rows, key=lambda item: (abs(item["token_count"] - target), item["token_count"]))
        if row not in chosen:
            chosen.append(row)
    return chosen


def estimate(timings, rows):
    import numpy

    counts, seconds = zip(*timings)
    degree = min(2, len(set(counts)) - 1)
    if degree == 0:
        rate = seconds[0] / max(counts[0], 1)
        return {"model": "per-token", "coefficients": [rate], "estimate_s": rate * sum(row["token_count"] for row in rows)}
    coefficients = numpy.polyfit(counts, seconds, degree)
    if degree == 2 and coefficients[0] < 0:
        degree, coefficients = 1, numpy.polyfit(counts, seconds, 1)
    total = float(sum(max(numpy.polyval(coefficients, row["token_count"]), 0.0) for row in rows))
    return {"model": "quadratic" if degree == 2 else "linear", "coefficients": [float(value) for value in coefficients],
            "estimate_s": total}


def measure(run, rows, device, forwarded):
    run(rows[0], device)
    timings = [(row["token_count"], run(row, device)[4]) for row in rows]
    return {"rows": [{"id": row["id"], "tokens": count, "seconds": round(seconds, 3)} for row, (count, seconds)
                     in zip(rows, timings)],
            "tokens_per_s": round(sum(count for count, _ in timings) / sum(seconds for _, seconds in timings), 2),
            **estimate(timings, forwarded)}


def largest(first, second):
    return max(abs(a - b) for a, b in zip(first, second, strict=True))


def choose_device(run, model, rows, requested, started):
    forwarded = [row for row in rows if not row["rejected"]]
    plan = {"requested": requested, "budget_s": BUDGET_S, "mps_tolerance": MPS_TOLERANCE}
    if requested != "auto":
        if requested == "mps":
            model.to("mps")
        return requested, plan
    plan["cpu"] = measure(run, speed_rows(forwarded), "cpu", forwarded)
    plan["elapsed_before_rows_s"] = round(time.perf_counter() - started, 1)
    if plan["elapsed_before_rows_s"] + plan["cpu"]["estimate_s"] <= BUDGET_S:
        return "cpu", plan
    short = sorted(forwarded, key=lambda row: (row["token_count"], row["id"]))[:3]
    on_cpu = [run(row, "cpu") for row in short]
    model.to("mps")
    on_mps = [run(row, "mps") for row in short]
    plan["mps_check"] = [{"id": row["id"], "tokens": row["token_count"], "cpu": first[2], "mps": second[2],
                          "cpu_logits": first[1], "mps_logits": second[1],
                          "difference": largest(first[2], second[2]) if first[2] and second[2] else None}
                         for row, first, second in zip(short, on_cpu, on_mps)]
    differences = [entry["difference"] for entry in plan["mps_check"]]
    plan["mps_max_difference"] = None if None in differences else max(differences)
    if plan["mps_max_difference"] is None or plan["mps_max_difference"] > MPS_TOLERANCE:
        return None, {**plan, "stopped": "MPS float32 differs from CPU float32 beyond the tolerance"}
    plan["mps"] = measure(run, speed_rows(forwarded), "mps", forwarded)
    plan["elapsed_before_rows_s"] = round(time.perf_counter() - started, 1)
    if plan["elapsed_before_rows_s"] + plan["mps"]["estimate_s"] > BUDGET_S:
        return None, {**plan, "stopped": "Neither CPU nor MPS fits the stage budget"}
    return "mps", plan


def forward_all(run, rows, stage, device, hidden):
    keys = (f"logits{stage}", f"p{stage}", f"p{stage}_T", f"elapsed{stage}_s")
    totals = {name: 0.0 for name in SETS}
    nonfinite = []
    index = 0
    for row in rows:
        if row["rejected"]:
            row.update(dict.fromkeys(keys))
        else:
            state, logits, plain, shipped, elapsed = run(row, device)
            row.update(zip(keys, (logits, plain, shipped, round(elapsed, 4))))
            totals[row["set"]] += elapsed
            if logits is None:
                nonfinite.append(row["id"])
            elif row["set"] == "F":
                hidden[index] = state
            print(stage, row["set"], row["id"], row["token_count"], f"{elapsed:.2f}s", flush=True)
        index += row["set"] == "F"
    tokens = sum(row["token_count"] for row in rows if not row["rejected"])
    return {**{f"forward_{name}_s": round(value, 1) for name, value in totals.items()},
            "tokens_per_s": round(tokens / max(sum(totals.values()), 1e-9), 2)}, nonfinite


def main():
    args = arguments()
    stage = STAGES[args.stage]
    output = args.output / f"reference_q-{args.stage}.json"
    if output.exists():
        raise FileExistsError(f"{output.name} must be written to a new directory")
    contract = load_contract(args.model)
    check_upstream(args.upstream, contract)
    source, folder = identity(args, contract)
    recipe = args.upstream.resolve() / RECIPE
    head = decision_head(args.snapshot, contract)
    started = time.perf_counter()
    timing = {}
    probe = {"stage": stage, "model_id": source["model_id"], "revision": source["revision"], "requested_device": args.device}
    args.output.mkdir(parents=True, exist_ok=True)
    monitor = Monitor(args.output / "monitor.tsv", [args.output, args.scratch],
                      [str(args.scratch / "layers"), str(args.scratch / "quantized")],
                      lambda reason: write_json(args.output / "probe.json", {**probe, "aborted": reason}))
    monitor.start()
    try:
        with pinned_hub(folder, source):
            import torch

            from . import backbone

            backbone.check_location(args.upstream)
            model_type = backbone.COMPOSITE if args.stage == "merged" else backbone.FLAT
            backbone.install(backbone.pipeline, model_type)
            rows = fixed_rows(args, contract, distributed_tokenizer(folder))
            timing["rows_s"] = round(time.perf_counter() - started, 1)
            config = backbone.export_config(contract, recipe, source["model_id"], args.scratch / "export", "probe")
            resolved = backbone.pipeline.resolve_model(config)
            if resolved.model_type != model_type or resolved.model_class is not backbone.TypedDecisionBackbone:
                raise ValueError(f"Unexpected resolution: {resolved.model_type}")
            loaded, marks = {}, {"built": time.perf_counter()}
            quantize = backbone.pipeline.quantize_for_export

            def quantized(model, hf_config, *rest, **options):
                loaded.update(backbone.load_check(model, hf_config, torch.bfloat16))
                if not loaded["passed"]:
                    raise ValueError(f"Load check failed: {loaded['checks']}")
                marks["quantize"] = time.perf_counter()
                return quantize(model, hf_config, *rest, **options)

            backbone.pipeline.quantize_for_export = quantized
            model = backbone.pipeline.build_model(resolved, temp_dir=str(args.scratch)).model
            hashed = time.perf_counter()
            timing.update(load_s=round(marks["quantize"] - marks["built"], 1), quantize_s=round(hashed - marks["quantize"], 1))
            devices = sorted({tensor.device.type for tensor in [*model.parameters(), *model.buffers()]})
            record = backbone.hash_record(model, {**source, "model_type": model_type, "device": ",".join(devices),
                                                  "recipe_sha256": sha256(recipe), "compute_precision": contract["precision"],
                                                  "compression": contract["compression"],
                                                  "context_length": contract["context_length"]})
            write_json(args.output / ("quantized-hashes.json" if args.stage == "merged" else "quantized-hashes-base.json"),
                       record)
            timing["hash_s"] = round(time.perf_counter() - hashed, 1)
            probe.update(load_check=loaded, hashes={"items": record["count"], "bytes": record["bytes"]})
            if args.stage == "merged":
                comparison = compare_hashes(read_json(args.export / "quantized-hashes.json"), record)
                probe.update(hashes_match=comparison["hashes_match"], hash_comparison=comparison)
                if not comparison["hashes_match"]:
                    write_json(args.output / "probe.json", {**probe, "passed": False, "timing": timing,
                                                            "monitor": monitor.finish()})
                    return 1
            floated = time.perf_counter()
            model = model.float()
            timing["float_s"] = round(time.perf_counter() - floated, 1)
            probe["dtypes_after_float"] = sorted({str(t.dtype) for t in [*model.parameters(), *model.buffers()]})
            run = Runner(backbone, torch, model, head)
            device, plan = choose_device(run, model, rows, args.device, started)
            probe.update(device=device, plan=plan)
            if device is None:
                write_json(args.output / "probe.json", {**probe, "passed": False, "timing": timing,
                                                        "monitor": monitor.finish()})
                return 1
            count = sum(row["set"] == "F" for row in rows)
            hidden = torch.zeros(count, model.config.hidden_size, dtype=torch.float32)
            forward, nonfinite = forward_all(run, rows, stage, device, hidden)
            timing.update(forward)
            mask = torch.tensor([row[f"logits{stage}"] is not None for row in rows if row["set"] == "F"], dtype=torch.uint8)
            versions = record["versions"]
        from safetensors.torch import save_file

        save_file({"hidden": hidden, "mask": mask}, str(args.output / f"hidden{stage}.safetensors"),
                  metadata={"format": "pt", "stage": stage, "model_id": source["model_id"], "revision": source["revision"],
                            "ids": json.dumps([row["id"] for row in rows if row["set"] == "F"])})
    except BaseException as error:
        write_json(args.output / "probe.json", {**probe, "passed": False, "failed": f"{type(error).__name__}: {error}",
                                                "timing": timing, "monitor": monitor.finish()})
        raise
    timing["total_s"] = round(time.perf_counter() - started, 1)
    record_monitor = monitor.finish()
    rejected = sum(row["rejected"] for row in rows)
    forwarded = {name: sum(row["set"] == name and not row["rejected"] for row in rows) for name in SETS}
    passed = probe.get("hashes_match", True) and not nonfinite and record_monitor["aborts"] == 0
    memory = {"peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "monitor": record_monitor}
    write_json(output, {"stage": stage, "model_id": source["model_id"], "revision": source["revision"], "device": device,
                        "dtype": "float32", "temperatures": head[2], "versions": versions, "timing": timing,
                        "memory": memory, "hashes_match": probe.get("hashes_match"), "rows": rows})
    write_json(args.output / "probe.json", {**probe, "passed": passed, "rows": {"total": len(rows), "rejected": rejected,
                                                                                "forwarded": forwarded},
                                            "nonfinite": nonfinite, "timing": timing, "memory": memory})
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
