import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from ...storage import read_json, write_json
from .checks import Checks, aggregate, fsq_summary, verdict
from .runtime import Components
from .source import load_contract, with_capacity

DTYPES = {"fp32": np.float32, "fp16": np.float16}


def measure(call, warmup, repeats):
    for _ in range(warmup):
        call()
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        values.append(1000 * (time.perf_counter() - started))
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
            "min_ms": float(np.min(values)), "count": repeats}


def workloads(components, contract, reference, record, dtype, assets):
    checks = Checks(components, contract, reference, record, dtype)
    size, length = contract["execution"]["prefill_length"], record["length"]
    load, floats = checks.load, checks.floats

    def padded(name):
        value = load(name)
        return checks.padded(value, np.zeros((1, size - length, value.shape[-1]), np.float32))

    first = {"length": np.array([length], np.int32)}
    position = {"position": np.array([length], np.int32)}
    rows = {
        ("tslm", "embed"): {"ids": np.concatenate((load("ids"), np.zeros((1, size - length), np.int32)), axis=1)},
        ("tslm", "prefill"): {"embeds": padded("embeds"), **first},
        ("tslm", "step"): {"embed": floats(load("tslm_step_in")[0]), **position},
        ("ralm", "prefill"): {"embeds": padded("prefill_fuse_out"), **first},
        ("ralm", "step"): {"embed": floats(load("ralm_step_in")[0]), **position},
        ("proj", "heads"): {"lm_hidden": floats(load("lm_hidden_in")[0]), "residual_hidden": floats(load("residual_hidden_in")[0])},
        ("proj", "fuse"): {"lm_raw": floats(load("tslm_step_out")[0]), "curr_embed": floats(load("tslm_step_in")[0])},
        ("proj", "prefill_fuse"): {"hidden": padded("tslm_prefill_out")},
        ("locenc", "main"): {"patch": floats(load("patch")[0])},
        ("dit", "main"): {name: floats(load(f"dit_{name}")[0, 1]) for name in ("x", "mu", "t", "cond", "dt")},
        ("vae", "main"): {"latent": floats(load("latent")[..., :contract["architecture"]["patch_size"]])},
    }
    return {key: value for key, value in rows.items() if key[0] in assets}


def timings(components, contract, reference, record, dtype, assets, rounds, warmup, repeats, keys=None):
    work = workloads(components, contract, reference, record, dtype, assets)
    result = {}
    for round_index in range(rounds):
        for (asset, function), values in work.items():
            if keys is not None and (asset, function) not in keys:
                continue
            state = components.state(asset, function)
            result.setdefault(f"{asset}.{function}", []).append(
                measure(lambda: components.call(asset, function, values, state), warmup, repeats))
    return result


def environment():
    def run(command):
        return subprocess.run(command, capture_output=True, text=True, timeout=30).stdout.strip()

    return {"machine": run(["/usr/sbin/sysctl", "-n", "hw.model"]), "os_build": run(["/usr/bin/sw_vers", "-buildVersion"]),
            "platform": platform.platform(), "load_average": list(os.getloadavg())}


def check(args):
    contract = load_contract(args.contract)
    reference = read_json(args.reference / "reference.json")
    if not reference["complete"] or reference["contract"] != contract:
        raise ValueError("The reference is incomplete or was captured for another contract")
    args.output.mkdir(parents=True)
    report = {"complete": False, "passed": False, "precision": args.precision,
              "criterion": {"normalized_max_limit": contract["criteria"]["normalized_max_limit"], "nonfinite": 0,
                            "gates_result": args.precision == "fp32"}, "environment_before": environment()}
    write_json(args.output / "comparison.json", report)
    components = Components(args.export, args.precision, contract)
    try:
        assets = [asset for asset in contract["assets"] if asset in components.exports]
        if sorted(assets) != sorted(contract["assets"]):
            raise ValueError("Every contract asset is required")
        rows = []
        for record in reference["records"]:
            rows.extend(Checks(components, contract, args.reference, record, DTYPES[args.precision]).run(assets))
        write_json(args.output / "rows.json", rows)
        report["summary"] = aggregate(rows)
        report["fsq"] = fsq_summary(rows, contract["criteria"])
        report["criterion_met"] = verdict(rows, contract["criteria"])
        report["fixtures"] = {record["id"]: {"length": record["length"], "steps": record["steps"]} for record in reference["records"]}
        report["timing"] = timings(components, contract, args.reference, reference["records"][0], DTYPES[args.precision],
                                   assets, args.rounds, args.warmup, args.repeats)
        report["load_seconds"] = components.load_seconds
        report["assets"] = {asset: {key: export["precisions"][args.precision][key] for key in ("file", "bytes", "main_mlirb_sha256")}
                            for asset, export in components.exports.items()}
    finally:
        components.close()
    report["environment_after"] = environment()
    report["passed"] = report["criterion_met"] if args.precision == "fp32" else None
    report["complete"] = True
    write_json(args.output / "comparison.json", report)
    print(json.dumps({"precision": args.precision, "criterion_met": report["criterion_met"],
                      "gated_rows": sum(row["gate"] for row in rows)}), flush=True)
    if args.precision == "fp32" and not report["passed"]:
        raise RuntimeError("The Core AI FP32 comparison did not meet the criterion")


def capacity(args):
    contract = load_contract(args.contract)
    reference = read_json(args.reference / "reference.json")
    if not reference["complete"] or reference["contract"] != contract:
        raise ValueError("The reference is incomplete or was captured for another contract")
    args.output.mkdir(parents=True)
    report = {"complete": False, "precision": "fp32", "environment_before": environment(), "capacities": {}}
    write_json(args.output / "capacity.json", report)
    groups = {contract["execution"]["capacity"]: (args.export, contract), args.capacity: (args.compare, with_capacity(contract, args.capacity))}
    loaded = {size: Components(directories, "fp32", expected) for size, (directories, expected) in groups.items()}
    try:
        for size, components in loaded.items():
            rows = []
            for record in reference["records"]:
                rows.extend(Checks(components, groups[size][1], args.reference, record, np.float32).run(["tslm", "ralm"]))
            report["capacities"][str(size)] = {"criterion_met": verdict(rows, contract["criteria"]), "summary": aggregate(rows),
                                               "load_seconds": components.load_seconds, "timing": {}}
        keys = [("tslm", "prefill"), ("tslm", "step"), ("ralm", "prefill"), ("ralm", "step")]
        for _ in range(args.rounds):
            for size, components in loaded.items():
                measured = timings(components, groups[size][1], args.reference, reference["records"][0], np.float32,
                                   ["tslm", "ralm"], 1, args.warmup, args.repeats, keys)
                for key, values in measured.items():
                    report["capacities"][str(size)]["timing"].setdefault(key, []).extend(values)
    finally:
        for components in loaded.values():
            components.close()
    report["environment_after"] = environment()
    report["complete"] = True
    write_json(args.output / "capacity.json", report)
    print(json.dumps({size: value["criterion_met"] for size, value in report["capacities"].items()}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["check", "capacity"])
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--export", type=Path, action="append", required=True)
    parser.add_argument("--compare", type=Path, action="append")
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    check(args) if args.mode == "check" else capacity(args)


if __name__ == "__main__":
    main()
