import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from Tools.Conversion.local import default_lock
from . import reference
from .score import load_set, result_prediction, score_set


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepared_inputs(sets, out):
    from PIL import Image

    records = {}
    for label, directory in sets.items():
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        expected = {row["id"]: row for row in manifest["samples"]}
        gts = load_set(directory)
        if set(expected) != set(gts):
            raise ValueError("Evaluation inputs differ from the manifest")
        rows = {}
        for sid, gt in gts.items():
            if Path(sid).name != sid or sid in (".", ".."):
                raise ValueError("Invalid sample identifier")
            if reference.sha256(directory / f"{sid}.json") != expected[sid]["gt_sha256"]:
                raise ValueError(f"Input hash mismatch: {sid}")
            source = reference.image_path(directory, gt)
            if reference.sha256(source) != expected[sid]["image_sha256"]:
                raise ValueError(f"Input hash mismatch: {sid}")
            image = Image.open(source).convert("RGB")
            crop = gt.get("crop_px")
            if crop:
                image = image.crop(tuple(crop))
            target = out / label / f"{sid}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target)
            write(target.with_suffix(".json"), gt)
            rows[sid] = {"image": str(target), "image_sha256": reference.sha256(target), "source_sha256": reference.sha256(source), "crop_px": crop}
        records[label] = rows
    write(out / "inputs.json", records)
    return records


def output_comparison(left, right):
    def signature(record):
        return {"layout_text": record["layout"]["text"], "layout_status": record["layout"]["status"],
                "skipped_boxes": record["layout"].get("skipped_boxes", []),
                "blocks": [{key: b.get(key) for key in ("label", "box_norm", "box_px", "prompt", "text", "otsl")} for b in record["blocks"]]}
    def complete(record):
        return all(row["finished_by"] == "eos" for row in [record["layout"], *record["blocks"]])
    return {"exact_prediction": signature(left) == signature(right), "both_eos": complete(left) and complete(right),
            "text_equal": result_prediction(left)["text"] == result_prediction(right)["text"]}


def size(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def metrics(record, variant, cache_root, cold):
    result = {"latency_s": {"value": record["generate_s"], "unit": "s"},
              "first_load_s" if cold else "warm_load_s": {"value": record["load_s"], "unit": "s"},
              "bundle_bytes": {"value": size(variant["bundle"]), "unit": "B"}}
    for source, target in [("peak_rss_bytes", "peak_rss_bytes"), ("footprint_peak_bytes", "peak_footprint_bytes")]:
        if record.get(source) is not None:
            result[target] = {"value": record[source], "unit": "B"}
    hashes = {row["main_hash"] for row in record["assets"]}
    if not all((Path(cache_root) / key).is_dir() for key in hashes):
        raise ValueError("Specialization cache entries could not be located")
    result["prepared_bytes"] = {"value": sum(size(Path(cache_root) / key) for key in hashes), "unit": "B"}
    result["generated_tokens"] = {"value": sum(row["tokens"]["output"] for row in [record["layout"], *record["blocks"]]), "unit": "tokens"}
    return result


def run_child(plan, variant, image, directory, cold, owner):
    directory.mkdir(parents=True, exist_ok=False)
    if variant["engine"] == "coreai":
        command = [plan["cli"], "ocr", image, "--out", str(directory)]
        command.insert(2, variant["bundle"])
        if cold:
            command.append("--clear-cache")
        output = directory / (Path(image).stem + ".json")
    else:
        config = directory / "variant.json"
        write(config, variant)
        output = directory / "reference.json"
        command = [sys.executable, "-m", "Tools.OCREval.compare", "--child", str(config), image, str(output)]
    with (directory / "process.log").open("w") as log:
        process = subprocess.Popen(command, cwd=reference.ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   env=os.environ | {"HF_HUB_OFFLINE": "1", "OCR_EVAL_LOCK_OWNER": owner})
        try:
            code = process.wait(timeout=1800)
        except BaseException:
            reference.stop(process)
            raise
    if code:
        raise RuntimeError(f"Child process exited with {code}; see {directory / 'process.log'}")
    result = json.loads(output.read_text())
    if result["image_sha256"] != reference.sha256(image):
        raise ValueError("The child read a different image")
    if variant["engine"] == "coreai":
        cached = [asset["cached"] for asset in result["assets"]]
        if cold and any(cached):
            raise ValueError("A cold-load measurement reused specialization cache")
        if result["executable_sha256"] != plan["cli_sha256"]:
            raise ValueError("The measured executable differs from the fixed plan")
    return result


def compare_records(plan, inputs, out):
    result = {}
    for label, samples in inputs.items():
        result[label] = {}
        for pair in plan["comparisons"]:
            left, right = pair["left"], pair["right"]
            rows = {}
            for sid in samples:
                paths = [out / "results" / label / variant / f"{sid}.json" for variant in (left, right)]
                rows[sid] = output_comparison(*(json.loads(p.read_text()) for p in paths)) if all(p.exists() for p in paths) else {"missing": True}
            result[label][pair["id"]] = {"axis": pair["axis"], "samples": rows}
    write(out / "comparisons.json", result)


def run_plan(plan_path, out, only=None, performance=True):
    plan = json.loads(plan_path.read_text())
    out.mkdir(parents=True, exist_ok=False)
    plan_hash = reference.sha256(plan_path)
    write(out / "plan.json", plan)
    state, runs, aborted = {}, [], None
    lock = default_lock()
    owner = f"session: Tools.OCREval.compare\njob: teleocr-333\nstarted: {reference.now()}\npid: {os.getpid()}\nexpected: fixed evaluation plan {plan_hash}\n"
    def on_signal(signum, _):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, on_signal)
    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:
        signal.signal(signal.SIGHUP, on_signal)
    try:
        reference.acquire(lock, lambda: owner, 0, print, state)
        if reference.sha256(Path(plan["cli"])) != plan["cli_sha256"]:
            raise ValueError("CLI hash differs from the plan")
        if plan.get("source_cache"):
            os.environ["HF_HUB_CACHE"] = plan["source_cache"]
        env = reference.environment()
        env["power"] = reference.command("pmset", "-g", "batt")
        env["thermal"] = reference.command("pmset", "-g", "therm")
        env["power_settings"] = reference.command("pmset", "-g")
        env["xcode"] = reference.command("xcodebuild", "-version")
        env["swift"] = reference.command("xcrun", "swift", "--version")
        write(out / "environment.json", env)
        inputs = prepared_inputs(plan["sets"], out / "inputs")
        unknown = set(only or []) - {sid for samples in inputs.values() for sid in samples}
        if unknown:
            raise ValueError(f"Unknown samples: {sorted(unknown)}")
        for variant in plan["variants"]:
            if variant["engine"] == "coreai":
                for file, expected in variant["asset_hashes"].items():
                    if reference.sha256(Path(variant["bundle"]) / file) != expected:
                        raise ValueError("Bundle differs from the fixed plan")
        def execute(variant, label, sid, phase, iteration=0):
            run_id = f"{phase}-{iteration}-{label}-{sid}-{variant['id']}"
            row = {"run_id": run_id, "variant": variant["id"], "set": label, "sample": sid, "phase": phase, "iteration": iteration,
                   "context": {"model": "teleocr", "variant": variant["id"], "device": env["machine"]["cpu"], "os": env["os"]["build"],
                               "workload": f"two-stage:{label}:{sid}:4096", "cache": "cold" if phase == "cold" else "observed-per-asset" if phase == "quality" else "prepared", "plan_sha256": plan_hash}}
            started = time.monotonic()
            try:
                if phase != "quality":
                    row["conditions_before"] = {"power": reference.command("pmset", "-g", "batt"), "thermal": reference.command("pmset", "-g", "therm")}
                record = run_child(plan, variant, inputs[label][sid]["image"], out / "processes" / run_id, phase == "cold", owner)
                if phase in ("warmup", "measured") and not all(asset["cached"] for asset in record["assets"]):
                    raise ValueError("A prepared-load measurement specialized an asset")
                eos = all(item["finished_by"] == "eos" for item in [record["layout"], *record["blocks"]])
                row["status"] = "warmup" if phase == "warmup" and eos else "passed" if eos else "failed"
                row["all_eos"] = eos
                row["metrics"] = metrics(record, variant, plan["cache_root"], phase == "cold") if phase != "quality" else {}
                if phase == "quality":
                    write(out / "results" / label / variant["id"] / f"{sid}.json", record)
                else:
                    row["assets_cached"] = {a["name"]: a["cached"] for a in record["assets"]}
                    row["conditions_after"] = {"power": reference.command("pmset", "-g", "batt"), "thermal": reference.command("pmset", "-g", "therm")}
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                row |= {"status": "failed", "error": f"{type(error).__name__}: {error}", "metrics": {}}
            row["wall_s"] = time.monotonic() - started
            runs.append(row)
            write(out / "runs.json", runs)
            print(run_id, row["status"], round(row["wall_s"], 2), flush=True)
        for label, samples in inputs.items():
            for sid in samples:
                if only and sid not in only:
                    continue
                for variant in plan["variants"]:
                    execute(variant, label, sid, "quality")
        if performance:
            label, sid = plan["performance"]["set"], plan["performance"]["sample"]
            variants = [v for v in plan["variants"] if v["engine"] == "coreai"]
            for variant in variants:
                execute(variant, label, sid, "cold")
            for phase, count in [("warmup", 1), ("measured", 20)]:
                for iteration in range(count):
                    for variant in variants:
                        execute(variant, label, sid, phase, iteration)
        compare_records(plan, inputs, out)
        for label in plan["sets"]:
            for variant in plan["variants"]:
                write(out / "scores" / label / f"{variant['id']}.json", score_set(out / "inputs" / label, results=out / "results" / label / variant["id"]))
    except BaseException as error:
        aborted = f"{type(error).__name__}: {error}"
        raise
    finally:
        write(out / "invocation.json", {"plan_sha256": plan_hash, "only": only, "performance": performance,
                                       "aborted": aborted, "finished_at": reference.now(), "runs_recorded": len(runs)})
        if state.get("owner"):
            if (lock / "owner.txt").read_text() != owner:
                raise RuntimeError("Lock ownership changed; leaving the lock in place")
            lock.rename(lock.with_name("heavy-job-done-333-" + datetime.now().strftime("%Y%m%dT%H%M%S%f")))
    return runs


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        owner = os.environ.get("OCR_EVAL_LOCK_OWNER")
        if not owner or (default_lock() / "owner.txt").read_text() != owner:
            raise RuntimeError("Reference child requires the parent's shared model lock")
        from .page_reference import run
        run(Path(sys.argv[3]), Path(sys.argv[4]), json.loads(Path(sys.argv[2]).read_text()))
        return
    parser = argparse.ArgumentParser(description="Compare fixed TeleOCR precision and reference conditions under the shared lock.")
    parser.add_argument("plan", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--no-performance", action="store_true")
    args = parser.parse_args()
    runs = run_plan(args.plan.resolve(), args.out.resolve(), args.only, not args.no_performance)
    if any(row["status"] == "failed" for row in runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
