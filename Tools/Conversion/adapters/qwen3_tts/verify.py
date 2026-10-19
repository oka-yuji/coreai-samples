import argparse
import json
from pathlib import Path
import time

import soundfile as sf

from ...storage import write_json
from .convert import load_record
from .runtime import CoreAIComponents
from .source import compare_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--export", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = json.loads((args.reference / "reference.json").read_text())
    if not reference["complete"]:
        raise ValueError("The reference is incomplete")
    args.output.mkdir(parents=True)
    report = {"complete": False, "passed": False, "records": []}
    write_json(args.output / "comparison.json", report)
    contract = reference["contract"]
    components = CoreAIComponents(args.export, contract)
    try:
        for record in reference["records"]:
            inputs = load_record(args.reference, record, "inputs")
            expected = load_record(args.reference, record, "outputs")
            started = time.perf_counter()
            actual = components.call(record["kind"], inputs)
            seconds = time.perf_counter() - started
            metrics = [compare_tensor(wanted, observed, contract["normalized_max_limit"])
                       for wanted, observed in zip(expected, actual, strict=True)]
            passed = all(value["passed"] for value in metrics)
            if record["kind"] == "codec":
                metric = metrics[0]
                passed = passed and metric.get("cosine") is not None and metric["cosine"] >= contract["audio_cosine_min"] and metric["rms"] <= contract["audio_rms_max"]
                sf.write(args.output / f"{record['id']}.wav", actual[0].reshape(-1), contract["sample_rate"], subtype="FLOAT")
            row = {"id": record["id"], "kind": record["kind"], "passed": bool(passed), "seconds": seconds, "outputs": metrics}
            report["records"].append(row)
            write_json(args.output / "comparison.json", report)
            print(json.dumps(row), flush=True)
        kinds = {item["kind"] for item in report["records"]}
        positions = {position for position in (0, 255, 256, 511)
                     if any(item["id"].endswith(f"position-{position}") for item in report["records"])}
        report["complete"] = kinds == {"projection", "talker", "predictor", "codec"} and len(positions) == 4
        report["passed"] = report["complete"] and all(item["passed"] for item in report["records"])
        report["components"] = components.metrics
        write_json(args.output / "comparison.json", report)
    finally:
        components.close()
    if not report["passed"]:
        raise RuntimeError("The Core AI fidelity check did not pass")


if __name__ == "__main__":
    main()
