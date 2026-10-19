import argparse
import asyncio
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf

from ...storage import write_json
from .artifacts import compare_tensor
from .convert import load_record


async def verify(reference_root, exports, output):
    from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions

    if not SpecializationOptions.is_supported():
        raise RuntimeError("Verification requires the macOS Core AI runtime")
    reference = json.loads((reference_root / "reference.json").read_text())
    if not reference["complete"]:
        raise ValueError("The reference is incomplete")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    report = {"complete": False, "passed": False, "records": [], "loads": []}
    write_json(output / "comparison.json", report)
    for directory in exports:
        export = json.loads((directory / "export.json").read_text())
        if not export["complete"] or export["contract"] != reference["contract"]:
            raise ValueError("Export and reference identities do not match")
        options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
        started = time.perf_counter()
        model = await AIModel.load(directory / export["asset"], specialization_options=options)
        report["loads"].append({"kind": export["kind"], "seconds": time.perf_counter() - started})
        function = model.load_function("main")
        if function.desc.state_names:
            raise ValueError("This profile requires stateless functions")
        for record in reference["records"]:
            if record["kind"] != export["kind"]:
                continue
            inputs = load_record(reference_root, record, "inputs")
            expected = load_record(reference_root, record, "outputs")
            started = time.perf_counter()
            result = await function(inputs={name: NDArray(np.ascontiguousarray(value))
                                            for name, value in zip(export["input_names"], inputs, strict=True)}, state={})
            seconds = time.perf_counter() - started
            actual = [np.asarray(result[name].numpy()).copy() for name in export["output_names"]]
            metrics = [compare_tensor(wanted, observed, export["contract"]["normalized_max_limit"])
                       for wanted, observed in zip(expected, actual, strict=True)]
            passed = all(item["passed"] for item in metrics)
            if record["kind"] == "codec":
                audio = metrics[0]
                audio_passed = audio["finite"] and audio.get("cosine") is not None and audio["cosine"] >= export["contract"]["audio_cosine_min"] and audio["rms"] <= export["contract"]["audio_rms_max"]
                passed = passed and audio_passed
                sf.write(output / f"{record['id']}.wav", actual[0].reshape(-1), export["contract"]["sample_rate"], subtype="FLOAT")
            row = {"id": record["id"], "kind": record["kind"], "passed": passed,
                   "seconds": seconds, "outputs": metrics}
            report["records"].append(row)
            write_json(output / "comparison.json", report)
            print(json.dumps(row), flush=True)
        del function, model
    kinds = {item["kind"] for item in report["records"]}
    report["complete"] = kinds == {"encoder", "decoder", "local", "codec"}
    report["passed"] = report["complete"] and all(item["passed"] for item in report["records"])
    write_json(output / "comparison.json", report)
    if not report["passed"]:
        raise RuntimeError("The Core AI fidelity check did not pass")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--export", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(verify(args.reference, args.export, args.output))


if __name__ == "__main__":
    main()
