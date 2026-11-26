import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from ...runner import deferred_interrupts
from ...storage import write_json
from . import mtp
from .layers import WEIGHTS

SCHEMA = "gemma4-31b-mtp-native-inputs/1"
INPUTS = ("input_ids", "hidden_states", "position_ids", "k_sliding", "v_sliding", "k_full", "v_full")
TYPES = {"input_ids": "int32", "hidden_states": "float16", "position_ids": "int32", "k_sliding": "float16",
         "v_sliding": "float16", "k_full": "float16", "v_full": "float16"}
FLOATS = tuple(name for name in INPUTS if TYPES[name] == "float16")
ARRAYS = {"logits": "float32", "drafts": "int32", "top2_ids": "int32", "top2_probabilities": "float64"}
LITTLE = {"int32": "<i4", "float16": "<f2", "float32": "<f4", "float64": "<f8"}
PACKAGES = ("numpy", "safetensors", "torch")


def rounded(arguments):
    import torch

    half, reference, statistics = [], [], {}
    for name, value in zip(INPUTS, arguments):
        if name not in FLOATS:
            half.append(value)
            reference.append(value)
            continue
        low = value.to(torch.float16)
        restored = low.to(torch.float32)
        finite = torch.isfinite(restored)
        difference = (restored - value).abs()[finite]
        statistics[name] = {"elements": value.numel(), "changed": int((restored != value).sum()),
                            "largest_difference": float(difference.max()) if difference.numel() else 0.0,
                            "non_finite": int((~finite).sum())}
        half.append(low)
        reference.append(restored)
    return tuple(half), tuple(reference), statistics


def top_two(logits):
    import torch

    wide = logits.to(torch.float64)
    ids = torch.sort(wide, descending=True, stable=True).indices[:2]
    return ids.tolist(), torch.softmax(wide, dim=-1)[ids].tolist()


def reference_row(model, arguments, steps):
    import torch

    with torch.no_grad():
        logits, _ = model.outputs(*arguments)
        drafts = model(*arguments)[0].tolist()
        token, hidden, repeated, ids, probabilities = arguments[0], arguments[1], [], [], []
        for _ in range(steps):
            step, hidden = model.outputs(token, hidden, *arguments[2:])
            best, chances = top_two(step[0, 0])
            token = step.argmax(dim=-1).to(torch.int32)
            repeated.append(int(token))
            ids.append(best)
            probabilities.append(chances)
    return {"logits": logits[0, 0], "drafts": drafts, "repeated": repeated, "ids": ids, "probabilities": probabilities}


def row_problems(number, drafts, repeated, ids):
    problems = []
    if drafts != repeated:
        problems.append(f"row {number}: the unrolled drafts differ from the drafts made step by step")
    if repeated != [best[0] for best in ids]:
        problems.append(f"row {number}: a draft made step by step is not the best token of its step")
    return problems


def write_bytes(path, data):
    with open(path, "xb") as handle:
        handle.write(data)
    return mtp.digest(path)


def write_row(folder, tensors):
    import numpy as np

    folder.mkdir()
    listed, files = [], {}
    for name, tensor in zip(INPUTS, tensors):
        array = np.ascontiguousarray(tensor.numpy())
        if array.dtype.str != LITTLE[TYPES[name]]:
            raise ValueError(f"{name} is not {TYPES[name]}")
        shape = list(array.shape)
        listed.append({"name": name, "dtype": TYPES[name], "shape": shape, "file": f"{name}.bin"})
        files[name] = {"dtype": TYPES[name], "shape": shape, **write_bytes(folder / f"{name}.bin", array.tobytes())}
    write_json(folder / "inputs.json", {"inputs": listed})
    return {"inputs_json": mtp.digest(folder / "inputs.json"), "files": files}


def write_arrays(output, columns):
    import numpy as np

    described, problems = {}, []
    for name, values in columns.items():
        array = np.asarray(values, dtype=LITTLE[ARRAYS[name]])
        described[name] = {"dtype": ARRAYS[name], "shape": list(array.shape),
                           **write_bytes(output / f"{name}.bin", array.tobytes())}
        count = int(array.size - np.isfinite(array).sum()) if array.dtype.kind == "f" else 0
        if count:
            problems.append(f"{name} holds {count} values that are not finite")
    return described, problems


def compute(plan, output):
    started, config, steps = time.perf_counter(), plan["config"], plan["steps"]
    place = mtp.layout(config)
    model = mtp.class_drafter(plan["assistant"], mtp.table(plan["shard"], config), config, steps, plan["config_bytes"])
    loaded = time.perf_counter()
    (output / "rows").mkdir()
    rows, problems, columns = [], [], {name: [] for name in ARRAYS}
    rounding = {name: {"elements": 0, "changed": 0, "largest_difference": 0.0, "non_finite": 0} for name in FLOATS}
    for item in plan["items"]:
        features = mtp.loaded_features(item)
        for anchor in mtp.anchors(len(item["input_ids"]), item["outputs"]):
            number = len(rows)
            half, reference, statistics = rounded(mtp.row_inputs(item, features, anchor, place["window"])["arguments"])
            found = reference_row(model, reference, steps)
            rows.append({"row": number, "id": item["id"], "category": item["category"], "anchor": anchor,
                         "output_index": anchor - len(item["input_ids"]), "sliding": half[3].shape[3],
                         "full": half[5].shape[3], **write_row(output / "rows" / f"{number:03d}", half)})
            for name, entry in statistics.items():
                total = rounding[name]
                total["elements"] += entry["elements"]
                total["changed"] += entry["changed"]
                total["largest_difference"] = max(total["largest_difference"], entry["largest_difference"])
                total["non_finite"] += entry["non_finite"]
            problems += row_problems(number, found["drafts"], found["repeated"], found["ids"])
            for name, value in zip(ARRAYS, (found["logits"].numpy(), found["drafts"], found["ids"],
                                            found["probabilities"])):
                columns[name].append(value)
        del features
    problems += [f"{name} holds {entry['non_finite']} values that are not finite after the rounding to float16"
                 for name, entry in rounding.items() if entry["non_finite"]]
    arrays, array_problems = write_arrays(output, columns)
    computed, problems = time.perf_counter(), problems + array_problems
    return {"schema": SCHEMA, "steps": steps, **plan["sources"], "weights": WEIGHTS, "rows": rows, "arrays": arrays,
            "rounding": rounding, "versions": {name: importlib.metadata.version(name) for name in PACKAGES},
            "timing": {"load_s": loaded - started, "rows_s": computed - loaded,
                       "total_s": time.perf_counter() - started},
            "memory": mtp.memory(), "problems": problems, "passed": not problems}


def write_inputs(contract_path, assistant_path, snapshot, assistant_snapshot, records, inputs, outputs, steps, output):
    output, made = Path(output), False
    try:
        with mtp.terminating_as_interruption():
            if os.path.lexists(output):
                raise FileExistsError(f"The inputs must be written to a new folder: {output}")
            mtp.check_export_steps(steps)
            target, pinned = mtp.read_contracts(contract_path, assistant_path)
            contract, _, items, sources, shard = mtp.prepared_from(
                target, pinned, snapshot, assistant_snapshot, records, inputs, outputs, mtp.first_per_category)
            plan = {"config": contract["config"], "items": items, "sources": sources, "shard": shard,
                    "assistant": Path(assistant_snapshot), "steps": steps,
                    "config_bytes": pinned[0]["files"]["config.json"]["bytes"]}
            with deferred_interrupts():
                output.mkdir(parents=True)
                made = True
            record = compute(plan, output)
            if mtp.prepared_from(target, pinned, snapshot, assistant_snapshot, records, inputs, outputs,
                                 mtp.first_per_category)[3] != sources:
                raise ValueError("A file that was read changed while the rows were made")
            with deferred_interrupts():
                write_json(output / "record.json", record)
        return record
    except BaseException as error:
        if made:
            with deferred_interrupts():
                (output / "record.json").unlink(missing_ok=True)
                write_json(output / "failure.json", {"schema": SCHEMA, "failure": type(error).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description="The input files and the FP32 reference values for checking the MTP "
                                                 "drafter assets of Gemma 4 31B with Core AI.")
    command = parser.add_subparsers(dest="command", required=True).add_parser("inputs")
    for option in ("model", "assistant", "snapshot", "assistant-snapshot", "records", "inputs", "outputs", "output"):
        command.add_argument(f"--{option}", type=Path, required=True)
    command.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()
    record = write_inputs(args.model, args.assistant, args.snapshot, args.assistant_snapshot, args.records,
                          args.inputs, args.outputs, args.steps, args.output)
    print(json.dumps({"passed": record["passed"], "rows": len(record["rows"])}), flush=True)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
