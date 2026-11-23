import argparse
import json
from pathlib import Path
import sys

from .compare import CONTRACT
from .export import quantized_names
from .layers import CONDITIONS, PARTS, RESTORED, WEIGHTS, kept_rows
from .native import NEAR_TIE, load_reference, measure
from .prepare import load_contract
from .summary import HEX

SCHEMA = "gemma4-31b-quantization-effect/1"
KEPT = ("rows", "non_finite", "reference_non_finite", "max_abs_logit", "top1_mismatches")


def checked(record):
    quantized = record.get("quantized")
    return isinstance(quantized, dict) and isinstance(quantized.get("export_sha256"), str) \
        and HEX[64].fullmatch(quantized["export_sha256"]) is not None \
        and type(quantized.get("tensors")) is int \
        and quantized["tensors"] == len(PARTS) * len(quantized_names(record["layer_types"]))


def effect(restored, rounded, contract_path=CONTRACT):
    import numpy as np

    contract, problems, loaded = load_contract(contract_path), [], {}
    for kind, folder, weights in (("restored", restored, RESTORED), ("rounded", rounded, WEIGHTS)):
        record, hashed, arrays, found = load_reference(folder, contract, contract_path)
        problems += [f"the {kind} reference: {problem}" for problem in found]
        if record.get("weights") != weights:
            problems.append(f"the {kind} reference does not use the {kind} weights")
        loaded[kind] = record, hashed, arrays
    if loaded["restored"][0].get("layers") != loaded["rounded"][0].get("layers"):
        problems.append("the references are of other layer counts")
    if not problems and not checked(loaded["restored"][0]):
        problems.append("the restored reference did not check the data, scales and minimums of every quantized weight "
                        "of a passed export")
    conditions = []
    for name, source, length, start, chunk in [] if problems else CONDITIONS:
        index = {row: position for position, row in enumerate(kept_rows(source))}
        rows = np.array([index[position] for position in range(start, length)])
        values = measure(loaded["restored"][2][source][rows], loaded["rounded"][2][source], rows,
                         np.zeros(len(rows), dtype=bool))
        conditions.append({"name": name, "input": source, "tokens": length, "start": start, "chunk": chunk,
                           **{key: values[key] for key in KEPT}})
    return {"schema": SCHEMA, "problems": problems,
            **{kind: {"sha256": hashed, "layers": record.get("layers")} for kind, (record, hashed, _) in loaded.items()},
            "criteria": {"near_tie": NEAR_TIE}, "conditions": conditions}


def main():
    parser = argparse.ArgumentParser(description="Print, without a judgment, how far the FP32 reference restored from the "
                                                 "8-bit weights is from the FP32 reference with float16-rounded weights "
                                                 "on the rows of the twelve conditions.")
    for name in ("restored", "rounded"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = effect(args.restored, args.rounded, CONTRACT)
        text = json.dumps(result, indent=2, allow_nan=False)
    except Exception as error:
        print(f"Cannot compare the references ({type(error).__name__}).", file=sys.stderr)
        return 1
    print(text)
    return 0 if not result["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
