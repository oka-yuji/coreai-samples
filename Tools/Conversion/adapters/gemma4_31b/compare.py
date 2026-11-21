import argparse
from pathlib import Path
import sys

from ...storage import read_json, write_json
from ..pinned import sha256

SCHEMA = "gemma4-31b-rerun/1"
SAME = ("model", "dtype", "device", "fixtures_sha256", "environment")


def compare_case(first, second, name):
    import numpy as np

    left, right = (read_json(Path(folder) / "cases" / f"{name}.tokens.json") for folder in (first, second))
    outputs = left["output_tokens"], right["output_tokens"]
    differs = next((index for index, (a, b) in enumerate(zip(*outputs)) if a != b), None)
    if differs is None and len(outputs[0]) != len(outputs[1]):
        differs = min(map(len, outputs))
    compared = len(outputs[0]) if differs is None else differs + 1
    logits = [np.load(Path(folder) / "cases" / f"{name}.logits.npy")[:compared] for folder in (first, second)]
    compared = min(compared, *(len(array) for array in logits))
    difference = np.abs(logits[0][:compared].astype(np.float64) - logits[1][:compared].astype(np.float64))
    return {"id": name, "input_tokens_match": left["input_tokens"] == right["input_tokens"],
            "output_tokens_match": differs is None, "first_difference": differs, "compared_positions": compared,
            "max_abs_logit_difference": float(difference.max()) if difference.size else 0.0,
            "argmax_changes": int((logits[0][:compared].argmax(-1) != logits[1][:compared].argmax(-1)).sum())}


def compare(first, second):
    records = [read_json(Path(folder) / "reference.json") for folder in (first, second)]
    if any(records[0][key] != records[1][key] for key in SAME):
        raise ValueError("The two runs used different models, inputs or environments")
    names = [case["id"] for case in records[0]["cases"]]
    if names != [case["id"] for case in records[1]["cases"]]:
        raise ValueError("The two runs hold different cases")
    cases = [compare_case(first, second, name) for name in names]
    return {"schema": SCHEMA, "passed": all(case["input_tokens_match"] and case["output_tokens_match"] for case in cases),
            "references_sha256": [sha256(Path(folder) / "reference.json") for folder in (first, second)],
            "max_abs_logit_difference": max(case["max_abs_logit_difference"] for case in cases), "cases": cases}


def main():
    parser = argparse.ArgumentParser(description="Compare two reference artifacts of the same inputs.")
    for name in ("first", "second", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("The comparison must be written to a new file")
    result = compare(args.first, args.second)
    write_json(args.output, result)
    print(f"passed={result['passed']} max_abs_logit_difference={result['max_abs_logit_difference']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
