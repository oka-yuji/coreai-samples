import argparse
from pathlib import Path
import sys

from ...storage import read_json, write_json
from ..pinned import sha256
from .prepare import load_contract

SCHEMA = "gemma4-31b-rerun/1"
SAME = ("model", "dtype", "device", "fixtures_sha256", "environment")
CONTRACT = Path(__file__).resolve().parents[2] / "recipes/gemma4_31b/model.json"
BLOCK_ROWS = 256


def load_case(folder, name, vocabulary):
    import numpy as np

    try:
        tokens = read_json(Path(folder) / "cases" / f"{name}.tokens.json")
        logits = np.load(Path(folder) / "cases" / f"{name}.logits.npy", mmap_mode="r")
    except (OSError, ValueError) as error:
        return None, None, [f"{name}: unreadable case files ({type(error).__name__})"]
    expected = (len(tokens["input_tokens"]) + len(tokens["output_tokens"]), vocabulary)
    problems = [] if logits.dtype == np.float32 else [f"{name}: logits dtype {logits.dtype}"]
    if logits.shape != expected:
        problems.append(f"{name}: logits shape {list(logits.shape)} instead of {list(expected)}")
    return tokens, logits, problems


def compare_case(first, second, name, vocabulary):
    import numpy as np

    (left, a, left_problems), (right, b, right_problems) = (load_case(folder, name, vocabulary) for folder in (first, second))
    problems = left_problems + right_problems
    if left is None or right is None:
        return {"id": name, "problems": problems, "input_tokens_match": False, "output_tokens_match": False,
                "first_difference": None, "compared_positions": 0, "max_abs_logit_difference": None, "argmax_changes": 0}
    sequences = [tokens["input_tokens"] + tokens["output_tokens"] for tokens in (left, right)]
    differs = next((index for index, (x, y) in enumerate(zip(*sequences)) if x != y), None)
    if differs is None and len(sequences[0]) != len(sequences[1]):
        differs = min(map(len, sequences))
    compared = 0 if problems else len(sequences[0]) if differs is None else differs
    maximum, changes = (None if problems else 0.0), 0
    for start in range(0, compared, BLOCK_ROWS):
        rows = slice(start, min(start + BLOCK_ROWS, compared))
        x, y = (np.asarray(array[rows], dtype=np.float64) for array in (a, b))
        maximum = max(maximum, float(np.max(np.abs(x - y))))
        changes += int((x.argmax(-1) != y.argmax(-1)).sum())
    return {"id": name, "problems": problems, "input_tokens_match": left["input_tokens"] == right["input_tokens"],
            "output_tokens_match": left["output_tokens"] == right["output_tokens"], "first_difference": differs,
            "compared_positions": compared, "max_abs_logit_difference": maximum, "argmax_changes": changes}


def compare(first, second, vocabulary=None):
    vocabulary = vocabulary or load_contract(CONTRACT)["config"]["vocab_size"]
    records = [read_json(Path(folder) / "reference.json") for folder in (first, second)]
    problems = [f"{label} run: reference.json did not pass" for label, record in zip(("first", "second"), records)
                if record.get("passed") is not True]
    problems += [f"different {key}" for key in SAME if records[0].get(key) != records[1].get(key)]
    names = [[case["id"] for case in record.get("cases", [])] for record in records]
    if names[0] != names[1]:
        problems.append(f"different cases: {sorted(set(names[0]) ^ set(names[1]))}")
    cases = [compare_case(first, second, name, vocabulary) for name in names[0] if name in names[1]]
    problems += [problem for case in cases for problem in case["problems"]]
    differences = [case["max_abs_logit_difference"] for case in cases if case["max_abs_logit_difference"] is not None]
    return {"schema": SCHEMA, "vocabulary": vocabulary, "problems": problems,
            "passed": not problems and bool(cases)
            and all(case["input_tokens_match"] and case["output_tokens_match"] for case in cases),
            "references_sha256": [sha256(Path(folder) / "reference.json") for folder in (first, second)],
            "max_abs_logit_difference": max(differences) if differences else None, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description="Compare two reference artifacts of the same inputs.")
    for name in ("first", "second", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("The comparison must be written to a new file")
    result = compare(args.first, args.second)
    write_json(args.output, result)
    print(f"passed={result['passed']} problems={len(result['problems'])} "
          f"max_abs_logit_difference={result['max_abs_logit_difference']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
