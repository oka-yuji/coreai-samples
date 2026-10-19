import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from .contracts import compare_vectors


def comparison(specification, reference, records):
    fixtures = specification["fixtures"]
    identifiers = [item["id"] for item in fixtures]
    if len(identifiers) != 20 or len(set(identifiers)) != len(identifiers):
        raise ValueError("The pinned smoke set must contain twenty distinct cases")
    if not isinstance(records, list) or [row.get("id") for row in records] != identifiers:
        raise ValueError("The native result has missing, duplicate or reordered cases")
    gate = specification["low_precision_gate"]
    if gate != {"cosine_min": 0.999, "max_abs_max": 0.01}:
        raise ValueError("The fixed fidelity criteria changed")
    cases = []
    for fixture, row in zip(fixtures, records):
        if row.get("tokensMatch") is not True or row.get("passed") is not True:
            raise ValueError("Native preprocessing or inference did not pass")
        values = row.get("vector")
        if not isinstance(values, list) or len(values) != 768 or any(type(v) not in (int, float) for v in values):
            raise ValueError("Missing native numeric vector")
        actual = np.asarray(values, dtype=np.float64)[None, :]
        expected = np.load(reference / fixture["id"] / "reference.npy", allow_pickle=False)
        result = compare_vectors(expected, actual, **gate)
        cases.append({"id": fixture["id"], **result})
    return {"passed": all(case["pass"] for case in cases), "count": len(cases), "gate": gate, "cases": cases,
            "scope": "Fixed text/image S128/vision70 native-input fidelity; no general search-quality certification."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--original-reference", type=Path)
    args = parser.parse_args()
    specification = json.loads((args.reference / "fixtures.json").read_text())
    samples = []
    for fixture in specification["fixtures"]:
        folder = args.reference / fixture["id"]
        ids = np.load(folder / "input_ids.npy", allow_pickle=False)
        mask = np.load(folder / "attention_mask.npy", allow_pickle=False)
        row = {"id": fixture["id"], "tokens": ids[mask == 1].tolist(),
               "reference": np.load(folder / "reference.npy", allow_pickle=False).reshape(-1).tolist()}
        if "image" in fixture:
            row.update(image=str(args.reference / fixture["image"]), pixelFile=str(folder / "pixel_values.f32"))
        else:
            row.update(text=fixture["raw_text"], role={"SearchQuery": "query", "Document": "document"}[fixture["role"]])
        samples.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    plan = args.out / "plan.json"
    output = args.out / "vectors.json"
    plan.write_text(json.dumps({"samples": samples}, ensure_ascii=False, indent=2) + "\n")
    subprocess.run([str(args.runner), str(args.bundle), str(plan), str(output)], check=True, timeout=600)
    result = comparison(specification, args.reference, json.loads(output.read_text()))
    result["precision"] = specification.get("precision", "fp16_safe")
    if args.original_reference:
        original_specification = json.loads((args.original_reference / "fixtures.json").read_text())
        result["original_precision_comparison"] = comparison(
            original_specification, args.original_reference, json.loads(output.read_text()))
        result["scope"] = "Native fidelity to the PTQ reference; original-precision degradation reported separately."
    (args.out / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["passed"]:
        raise SystemExit("The native vectors did not meet the fixed fidelity criteria")


if __name__ == "__main__":
    main()
