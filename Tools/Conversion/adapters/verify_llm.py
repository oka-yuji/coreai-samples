import argparse
import base64
import json
from pathlib import Path
import subprocess


def probabilities(logits):
    import numpy as np
    shifted = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / numerator.sum(axis=-1, keepdims=True)


def compare(reference, observed, criteria):
    import numpy as np
    if reference.shape != observed.shape or reference.ndim != 2 or not reference.size:
        raise ValueError(f"Logit shape mismatch: {reference.shape} vs {observed.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(observed).all():
        raise ValueError("Non-finite logits")
    a, b = probabilities(reference), probabilities(observed)
    reference_top = a.argmax(axis=-1)
    observed_top = b.argmax(axis=-1)
    top_two_ids = np.argpartition(a, -2, axis=-1)[:, -2:]
    observed_two_ids = np.argpartition(b, -2, axis=-1)[:, -2:]
    same_candidates = (np.sort(top_two_ids, axis=-1) == np.sort(observed_two_ids, axis=-1)).all(axis=-1)
    winner_gap = a[np.arange(len(a)), reference_top] - a[np.arange(len(a)), observed_top]
    mismatch = reference_top != observed_top
    maximum = float(np.abs(a - b).max())
    near_tie = mismatch & same_candidates & (winner_gap <= criteria["near_tie_probability_margin"])
    significant = int((mismatch & ~near_tie).sum())
    return {"positions": len(a), "top1_matches": int((~mismatch).sum()),
            "near_tie_mismatches": int(near_tie.sum()),
            "significant_mismatches": significant, "max_probability_error": maximum,
            "passed": significant == 0 and maximum <= criteria["max_probability_error"]}


def decode_swift(data, case):
    import numpy as np
    targets = case["tokens"][1:]
    if data["context_tokens"] != case["tokens"][:1] or data["continuation_tokens"] != targets:
        raise ValueError("Swift evaluated different token IDs")
    positions = data["positions"]
    if len(positions) != len(targets):
        raise ValueError("Missing Swift output positions")
    width = case["shape"][1]
    arrays = []
    for index, (row, target) in enumerate(zip(positions, targets)):
        if row["index"] != index or row["target_token"] != target:
            raise ValueError("Swift output alignment changed")
        raw = base64.b64decode(row["logits_base64"], validate=True)
        if len(raw) != width * 2:
            raise ValueError("Expected Apple Silicon llm-runner Float16 logits")
        arrays.append(np.frombuffer(raw, dtype="<f2").astype(np.float32))
    return np.stack(arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--criteria", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from pinned import sha256
    reference = json.loads((args.reference / "reference.json").read_text())
    criteria = json.loads(args.criteria.read_text())
    if set(criteria) != {"near_tie_probability_margin", "max_probability_error"}:
        raise ValueError("Unknown or missing numerical criteria")
    if any(type(v) not in (float, int) or not np.isfinite(v) or not 0 <= v < 1 for v in criteria.values()):
        raise ValueError("Invalid numerical criteria")
    metadata = json.loads((args.bundle / "metadata.json").read_text())
    contract = reference["contract"]
    if (metadata["source"]["hf_model_id"], metadata["source"]["revision"]) != (
        contract["model_id"], contract["revision"]
    ):
        raise ValueError("Bundle and reference source identities differ")
    provenance = json.loads((args.bundle.parent / "provenance.json").read_text())
    if provenance["contract"] != contract:
        raise ValueError("Export and reference contracts differ")
    for name, expected in provenance["tokenizer_sha256"].items():
        if sha256(args.bundle / "tokenizer" / name) != expected:
            raise ValueError(f"Tokenizer content mismatch: {name}")
    (args.out / "swift").mkdir(parents=True)
    results = []
    for case in reference["cases"]:
        name = case["id"]
        output = args.out / "swift" / f"{name}.json"
        command = [
            str(args.runner), "--model", str(args.bundle), "--raw-tokens",
            str(args.reference / f"{name}.tokens.json"), "--save-logits", str(output),
            "--save-logits-length", "full", "--inference-engine-variant", "coreai-sequential",
            "--sampling-strategy", "greedy", "--warmup", "off",
        ]
        with (args.out / "swift" / f"{name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
        observed = decode_swift(json.loads(output.read_text()), case)
        expected = np.load(args.reference / f"{name}.npy", allow_pickle=False)
        results.append({"id": name, **compare(expected, observed, criteria)})
    passed = bool(results) and all(row["passed"] for row in results)
    (args.out / "comparison.json").write_text(json.dumps(
        {"passed": passed, "criteria": criteria, "cases": results,
         "scope": "fixed-input integration smoke; not full production qualification",
         "limits": ["Swift tokenizer path is bypassed with raw IDs", "No long-context or endurance evaluation",
                    "No quality claim for other checkpoints, quantization recipes or devices"]}, indent=2) + "\n")
    if not passed:
        raise SystemExit("Numerical verification did not meet the fixed criteria")


if __name__ == "__main__":
    main()
