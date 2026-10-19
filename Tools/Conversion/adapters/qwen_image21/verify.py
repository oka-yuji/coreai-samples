import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .contracts import KINDS, PROFILE, check_profile, component_inventory, digest, read_json, tensor_metrics, write_json
from .examples import arrays
from .generate import generate
from .preprocessing import prompt_inputs


def compare_source_conditions(reference, cache_on, bf16):
    for directory, precision in ((cache_on, "fp32"), (bf16, "bf16")):
        metadata = read_json(directory / "reference.json")
        check_profile(metadata["target_profile"])
        if not metadata["complete"] or metadata["cache"] != "on" or metadata["precision"] != precision:
            raise ValueError("Source comparisons must separate cache and precision conditions")
    rows = []
    for case in ("generate", "edit"):
        for directory in (cache_on, bf16):
            for name in ("initial-noise.npy", "prompt.txt"):
                if digest(reference / case / name) != digest(directory / case / name):
                    raise ValueError("Source condition comparisons require the same noise and prompt")
            if digest(reference / "edit-input.png") != digest(directory / "edit-input.png"):
                raise ValueError("Source condition comparisons require the same reference image")
        result = {"case": case}
        for name, baseline, candidate in (("fp32_cache_on_vs_off", reference, cache_on), ("bf16_vs_fp32_cache_on", cache_on, bf16)):
            values = {}
            for field, filename, gate in (("latent", "final-latents.npy", 5e-3), ("rgba", "vae-decode-output.npy", np.inf)):
                measured = tensor_metrics(np.load(baseline / case / filename, allow_pickle=False),
                                          np.load(candidate / case / filename, allow_pickle=False),
                                          max_limit=np.inf, rmse_limit=gate)
                if field == "rgba":
                    measured["passed"] = measured["finite"] and measured["absolute_rmse"] <= 1e-2
                if name.startswith("bf16"):
                    measured.pop("passed")
                values[field] = measured
            result[name] = values
        rows.append(result)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-on", type=Path, required=True)
    parser.add_argument("--reference-bf16", type=Path, required=True)
    parser.add_argument("--components", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = read_json(args.reference / "reference.json")
    check_profile(reference["target_profile"])
    if not reference["complete"] or reference["precision"] != "fp32" or reference["cache"] != "off":
        raise ValueError("End-to-end validation requires the independent FP32 cache-off source reference")
    components = {}
    for directory in args.components:
        contract = read_json(directory / "component.json")
        check_profile(contract["profile"])
        kind = contract["kind"]
        if kind not in KINDS or kind in components or contract["passed"] is not True:
            raise ValueError("Unexpected, duplicate, or unverified component")
        components[kind] = directory
    if set(components) != set(KINDS):
        raise ValueError("All five components must be supplied")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "comparison.json").exists():
        raise FileExistsError("Verification must use a new output directory")
    conditions = compare_source_conditions(args.reference, args.reference_on, args.reference_bf16)
    write_json(args.output / "source-conditions.json", {"conditions": conditions, "precision_differences": "descriptive; not conversion-error gates"})
    if not all(item["passed"] for row in conditions for item in row["fp32_cache_on_vs_off"].values()):
        raise RuntimeError("The independent FP32 cache-on/off source comparison failed")
    rows = []
    write_json(args.output / "criteria.json", {"profile": PROFILE, "latent_normalized_rmse": 5e-3,
               "rgba_absolute_rmse": 1e-2, "shape": "exact", "finite": True,
               "step_differences": "diagnostic only; source trajectories use their own previous outputs"})
    identity = read_json(args.reference / "identity.json")
    provenance = {"profile": PROFILE, "components": {kind: component_inventory(path) for kind, path in components.items()},
                  "source_reference": {key: identity[key] for key in ("source", "revision", "diffusers_commit", "versions", "dtype", "device", "resolution", "output_resolution", "seed", "use_kv_cache")},
                  "runtime_code_sha256": {path.name: digest(path) for path in Path(__file__).parent.glob("*.py")},
                  "fixtures": {case: {name: digest(args.reference / case / name)
                                     for name in ("prompt.txt", "initial-noise.npy", "encoder-inputs.npz", "final-latents.npy", "vae-decode-output.npy")}
                               for case in ("generate", "edit")},
                  "edit_input_sha256": digest(args.reference / "edit-input.png")}
    write_json(args.output / "provenance.json", provenance)
    for case in ("generate", "edit"):
        reference_case = args.reference / case
        output = args.output / case
        prompt = (reference_case / "prompt.txt").read_text().rstrip("\n")
        image = Image.open(args.reference / "edit-input.png").convert("RGBA") if case == "edit" else None
        encoded, _, _ = prompt_inputs(args.snapshot, prompt, image)
        original = arrays(reference_case / "encoder-inputs.npz")
        preprocessing = {name: bool(name in encoded and np.array_equal(value, encoded[name]))
                         for name, value in original.items()}
        if not all(preprocessing.values()):
            raise RuntimeError(f"Raw input preprocessing differs from the source: {preprocessing}")
        noise = np.load(reference_case / "initial-noise.npy", allow_pickle=False)
        latents, rgba = generate(components, args.snapshot, output, prompt, image, noise)
        expected_latents = np.load(reference_case / "final-latents.npy", allow_pickle=False)
        expected_rgba = np.load(reference_case / "vae-decode-output.npy", allow_pickle=False)
        latent_result = tensor_metrics(expected_latents, latents, max_limit=np.inf, rmse_limit=5e-3)
        rgba_result = tensor_metrics(expected_rgba, rgba, max_limit=np.inf, rmse_limit=np.inf)
        rgba_result["passed"] = rgba_result["finite"] and rgba_result["absolute_rmse"] <= 1e-2
        steps = []
        for index in range(8):
            expected = np.load(reference_case / f"noise-{index}.npy", allow_pickle=False)[:, -1024:]
            actual = np.load(output / f"noise-{index}.npy", allow_pickle=False)
            steps.append(tensor_metrics(expected, actual, max_limit=np.inf, rmse_limit=np.inf))
        rows.append({"case": case, "passed": latent_result["passed"] and rgba_result["passed"],
                     "preprocessing": preprocessing, "latent": latent_result, "rgba": rgba_result,
                     "steps": steps, "manual_image_review": "pending"})
        write_json(args.output / "comparison.json", {"passed": len(rows) == 2 and all(row["passed"] for row in rows), "cases": rows})
        print(rows[-1], flush=True)
    if not all(row["passed"] for row in rows):
        raise RuntimeError("The complete image pipeline does not satisfy the fixed fidelity gate")


if __name__ == "__main__":
    main()
