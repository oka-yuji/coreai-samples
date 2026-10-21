import argparse
import json
from pathlib import Path
import shutil

from .fixtures import save_json
from .mobile_artifacts import CONVERTER_FILES
from .mobile_profile import CANDIDATES, comparison_candidate


def recipe(candidate, *, export_only=False, reviewed=False):
    if candidate not in CANDIDATES:
        raise ValueError("Unknown mobile recipe")
    if reviewed and export_only:
        raise ValueError("An export-only recipe cannot declare quality review")
    prefix = "$repo/Tools/Conversion/adapters/qwen_image21/"
    source_inputs = [prefix + name for name in CONVERTER_FILES]
    steps = [
        {"id": "prepare", "role": "prepare", "command": ["$python", "-m", "Tools.Conversion.adapters.qwen_image21.prepare",
         "--snapshot", "$snapshot", "--model", "$recipe_dir/model.json", "--output", "$out"],
         "inputs": [prefix + "prepare.py", prefix + "contracts.py", "$recipe_dir/model.json", "$snapshot"],
         "outputs": ["source.json"], "heavy": True, "timeout_s": 600, "min_free_bytes": 80000000000, "cwd": "$repo"},
        {"id": "profile", "role": "prepare", "depends": ["prepare"],
         "command": ["$python", "-m", "Tools.Conversion.adapters.qwen_image21.mobile_convert", "profile", "--candidate", candidate,
                     "--dependency-lock", "$recipe_dir/uv.lock", "--output", "$out"],
         "inputs": [*source_inputs, "$recipe_dir/uv.lock"], "outputs": ["profile.json"], "cwd": "$repo"},
    ]
    for kind in ("vision", "text", "denoiser", "vae_encoder", "vae_decoder"):
        group = "text_encoder" if kind in ("vision", "text") else "transformer" if kind == "denoiser" else "vae"
        steps.append({
            "id": kind, "role": "export", "depends": ["prepare", "profile"],
            "command": ["$python", "-m", "Tools.Conversion.adapters.qwen_image21.mobile_convert", "export",
                        "--snapshot", "$snapshot", "--profile", "$profile/profile.json", "--dependency-lock", "$recipe_dir/uv.lock",
                        "--kind", kind, "--output", "$out"],
            "inputs": [*source_inputs, f"$snapshot/{group}", "$snapshot/processor", "$snapshot/scheduler",
                       "$recipe_dir/model.json", "$recipe_dir/uv.lock", "$snapshot/LICENSE", "$snapshot/model_index.json"],
            "outputs": ["component.json", "compression.json", kind + ".aimodel"],
            "checks": [{"file": "component.json", "key": ["conversion_complete"], "equals": True}],
            "heavy": True, "timeout_s": 3600, "min_free_bytes": 80000000000, "cwd": "$repo",
        })
    if not export_only:
        original_recipe = "$repo/Tools/Conversion/recipes/qwen_image21/"
        comparison_files = [prefix + name for name in (
            "mobile_reference.py", "mobile_source.py", "mobile_capture.py", "mobile_evidence.py",
            "fixtures.py", "preprocessing.py", "numeric_policy.py", "examples.py", "mobile_verify.py",
            "mobile_runtime.py", "mobile_generate.py", "mobile_examples.py", "measurement.py")]
        inputs = [*source_inputs, *comparison_files, "$snapshot", "$recipe_dir/uv.lock",
                  original_recipe + "evaluation-cases.json", original_recipe + "numeric-gates-v1.json"]
        components = ["vision", "text", "denoiser", "vae_encoder", "vae_decoder"]
        bindings = [item for kind in components for item in ("--component", kind + "=$" + kind)]
        steps.append({
            "id": "fixtures", "role": "prepare", "depends": ["prepare"],
            "command": ["$python", "-m", "Tools.Conversion.adapters.qwen_image21.fixtures",
                        "--definition", original_recipe + "evaluation-cases.json", "--snapshot", "$snapshot",
                        "--output", "$out/fixtures"],
            "inputs": [*source_inputs, *comparison_files, "$snapshot/processor", "$snapshot/text_encoder/config.json",
                       original_recipe + "evaluation-cases.json"],
            "outputs": ["fixtures"], "cwd": "$repo", "timeout_s": 600,
        })
        steps.append({
            "id": "reference", "role": "reference", "depends": ["profile", "fixtures", *components],
            "command": ["$python", "-m", "Tools.Conversion.adapters.qwen_image21.mobile_reference",
                        "--snapshot", "$snapshot", "--profile", "$profile/profile.json", "--dependency-lock", "$recipe_dir/uv.lock",
                        "--fixtures", "$fixtures/fixtures", "--output", "$out/reference", *bindings],
            "inputs": inputs, "outputs": ["reference"], "heavy": True, "timeout_s": 14400,
            "min_free_bytes": 80000000000, "cwd": "$repo",
            "checks": [{"file": "reference/summary.json", "key": ["complete"], "equals": True}],
        })
        steps.append({
            "id": "numeric", "role": "verify", "depends": ["profile", "fixtures", "reference", *components],
            "command": ["$python", "-m", "Tools.Conversion.adapters.qwen_image21.mobile_verify",
                        "--snapshot", "$snapshot", "--profile", "$profile/profile.json", "--dependency-lock", "$recipe_dir/uv.lock",
                        "--fixtures", "$fixtures/fixtures", "--reference", "$reference/reference", "--output", "$out/numeric", *bindings],
            "inputs": inputs, "outputs": ["numeric"], "heavy": True, "timeout_s": 14400,
            "min_free_bytes": 80000000000, "cwd": "$repo",
            "checks": [{"file": "numeric/verification.json", "key": ["numeric_complete"], "equals": True},
                       {"file": "numeric/verification.json", "key": ["numeric_passed"], "equals": True}],
        })
        steps.append({
            "id": "mixed_profile", "role": "prepare", "depends": ["prepare"],
            "command": ["$python", "-m", "Tools.Conversion.adapters.qwen_image21.mobile_convert", "profile", "--candidate",
                        comparison_candidate({"id": "qmi-mobile-" + candidate + "-v1"}),
                        "--dependency-lock", "$recipe_dir/uv.lock", "--output", "$out"],
            "inputs": [*source_inputs, "$recipe_dir/uv.lock"], "outputs": ["profile.json"], "cwd": "$repo",
        })
        steps.append({
            "id": "mixed_reference", "role": "reference", "depends": ["mixed_profile", "fixtures"],
            "command": ["$python", "-m", "Tools.Conversion.adapters.qwen_image21.mobile_reference",
                        "--snapshot", "$snapshot", "--profile", "$mixed_profile/profile.json", "--dependency-lock", "$recipe_dir/uv.lock",
                        "--fixtures", "$fixtures/fixtures", "--output", "$out/reference"],
            "inputs": inputs, "outputs": ["reference"], "heavy": True, "timeout_s": 14400,
            "min_free_bytes": 80000000000, "cwd": "$repo",
            "checks": [{"file": "reference/summary.json", "key": ["complete"], "equals": True}],
        })
        checks = [{"file": "receipt.json", "key": ["bundle_ready"], "equals": True},
                  {"file": "receipt.json", "key": ["numeric_passed"], "equals": True}]
        if reviewed:
            checks.append({"file": "receipt.json", "key": ["verified"], "equals": True})
        steps.append({
            "id": "bundle", "role": "export", "depends": ["profile", "fixtures", "reference", "numeric", "mixed_reference", *components],
            "command": ["$python", "-m", "Tools.Conversion.adapters.qwen_image21.mobile_pack",
                        "--snapshot", "$snapshot", "--profile", "$profile/profile.json", "--dependency-lock", "$recipe_dir/uv.lock",
                        "--fixtures", "$fixtures/fixtures", "--reference", "$reference/reference", "--mixed-reference", "$mixed_reference/reference",
                        "--baseline", "$baseline_reference", "--baseline-bundle", "$baseline_bundle",
                        "--verification", "$numeric/numeric", "--output", "$out", *bindings, *(["--quality", "$quality"] if reviewed else [])],
            "inputs": [*inputs, prefix + "mobile_pack.py", prefix + "mobile_compare.py", prefix + "pack.py",
                       "$repo/docs/research/qwen-image21-evaluation-20261021.json", "$baseline_reference", "$baseline_bundle",
                       *(["$quality"] if reviewed else [])],
            "outputs": ["bundle", "receipt.json", "cohort-comparison"], "heavy": True, "timeout_s": 3600,
            "min_free_bytes": 40000000000, "cwd": "$repo", "checks": checks,
        })
    variables = {"python": "$python", "snapshot": "$snapshot", "heavy_lock": "$heavy_lock"}
    if not export_only:
        variables.update(baseline_reference="$baseline_reference", baseline_bundle="$baseline_bundle")
    if reviewed:
        variables["quality"] = "$quality"
    return {"version": 1, "name": "qwen_image21_mobile_" + candidate,
            "variables": variables, "python": "$python",
            "environment": {"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "COREAI_HEAVY_JOB_LOCK": "$heavy_lock"},
            "steps": steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parents[2] / "recipes"
    target = root / "qwen_image21_mobile"
    if not args.check:
        target.mkdir(exist_ok=True)
    for name in ("model.json", "pyproject.toml", "uv.lock"):
        if args.check:
            if (root / "qwen_image21" / name).read_bytes() != (target / name).read_bytes():
                raise ValueError("The copied source specification or dependency lock changed: " + name)
        else:
            shutil.copy2(root / "qwen_image21" / name, target / name)
    for candidate in CANDIDATES:
        for mode in ("", "-export", "-reviewed"):
            filename = "recipe" + mode + ("" if candidate == "w4" else "-" + candidate) + ".json"
            expected = recipe(candidate, export_only=mode == "-export", reviewed=mode == "-reviewed")
            if args.check:
                if json.loads((target / filename).read_text()) != expected:
                    raise ValueError("A generated mobile recipe is out of date: " + filename)
            else:
                save_json(target / filename, expected)


if __name__ == "__main__":
    main()
