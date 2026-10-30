import argparse
import json
from pathlib import Path
import shutil

from .fixtures import save_json
from .mobile_acceptance import OWNER_IDENTICAL_REVIEW, OWNER_REVIEW
from .mobile_artifacts import CONVERTER_FILES
from .mobile_profile import CANDIDATES, NATIVE_COMPLEX_CANDIDATE, comparison_candidate


OWNER_METHODS = {"w4-dit-g16-blocks16-23-f32": OWNER_REVIEW, NATIVE_COMPLEX_CANDIDATE: OWNER_IDENTICAL_REVIEW}


def recipe(candidate, *, export_only=False, reviewed=False, owner_review=False):
    if candidate not in CANDIDATES:
        raise ValueError("Unknown mobile recipe")
    if reviewed and export_only:
        raise ValueError("An export-only recipe cannot declare quality review")
    if owner_review and (export_only or reviewed or candidate not in OWNER_METHODS):
        raise ValueError("Owner review is bound to supported candidates and a separate acceptance method")
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
        if reviewed or owner_review:
            checks.append({"file": "receipt.json", "key": ["verified"], "equals": True})
        steps.append({
            "id": "bundle", "role": "export", "depends": ["profile", "fixtures", "reference", "numeric", "mixed_reference", *components],
            "command": ["$python", "-m", "Tools.Conversion.adapters.qwen_image21.mobile_pack",
                        "--snapshot", "$snapshot", "--profile", "$profile/profile.json", "--dependency-lock", "$recipe_dir/uv.lock",
                        "--fixtures", "$fixtures/fixtures", "--reference", "$reference/reference", "--mixed-reference", "$mixed_reference/reference",
                        "--baseline", "$baseline_reference", "--baseline-bundle", "$baseline_bundle",
                        "--verification", "$numeric/numeric", "--output", "$out", *bindings, *(["--quality", "$quality"] if reviewed else []),
                        *(["--owner-review", OWNER_METHODS[candidate]] if owner_review else [])],
            "inputs": [*inputs, prefix + "mobile_pack.py", prefix + "mobile_compare.py", prefix + "mobile_ratings.py", prefix + "mobile_acceptance.py", prefix + "pack.py",
                       "$repo/docs/research/qwen-image21-evaluation-20261021.json", "$baseline_reference", "$baseline_bundle",
                       *(["$repo/docs/research/qwen-image21-owner-quality-20261028.json"] if owner_review else []),
                       *([prefix + "mobile_visual_equivalence.py", prefix + "mobile_gallery.py",
                          "$repo/docs/research/qwen-image21-native-complex-visual-equivalence-20261029.json"]
                         if owner_review and candidate == NATIVE_COMPLEX_CANDIDATE else []),
                       *(["$quality"] if reviewed else [])],
            "outputs": ["bundle", "receipt.json", "cohort-comparison"], "heavy": True, "timeout_s": 3600,
            "min_free_bytes": 40000000000, "cwd": "$repo", "checks": checks,
        })
    variables = {"python": "$python", "snapshot": "$snapshot", "heavy_lock": "$heavy_lock"}
    if not export_only:
        variables.update(baseline_reference="$baseline_reference", baseline_bundle="$baseline_bundle")
    if reviewed:
        variables["quality"] = "$quality"
    return {"version": 1, "name": "qwen_image21_mobile_" + candidate.replace("-", "_"),
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
        modes = ("", "-export", "-reviewed", "-owner-reviewed") if candidate in OWNER_METHODS else ("", "-export", "-reviewed")
        for mode in modes:
            filename = "recipe" + mode + ("" if candidate == "w4" else "-" + candidate) + ".json"
            expected = recipe(candidate, export_only=mode == "-export", reviewed=mode == "-reviewed", owner_review=mode == "-owner-reviewed")
            if args.check:
                if json.loads((target / filename).read_text()) != expected:
                    raise ValueError("A generated mobile recipe is out of date: " + filename)
            else:
                save_json(target / filename, expected)


if __name__ == "__main__":
    main()
