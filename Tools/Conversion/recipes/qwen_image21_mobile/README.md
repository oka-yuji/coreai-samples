# Qwen Image 2.1 Turbo mobile candidates

Maintainer tooling for [#216](https://github.com/oka-yuji/coreai-samples/issues/216). The recipes preserve the pinned source, 512-square generation and one-reference editing, eight Turbo steps, CFG one, text capacity 512, joint capacity 2304 and no prefix KV cache. Export success does not establish numerical fidelity, image quality or iPhone support.

| Candidate | Text | DiT | Vision | VAE |
|---|---|---|---|---|
| `fp16` | Dense F16 | Dense F16 | Dense F16 | Dense F16 |
| `bf16` | Dense BF16 | Dense BF16 | Dense BF16 | Dense BF16 |
| `mixed` | F16 Linear/embedding storage, F32 math | Dense F16 | F16 Linear storage, F32 math | Dense F32 |
| `w8` | W8/group64, F16 scales, F32 math | W8/group64, F16 math | W8/group16, F32 math | Dense F32 |
| `w4` | W4/group64, F16 scales, F32 math | W4/group64, F16 math | W8/group16, F32 math | Dense F32 |

The physical floating-point inputs and outputs remain Float32. Integer indices and Boolean masks keep their declared types. DiT returns the complete `[1,2304,64]` joint output. Text returns hidden states before the final RMSNorm. DiT temporal frequencies and Vision trigonometric tables retain the source Float32 precision. The Vision feed-forward width is 4304, requiring the separate group16 setting.

`mixed` is the unquantized A1 counterpart for the W4/W8 candidates. The previous FP16-network/FP32-VAE source cohort is a different condition. Full FP16 VAE overflow and earlier FP16 text fidelity failures remain failures; these recipes do not change the frozen numerical policy.

## Prepare and run

Use an existing pinned source snapshot and the accepted #215 Float32 source cohort and bundle. Model files and raw evidence stay in the local workspace.

```sh
python3 -m Tools.Conversion.local setup --recipe qwen_image21_mobile --workspace "$QI_WORKSPACE" --snapshot "$QI_SNAPSHOT"
```

Setup writes `qwen_image21_mobile.paths.json` with the fixed Python environment, snapshot and machine lock. Add `baseline_reference` pointing to the accepted `reference-fp32-v1` directory and `baseline_bundle` pointing to the existing verified Float32 bundle. The baseline source summary must match the pinned #215 measurement record. Setup rewrites the bindings file, so retain these two local bindings when preparing the environment again.

```sh
python3 -m Tools.Conversion.local plan --recipe qwen_image21_mobile --workspace "$QI_WORKSPACE"
python3 -m Tools.Conversion.local run --recipe qwen_image21_mobile --workspace "$QI_WORKSPACE"
```

The default candidate is W4. Select `--precision fp16`, `bf16`, `mixed` or `w8` for the other profiles. The runner preserves failed attempts and verifies stage inputs, outputs and dependencies before reuse. Model stages use the shared heavy-job lock. Keep other model processes stopped and monitor free disk space; individual recipe stages also require a free-space reserve.

For an initial export-only run, use the common runner with `recipe-export.json` (or `recipe-export-<candidate>.json`), the same bindings, work directory and lock:

```sh
python3 -m Tools.Conversion run Tools/Conversion/recipes/qwen_image21_mobile/recipe-export.json \
  --bindings "$QI_WORKSPACE/qwen_image21_mobile.paths.json" \
  --work-dir "$QI_WORKSPACE/work/qwen_image21_mobile" \
  --lock-dir "$QI_HEAVY_LOCK"
```

Export-only runs have no verification stage and retain pending numerical/device states. Their validated export stages can be reused by the complete recipe.

## Independent comparisons and acceptance

The original Diffusers pipeline constructs B. Every compressed Linear must reproduce C's finalized integer, scale, zero-point and effective-weight hashes. B then uses ordinary PyTorch Linear weights; it does not use the Core AI export wrapper's forward control. The effective stored weight is `round16((q-zp)*scale16)`, promoted to Float32 for components declaring Float32 math.

The fixture stage must reproduce the frozen 30-input/two-seed manifest. The source stage records all 60 outcomes, the required two repeats, pre-norm text, all Vision taps, every DiT input/full joint output, all Euler steps and unclipped RGBA. Non-finite outputs are saved with failure records. A partial or failed cohort cannot pass complete verification.

Fixed-input comparisons check all applicable outputs and every denoising step, selecting both the complete valid joint region and the exact 1024 target positions. Free rollout uses C's own previous latent. Each component uses its declared compute-dtype budget; the final latent/RGBA budgets stay unchanged. Packaging recomputes the recorded metrics from saved tensors and checks their complete inventories and source/input/code bindings.

A0-to-A1 and A1-to-B differences are descriptive precision/quantization effects. Human ratings cover every applicable axis of every output and remain separate from conversion error. The first complete recipe produces a reviewable bundle with `verified=false` until quality ratings pass. Its `cohort-comparison/quality-ratings.json` binds the ratings to the exact A0/A1/B/C cohorts.

After human review, run `recipe-reviewed.json` (or `recipe-reviewed-<candidate>.json`) through the common runner with `--set quality=/path/to/reviewed-ratings.json`. This reuses unchanged numerical evidence and creates a new bundle; it does not edit an earlier bundle's manifest. Missing rows, changed identities/axes, unreviewed values or a failed quality gate cannot pass the reviewed recipe.

The bundle contains the complete asset inventory, source configuration, tokenizer/processor, dependency lock, numerical/quality summaries and original LICENSE/NOTICE. `receipt.json` gives the SHA-256 of the saved raw `bundle.json` for #263 to pin independently. Grouped payload bits per selected Linear weight and serialized asset-size ratios have explicit scopes; neither is runtime memory usage. The unused LM head was already excluded from the old conversion and is not a new compression gain.

`verified` here covers candidate numerical and human-quality acceptance. `device_verification` stays pending until the separate native integration/device work supplies evidence. Export metadata's minimum OS is a target declaration, not an executed-device claim. No model publication is performed by these recipes.

## Maintenance checks

```sh
python3 -m Tools.Conversion.adapters.qwen_image21.mobile_recipe --check
"$QI_PYTHON" -m unittest discover -s Tools/Conversion/tests -v
```

Regenerate the recipe JSON with `python3 -m Tools.Conversion.adapters.qwen_image21.mobile_recipe`. The generator is the source of the five candidate and export/review variants. The existing `qwen_image21` Float32 recipe retains its separate identity and behavior.
