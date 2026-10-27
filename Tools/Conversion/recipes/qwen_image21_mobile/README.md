# Qwen Image 2.1 Turbo mobile candidates

Maintainer tooling for [#216](https://github.com/oka-yuji/coreai-samples/issues/216). The recipes preserve the pinned source, 512-square generation and one-reference editing, eight Turbo steps, CFG one, text capacity 512, joint capacity 2304 and no prefix KV cache. Export success does not establish numerical fidelity, image quality or iPhone support.

| Candidate | Text | DiT | Vision | VAE |
|---|---|---|---|---|
| `fp16` | Dense F16 | Dense F16 | Dense F16 | Dense F16 |
| `bf16` | Dense BF16 | Dense BF16 | Dense BF16 | Dense BF16 |
| `mixed` | F16 Linear/embedding storage, F32 math | Dense F16 | F16 Linear storage, F32 math | Dense F32 |
| `w8` | W8/group64, F16 scales, F32 math | W8/group64, F16 math | W8/group16, F32 math | Dense F32 |
| `w4` | W4/group64, F16 scales, F32 math | W4/group64, F16 math | W8/group16, F32 math | Dense F32 |
| `mixed-f32` | F16 Linear/embedding storage, F32 math | F16 Linear storage, F32 math | F16 Linear storage, F32 math | Dense F32 |
| `w4-f32` | W4/group64, F16 scales, F32 math | W4/group64, F16 scales, F32 math | W8/group16, F32 math | Dense F32 |
| `w4-blocks20-21-f32` | W4/group64, F32 math | W4 except the 14 Linear weights in blocks 20–21, retained as F16; F32 math | W8/group16, F32 math | Dense F32 |
| `w4-blocks16-23-f32` | W4/group64, F32 math | W4 except the 56 Linear weights in blocks 16–23, retained as F16; F32 math | W8/group16, F32 math | Dense F32 |
| `w4-dit-g16-blocks16-23-f32` | W4/group64, F32 math | W4/group16 except the same 56 retained F16 Linear weights; F32 math | W8/group16, F32 math | Dense F32 |

The physical floating-point inputs and outputs remain Float32. Integer indices and Boolean masks keep their declared types. DiT returns the complete `[1,2304,64]` joint output. Text returns hidden states before the final RMSNorm. DiT temporal frequencies and Vision trigonometric tables retain the source Float32 precision. The Vision feed-forward width is 4304, requiring the separate group16 setting.

Mobile Float32 DiT exports use a custom Metal RoPE kernel with explicit fused multiply-add ordering. The kernel matches the original MPS complex multiplication on the captured query/key tensors; ordinary `addcmul` decomposition does not preserve that ordering. Global safe256 attention, weights, full output ABI and frozen thresholds are retained. The kernel is registered before authoring and its source is included in converter identity. These assets require GPU-path validation; CPU-only execution is not established. Low-precision mobile exports and the separate legacy FP32 recipe retain their previous rotary path.

`mixed` is the unquantized A1 counterpart for the W4/W8 candidates. The previous FP16-network/FP32-VAE source cohort is a different condition. Full FP16 VAE overflow and earlier FP16 text fidelity failures remain failures; these recipes do not change the frozen numerical policy.

`mixed-f32` is the unquantized A1 counterpart for `w4-f32`. These additional profiles retain Float32 DiT normalization parameters, temporal frequencies, activations and scheduler outputs, with Float16 Linear weight storage. The two-case W4/F16 smoke failed fixed-input DiT fidelity while its final latent/RGBA passed. A Float32 arithmetic diagnostic reduced the original-to-wrapper discrepancy; the new candidate must independently pass complete numerical, quality and device checks. It uses the existing Float32 numerical budget.

The complete W4/F32 fixed comparison subsequently failed 32 full-joint tensors, despite passing all 480 target comparisons. The two selective candidates preserve the full output ABI and frozen Float32 thresholds. A four-input MPS source/wrapper diagnostic supported trying retained weights in blocks 20–21 and 16–23; that source/wrapper result alone does not establish Core AI fidelity. The subsequent four-case Core AI probe passed both full-joint and target gates for `w4-blocks16-23-f32`, while the two-block candidate still failed e08/e09 full-joint. Their measured DiT assets are 6,342,973,880 B and 4,420,941,001 B respectively. These fixed-step results do not establish full-suite quality or mobile memory fit; see the [scoped probe record](../../../../docs/research/qwen-image21-selective-dit-20261022.json). Block numbers are the zero-based `transformer_blocks` indices. Both use `mixed-f32` as A1. The profile lists every retained module, compression records its exact F16 weight and bias hashes, and the independent reference rejects missing or changed records. The compression summary distinguishes quantized-only payload estimates from the combined quantized/retained Linear-weight estimate.

## Prepare and run

The group16 DiT candidate has a distinct profile identity and must generate its own matched source cohort and exported assets. It tests finer weight quantization after the group64 candidate passed all 1,230 fixed comparisons but failed one of 60 free rollouts. It does not change any numerical threshold, the preserved block set, other component settings, or the full output ABI. Improved numerical stability, image quality and device fit remain unverified. See the [fidelity record](../../../../docs/research/qwen-image21-attention-20261026.json).

Use an existing pinned source snapshot and the accepted #215 Float32 source cohort and bundle. Model files and raw evidence stay in the local workspace.

```sh
python3 -m Tools.Conversion.local setup --recipe qwen_image21_mobile --workspace "$QI_WORKSPACE" --snapshot "$QI_SNAPSHOT"
```

Setup writes `qwen_image21_mobile.paths.json` with the fixed Python environment, snapshot and machine lock. Add `baseline_reference` pointing to the accepted `reference-fp32-v1` directory and `baseline_bundle` pointing to the existing verified Float32 bundle. The baseline source summary must match the pinned #215 measurement record. Setup rewrites the bindings file, so retain these two local bindings when preparing the environment again.

```sh
python3 -m Tools.Conversion.local plan --recipe qwen_image21_mobile --workspace "$QI_WORKSPACE"
python3 -m Tools.Conversion.local run --recipe qwen_image21_mobile --workspace "$QI_WORKSPACE"
```

The default candidate is W4. Select a candidate name from the table with `--precision` for the other profiles. The runner preserves failed attempts and verifies stage inputs, outputs and dependencies before reuse. Model stages use the shared heavy-job lock. Keep other model processes stopped and monitor free disk space; individual recipe stages also require a free-space reserve.

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

For offline review, create a separate gallery after the complete comparison exists:

```sh
"$QI_PYTHON" -m Tools.Conversion.adapters.qwen_image21.mobile_gallery \
  --comparison /path/to/cohort-comparison --fixtures /path/to/fixtures \
  --baseline /path/to/A0 --mixed-reference /path/to/A1 --reference /path/to/B \
  --verification /path/to/C-verification --output /path/to/new-review-folder
```

Open `index.html` to compare the four outputs with their prompt and editing reference. The gallery verifies the comparison's cohort identities and copies each PNG against its recorded hash. It does not load models or certify numerical fidelity. Scores remain null until reviewed; browser storage and JSON export preserve partial work. Resume accepts only matching cohort identities, all 60 outputs and unchanged axes. Export the reviewed JSON and pass it to the reviewed recipe above. Keep this research-only gallery local alongside its copied images.

The review page presents Japanese display translations, the unchanged original prompt, and case-specific inspection points from `quality-review-ja.json`. Each guide entry must match its frozen source prompt and applicable axes. Instruction adherence appears first; shape/detail, lettering, preservation of the editing reference and transparency have separate explanations. The editing reference is shown beside the instruction, and narrow displays retain the same controls. Display changes keep the comparison identity and browser storage key, so existing scores are restored without regrading. The gallery receipt records the display guide and template independently of the model evidence.

Browser JSON may serialize a cohort identity's `1.0` as `1`. Rating import compares these as exactly equal finite numeric values while retaining strict field, string, Boolean and hash checks. Changed values, rounded large integers, nonfinite numbers and Boolean/number substitutions are rejected. This handling applies only to rating identities; shared artifact hashes, frozen inputs and numerical gates are unchanged.

The bundle contains the complete asset inventory, source configuration, tokenizer/processor, dependency lock, numerical/quality summaries and original LICENSE/NOTICE. `receipt.json` gives the SHA-256 of the saved raw `bundle.json` for #263 to pin independently. Grouped payload bits per selected Linear weight and serialized asset-size ratios have explicit scopes; neither is runtime memory usage. The unused LM head was already excluded from the old conversion and is not a new compression gain.

`verified` here covers candidate numerical and human-quality acceptance. `device_verification` stays pending until the separate native integration/device work supplies evidence. Export metadata's minimum OS is a target declaration, not an executed-device claim. No model publication is performed by these recipes.

## Maintenance checks

```sh
python3 -m Tools.Conversion.adapters.qwen_image21.mobile_recipe --check
"$QI_PYTHON" -m unittest discover -s Tools/Conversion/tests -v
```

Regenerate the recipe JSON with `python3 -m Tools.Conversion.adapters.qwen_image21.mobile_recipe`. The generator is the source of the five candidate and export/review variants. The existing `qwen_image21` Float32 recipe retains its separate identity and behavior.
