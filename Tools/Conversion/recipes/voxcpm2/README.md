# VoxCPM2 component conversion

This maintenance recipe converts the parts of
[OpenBMB VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) that a native host needs
for text-only (zero-shot) speech generation into six macOS Core AI assets, in FP32
and FP16. Each function is compared with the original PyTorch implementation, run
on the CPU in FP32. The comparison uses inputs and outputs recorded from five
generations: four Japanese sentences and one English sentence with a voice
description. The comparison is per component. It is not an end-to-end generation
check, and it does not cover the Swift runtime, iPhone, quantization, reference
audio or listening quality. `model.json` is the asset contract that a native host
follows.

## Fixed inputs

`model.json` pins the Hugging Face revision, the sizes and SHA-256 values of all
nine snapshot files, and the commit of the
[official implementation](https://github.com/OpenBMB/VoxCPM). The snapshot hashes
were also compared with the revision's LFS SHA-256 values and Git blob IDs on
2026-11-16. Model weights and code use Apache-2.0. Apple's separately installed
Core AI tools keep their own licenses.

Every model stage checks the inputs again before it loads anything:

- The snapshot directory must be named after the revision.
- Every pinned file must match its size and SHA-256.
- Any other file is rejected, except Hugging Face download metadata under
  `.cache/huggingface`. For example, the upstream loader would prefer an extra
  `audiovae.safetensors` over the pinned `audiovae.pth`.
- Links that leave the snapshot are rejected.
- `config.json` must match the architecture in `model.json`.
- The upstream checkout must be clean, at the pinned commit, and the source that
  Python imports.

The model is built from the pinned configuration in float32. The checkpoint is
loaded with `strict=True` (the upstream loader uses `strict=False`), and
`audiovae.pth` is read only with `weights_only=True`. The tokenizer is
`LlamaTokenizerFast` wrapped by the upstream multi-character CJK split, without
`trust_remote_code`. The bundled `VoxCPM2Tokenizer` adds a BOS token; the upstream
generation path does not.

## Environment

Use a dedicated Python 3.11 environment and the clean checkout:

```sh
uv venv --python 3.11 .local/conversion/voxcpm2/.venv
uv pip install --python .local/conversion/voxcpm2/.venv/bin/python --require-hashes \
  -r Tools/Conversion/recipes/voxcpm2/requirements.lock
uv pip install --python .local/conversion/voxcpm2/.venv/bin/python --no-deps -e /path/to/VoxCPM
```

The lock pins 64 packages (torch 2.9.0, coreai-core 1.0.0b3, coreai-torch 0.4.3,
coreai-opt 0.3.0, transformers 4.57.3, numpy 2.3.5) with SHA-256 hashes. It was
made from the package set of the 2026-11-16 investigation with
`MACOSX_DEPLOYMENT_TARGET=26.0 uv pip compile --generate-hashes --no-header --no-annotate --python-version 3.11 --python-platform aarch64-apple-darwin`.
The deployment target matches the `macosx_26_0_arm64` coreai-core wheel. The
checkout declares interface, ASR and training dependencies that generation does
not import; they are intentionally not installed, so `uv pip check` lists them.

Download the snapshot into a directory named after the revision, without links
into a shared cache (`hf download openbmb/VoxCPM2 --revision <revision> --local-dir <directory>`).
Create a git-ignored bindings file:

```json
{
  "python": "/path/to/voxcpm2/.venv/bin/python",
  "snapshot": "/path/to/7936080b41163874f38610f46f438e2c0757f9fd",
  "upstream": "/path/to/pinned/VoxCPM"
}
```

From the repository root:

```sh
.local/conversion/voxcpm2/.venv/bin/python -m Tools.Conversion plan Tools/Conversion/recipes/voxcpm2/recipe.json \
  --bindings .local/conversion/voxcpm2.paths.json
.local/conversion/voxcpm2/.venv/bin/python -m Tools.Conversion run Tools/Conversion/recipes/voxcpm2/recipe.json \
  --bindings .local/conversion/voxcpm2.paths.json \
  --work-dir .local/conversion/work/voxcpm2 \
  --lock-dir "$HOME/Library/Caches/coreai-samples/heavy-job"
```

The generic runner owns the shared lock, process groups, timeouts and reuse. All
stages run offline and reject existing output directories. A complete FP32 and
FP16 run writes about 21 GB of assets, plus Core AI specialization caches.

## Stages

1. `reference` generates the five sentences in `fixtures.json` with the original
   model: seed 1708, 10 solver steps, CFG 2.0, no text normalization and no
   bad-case retry. The text is reduced to single spaces, as the upstream
   `VoxCPM.generate` does.
   - It records every component input and output, the prefill caches, the solver
     noise, and the streaming and full VAE decodes.
   - It also records the quantizer values before rounding and their integer
     codes. Re-encoding the codes must reproduce the recorded quantizer output
     bit for bit.
   - By exact equality on the recorded tensors, it checks the connections that a
     host must reproduce (listed below).
2. `wrapper_check` compares the PyTorch wrappers with the recorded originals on
   the CPU.
3. Six export stages write `<asset>-fp32.aimodel` and `<asset>-fp16.aimodel`.
   - They read only `convert.py`, `wrappers.py`, `source.py`, `model.json`, the
     lock, the snapshot and the checkout.
   - They do not depend on the fixtures or the comparison code, so changing either
     does not export the assets again.
   - They confirm that the cache state order equals the contract and that the
     exported graphs use every input.
   - Two more stages export the language models with capacity 2048, used only for
     the timing record.
4. `verify_fp16` repeats the comparisons with the FP16 assets against the same FP32
   originals. It records errors and times without a pass or fail.
5. `capacity` compares capacities 1024 and 2048 for the two language models and
   records their step and prefill times.
6. `verify_fp32` loads the FP32 assets with GPU-preferred specialization. It
   checks every function descriptor against `model.json` and repeats all
   comparisons. It fails unless the criteria below are met.

## Asset contract

All shapes are fixed. FP32 assets use float32 for every floating input, output and
state. In FP16 assets they are float16; `ids`, `length` and `position` stay int32.

| Asset | Function | Inputs | Outputs | States |
|---|---|---|---|---|
| `tslm` | `embed` | `ids` int32 [1, 145] | `embeds` [1, 145, 2048] | none |
| `tslm` | `prefill` | `embeds` [1, 145, 2048], `length` int32 [1] | `hidden` [1, 145, 2048] | `k_cache`, `v_cache` [28, 2, 1024, 128] |
| `tslm` | `step` | `embed` [1, 2048], `position` int32 [1] | `hidden` [1, 2048] | same as `prefill` |
| `ralm` | `prefill` | `embeds` [1, 145, 2048], `length` int32 [1] | `hidden` [1, 145, 2048] | `k_cache`, `v_cache` [8, 2, 1024, 128] |
| `ralm` | `step` | `embed` [1, 2048], `position` int32 [1] | `hidden` [1, 2048] | same as `prefill` |
| `proj` | `heads` | `lm_hidden`, `residual_hidden` [1, 2048] | `dit_hidden` [1, 2048], `stop_logits` [1, 2] | none |
| `proj` | `fuse` | `lm_raw`, `curr_embed` [1, 2048] | `lm_hidden`, `residual_input` [1, 2048] | none |
| `proj` | `prefill_fuse` | `hidden` [1, 145, 2048] | `residual_inputs` [1, 145, 2048] | none |
| `locenc` | `main` | `patch` [1, 64, 4] | `curr_embed` [1, 2048] | none |
| `dit` | `main` | `x` [2, 64, 4], `mu` [2, 2048], `t` [2], `cond` [2, 64, 4], `dt` [2] | `velocity` [2, 64, 4] | none |
| `vae` | `main` | `latent` [1, 64, 4] | `audio` [1, 1, 7680] | 26 causal convolution states, listed in `model.json` |

Every state starts at zero. The host allocates new zeroed states for each
utterance, from the descriptors. A patch is always `[1, 64, 4]` (latent channels by
frames): the solver result, the next `cond`, the LocEnc input and the VAE input.
One patch is four 48 kHz frames of 1920 samples, 0.16 seconds of audio.

A zero-shot utterance runs as follows:

1. Tokenize without BOS and append the audio start token 101. `length` is the
   number of text tokens plus one; at most 144 text tokens are supported. Pad the
   ids to 145 with token 0 and call `tslm.embed`.
2. `tslm.prefill(embeds, length)` writes cache rows 0 to 144; rows from `length`
   are zero. Values at padded positions do not affect the result. `lm_hidden` is
   `hidden[length - 1]`. Text positions do not pass through FSQ.
3. `proj.prefill_fuse(hidden)` gives the residual model input, with the audio
   feature half set to zero. `ralm.prefill(residual_inputs, length)` gives
   `residual_hidden = hidden[length - 1]`. The first `cond` is zero.
4. For patch `i` from 0, call `proj.heads`. Solve the patch from fresh standard
   normal noise `x` [1, 64, 4] with 10 Euler steps over
   `t_span = linspace(1, 0, 11) + (cos(pi/2 * t_span) - 1 + t_span)`. Start with
   `t = t_span[0]` and `dt = t_span[0] - t_span[1]`. Step 1 leaves `x` unchanged
   (CFG-Zero* initialization). Steps 2 to 10 call `dit.main` with the batch
   `x = [x, x]`, `mu = [dit_hidden, 0]`, `t = [t, t]`, `cond = [cond, cond]` and
   `dt = [0, 0]`. With `v_c` and `v_u` the two velocities and
   `s = <v_c, v_u> / (|v_u|^2 + 1e-8)`, apply
   `x -= dt * (v_u * s + 2.0 * (v_c - v_u * s))`. After every step, set
   `t -= dt` and then `dt = t - t_span[step + 1]`.
5. `curr_embed = locenc.main(patch)` and `audio = vae.main(patch)`; the VAE keeps
   its state across the patches of one utterance. If `i > 2` and the arg max of
   `stop_logits` is 1, or the patch count reaches `min(int(6 * T + 10), 4096)` for
   `T` text tokens, the utterance ends with this patch.
6. Otherwise `lm_raw = tslm.step(curr_embed, length + i)`,
   `(lm_hidden, residual_input) = proj.fuse(lm_raw, curr_embed)`,
   `residual_hidden = ralm.step(residual_input, length + i)`, `cond = patch`, and
   the next patch starts.

The reference stage checks these connections on the original computation:
prefill inputs and outputs, the zero audio half, the first and later hidden
states, `cond`, step positions, the solver batch layout, the solver times, and the
latent made from the patches. The FP16 `dit` asset receives `t` as float16, so the
time is rounded before its sinusoidal embedding, even though that embedding runs
in float32.

## Decisions

`model.json` records the same decisions under `decisions`.

- **Prefill shares the asset and the cache with step.** `tslm` holds `embed`,
  `prefill` and `step`, and `ralm` holds `prefill` and `step`. A weight-free check
  with the original MiniCPM layers found:
  - the same state descriptors in both functions;
  - prefill-then-step outputs equal to the original up to the last cache position;
  - bit-equal repeats with new or reused states;
  - a two-function asset only 0.3% larger than the step-only asset, because the
    converter stores shared weights once.

  Separate prefill assets would store the language-model weights twice. Each
  export records its asset size next to the unique parameter bytes
  (`parameter_bytes_fp32` in `export.json`).
- **Embedding is a separate function.** `prefill` takes embeddings. A later
  reference-audio prompt can then pass mixed text and audio features without a new
  prefill function.
- **Two cache states per language model.** Each state is
  `[layers, 2, capacity, 128]`. Each layer reads its cache through `where`, and a
  step writes every layer with one `scatter_`. Index assignment aborts
  specialization. In weight-free models with the real cache shapes, this form
  gave the same results as per-layer states. At capacity 1024 its step time
  stayed within about 0.3 ms of theirs, and the host handles 4 arrays instead of 72.
  Scattering into each layer's slice was fast in three of four configurations but
  2.5 times slower in the fourth, for an unknown reason, so it is not used.
- **Capacity 1024.** `T + 1 + min(int(6 * T + 10), 4096) <= 1024` holds for at
  most 144 text tokens. The `capacity` stage records step and prefill times at
  capacity 2048 only for comparison; they do not change the contract.
- **Prefill length 145.** This is 144 text tokens and the audio start token. Cache
  rows from `length` onward are zero, and padded values do not change the outputs.
- **Float time inputs.** `t` and `dt` stay float inputs, as in the original
  estimator. In the 2026-11-16 investigation, precomputed time tables selected by
  step number did not change chained FP16 results. FP16 assets round `t` to
  float16 before the float32 sinusoidal embedding.
- **Streaming VAE only.** The VAE asset decodes one patch with 26
  causal-convolution states. A native host emits audio patch by patch, so a
  whole-utterance decode is not part of the contract.
- **FP16 for every asset, recorded only.** Inside FP16 assets, RMSNorm and the
  sinusoidal time embedding still compute in float32. FP16 errors and times are
  recorded without acceptance. A mixed-precision host combines FP32 and FP16
  assets; there are no separate mixed assets.

## Verification

### Criteria

The FP32 criteria are in `model.json` under `criteria`.

- Every compared output must be finite and have its maximum absolute error, divided
  by the largest absolute reference value, at most 0.001.
- After the prefill, the language models are compared at every generation step:
  14 to 104 steps per sentence, against the required minimum of 8.
- The arg max of `stop_logits` must match the original at every patch.
- The joined VAE waveform must meet the criterion. Per-patch VAE errors are
  recorded only, because near-silent patches have tiny reference values.
- FP16 results are recorded without a pass or fail.

`proj.fuse` contains the scalar quantizer, `round(tanh(x) * 9)`. Its two outputs
are judged together per call and classified as follows:

1. **Match**: both outputs meet the criterion.
2. **Boundary-consistent**: all of the following hold.
   - Candidates come from the reference only. They are the dimensions whose
     original float32 value before rounding lies within `fsq_boundary_distance`
     (1e-5) of a rounding boundary.
   - Each candidate replaces the original code of exactly one such dimension with
     the integer on the other side of that boundary (`fsq_boundary_max_dims` = 1).
   - The reference stage computes the original module outputs for each candidate.
   - Both Core AI outputs meet the criterion against exactly one candidate, using
     the original reference maxima as the scale. If none or several candidates
     match, the call fails.
3. **Fail**: anything else.

The result passes only if no call fails and the boundary-consistent calls are at
most `fsq_boundary_max_fraction` (2%) of all `proj.fuse` calls. Core AI outputs are
never decoded back into codes.

The distance 1e-5 was chosen after one disagreement had been observed in the first
three sentences, at 4.77e-7 from a boundary. It is a provisional acceptance policy,
not a derived numerical bound. Boundary consistency shows that the Core AI output
equals the original output for the neighboring code of one dimension very close
to a rounding boundary. It does not prove that the Core AI value before rounding is
within that distance of the original value.

Before this rule, runs on the first three sentences failed exactly one of 2,527
gated comparisons. The failure was `lm_hidden` of `proj.fuse` at step 8 of
`long`, with a normalized error of 3.8e-3. GPU-preferred and CPU-only
specializations gave the same result.

### Results

The run of 2026-11-17 completed all 13 stages; a second run reused every stage.
Values come from the `comparison.json` files of `verify_fp32` and `verify_fp16`,
and from `capacity.json`.

- **Timing conditions**: the Python host, with GPU-preferred specialization. Each
  time is the median of 20 calls after 3 warm-up calls, measured in two rounds; the
  load average was about 4.
- **Errors**: normalized as in the criteria above.
- **Model**: the original runs on the CPU in FP32.

| Function | FP32 comparisons within | FP32 largest error | FP16 largest error | FP32 ms | FP16 ms |
|---|---:|---:|---:|---:|---:|
| `tslm.embed` | 5/5 | 0 | 8.6e-08 | 0.60 / 0.59 | 0.53 / 0.53 |
| `tslm.prefill` | 15/15 | 2.2e-06 | 3.8e-03 | 39.80 / 40.22 | 13.24 / 13.31 |
| `tslm.step` | 234/234 | 1.1e-06 | 3.2e-03 | 13.30 / 13.53 | 8.10 / 8.21 |
| `ralm.prefill` | 15/15 | 1.1e-06 | 2.6e-03 | 11.66 / 11.68 | 4.25 / 4.17 |
| `ralm.step` | 234/234 | 1.6e-06 | 6.0e-03 | 4.24 / 4.25 | 2.75 / 2.78 |
| `proj.heads` | 478/478 | 3.4e-07 | 6.1e-04 | 0.67 / 0.67 | 0.64 / 0.65 |
| `proj.fuse` | 234/234 calls | see below | 3.8e-03 | 0.54 / 0.54 | 0.50 / 0.51 |
| `proj.prefill_fuse` | 5/5 | 1.1e-07 | 4.5e-04 | 0.85 / 0.87 | 0.69 / 0.70 |
| `locenc.main` | 239/239 | 3.1e-06 | 1.7e-03 | 3.12 / 3.11 | 2.08 / 2.09 |
| `dit.main` | 2151/2151 | 4.6e-06 | 7.9e-03 | 4.84 / 4.82 | 3.07 / 3.07 |
| `vae.main`, joined | 5/5 | 4.2e-06 | 3.6e-03 | 4.00 / 4.00 | 3.11 / 3.11 |

Results for `proj.fuse` in FP32:

- Of 234 calls, 233 match (largest error 2.0e-7), 1 is boundary-consistent and
  none fail. The limit is 4.68 calls.
- The boundary-consistent call is step 8 of `long`, dimension 283. Its original
  value is 4.77e-7 from a boundary. The original code is 3 and the candidate code
  is 2. The Core AI outputs differ from that candidate's original outputs by
  5.8e-8 and 1.0e-7.

Other results:

- All 239 stop decisions match.
- 229 of 239 VAE patches meet the criterion on their own; the largest
  per-patch error, 1.29e-3, is in near-silent audio.
- With noise in the padded prefill positions, the outputs and caches are bitwise
  equal.
- The FP16 assets give 188 matching, 2 boundary-consistent and 44 failing
  `proj.fuse` calls; this is recorded only.

The sentences `heldout_ja` (51 steps) and `heldout_en` (32 steps) were added after
the rule and its numbers were fixed, and were evaluated once. Every FP32
comparison on them meets the criteria without boundary-consistent calls. One
`heldout_ja` step had a candidate dimension (1.6e-6 from a boundary) whose Core AI
output still matched the original code.

Assets total 9,345,658,961 bytes in FP32 and 4,673,736,753 bytes in FP16. The FP32
`tslm` asset (5,888,752,611 bytes) is as large as its unique parameters
(5,889,073,152 bytes), so the three functions share one copy of the weights.

Capacity 2048 passes the same comparisons and is slower:

| Function | Capacity 1024 (ms) | Capacity 2048 (ms) |
|---|---:|---:|
| `tslm.step` | 13.27 / 13.35 | 14.35 / 14.24 |
| `ralm.step` | 4.23 / 4.22 | 4.61 / 4.51 |
| `tslm.prefill` | 39.76 / 39.81 | 40.22 / 40.13 |
| `ralm.prefill` | 11.54 / 11.66 | 12.04 / 11.82 |

## Limits

The checks start from recorded inputs. They do not show that Core AI parts
chained by a host reproduce the original trajectory; that belongs to the
end-to-end work. FP16 parts are recorded, not accepted. In the 2026-11-16
investigation, chaining any FP16 part moved the generated latent trajectory
beyond 0.001 within a few patches. The Python runtime cannot set
`expectFrequentReshapes`; all shapes here are static.
Reference-audio prompts, the VAE encoder, quantization, iPhone and listening
quality are out of scope.
