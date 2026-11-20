# VoxCPM2 component conversion

This maintenance recipe converts the parts of
[OpenBMB VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) that a native host needs
for text-only (zero-shot) speech generation into six macOS Core AI assets, in FP32
and FP16. Each function is compared with the original PyTorch implementation, run
on the CPU in FP32. The comparison uses inputs and outputs recorded from five
generations: four Japanese sentences and one English sentence with a voice
description. Two more stages join the assets from text to audio with a Python host:
one compares the FP32 generation of the same five sentences with the original, and
one records FP32 and FP16 configurations and writes audio for listening. The last
three stages remove debug information from the assets, assemble three app bundles
with a manifest and fixed inventories, and compare the FP32 bundle with the original
again. The six-asset recipe does not cover the Swift runtime, iPhone, quantization,
reference-audio generation or listening quality. A separate FP32 reference encoder
check, and the assembly of a reference bundle from one of its runs, are described
below. `model.json` is the asset contract that a native host follows.

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
FP16 run writes about 21 GB of assets and 14 GB of assets without debug information,
plus Core AI specialization caches. The bundles hold another 19 GB of asset files,
which take no further space when they are cloned (see [App bundles](#app-bundles)).

## Reference-audio encoder

`reference-recipe.json` is a separate FP32 encoder check. It uses the same pinned
model snapshot, upstream checkout and Python environment as the component recipe,
plus the three files listed by `reference-voices.json`: the voice-donation README
and two CC0 recordings at their fixed revision. The recordings are inputs and are
not stored in this repository.

Add `voices` to a separate bindings file, pointing to a folder that contains
`README.md` and `voice-donations/0a67.wav` and `voice-donations/1410.wav`. Keep the
existing component bindings unchanged. Download the files at revision
`323332d33f997de8394f24a193e1a76df720e01a` from `kyutai/tts-voices`, without changing
their bytes.

```sh
.local/conversion/voxcpm2/.venv/bin/python -m Tools.Conversion plan Tools/Conversion/recipes/voxcpm2/reference-recipe.json \
  --bindings .local/conversion/voxcpm2-reference.paths.json
.local/conversion/voxcpm2/.venv/bin/python -m Tools.Conversion run Tools/Conversion/recipes/voxcpm2/reference-recipe.json \
  --bindings .local/conversion/voxcpm2-reference.paths.json \
  --work-dir .local/conversion/work/voxcpm2-reference \
  --lock-dir "$HOME/Library/Caches/coreai-samples/heavy-job"
```

The verify stage checks both recordings at five and ten seconds. It requires four
complete comparisons, finite outputs, normalized maximum error at most 0.001, the
same original PCM inputs and an unchanged source encoder. The fixed encoder input
is Float32 `[1,1,161280]`; its `mu` output is `[1,64,252]`, with no states. Only the
frames corresponding to the recording are compared. The runner checks required
outputs and their hashes before reusing a successful run; use `--refresh` for a
new conversion and comparison.

The stage preserves original, fixed-length PyTorch and Core AI arrays, cropped
recordings, descriptors and timings under its `encoder` output. These are research
outputs. [Reference bundle](#reference-bundle) removes the debug information of the
encoder of one selected run and assembles a bundle; the distribution checks still
follow before publication. This recipe does not establish Swift,
normal-UI or listening acceptance. FP16 encoder comparisons failed three of the
four fixed cases at the separately recorded 0.01 threshold; this recipe exports
only the accepted FP32 candidate.

## Reference bundle

`reference-bundle-recipe.json` makes `reference_fp32`: the published FP32 bundle with
the reference encoder of one selected, passing `reference-recipe.json` run added. It
converts and compares nothing again, and it does not run the reference recipe, so
the selected files keep their bytes. Its bindings are `python`, `base_bundle` (the
`fp32/` folder of [okayuji/voxcpm2-coreai](https://huggingface.co/okayuji/voxcpm2-coreai)
at revision `afe11d61de9dd7c3609140aa040b777ef77b2555`), `encoder_run` (the `encoder`
folder of the selected run) and `selection`:

```sh
.local/conversion/voxcpm2/.venv/bin/python -m Tools.Conversion plan Tools/Conversion/recipes/voxcpm2/reference-bundle-recipe.json \
  --bindings .local/conversion/voxcpm2-reference-bundle.paths.json
.local/conversion/voxcpm2/.venv/bin/python -m Tools.Conversion run Tools/Conversion/recipes/voxcpm2/reference-bundle-recipe.json \
  --bindings .local/conversion/voxcpm2-reference-bundle.paths.json \
  --work-dir .local/conversion/work/voxcpm2-reference-bundle \
  --lock-dir "$HOME/Library/Caches/coreai-samples/heavy-job"
```

The stage runs `python -m Tools.Conversion.adapters.voxcpm2.reference_bundle
--base-bundle <fp32> --encoder-run <encoder> --selection <selection.json> --output
<new folder>`, which can also be run directly.

- **Selection**: a JSON object written when the run is chosen, with exactly `format`
  1, `profile` `reference_fp32`, `report` and `files`. `report` and each of
  `files["main.hash"]`, `files["main.mlirb"]` and `files["metadata.json"]` hold the
  `bytes` (a positive integer) and `sha256` (64 lowercase hexadecimal digits) of that
  file, read from the run itself. It names no paths: the report is always
  `report.json` and the asset always `refenc-fp32.aimodel/` in the run. Duplicate
  keys, other fields or files, booleans as sizes and other digest forms are refused.
  Another run needs a new selection file; neither the recipe nor the code holds the
  fingerprint of a run.
- **Before loading anything**: the output must not exist, not even as a broken link,
  and must be outside both inputs. `report.json` and the three asset files must be
  regular files, the asset folder must hold exactly those three, and their sizes and
  SHA-256 values must equal the selection. The report must be version 1, at stage
  `complete`, complete and passed, in FP32, for the model and code revisions and the
  source files of `model.json`, with the encoder descriptor below as its contract and
  its Core AI descriptor, four of four cases passed with matching original inputs,
  no non-finite value and a normalized maximum error of at most 0.001 for both the
  fixed-shape PyTorch and the Core AI outputs, and an unchanged source encoder. Its
  `export.file`, the three sizes, their total and `main_mlirb_sha256` must equal the
  files. The base must have exactly the 22 files, sizes and SHA-256 values of
  `voxcpm2-fp32.inventory.json` in
  [`CoreAISamplesKit/Sources/SpeechAssets/Resources`](../../../../CoreAISamplesKit/Sources/SpeechAssets/Resources),
  with no link, and its `manifest.json` must be the format 1 FP32 manifest that
  `bundle.py` writes from `model.json`. The `manifest` field of the report names the
  voices of the comparison, not a bundle manifest; the recordings, crops and arrays of
  the run are not copied.
- **Assembly**: `strip_one` removes the debug information of the encoder into the new
  bundle with the same three-way comparison as `strip` (see
  [App bundles](#app-bundles)). The 18 files of the six FP32 assets, `LICENSE` and
  `tokenizer.json` are cloned or copied with their checked bytes; copying them needs
  about 9.3 GB, and the stage asks for 12 GB of free space. `manifest.json`
  and `NOTICE` (from `bundle-notice.txt`, naming `reference_fp32`) are written. Every
  file is then read again: the 20 kept files must equal the repository inventory,
  `manifest.json` and `NOTICE` the written bytes and the encoder files the record
  after the removal. All 25 files are scanned for absolute paths with the canaries of
  `bundle`.

```text
bundle/
  reference_fp32/                  the bundle
    manifest.json, NOTICE, LICENSE, tokenizer.json
    artifacts/{tslm,ralm,proj,locenc,dit,vae,refenc}-fp32.aimodel/main.mlirb, main.hash, metadata.json
  reference_fp32.inventory.json    format 1 listing of the 25 files
  check.json
```

The manifest copies `model`, `execution`, `generation`, `structure`, `tokenizer` and
the six assets from the base manifest and changes the following:

| Key | Content |
|---|---|
| `format` | 2 |
| `configuration` | `name` `reference_fp32`; the precision of all seven assets is `fp32` |
| `assets.refenc` | `path` `artifacts/refenc-fp32.aimodel`, `precision` `fp32`, function `main` with input `pcm` float32 `[1,1,161280]`, output `mu` float32 `[1,64,252]` and no states |
| `reference` | `sample_rate` 16000, `minimum_samples` 80000, `maximum_samples` 160000, `hop_samples` 640, `start_token` 103, `end_token` 104, `padding` `right_zero` |

Reference audio is 5 to 10 seconds of 16 kHz mono, padded with zeros on the right to
the fixed encoder input of 161,280 samples; the input length is not the audio limit.
`check.json` records `complete`, `passed`, `selection_sha256`,
`base_inventory_sha256`, `manifest_sha256`, `inventory_sha256`, the `strip` record,
the cloned and copied counts, `path_scan`, the scan of the encoder before the removal
and the canaries. A failed run exits with an error and writes no inventory; a
rejected input leaves no output at all, and a later failure leaves `check.json` with
`passed` false. The inputs and an existing folder are not changed. A pass covers this
assembly only. Before publication, the encoder inside the finished bundle still has
to run in a new process with the OS Core AI runtime on the two voices at 5 and 10
seconds against the original PyTorch features, the manifest and inventory handed on
must be checked against the 25 files, and the model release must be audited. Swift,
normal-UI and listening acceptance are separate.

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
7. `synthesis_compare` generates five sentences from text with three seeds in three
   precision configurations, records the results without a pass or fail, and
   writes audio for listening.
8. `synthesis_check` generates the five sentences of `fixtures.json` from text with
   the FP32 assets and compares the `proj.fuse` outputs and the patch latents with
   the original generation. It fails unless the rule in
   [End-to-end generation](#end-to-end-generation) is met.
9. `strip` removes debug information from the twelve FP32 and FP16 assets and checks
   that the programs without location information stay the same. `main.hash` and
   the creation date in `metadata.json` change.
10. `bundle` assembles the bundles `fp32`, `fp16` and `fp16_proj32`, writes their
    inventories and scans every byte for absolute paths.
11. `bundle_verify` checks every bundled asset against the contract and repeats the
    comparisons of `verify_fp32` with the FP32 bundle. See
    [App bundles](#app-bundles).

Every stage declares only the adapter modules it imports, so changing or adding a
module runs again only the stages that import it.

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

The runs of 2026-11-17 completed all 15 stages; a second run reused every stage.
Values come from the `comparison.json` files of `verify_fp32` and `verify_fp16`,
and from `capacity.json`.

- **Timing conditions**: the Python host, with GPU-preferred specialization. Each
  time is the median of 20 calls after 3 warm-up calls, measured in two rounds; the
  load average was about 4.
- **Errors**: normalized as in the criteria above.
- **Model**: the original runs on the CPU in FP32.

| Function | FP32 comparisons within | FP32 largest error | FP16 largest error | FP32 ms | FP16 ms |
|---|---:|---:|---:|---:|---:|
| `tslm.embed` | 5/5 | 0 | 8.6e-08 | 0.60 / 0.60 | 0.54 / 0.53 |
| `tslm.prefill` | 15/15 | 2.2e-06 | 3.8e-03 | 39.85 / 39.72 | 14.31 / 13.35 |
| `tslm.step` | 234/234 | 1.1e-06 | 3.2e-03 | 13.37 / 13.35 | 8.57 / 8.22 |
| `ralm.prefill` | 15/15 | 1.1e-06 | 2.6e-03 | 11.66 / 11.51 | 4.25 / 4.25 |
| `ralm.step` | 234/234 | 1.6e-06 | 6.0e-03 | 4.23 / 4.23 | 2.85 / 2.81 |
| `proj.heads` | 478/478 | 3.4e-07 | 6.1e-04 | 0.66 / 0.66 | 0.66 / 0.68 |
| `proj.fuse` | 234/234 calls | see below | 3.8e-03 | 0.54 / 0.55 | 0.51 / 0.51 |
| `proj.prefill_fuse` | 5/5 | 1.1e-07 | 4.5e-04 | 0.84 / 0.86 | 0.70 / 0.71 |
| `locenc.main` | 239/239 | 3.1e-06 | 1.7e-03 | 3.13 / 3.16 | 2.13 / 2.25 |
| `dit.main` | 2151/2151 | 4.6e-06 | 7.9e-03 | 4.88 / 4.92 | 3.08 / 3.12 |
| `vae.main`, joined | 5/5 | 4.2e-06 | 3.6e-03 | 4.00 / 4.01 | 3.13 / 3.13 |

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
| `tslm.step` | 13.30 / 13.64 | 14.13 / 14.48 |
| `ralm.step` | 4.26 / 4.25 | 4.83 / 4.52 |
| `tslm.prefill` | 39.64 / 39.70 | 40.07 / 40.33 |
| `ralm.prefill` | 11.53 / 11.68 | 12.00 / 11.85 |

## End-to-end generation

`synthesis.py` joins the assets from text to audio. It follows the zero-shot steps
of the asset contract and uses the model-free calculations in `host.py`.

- **Text**: whitespace is reduced as in the reference stage. `tokenizer.json` is read
  with `tokenizers` after its size and SHA-256 are checked against `model.json`. No
  BOS is added, `host.prompt_ids` splits the multi-character kanji tokens, and the
  audio start token is appended.
- **Patch limit**: a text with more than 144 tokens, or without room for one patch in
  the cache, is rejected before any model call. A generation has at most
  `min(int(6 * T + 10), 4096, 1024 - T - 1)` patches for `T` text tokens.
- **Noise**: every patch draws `[1, 64, 4]` standard normal values from a CPU
  `torch.Generator` seeded with the request seed, in the order of the original
  solver.
- **States**: every request starts from new zeroed states. Prefill outputs are passed
  on unchanged, and step `i` uses position `length + i`.
- **End**: a generation ends with `stop` (the stop head after patch 2), `limit` (the
  patch limit without a stop) or `error`. An `error` is a non-finite output, an input
  that is not finite after conversion to the asset precision (for example a float32
  value beyond the float16 range), or a disagreement that the rule below cannot
  explain. Only `stop` is a success. The other two return no audio and are not
  retried.
- **Precision**: a configuration maps every asset to FP32 or FP16. FP16 assets
  receive float16 inputs; the host itself computes in float32.
- **Processes**: every generation runs in its own worker process. A patch allocates
  17 output arrays (2 from `proj.heads`, 9 from the solver calls, 2 from
  `proj.fuse`, and 1 each from LocEnc, the VAE and the two steps), and the prefill
  allocates 4. In one process that ran an FP32 configuration and part of an FP16
  one, the Python runtime stopped with `SIGTRAP` while allocating an output array
  (`NDArray+Pool.swift`); by this count, the generations that finished before the
  stop had allocated about 15,700. A
  worker of the precision comparison runs one generation after a warm-up
  generation of `plain`; at the patch limits of `plain` (124) and `long` (592) it
  allocates at most 2,112 + 10,068 = 12,180 arrays.

### FP32 check

`synthesis_check` generates the five sentences of `fixtures.json` with seed 1708 and
the FP32 assets, and compares them with the original generation recorded by
`reference`. The token IDs and the noise of every patch must equal the original bit
for bit.

The generation that is judged replaces values at two kinds of comparison points.
Each point is judged on the value before any replacement, and a value that equals
the original only because it was replaced is not counted as agreement.

1. **`proj.fuse` outputs**, at every step: both outputs within the criterion
   continue unchanged; a boundary-consistent call (rule above) continues with the
   original outputs of that call; anything else fails the sentence.
2. **Patch latents**, at every patch, before LocEnc, the VAE and the next `cond` use
   them. A latent within the criterion continues unchanged. A latent outside the
   criterion is solver-consistent on the same inputs only if both of the following
   hold, and then continues as the original latent:
   - The solver inputs of that patch match the original: the noise bit for bit, and
     `dit_hidden` and `cond` within the criterion. The time schedule, CFG value,
     types and shapes follow the contract, and every value is finite.
   - The original solver (`UnifiedCFM.solve_euler` with the original estimator, in
     PyTorch on the CPU in FP32) is given the inputs that Core AI actually used for
     that patch, before any replacement. Its result and the Core AI latent differ by
     at most the criterion. The original solver runs in its own child process, only
     when a latent needs it. In the same process it also solves the original inputs
     of that patch, and its result must equal the original latent bit for bit.

   Anything else fails the sentence. The record keeps the latent before replacement,
   its error, the input errors, the difference `a` between the original solver's
   result and the Core AI latent, and the difference `b` between that result and
   the original latent. `a` and `b` are given on the scale of the original solver's
   result and on the scale of the original latent.

The replaced patches are the union of both kinds; a `proj.fuse` replacement at step
`k` counts as patch `k + 1`, which uses its outputs. A patch whose `proj.fuse`
outputs and latent are both replaced counts once. At most one patch per sentence
and at most 2% of the original patches (4 of 239) may be replaced. These limits are
in `synthesis.json`. They were set after the observations below and are operating
limits, not a proof of correctness. A sentence passes when every comparison point
holds within these limits, every value is finite, the original generation stopped
before its patch limit, and the replacing generation ends with a stop at the
original stop patch. Its joined audio must also meet the criterion against the
original streaming decode.

Solver consistency on the same inputs does not show that the language-model side
that produced those inputs is correct. The per-component comparisons above and the
comparison of the replacing generation cover that side.

The stage also generates all five sentences without any replacement and records,
without a pass or fail, the latent error of every patch, the first patch outside the
criterion, the largest error, the stop patch, the end, the audio length and the peak
and RMS level. These generations are saved for the Swift runtime under
`check/swift_reference/<sentence>/`:

- `ids.npy` and the joined `audio.npy`;
- `patches.npz`: the noise, `dit_hidden`, `cond`, latent and stop logits of every
  patch;
- `calls.npz` and `calls.json`: the inputs and outputs of every call in order,
  including the prefill, each solver call and every `heads`, `locenc`, `vae`,
  `tslm.step`, `fuse` and `ralm.step` call. Stateful calls (`tslm` and `ralm` steps
  and the VAE) can be compared one by one only when they are replayed in order from
  the first call, because their states are not saved.

`plain` limited to 8 patches must end with `limit`, must not be a success and must
write no audio.

### Precision comparison

`synthesis_compare` generates `plain`, `long` and `design`, and the two sentences in
`synthesis.json` (a Japanese sentence with a date, times, a number and a price, and
an English sentence). It uses seeds 1708, 1709 and 1710 in three configurations:
all FP32, all FP16, and FP16 with the `proj` asset (projections, FSQ and stop head)
in FP32. The same sentence and seed use the same noise in every configuration.

The stage records the end, the stop patch, the patch count, the audio length, the
latent errors against FP32, the time after one warm-up generation in the same
worker, the peak and RMS level, the samples clipped when rounding to 16 bits, and
the load average. There is no pass or fail.

The audio is written as 48 kHz mono 16-bit WAV files without level changes. Seed
1708 goes to `compare/listen/<sentence>/<label>.wav`, the other seeds to
`compare/listen-extra/<seed>/<sentence>/<label>.wav`. Labels A to C are assigned to
the configurations per sentence with a recorded random seed. The files are written
after all generations, in label order, and get one shared modification time, so
the times do not show the configuration. The key is `compare/listen-key.json`,
outside the listening folders. The files stay outside Git.

The audio comes from the same weights run with Core AI. Numerical differences near
rounding boundaries inside the model can make the generated result differ from the
reference implementation, even with the same seed.

### End-to-end results

The run of 2026-11-17 passed `synthesis_check`. Token IDs and noise equal the
original for all five sentences, and `plain` limited to 8 patches ended with
`limit`, was not a success and wrote no audio. Three of the 239 original patches
were replaced (two `proj.fuse` and one latent), at most one per sentence.

| Sentence | Patches (Core AI / original) | Stop patch | Within the criterion without replacement | `proj.fuse` outputs replaced with the original | Latent replaced with the original | Without replacement: first patch outside, largest error, stop patch |
|---|---:|---:|---|---|---|---|
| `plain` | 15 / 15 | 14 / 14 | no | — | patch 9 | 9, 6.4e-3, 14 |
| `design` | 34 / 34 | 33 / 33 | yes | — | — | none, 4.0e-5, 33 |
| `long` | 105 / 105 | 104 / 104 | no | step 8 (patch 9) | — | 9, 2.19, 104 |
| `heldout_ja` | 52 / 52 | 51 / 51 | no | step 29 (patch 30) | — | 30, 0.042, 51 |
| `heldout_en` | 33 / 33 | 32 / 32 | yes | — | — | none, 9.0e-6, 32 |

- `long`, step 8: dimension 283, 4.77e-7 from a boundary, code 3 to 2; the remaining
  errors are 5.8e-8 and 1.1e-6. `heldout_ja`, step 29: dimension 328, 1.61e-6 from a
  boundary, code 1 to 0; the remaining errors are 9.8e-8 and 1.2e-6.
- `plain`, patch 9: before replacement the latent was 3.87e-3 from the original. Its
  inputs matched the original (noise bit for bit, `dit_hidden` 1.35e-6, `cond`
  7.3e-6). With those inputs the original solver gave a latent 2.6e-5 from the Core
  AI latent (`a`) and 3.84e-3 from the original latent (`b`). From the original
  inputs, the same solver process reproduced the original latent bit for bit.
- In the replacing generations, the other latents stayed within the criterion
  before replacement (largest 4.3e-4, in `long`), and the joined audio differed from
  the original streaming decode by 9.1e-6 to 8.1e-4.
- Without replacement, `long` and `heldout_ja` leave the original trajectory from
  the patch that uses the `proj.fuse` call replaced above (patches 9 and 30), and
  `plain` from patch 9. All five sentences still stop at the original patch.

Two experiments with the original solver looked at `plain` patch 9 (records outside
Git). With independent random perturbations of 4e-6 of each estimator output, 32
runs changed the latent by at most 3.2e-4, about 1/12 of the observed difference;
the verdict fixed before the experiment was undetermined. Given the inputs that the
Core AI generation without replacement actually used, the original solver agreed
with the Core AI latent within 2.6e-5 at patch 9 and within 2.6e-5 at every one of
the 239 patches, and differed from the original latent by 3.84e-3 at patch 9.

The precision comparison (15 generations per configuration, Python host,
GPU-preferred specialization, each time after one warm-up generation in its worker,
load average 4.1 to 6.1):

| Configuration | Stopped | Stop patch as FP32 | Latent error against FP32 (median of medians / largest) | ms per patch (median) | Seconds per second of audio (median) |
|---|---:|---:|---:|---:|---:|
| FP32 | 15/15 | — | — | 72.9 | 0.456 |
| FP16 | 15/15 | 10/15 | 0.118 / 2.14 | 47.1 | 0.294 |
| FP16 with `proj` in FP32 | 15/15 | 10/15 | 0.055 / 2.67 | 47.1 | 0.294 |

No sample was clipped at 16 bits. In both FP16 configurations every generation
leaves the FP32 latent criterion early (from patch 1 in the median generation).
On 2026-11-18 the owner kept FP16 (`fp16`, all components in FP16) as the default;
no scored listening evaluation has been recorded.

## App bundles

`strip`, `bundle` and `bundle_verify` turn the exported assets into folders that an
app reads.

- **Debug information**: `strip` loads each of the twelve FP32 and FP16 assets with
  `AIModelAsset.load`, applies `strip_debug_info` to the program and saves that same
  program with `save_asset` and the original metadata. `AIModelAsset.program` reads the
  bytecode again on every access, so the removal and the save must use one program
  object; the stage fails unless `main.mlirb` changes. It prints each program as
  generic MLIR without locations and with every constant and resource, and compares
  the SHA-256 of that text after loading, after the removal and after loading the
  saved asset in a new process. The three must be equal. `strip_one` does this for one
  asset into a new folder and also requires the source folder to be unchanged
  afterwards; `strip` calls it for each of the twelve assets, and the reference bundle
  calls it for the encoder. `strip.py` is a module of its
  own, so changes to the bundle code do not run the removal again; every removal
  writes a new creation date into `metadata.json`.
- **Bundles**: `bundle` makes `fp32`, `fp16` and `fp16_proj32` with the precisions of
  `synthesis.json`. The default is `fp16`, which the owner kept on 2026-11-18; no scored
  listening evaluation has been recorded. Each bundle is a self-contained folder:

  ```text
  manifest.json
  tokenizer.json   the unmodified file of the pinned revision
  LICENSE          the Apache-2.0 text of the official checkout
  NOTICE           the changes made and the conditions of use
  artifacts/<asset>-<precision>.aimodel/main.mlirb, main.hash, metadata.json
  ```

  The asset files are cloned with `clonefile` when the file system allows it and
  copied otherwise; `check.json` records how many files were cloned and copied.
  Copying all three bundles would need about 19 GB, and the stage asks for 25 GB of
  free space. The stage reads only `tokenizer.json` from the snapshot and `LICENSE`
  from the checkout. The notice comes from `bundle-notice.txt`.
- **Inventories**: `bundle` checks every file of a bundle against its source: the
  asset files against the sizes and SHA-256 values that `strip` recorded after the
  removal, `tokenizer.json` against `model.json`, `LICENSE` against the size and
  SHA-256 of the file at the pinned commit (kept in `bundle.py`), and `manifest.json`
  and `NOTICE` against the bytes it wrote. A missing, extra or changed file, or a
  symbolic link, fails the stage. The checked list, with the relative path, size and
  SHA-256 of every file, becomes `<configuration>.inventory.json`.
- **Absolute paths**: `bundle` reads every byte of every bundle and counts the home,
  external-volume and temporary-folder prefixes, written with `/` or with the escaped
  separators `\/`, `\u002f`, `\u002F`, `%2f` and `%2F`, and the current user name, in
  UTF-8, UTF-16LE and UTF-16BE. It records only counts and file names, never the
  matched text, and fails on any match. It also records the same scan of the exported
  assets before the removal, and checks with planted strings that the scan finds a
  home path in UTF-8, UTF-16LE and UTF-16BE and with the JSON and percent escapes, an
  external volume, a temporary folder, the user name and a match across a read
  boundary, and that it ignores a clean relative path.
- **Re-check**: `bundle_verify` loads every asset of the three bundles and checks its
  function descriptors against `model.json`. With the FP32 bundle it repeats the
  comparisons of `verify_fp32` and requires the same counts, results and `proj.fuse`
  classes as before the removal.

### Manifest

`manifest.json` is the contract that a native host reads. Its values are copied from
`model.json`, and a test compares them:

| Key | Content |
|---|---|
| `format` | 1 |
| `model` | `model_id`, `revision`, `upstream_revision`, `license` |
| `configuration` | `name` and the precision of every asset |
| `assets` | per asset: `path` relative to the bundle, `precision`, and per function the `inputs`, `outputs` and `states` as `[name, dtype, shape]` |
| `execution` | `capacity`, `prefill_length`, `max_text_tokens`, `audio_start_token`, `pad_token`, `state_initial_value`, `specialization` |
| `generation` | the generation settings of `model.json`, including `samples_per_patch` 7680 |
| `structure` | `patch` `[1, 64, 4]`, `sample_rate` 48000 |
| `tokenizer` | `tokenizer.json` |

In FP16 assets every floating input, output and state is float16, and `ids`,
`length` and `position` stay int32. The manifest contains no paths of the working
environment, dates or host names.

### Inventories in the repository

The inventories of the bundles made on 2026-11-17 and recorded in the
[validation record](../../../../docs/validation/voxcpm2-conversion-20261117.md) are
Swift resources, `voxcpm2-<configuration>.inventory.json` in
[`CoreAISamplesKit/Sources/SpeechAssets/Resources`](../../../../CoreAISamplesKit/Sources/SpeechAssets/Resources).
The Swift bundle check uses them, and the repository keeps no other copy. They
describe those bundles only. Every export and every `strip` run writes a new creation
date into `metadata.json`, so the hashes change even when the programs do not (three
`strip` runs on the same exports gave the same `main.mlirb` and `main.hash`). Replace
the resources with the `<configuration>.inventory.json` files that the `bundle` stage
writes, together with the bundles. To keep a bundle outside the work
directory, clone it (for example with `cp -cR` to
`.local/conversion/voxcpm2/bundles/<configuration>/`) and check the copy against its
inventory.

### Bundle results

The run of 2026-11-17 passed all three stages, and a second run reused all 18.

- All twelve programs have the same digest before and after the removal. `main.mlirb`
  became 177 to 18,277 bytes smaller per asset, and in `metadata.json` only
  `creationDate` changed. A probe before the stage found equal descriptors and bitwise
  equal outputs and states for `proj` FP32, `vae` FP32 (26 states) and `ralm` FP16
  (two functions sharing two states).
- `fp32` has 22 files and 9,349,325,520 bytes, `fp16` 22 files and 4,677,413,664 bytes,
  and `fp16_proj32` 22 files and 4,715,188,166 bytes. All 18 asset files of each
  bundle were cloned; none was copied.
- No match for the 132 patterns in the 18,741,927,350 bytes of the bundles. The
  exported assets had no match before the removal either; the planted strings were all
  found.
- With the FP32 bundle, all 3,934 comparisons that decide the result pass, and
  `proj.fuse` gives 233 matches, 1 boundary-consistent call and no failure, as before
  the removal.

## Limits

The component checks start from recorded inputs. The end-to-end check covers five
sentences with one seed on the Python host. Its replacing generation agrees with
the original only along the path with the recorded replacements, and the
generations without replacement leave the original trajectory (see the results
above). Not checked:

- same-input comparisons of the language models, their caches and `proj.fuse` along
  the generations without replacement;
- how often replacements are needed for other sentences and seeds;
- whether the Swift runtime behaves the same way;
- listening quality; no scored listening evaluation has been recorded.

The macOS app downloads the `fp16` bundle, published in
[okayuji/voxcpm2-coreai](https://huggingface.co/okayuji/voxcpm2-coreai), and uses it.
FP16 parts and configurations are recorded, not
accepted. The Python runtime cannot set `expectFrequentReshapes`; all shapes here are
static. The original six-asset recipe excludes reference-audio prompts, the VAE
encoder, quantization and iPhone. The separate reference recipe checks the FP32 VAE
encoder; it does not assemble a reference-audio app bundle.
