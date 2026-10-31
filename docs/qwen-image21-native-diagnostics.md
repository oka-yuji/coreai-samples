# Qwen Image native diagnostics

The explicit `--image-generation-check <request.json>` entry executes the same native service as the generation UI. Ordinary launch continues to Models. A diagnostic launch is a separate process with one Store-owned request and no other demo adapter. Phone scenes share that Store; the first scene owns start/stop/end actions, and another scene can only observe it. It does not register an artifact in the production catalog or bypass the runtime's verified, numeric and quality checks.

On macOS, pass an absolute request path to the Release app executable. On iPhone, the request path is relative to the app's Documents directory. Install the signed CoreAISamples-iOS target with its increased-memory-limit entitlement and preserve the app's existing data container. Model jobs still require the repository's shared heavy-job lock.

```sh
CoreAISamples.app/Contents/MacOS/CoreAISamples --image-generation-check /path/to/request.json
```

## Request and output contract

The UTF-8 JSON uses these fields:

| Field | Meaning |
|---|---|
| `format` | `1` |
| `runID` | A new lowercase ASCII letters/digits/hyphens identifier, up to 100 bytes |
| `bundlePath`, `bundleBytes` | Exact local verified bundle path and aggregate inventory plus raw manifest bytes |
| `budget` | `ImageMemoryBudget`: `profileID`, raw `manifestSHA256`, and all five components with positive `loadingAdditionalBytes` / `inferenceAdditionalBytes` |
| `budgetBasis`, `budgetEvidence` | `estimate` or `measured`, plus the concrete calculation or measurement reference; estimates are not device calibration |
| `hardwareModel`, `osBuild` | Exact `hw.model` on Mac / `hw.machine` on iPhone, and `kern.osversion`; mismatches are rejected |
| `prompt`, `seed` | Original prompt and exact integer seed within the native request contract |
| `reference` | Optional `{ "path": ..., "sha256": ... }` for one hashed image with a longest side at most 1,024; existing eight-bit RGB/orientation/center-crop preprocessing applies |
| `noise` | Optional `{ "path": ..., "sha256": ... }` for exactly 65,536 little-endian Float32 values in packed latent order; NaN/infinity are rejected |
| `trace` | Explicit Boolean; false for clean timing, true for intermediate tensor capture |
| `loadingMode` | Optional diagnostic comparison: `explicitCache` (default), `automatic`, or `aheadOfTime`, all using GPU preference. Automatic and compiled loading use `AIModel(contentsOf:options:)`; cache-hit state is unobserved and omitted, and initialization duration is recorded separately. Ordinary Models retains explicit cache lookup. |
| `stop` | Optional component, operation (`loading` or `inference`), DiT step (0–7; otherwise absent), and delayMilliseconds (1–60,000) |

Relative bundle/input paths resolve from the request's parent directory. The output is a new sibling directory named `image-check-<runID>`. Existing outputs are refused, including an existing symlink at that name. Source inputs and bundles are never modified.

The reference's exact hashed Data is passed to the shared image reader. Standard RGB/RGBA PNGs use the unassociated-pixel decoder described in [native preprocessing](qwen-image21-native-preprocessing.md); the diagnostic does not reopen the URL or pass the PNG through a premultiplying display decoder. The existing 16 MiB input-file and 1,024-pixel dimension limits remain in effect.

The output retains `request.json`, `execution.json`, and a terminal `completion.json`. Successful generation also writes a 512-square `output.png` and raw straight `output.rgba`; completion contains the raw RGBA SHA-256. With trace enabled, a new `trace` directory contains the runtime's shape/dtype/hash records, component inputs/outputs, initial noise and all eight latent updates. Diagnostic records include the prompt and local paths and are private measurement data.

The explicit diagnostic entry also writes `checkpoint.json` before preparation and each normal component load, inference and release boundary. Its actor completes an atomic replacement before the pipeline can proceed, then the pipeline rechecks cancellation and ownership. The checkpoint binds the request, profile and manifest to the process ID, system uptime, current phase, optional DiT step, point memory observation and already completed phase metrics. Unavailable memory fields remain absent rather than becoming zero. If recording fails, the next model operation is refused. Emergency cleanup does not depend on another journal write; it still releases resources and records its outcome in the terminal report when possible.

A checkpoint's `report.status` remains `running`, including the final checkpoint before VAE release. It proves only that execution reached that boundary, not that the upcoming operation completed, nor that its memory observation is the phase peak. An OS termination can leave a readable checkpoint without `execution.json` or `completion.json`; retain that partial run as incomplete and correlate its process ID with the OS termination record. The regular Models path does not create this journal. The standalone temporary QwenMobileProbe is not required; use the CoreAISamples app for these diagnostics.

`native-checkpoint.json` further separates function loading, each input allocation/copy and the call to `InferenceFunction.run`. It is bound to the same request, profile, manifest and latest pipeline checkpoint. Input records include the actual SDK descriptor's shape, preferred strides and minimum byte count, with memory observations before and after creation. A `run-before` record establishes that all inputs were prepared; `run-after` establishes that the awaited call returned. Neither record alone establishes valid output or a completed image. Emergency unloading does not depend on these writes.

Native checkpoints observe an already admitted backend operation: a load includes specialization and function loading, and a call includes native input buffers and execution. They do not add cancellation points inside that protected operation. Stop continues to join the current operation before release; the pipeline checkpoint before the next operation rechecks cancellation and ownership.

Input validation failures before output creation have no result directory. A record-writing failure preserves the original native and cleanup errors in the diagnostic outcome instead of replacing them with the final I/O error. A partial directory without a complete, parseable terminal record is not a successful run. Cancellation does not publish an unfinished image.

## Cancellation and evidence limits

The Store owns execution and optional delayed-stop tasks. Repeated stop and disappearance during stopping preserve the pending cleanup. Stop joins native release and terminal recording before its acknowledgement. Failed cleanup remains explicitly unconfirmed and can be retried on the diagnostic screen. A successful retry appends a separate cleanup record and leaves the original failure intact; the generation is still a failed run. If the record cannot be written, that additional recording failure is retained in the terminal outcome.

An optional stop plan schedules cancellation after a matching progress event. The completion record retains the last observed phase and stop-request times relative to diagnostic and runtime entry. That event precedes native submission, so a delayed-stop request alone does not prove that cancellation happened inside a GPU kernel or specialization. Inspect the actual phase records and release evidence; a stop that missed the intended operation is an inconclusive measurement. It must not satisfy QMR-05 by itself.

Mac command exit codes are 0 for completed, 2 for cancelled and 1 for failure or unconfirmed cleanup. The phone exits after a terminal acknowledged outcome; an unconfirmed cleanup keeps its stop control available. Diagnostic images, CPU fake tests and signed builds do not establish real-device acceptance.

The existing runtime reports point memory samples and process-lifetime peak values, not phase peaks. Use fresh-process measurements and external observation for phase/specialization peaks, and record power/thermal conditions before performance claims. A positive estimated budget is a stated admission policy for measurement, not evidence of an observed memory requirement. Production device profiles remain empty until the relevant measurements are accepted.

## Hosted consecutive calibration

`CoreAISamples-ImageTests` runs the shared `ImageDeviceCalibrationTests` inside the ordinary iPhone app. It requires the explicit `COREAI_IMAGE_CALIBRATION_CONFIG` environment variable with a Documents-relative configuration path. A new `runID` preserves earlier results. The configuration lists one or two request files and a repetition count; the service is released and observed for five seconds before each new service instance.

```json
{
  "runID": "native-complex-checkpoints-v5",
  "requests": ["generation/request.json", "editing/request.json"],
  "repetitions": 5,
  "checkpoints": true
}
```

The optional `checkpoints` flag defaults to disabled. Enabling it uses the existing runtime journal to save `checkpoint.json` and `native-checkpoint.json` before model operations, without tensor traces. Its execution reports explicitly set `diagnosticsEnabled`; compare its instrumentation conditions separately from trials with journaling disabled. A checkpoint identifies the last observed boundary, not a completed GPU operation or a successful image. Recording failures still propagate through the existing cleanup path.

Each finished attempt writes `measurement.json`, and only the complete sequence writes `completed.txt`. A missing terminal record remains incomplete even when earlier attempts passed. The [October 31 record](research/qwen-image21-device-series-partial-20261031.json) contains nine completed requests followed by a test-connection failure during the tenth attempt. It does not activate production device budgets or ordinary Models registration.

## Source-bound AOT comparison

A format-two manifest may add `compiled_assets`, keyed by an existing component name. Each entry declares `asset` (a relative `.aimodelc` directory), `architecture` and `source_sha256`. The source hash must match that component's raw `main.mlirb` in the inventory. Every compiled file is included in the complete inventory, and its metadata is limited to 1 MiB. Both Models import and the runtime compare the compiled `metadata.json` `sourceHash` with the raw model hash. Existing raw bundles remain supported.

Selecting `aheadOfTime` requires a nonempty declaration and an exact match between every declared architecture and `AIModel.deviceArchitectureName` before any model loads. Declared components use the compiled asset; other components retain explicit raw cache lookup. Missing, altered, linked or mismatched files fail validation. Native checkpoints name the actual selected asset. This diagnostic option does not register a model in Models or supply a production memory budget.

Use the pinned mobile conversion environment and the common runner's shared heavy-job lock for this pack-stage command:

```sh
xcrun coreai-build compile /path/to/accepted-raw-bundle/denoiser/denoiser.aimodel \
  --platform iOS --min-deployment-version 27.0 --preferred-compute gpu \
  --architecture h18p --output /new/compiled/denoiser.h18p.aimodelc

python -m Tools.Conversion.adapters.qwen_image21.mobile_aot_pack \
  --bundle /path/to/accepted-raw-bundle \
  --manifest-sha256 <recorded-raw-manifest-sha256> \
  --compiled denoiser /path/to/denoiser.h18p.aimodelc \
  --architecture h18p --output /new/aot-comparison
```

The command refuses existing or overlapping output directories, verifies the pinned raw manifest and both inventories, then copies into a new bundle and writes its new manifest and receipt. It preserves the raw bundle and its numerical/visual evidence. `compiled_verification` and `device_verification` remain `pending`; source binding does not certify the compiled output's numerical behavior.

The [October 29 AOT record](research/qwen-image21-aot-20261029.json) covers a 57-file, 20,630,914,733-byte candidate built from the same accepted raw DiT. The physical iPhone loads the selected h18p asset, but its loaded footprint remains about 3.31 GB. The generation trial reaches `run-before` after input preparation, then the OS terminates PID 5204 with `vm-pageshortage`, with a matching executable UUID. No complete image or production calibration was obtained; AOT alone did not resolve this workload's memory failure.

## Physical checkpoint verification, October 29, 2026

The signed CoreAISamples app on iPhone 17 Pro / iOS 27.0.1 (24A446) records checkpoints before model loading and each attempted stage. The [scoped device record](research/qwen-image21-native-checkpoints-20261029.json) preserves three admission refusals and a subsequent OS memory-pressure termination at the first DiT inference boundary. The last checkpoint remains readable without a terminal report or image, and its process ID and the installed executable UUID match the jetsam record. The explicit diagnostic entry uses the ordinary app binary; the temporary QwenMobileProbe was removed after archiving its diagnostic files.

Text loading, inference and release completed in the current native pipeline. DiT loading completed with approximately 3.31 GB footprint in both cache-miss and cache-hit runs. The v3 attempt under `2fac561` (PID 3132) ended with `vm-pageshortage`; no completed DiT step or complete generation is claimed. A later `84db75b` attempt saved `run-before` after every native input was prepared, then lost its control connection and ended without a terminal report. Its nearby system memory-pressure record does not identify an app kill reason, so that outcome is kept separately as unconfirmed. The positive diagnostic estimates and their revisions are documented in the record. They are not production calibration, and the 512 MiB / 20-percent reserve is unchanged. The observed app budget was 6 GiB despite the increased-memory-limit entitlement; device physical RAM alone does not determine that budget.

The `e71e655` loader comparison requested stop during DiT loading and joined load/release without DiT inference. The explicit cache route and convenience initializer produced loaded footprints of 3,311,230,288 and 3,311,263,104 bytes. This single pair does not support a substantial difference due to loader choice. Automatic cache-hit state is unobserved; timing and output equivalence are not established by these stopped trials.

QMJ-01 and QMJ-02 have CPU journal tests plus the existing pipeline/diagnostic lifecycle tests, including a real atomic-write refusal that preserves the previous checkpoint. QMJ-03 has the signed iOS build, actual plist/entitlement checks and the physical checkpoint evidence above. Ordinary launch was also opened on the device and displayed Models; its routing remains covered by `IOSDemoLaunchTests`. Checkpoint injection is confined to explicit diagnostics. Full generation/editing, normal Models operation and final device acceptance remain in #219, #220 and #221.
