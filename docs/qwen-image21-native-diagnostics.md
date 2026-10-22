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
| `stop` | Optional component, operation (`loading` or `inference`), DiT step (0–7; otherwise absent), and delayMilliseconds (1–60,000) |

Relative bundle/input paths resolve from the request's parent directory. The output is a new sibling directory named `image-check-<runID>`. Existing outputs are refused, including an existing symlink at that name. Source inputs and bundles are never modified.

The output retains `request.json`, `execution.json`, and a terminal `completion.json`. Successful generation also writes a 512-square `output.png` and raw straight `output.rgba`; completion contains the raw RGBA SHA-256. With trace enabled, a new `trace` directory contains the runtime's shape/dtype/hash records, component inputs/outputs, initial noise and all eight latent updates. Diagnostic records include the prompt and local paths and are private measurement data.

Input validation failures before output creation have no result directory. A record-writing failure preserves the original native and cleanup errors in the diagnostic outcome instead of replacing them with the final I/O error. A partial directory without a complete, parseable terminal record is not a successful run. Cancellation does not publish an unfinished image.

## Cancellation and evidence limits

The Store owns execution and optional delayed-stop tasks. Repeated stop and disappearance during stopping preserve the pending cleanup. Stop joins native release and terminal recording before its acknowledgement. Failed cleanup remains explicitly unconfirmed and can be retried on the diagnostic screen. A successful retry appends a separate cleanup record and leaves the original failure intact; the generation is still a failed run. If the record cannot be written, that additional recording failure is retained in the terminal outcome.

An optional stop plan schedules cancellation after a matching progress event. The completion record retains the last observed phase and stop-request times relative to diagnostic and runtime entry. That event precedes native submission, so a delayed-stop request alone does not prove that cancellation happened inside a GPU kernel or specialization. Inspect the actual phase records and release evidence; a stop that missed the intended operation is an inconclusive measurement. It must not satisfy QMR-05 by itself.

Mac command exit codes are 0 for completed, 2 for cancelled and 1 for failure or unconfirmed cleanup. The phone exits after a terminal acknowledged outcome; an unconfirmed cleanup keeps its stop control available. Diagnostic images, CPU fake tests and signed builds do not establish real-device acceptance.

The existing runtime reports point memory samples and process-lifetime peak values, not phase peaks. Use fresh-process measurements and external observation for phase/specialization peaks, and record power/thermal conditions before performance claims. A positive estimated budget is a stated admission policy for measurement, not evidence of an observed memory requirement. Production device profiles remain empty until the relevant measurements are accepted.
