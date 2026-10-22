# Qwen Image 2.1 Turbo native runtime

Implementation for [#218](https://github.com/oka-yuji/coreai-samples/issues/218), using practical v2 and the [native preprocessing](qwen-image21-native-preprocessing.md) contract. The loader, sequential pipeline, cancellation joining, diagnostics and CPU lifecycle tests are implemented. The [generation UI and common Models adapter](qwen-image21-generation-ui.md) add the Store-owned request path. The explicit [native diagnostic entry](qwen-image21-native-diagnostics.md) provides fixed-input replay and lifecycle measurements. Full candidate fidelity, calibrated device budgets, actual headless device measurements and QMR-03 through QMR-06 remain pending. Fake backend tests do not establish GPU release or iPhone support.

## Ownership and execution

`LiveImageGenerationService` is a MainActor entry for one immutable `ModelDemoSession`. Assembly must pass that session's common Models execution permission, including its owner, binding generation and active scene. The caller retains the resource lease and residency permission until `release()` acknowledges cleanup. Constructing the service performs no model loading or generation. Feature/Store owns the request task; the runtime creates no task of its own.

The runtime actor keeps its busy gate across every suspension. Generation uses text → DiT × 8 → VAE decoder; editing adds Vision before text and VAE encoder before DiT. A component's function and model are released before the next component loads. DiT keeps one loaded function across eight steps. Native buffers remain within the backend call, and outputs are copied into owned Swift arrays before they leave it.

Stop rejects further steps and waits for the operation already submitted. A load includes specialization and function loading. The OS 27 cancellation shield joins that native operation before unloading; it does not claim to interrupt a GPU kernel. Service stop also joins its request's return path before permitting immediate reuse. Concurrent stops share cleanup. An unload failure retains the component and prevents generation until explicit cleanup succeeds. Permanent release prevents the old binding from starting again.

A request-specific admission check also runs before the first validation or model load. A request with a reference image is refused after bundle validation and before any component loads unless the profile's operation is `generate-and-edit`. Component headers are read beside each component asset, and the inventory walk uses `readdir` so that AppleDouble and other hidden files outside the manifest are rejected. If stop reaches an idle runtime before an already-requested generation reaches its actor, that withdrawn request is rejected when it arrives.

Progress carries the request ID. Its callback is nonescaping throughout the service and pipeline; it is delivered only within the awaited request. The service checks its delivery ID and current permission on MainActor before forwarding progress or a result. Revocation prevents later stages and suppresses stale delivery. The common adapter must still be tested for backgrounding, scene destruction and A → B → A switching with real resources.

## Validation and memory budgets

Validation requires a local-manifest resource with the pinned upstream revision. It hashes the raw manifest, checks its complete asset inventory and aggregate size, rejects extra files, directories and links, verifies all five component headers and requires numerical/quality acceptance flags. The common catalog supplies the trusted manifest identity; flags alone do not establish trust. Tokenizer and VAE scaling configurations must be in the hashed inventory.

`ImageMemoryBudget` binds `profileID` and `manifestSHA256` to all five components. Every component supplies positive `loadingAdditionalBytes` and `inferenceAdditionalBytes`. Missing estimates are errors. No production default is supplied while device calibration is pending.

Each load and inference takes a new memory observation. On iOS, available dirty-memory headroom plus current footprint forms the observed process budget. The reserve is the greater of 512 MiB and 20% of that budget, rounded up. The phase's additional estimate must fit alongside that reserve. The inference estimate excludes weights already counted in the loaded footprint. On macOS, free/inactive pages provide estimated system headroom and physical RAM supplies the reserve basis; the report distinguishes this from an iOS process limit.

Reports preserve cache lookup, specialization on a cache miss, function loading, each call and release timings. They include point observations of footprint, RSS, visible Metal allocation, available memory, thermal state and low-power mode. The kernel peak is a process-lifetime peak. Neither point samples nor that lifetime peak are labeled as a phase-specific peak. Fresh-process specialization and release measurements remain required.

## Diagnostics and validation

Diagnostics are disabled by default. An explicit `ImageDirectoryTrace` writes to a new directory, records component inputs/outputs, the initial latent and all eight updated latents, and refuses overwrites. Raw little-endian tensors have shape, physical dtype, byte count, SHA-256 and nonfinite counts. Output capture precedes finite validation. Trace I/O is part of diagnostic request time and must be excluded from clean latency claims.

The physical boundary supports Float32, Int32 and Boolean inputs and Float32 outputs. Padded and permuted strides are copied by logical index; overlapping writable strides are rejected. Profile precision controls timestep, Euler and VAE scaling rounding. The versioned native RNG does not claim PyTorch seed equivalence; fidelity replay supplies frozen initial noise explicitly.

Run CPU verification with:

```sh
swift test --package-path CoreAISamplesKit --filter ImageGeneration
```

Tests cover eight-step ordering, generation/editing stage selection, busy rejection, cancellation during load/text/DiT/decode, repeated stop, cleanup failure and retry, permission revocation, stale delivery, memory refusal, corrupt resources, physical buffers and diagnostics. They retain the independent preprocessing goldens. `Tools.Conversion.adapters.qwen_image21.native_scaling_fixture` produces the Torch CPU VAE-scaling golden for F32/F16/BF16.

QMR-01 and QMR-02 have CPU evidence for these boundaries; common Models integration and device acceptance remain open. QMR-03 through QMR-06 require a verified candidate, calibrated budget and actual phone execution. No trusted catalog hash, device success or quality rating is synthesized.
