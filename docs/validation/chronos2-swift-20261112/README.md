# Chronos-2 Swift runtime verification

Issue [#543](https://github.com/oka-yuji/coreai-samples/issues/543), observed on 2026-11-12 (JST).

The Mac Swift `chronos-verify --verify` route compared all 18 fixed cases and 6,048 values with the official FP32 reference, then released the model, reloaded it and compared the first 336 values again. Every value passed `abs(actual - reference) <= 1e-4 + 1e-4 * abs(reference)`. The maximum absolute error was 0.00152587890625; the relative term is part of this criterion.

[Results](results.json) retain all raw quantiles, the manifest and file identities, and the tested code revision. The input and original reference are in the [conversion evidence](../chronos2-conversion-20261111/README.md). Both records use manifest `786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572` and graph `3094871fc431ae5d86d07c8b487ab53e48e60e9fb56efe803ba18c5bf88889c8`.

## Conditions and scope

- Apple M5 Max, 137438953472 B RAM, macOS 27.2.1 (26B131), Xcode 27.2 (27B52), Swift 6.4.2, macOS/iOS SDK 27.2.
- Swift language mode 6, nonisolated Core/Runtime, `NonisolatedNonsendingByDefault` and `InferIsolatedConformances` enabled.
- Debug Swift package/CLI, batch 1, one request at a time, GPU preference, `expectFrequentReshapes = false`. The actual delegate was not profiled. The machine-wide heavy-job lock was held.
- One pass over 18 cases, no warmup, then one release/reload repeat. A separate opt-in `ChronosNativeTests.nativeReference` invocation also passed the same gate.
- This is numerical and lifecycle validation, not a speed or memory benchmark. Power and thermal state were not sampled for this run. The common record retains these missing conditions rather than supplying inferred measurements.
- The JSON-input route returned all 21 quantiles, shape, real model identity and the explicit timing boundary. Invalid input, absent model, absent output parent and existing output all returned nonzero. The existing output remained byte-identical and failed requests left no report.
- CPU tests cover context boundaries, bundle inventory/hash/contract/path failures, all-output strides, noncooperative load/inference, busy rejection, cancellation, multiple release waiters and later reuse. The CPU run skips native inference explicitly.
- iOS Release library build succeeded without signing. This does not establish iPhone inference or app support. Models and normal UI remain assigned to the following Issues.

## Reproduce

Use the shared heavy-job lock described in `AGENTS.md` before either model command. Supply the nine-file bundle from the conversion recipe; output must be a new file in an existing directory.

```sh
swift run --package-path CoreAISamplesKit chronos-verify --bundle /path/to/chronos2-bundle --verify --output /path/to/new-result.json
COREAI_CHRONOS2_BUNDLE=/path/to/chronos2-bundle swift test --package-path CoreAISamplesKit --filter ChronosNativeTests.nativeReference
```

## Integration failure retained

The SF scratch environment's CPU tests passed, while the first full-package run failed on the host's standard temporary directory. Rejecting every symlink above the bundle rejected the operating system's parent alias. The fix keeps the bundle root and every internal entry non-symlink, with descriptor-relative reads and hash validation, while allowing an alias outside the bundle boundary. `ChronosBundleTests.parentAlias` and the ordinary host temporary-directory run verify this behavior. No bundle-content check or numerical tolerance was weakened.

## Asset ownership review

The initial runtime verified source files and then let Core AI reopen their original paths. A regular directory replacement between those operations could disconnect the reported identity from the loaded model. The final runtime copies or clones from open file descriptors into a private directory, validates that complete copy against the original identity, and retains its owner alongside the model and function. A stream-copy fallback remains bounded and checks cancellation. CPU regression tests replace the original graph after snapshot creation, corrupt the copy, inject copy failure, and verify cleanup. The final native gate below was repeated through this ownership path.

## Cache and teardown scope

The runtime deliberately makes a fresh private source snapshot for each load. Cross-load specialization-cache hits and warm-load performance are not promised. [Three additional fresh CLI processes](cache-observation.json) used new snapshot paths; the observed compiled-payload cache remained six files and 478028880 bytes with identical hashes before and after each process. This observation covers the compiled payload only, not cache-index metadata, internal specialization events or other devices. It does not establish an unlimited-run storage bound or a speed result.

The native owner now explicitly drops its function and model references before allowing the source snapshot to be destroyed. Input/output files accept an aliased parent directory and normalized relative paths; bundle roots and internal entries retain no-follow checks. The parent-alias command regression failed before the fix and passed afterward.
