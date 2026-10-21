# Bounded memory while checking model files

Issue #277, QMI practical v2 supplement, 2026-10-21. This change drains temporary objects after each 1 MiB SHA-256 chunk in `ArtifactTransfer`, `ManagedModelStore` and `EmbeddingBundle`. It preserves the digest, cancellation checks, file closure, and existing error/state transitions.

## Regression and acceptance

| Condition | Evidence | Result |
|---|---|---|
| QMH-01 | `ArtifactTransferMemoryTests.testGivenLargeArtifactWhenReusedThenHashMemoryIsBounded`, fresh filtered process | Before: 256 MiB input, 278,578,064 B kernel lifetime footprint peak, assertion failed. After: 11,059,920 B for 256 MiB and 11,059,896 B for 64 MiB; known SHA-256 accepted and the loader was never called. |
| QMH-02 | Existing `ArtifactTransferTests`, `ManagedModelStoreTests`, transfer runner tests in the package suite | Integrity failures, incomplete files, cancellation, restart, and recovery remain covered; no learned-model inference was run. |
| QMH-03 | Separate Swift 6 optimized CPU probe for each row below | All five patched paths below 128 MiB; reuse/fetch finish SHA verification; inspect/promote reach the deliberately failing learned-asset validator after hashes. |
| QMH-04 | Required lint, format, Irodori input library, package tests, macOS Release and iOS Release builds | Passed. Lint: 398 files, zero violations. Format: three Swift files. Package runners report 732 tests. The log contains 38 skip entries (34 tests and four suites, including the isolated memory test); these are not executed-model passes. The two memory opt-in invocations passed separately. iOS compile used `CODE_SIGNING_ALLOWED=NO`; this is not a signed-device integration test. |

Run the two isolated regressions separately:

```sh
COREAI_HASH_MEMORY_TEST=1 COREAI_HASH_MEMORY_MIB=64 swift test --package-path CoreAISamplesKit --filter ArtifactTransferMemoryTests
COREAI_HASH_MEMORY_TEST=1 COREAI_HASH_MEMORY_MIB=256 swift test --package-path CoreAISamplesKit --filter ArtifactTransferMemoryTests
```

The opt-in test uses sparse zero-filled files, an externally fixed SHA-256 for each size, and the real transfer reuse path. Run it in a fresh filtered process because the kernel counter is a process-lifetime high water mark. It does not load model weights or use the GPU.

## Independent entry-point probe

M4 Max, macOS 27.0 (26A428), arm64, Swift 6.4. The probe uses `-O -swift-version 6 -enable-upcoming-feature NonisolatedNonsendingByDefault -enable-upcoming-feature InferIsolatedConformances -parse-as-library`.

Production source files are copied from the base and patched worktree. Only cross-module imports are removed. Model definitions and hash logic are unchanged. The harness substitutes a sparse-file downloader and learned-asset validators that throw after file verification, so no model/GPU path executes. Every row is a separate process and its exit status is zero. The peak is an absolute kernel lifetime footprint value, not RSS, a phase allocation, or a difference between samples.

| Entry point | Input MiB | Variant | Kernel lifetime peak B | Verified boundary |
|---|---:|---|---:|---|
| reuse | 64 | original | 70,337,112 | all-hashes-passed |
| reuse | 64 | pooled | 6,242,856 | all-hashes-passed |
| reuse | 256 | original | 272,581,352 | all-hashes-passed |
| reuse | 256 | pooled | 5,145,128 | all-hashes-passed |
| fetch | 256 | original | 273,089,256 | all-hashes-passed |
| fetch | 256 | pooled | 5,571,112 | all-hashes-passed |
| inspect | 256 | original | 272,614,120 | all-hashes-checked-before-asset-validator |
| inspect | 256 | pooled | 5,145,128 | all-hashes-checked-before-asset-validator |
| promote | 256 | original | 272,614,120 | all-hashes-checked-before-asset-validator |
| promote | 256 | pooled | 5,128,744 | all-hashes-checked-before-asset-validator |

`original` is base `d5e6ce2575b6f4940a4f039d2632b8ba84f77f77`; `pooled` is this patch. The original failure, source snapshots, harness, compiler logs, commands, and complete results are preserved in the private knowledge record `projects/coreai-samples/qwen-image21-mobile-20261020/hash-memory-277/`.

Patched source SHA-256:

- `CoreAISamplesKit/Sources/ModelLibraryTransfer/ArtifactTransfer.swift`: `07d6cd8a4d1523aa056c21aab5700c83eb8af3640dab42261c0405c041c59b95`
- `CoreAISamplesKit/Sources/ModelLibraryStorage/ManagedModelStore.swift`: `1f0f816f7198d09e25302ada4e051d7ed984a0fb3853413b31273739632a9ce5`

Full probe evidence SHA-256: `d090cdc3b0d1b825d52a0cf6e2928e2e9a290dcd77590fab7ffbd9f03d9c0a1e`.

## Review follow-up (2026-10-22)

An independent review found that the inspect/promote probe stopped before `EmbeddingBundle(directory:)`, which hashes every file again with the same unpooled loop. On the production path, promoting an EmbeddingGemma 2 FP16 fixture whose `token_embeddings.bf16` is a 256 MiB sparse file reached 276,743,032 B. `EmbeddingBundle.digest` now uses the same per-chunk pool. With the patched sources, both paths completed every file hash and the full bundle validation, returning ready, in fresh processes: promote 8,995,488 B and inspect 9,077,432 B. Model loading also uses this digest.

The opt-in regression now requires the peak to stay below half of the synthetic file. The original loop at 64 MiB reached 75,104,976 B and therefore passed the earlier 128 MiB limit; it now fails. With the patch, 64 MiB reached 8,733,344 B and 256 MiB reached 8,700,576 B.

`HubDownloader` on the macOS Models screen and `DecisionBundle` have the same loop and are tracked in #281.

- `CoreAISamplesKit/Sources/EmbeddingRuntime/EmbeddingBundle.swift`: `a75bcd77208578471e290f625951889cf7449aaa27289e5624240f45ad12ac1d`

## Scope and next step

The fix is shared infrastructure for the planned Qwen bundle import (#263). It does not establish full mobile generation, bundle import acceptance, or end-to-end image quality. The earlier iPhone component probe used the same per-chunk lifetime correction and remains separate evidence; this PR was verified on synthetic CPU paths and by compiling the changed library for iOS.

A one-issue change keeps the two equivalent hash loops and their resource-lifetime contract together. No public API, actor, state, hash algorithm, or retry policy was added. The memory regression belongs to the actual file-verification path; unrelated implementation improvements remain outside this change.
