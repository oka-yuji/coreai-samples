# Shared model residency validation

Implementation: [#262](https://github.com/oka-yuji/coreai-samples/issues/262), [PR #264](https://github.com/oka-yuji/coreai-samples/pull/264). Plan: QMI practical v2, 2026-10-20. The source review covered `0c23a9d` through `3d9594d`, followed by the permission-change wording correction.

`ModelLibraryRunner` reserves one owner/session before lease acquisition can suspend. Rendering binds an adapter only after the session has this reservation and a file lease. A completed request retains ownership while its native model remains loaded. Successful adapter stop precedes lease release and handoff; stop failure retains both. Nonowners never stop the resident adapter.

The normal iOS Models adapters are comparison, extraction, inspection, image search, video search and object finder. All check the admitted session gate. Search wrappers transfer input/result values into a new owner wrapper without sharing the old execution service. Product and object-finder sessions use a binding generation as well as owner/resource identity, preventing a retired runtime from becoming permitted after A→B→A.

## Evidence by acceptance condition

| Condition | Evidence | Status |
| --- | --- | --- |
| QMA-01 | `ModelResidencyTests.rejectsAnotherOwnerOfTheSameSelection`; wrapper ownership tests | CPU regression passed |
| QMA-02 | `reservesBeforeSuspensionAndReleasesAbandonedPreparation`; reservation before awaiting storage | CPU regression passed |
| QMA-03 | `waitsForSuccessfulReleaseBeforeHandoff`; `RealModelResidencyTests.retainsUnloadsAndHandsOff` | CPU and real Mac model passed |
| QMA-04 | Lifecycle tests, stop-failure retry, old-stop and binding-generation tests | CPU regression passed |
| QMA-05 | Six-adapter wiring review, package suite, macOS and signed iOS Release builds | Code/build regression passed; normal iPhone UI still pending |
| QMA-06 | Real EmbeddingGemma 2 load/encode/retention/unload/reload through the shared runner | Passed within the Mac integration scope below |

The independent reviews found a P2 input/result loss during owner replacement and a P3 permission-retry message inconsistent with restored permission. Both were corrected. A possible old-runtime A→B→A permission reuse was also eliminated with binding generations and regression tests. The action/effect permission-change race now returns a retryable preparation state; it no longer leaves preparation stuck.

## Real model integration

Host: M4 Max, macOS 27.0 (26A428), Xcode 27, Swift 6.4. Bundle: `embeddinggemma2-fp16-safe-s128-v70`, published revision `daab6dbfbae1610a87d9f181458419e3ea5ce682`, upstream `914f7f89142e33e77833254d9c9b90c3cef7303b`.

The test hard-links immutable files into a temporary managed store and validates them against the actual catalog's hashes. It does not invent a file lease for a different model. Two scenes share the real `ModelLibraryRunner`; a test adapter invokes the production `EmbeddingModel`.

The final run passed in 7.727 seconds with two loads and two native unloads. After each completed request, a second owner cannot bind. Native unload causes `EmbeddingError.notLoaded`, the model's weak reference becomes nil, and the lease count becomes zero. The next owner then reloads and produces the same vector. A late stop for the prior owner does not affect it.

| Point observation | RSS B | Footprint B | Graphics footprint B | Default Metal device allocated B |
| --- | ---: | ---: | ---: | ---: |
| Before load | 1,893,580,800 | 1,845,331,936 | 0 | 81,920 |
| First request completed, retained | 2,903,031,808 | 1,441,172,552 | 282,214,400 | 345,833,472 |
| First unload and stop acknowledged | 2,290,221,056 | 1,205,815,912 | 45,973,504 | 33,259,520 |
| Second owner reloaded | 2,894,675,968 | 1,331,596,432 | 274,366,464 | 345,817,088 |
| Second unload acknowledged | 2,281,029,632 | 1,102,957,184 | 45,973,504 | 33,243,136 |

The kernel lifetime peak reached 2,185,316,632 B and remained there after release. These are point observations immediately after the relevant operations. They are not phase peaks, total GPU usage, or additive categories of RAM. File verification and allocator/cache behavior also affect RSS and footprint. Native reference release does not imply every driver allocation returns to zero.

This integration does not run the app's full `ImageSearchAdapter → LiveImageSearchService` UI route, real inference cancellation races, or iPhone UI acceptance. The signed iOS build is not evidence that those device checks ran.

With the shared heavy-job lock held and no other model process running:

```sh
COREAI_RESIDENCY_BUNDLE=/path/to/embeddinggemma2-fp16 \
COREAI_RESIDENCY_METRICS=/path/to/result.json \
swift test --package-path CoreAISamplesKit --filter RealModelResidencyTests
```

The input bundle must be on the same volume as the temporary directory. Default `swift test` skips this model test. Scalar metrics are opt-in; weights and raw validation logs remain outside Git.

The required package suite passed. SwiftLint reported zero violations in 400 files; swift-format checked 28 changed Swift files. Rust inputs, macOS Release and signed iOS Release built successfully. Runtime localization has all 19 keys translated into its configured Japanese locale. The remaining acceptance item is normal iPhone UI regression once the device can launch the installed app.
