# Managed Irodori validation

## Asset import — 2026-10-27

Scope: #335 / #337, Models Irodori v2, based on `ffa3296`. Apple Swift 6.4 (swiftlang-6.4.0.34.1), Xcode 27.0 (27A266a), Apple silicon Mac. The implementation keeps the existing prepared FP32 bundle and shares its definition through the Rust-independent `SpeechAssets` target.

| Acceptance | Evidence | Result |
|---|---|---|
| IOS-IR-02 | `IrodoriImportTests.definition`, `alteredDefinition`, `missing`, and `knownBundle` | Pass for managed storage; normal Models exposure belongs to #336 |
| IR-ASSET-01 | `manifest`, `unsafeSource`, `changedManifest`, and `knownBundle` | Exact manifest/inventory, source rejection, and installed-copy rejection pass |
| IR-ASSET-02 | `legacyManifest`, existing local-image imports and managed-store regressions | Pass |
| IR-ASSET-03 | Rust input archive build, strict SwiftLint (0 violations), strict format check, full package tests, Mac Release | Pass |

The full package test command exited 0 and reported 903 tests across its test products, including 43 existing conditional skips. The known-bundle test was explicitly enabled separately under the shared lock: one test ran and passed (4.207 seconds on the final asset-target implementation). It copied and hashed all 14 files, tested lease-protected deletion, and rejected modified, extra, missing and symbolic-link files. No model was specialized or executed. The final constant-reference-only change was covered by focused input/import test runs; the earlier full-suite and app-build results remain applicable to unchanged behavior.

The existing iOS app also completed an unsigned Release build without introducing Rust linkage. The signed SpeechPhone Release build succeeded. Its generated app contained `CoreAISamplesKit_SpeechAssets.bundle/irodori-assets.json`, both file-sharing keys set to true, and the increased-memory entitlement.

One full-suite attempt stalled in the unchanged AVFoundation video fixtures and was terminated after stack inspection. The 43 video tests then passed in a fresh process, followed by a successful full package run. The cause of that isolated wait was not established; no unrelated video code was changed.

Codex's independent review found the initial Storage-to-runtime native dependency, which was fixed by introducing `SpeechAssets`. Claude Code independently reviewed the final branch, then verified three follow-ups: default import-source rejection tests, use of the shared model ID, and a link to the import guide. Both reviews ended with no unresolved confirmed or plausible findings. Review coverage includes identity/Codable compatibility, import/promotion/inspection, leases, the package graph, old image/HF paths, and the associated tests and documentation.

The body change is smaller than the 350–550-line estimate because copy, hash, promotion, ledger and lease handling were reused. The two-issue boundary and the original provisional 5 SP remain unchanged. Device generation, memory, audio output and subjective listening were not measured by this host check; later device results are linked below.

## Normal Models speech connection — 2026-10-27

Scope: #336 / #338, Models Irodori v2, after #337 (`38cf883`). The usual iOS catalog registers the exact pinned artifact, and its diagnostic receives that same catalog. `SpeechDemoSessions` binds each Models ticket to a managed service and a Speech Store. Merely binding a screen does not open or execute the model. The first explicit generation opens the selected managed folder; managed operation does not persist a standalone-app model path.

| Acceptance | Evidence | Result |
|---|---|---|
| IR-SESSION-01 | `ManagedSpeechTests.lazyBinding`, `gateAndResults`, `LocalSpeechCatalogTests.catalog` | Binding does no inference; permitted work uses the selected folder; the supplied catalog includes Irodori |
| IR-SESSION-02 | `ManagedSpeechTests.failures`, `SpeechFeatureTests.managedLifetime`, existing input/export tests | Generation failure and external playback interruption can retry; ending invalidates audio and export tickets |
| IOS-IR-04 | `stopDuringOpen`, `sessionLifetime`, `stopPlayback`, `SpeechModelSwitchTests.switchAfterStop` | Late completion is dropped; cancelled stop callers still join cleanup; Models retains the lease until speech shutdown finishes before allowing the next model |
| IOS-IR-06 | Strict lint/format, package tests, Mac Release, signed CoreAISamples-iOS and SpeechPhone Release, localization/model-report checks | Pass for automated host checks; device behavior not measured by this check |

The full package command exited 0, reporting 914 tests including 43 conditional skips. Eight managed speech/model-switch tests also passed with Thread Sanitizer, with no sanitizer warnings. The 836-key app string catalog has 819 translatable entries in English and Japanese; compiled iOS lookups checked 1,638 values with no failures. The model-report validator and all 27 report-tool tests passed.

Both signed iOS apps contain the shared asset resource and speech examples. The normal app retains `com.coreai-samples.DemoApp.ios` and the increased-memory entitlement. Its host signature verification passed. These checks do not establish device installation, specialization, available memory or audible playback.

The source review identified missing diagnostic reports for catalog construction errors, old result messages in a closed-session preview, a stale demo count, and a local-bundle information view that applied Qwen's license notice to every local artifact. Catalog errors now pass through the reporting boundary, and retained previews contain only text, seed and examples. The two added regression tests cover reporting and completed-session previews. The notice and link now require the Qwen image contract; Irodori no longer inherits that notice. Source-kind identity is not treated as a license classification.

Codex and Claude Code completed independent source reviews and follow-up checks of the fixes. No confirmed or plausible implementation findings remain. The final review also reconciled the test counts and kept device-only checks pending.

The README, application setup, iPhone instructions, Models specification and bilingual model records now describe the normal Models entry. The existing SpeechPhone diagnostic, source model folders and WAV files remain available. The original provisional 8 SP and two-issue dependency plan are retained; implementation methods reused the existing speech reducer, runner, playback and exporter.

This historical host check did not run model inference or measure device interaction. Later generation measurements are in [reference-input validation](reference-input-validation.md); [current validation limits](iphone.md#validation-limits) distinguish those measurements from microphone capture and listening quality.
