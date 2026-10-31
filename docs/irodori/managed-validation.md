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

## Public download verification (2026-11-01)

PR #420 registers the fixed public revision `f5db5096365c7d1388a5002cfb3280f66cb19375` of [Irodori Small MF](https://huggingface.co/okayuji/irodori-v4.1-small-mf-coreai-fp32). On M5 Max 128 GiB, macOS 27.2 (26B123), Xcode 27.0 (27A266a), `LocalSpeechCatalogTests.downloadReference` downloaded the model into a new managed store through the normal Models transfer. The transfer uses an ephemeral URLSession without credentials or cookies. One explicitly enabled test ran and passed; the acquisition route was `download`, with both state and inspection `ready` and no failures.

All 22 installed files, totaling **3,835,396,408 bytes**, matched the audited release by size and SHA-256. These are the 20 model files plus LICENSE and NOTICE. The full Hugging Face repository additionally includes README, a synthetic audio sample and `.gitattributes`: 25 files totaling **3,836,457,825 bytes**. The diagnostic used the iOS catalog on the Mac; it does not establish iPhone screen interaction, microphone capture or listening quality.

The acquired files passed `SpeechReferenceIntegrationTests.nativeSmoke`: reference/no-reference generation, repeated-reference equality and the 0.5/30-second input boundaries. The outputs were 4.00 and 5.48 seconds. Their SHA-256 values differ from the earlier M4 Max/macOS 27.0 outputs, so the additional cross-environment byte-equality check failed. That result is retained, not converted into a pass.

A preregistered follow-up ran the original pre-strip assets and their original pinned manifest with baseline code `1ae9634e29729afa8595c6d9caaad6f63850f862` on the same M5 Max/macOS 27.2 host. The generation code, input processing, dependencies and nativeSmoke test match `ed013ff786fb9aa79c61c8addb8883ec321965b2`; only the asset identity/inventory differs. Both original-asset WAVs were byte-identical to the newly acquired release outputs, and the repeated-reference and boundary checks passed again. This supports output preservation within this environment; it does not identify whether GPU generation or OS version caused the historical difference. The frozen control plan retained two public-download inventory fields and a historical-comparison sentence from its template. The linked result records the actual 20-file/3,836,181,569-byte source and this metadata correction without rewriting the plan or its hash. Control thermal state was not sampled separately.

The public, anonymous information audit passed for the sole main ref, its one commit and all 25 objects, including repository history. Its file manifest and refs match the final private publication gate. The host CPU suite, strict lint/format, bundle/report tests, package pins, Mac Release and signed CoreAISamples-iOS/SpeechPhone Release builds also passed. Both iOS products contain the two file-sharing keys and increased-memory entitlement. Conditional model tests in the default CPU suite remained skipped; the download and native checks above were enabled separately.

The [machine-readable result](release-download-validation.json) retains the output hashes, failed cross-environment comparison, successful same-host control, environment and scope. Original-PyTorch parity, listening quality and iPhone normal-screen behavior remain outside these host checks.
