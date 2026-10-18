# iOS Models integration

2026-10-18 JST. Issue #198 (parent #199), PR #209. This record checks the ordinary `CoreAISamples-iOS` assembly, with the real runner, the catalog and the five demo providers, from acquisition through every demo. Automated results and device checks are kept apart. A build or a fake-based test does not stand in for a device check, and nothing below that is marked pending has been performed.

Update 2026-10-19: #210 (PR #212) adopted the published distributions. [Distribution adoption](#distribution-adoption-210) records the definitions, their tests, and the acquisition on the Mac and the iPhone. The target table below still describes the #198 tree.

## Target

| Item | Value |
|---|---|
| Source | The tree committed to PR #209 (`chore/198-ios-integration-check`) together with this record, on main `6adbb33` |
| Catalog | `IOSDemoCatalog` of that tree: GLiNER2.5-multi (comparison, extraction, inspection) and EmbeddingGemma 2 (image search, camera search). The artifacts `gliner2.5-multi-unconfigured` and `embeddinggemma2-unconfigured` have the placeholder contract `unconfigured`, no distribution and no files |
| Distribution and upstream revisions | None adopted (#188). The catalog contains no real repository, revision or checksum |
| Build host | Apple silicon Mac, macOS 27.0 (26A428), Xcode 27.0 (27A266a), Swift 6.4, iPhoneOS and iPhoneSimulator 27.0 SDKs |
| Device | None. This work installed nothing on an iPhone |

## Automated checks

| Check | Command | Result |
|---|---|---|
| Strict lint | `swiftlint lint --config .swiftlint.yml --baseline .swiftlint-baseline.json --strict --no-cache` | 0 violations in 381 files |
| Format | `python3 Tools/Format/check_format.py --base origin/main` | PASS (10 changed Swift files) |
| Irodori inputs | `./Tools/Irodori/build-inputs.sh` | exit 0 |
| CPU tests | `swift test --package-path CoreAISamplesKit` | 682 tests: 650 passed, 32 skipped, 0 failed. The skipped tests need model bundles, fixtures or environment variables (`CLAUDE.md`, heavy jobs) |
| macOS Release | `xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples -configuration Release -destination 'platform=macOS' build` | BUILD SUCCEEDED |
| iPhoneOS Release | `xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples-iOS -configuration Release -destination 'generic/platform=iOS' build` | BUILD SUCCEEDED, signed with the local development team |
| Native preview | `xcodegen generate --spec Examples/ModelLibraryPreview/project.yml`, then a simulator build of `ModelLibraryPreview` | BUILD SUCCEEDED (build only; the preview was not driven) |
| String Catalogs | `python3 Tools/Localization/check-catalogs.py` | 0 errors |

E2E-01 is covered by `ModelLibraryAssemblyTests` in `ModelLibraryRuntimeTests`, which assembles `ModelLibraryAssembly` from the real `IOSDemoCatalog` with five fake providers that carry its demo IDs:

- "E2E-01 The app assembly connects one provider to each demo of the iOS catalog, and browsing, refreshing and download requests for its unconfigured artifacts create no session, call no provider and leave legacy folders unchanged"
- "E2E-01 Direct operations of the app assembly with another owner's ticket, an artifact that was never acquired, or a session that is stopping or stopped take no lease and start no provider work"
- "E2E-01 With an acquired artifact the app assembly rejects another owner's ticket and operations during or after a stop while a valid session still runs" (the same assembly over the downloaded fixture artifact of `RuntimeHarness`)
- "The model library self-test reports the iOS catalog, its five providers, the capability states and missing unconfigured artifacts, fails when a catalog demo has no provider, and leaves the iOS-only checks out without a registry"

Each test failed when its behavior was broken on purpose: one adapter left unregistered, an empty connected-demo set, a lease kept for an identity that is not installed, a self-test that ignores a registry mismatch, and the self-test destination checked before the evaluations (`IOSDemoLaunchTests`). The acquired-artifact test alone failed when `permits` ignored the session ticket, when `access(to:)` ignored a pending stop, and when a released session could take a lease again; the tests with unconfigured artifacts still passed in those cases. The sources were restored afterwards.

The app's own five adapters are not in these tests, because they are iOS app types. `IOSModelLibraryAssembly.make(catalog:)` stops the app when the adapters and the catalog demos differ, and `--model-library-selftest` reports the registered demos on the device (pending below).

### Diagnostics

| Launch | Result |
|---|---|
| macOS `--model-library-selftest selftest.json` | `MODEL LIBRARY SELFTEST PASS`, exit 0. `failures` is empty, `catalogValidated` is true, 2 models and 5 demos are visible on iOS, both artifacts are `missing` with 0 files, and the managed root `~/Library/Application Support/com.coreai-samples.DemoApp/ModelLibrary` does not exist and was not created. `registry` and `capabilities` are absent, and `scope` says that they belong to the iOS app |
| macOS `--model-library-selftest=selftest.json` | Same result |
| macOS `--model-library-selftest` with an output folder that does not exist | Exit 1 with the write error |
| macOS `--check-localization` | Exit 0; 92 runtime UI keys, 0 missing Japanese translations |
| macOS `--agent-selftest --offline` | `AGENT SELFTEST PASS`; network cases skipped, 0 failed |
| macOS `--extract-eval gold.json` without `--asset` | Exit 2 with the usage line |
| `IOSDemoLaunchTests` | Destinations of the ordinary launch, `--extract-eval`, `--image-search-eval`, `--model-library-selftest` and the legacy arguments, including their precedence |
| `ExtractEvaluationTests` | Scoring of the extract evaluation (part of the CPU tests above) |

Not run, because they need a model, a device or the network: `--extract-eval` with an asset, including the iPhone's Documents-relative paths (the resolution is `ExtractEval.execute(relativeTo:)` in the app and has no automated test); `--image-search-eval`; `--selftest`; `--product-e2e`; `--product-source-check`; `--recipe-e2e`; `--decide-e2e`; the Hub self-tests; the model-gated package tests (`COREAI_TEST_BUNDLES`, `COREAI_DECISION_BUNDLE`, `EG2_TOKENIZER`, `GLINER_*`); and the iPhone `--model-library-selftest`.

## Distribution adoption (#210)

2026-10-19 JST, PR #212 on main `6499acd`. `IOSDemoCatalog` pins the definitions from the publication. [`distribution.json`](ios-models-distribution-20261019/distribution.json) is the copy that the tests compare with the catalog; the repositories are public and anonymously downloadable.

| Artifact | Distribution | Root | Files | Bytes | Upstream |
|---|---|---|---:|---:|---|
| `gliner2.5-multi-fp32-t256-t512` (FP32) | `okayuji/gliner2.5-multi-coreai-fp32@043e738f3a91222ec013b2d80e240e34abb46bdb` | repository root | 8 | 1,247,733,386 | `fastino/gliner2.5-multi-v1@2ca71aafb3446d9014e1c55c7ff51c9bc7209c47` |
| `embeddinggemma2-fp16-safe-s128-v70` (FP16, default) | `okayuji/embeddinggemma-2-coreai@daab6dbfbae1610a87d9f181458419e3ea5ce682` | `fp16` | 13 | 913,677,333 | `google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b` |
| `embeddinggemma2-int4-int8-ptq-s128-v70` (INT4/INT8 PTQ) | `okayuji/embeddinggemma-2-coreai@daab6dbfbae1610a87d9f181458419e3ea5ce682` | `int4-int8` | 16 | 341,880,041 | `google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b` |

### Automated checks

| Check | Result |
|---|---|
| ADOPT-01 | `IOSDemoLaunchTests` "ADOPT-01 The iOS catalog presents two models and five demos whose artifacts carry the published distribution definitions with every file, size, checksum and upstream" and `ManagedModelStoreTests` "ADOPT-01 The managed store resolves the contract of every artifact in the iOS catalog" pass |
| ADOPT-02 | `ModelLibraryAssemblyTests` "E2E-01 ADOPT-02 The app assembly connects one provider to each demo of the iOS catalog, passes the download request of every published artifact to the transfer with its catalog definition, and creates no session or provider work before the acquisition completes" and "ADOPT-02 The self-test with an acquisition target requests each catalog artifact in turn through the app assembly and fails with the route, defined bytes and inspection of every failed transfer" pass |
| Acquisition routes | "The self-test acquisition downloads a missing artifact, keeps an installed one and migrates a matching folder through the library store, with sizes equal to the definitions, the validated identity and a ready inspection" passes |
| Broken on purpose | A changed checksum in the catalog failed ADOPT-01; downloads that were not awaited one at a time failed both self-test acquisition tests; swapped route labels failed the routes test. The sources were restored afterwards |
| PR commands | Strict lint 0 violations in 382 files; format PASS (10 changed Swift files); `./Tools/Irodori/build-inputs.sh` exit 0; `swift test` 688 tests, 0 failed (the skipped suites need model bundles, fixtures or environment variables); macOS Release and signed iPhoneOS Release BUILD SUCCEEDED; native preview simulator build BUILD SUCCEEDED; `check-catalogs.py` 0 errors |

No test connects to the network: every `ArtifactTransfer` in the tests uses an injected file loader.

### Acquisition

`--model-library-selftest selftest.json --acquire all` loads no model. The transfer requests each listed file anonymously over HTTPS; the resolve URLs answer with redirects, which it follows. Mac: Apple M4 Max, macOS 27.0 (26A428), Xcode 27.0 (27A266a). iPhone: iPhone 17 Pro (iPhone18,1), iOS 27.0.1 (24A446), installed with `devicectl device install app` and launched with `devicectl device process launch --console`.

| Run | Artifact | Route | Defined bytes | Measured bytes | Seconds | State | Inspection |
|---|---|---|---:|---:|---:|---|---|
| Mac | GLiNER2.5-multi FP32 | download | 1,247,733,386 | 1,247,733,386 | 53.6 | ready | ready |
| Mac | EmbeddingGemma 2 FP16 | download | 913,677,333 | 913,677,333 | 37.6 | ready | ready |
| Mac | EmbeddingGemma 2 INT4/INT8 PTQ | download | 341,880,041 | 341,880,041 | 15.9 | ready | ready |
| iPhone | GLiNER2.5-multi FP32 | download | 1,247,733,386 | 1,247,733,386 | 38.1 | ready | ready |
| iPhone | EmbeddingGemma 2 FP16 | migration | 913,677,333 | 913,677,333 | — | ready | ready |
| iPhone | EmbeddingGemma 2 INT4/INT8 PTQ | migration | 341,880,041 | 341,880,041 | — | ready | ready |

- Mac ([report](ios-models-distribution-20261019/selftest-macos.json)): exit 0 in 109.1 s, refresh 0.0 s, managed root `~/Library/Application Support/com.coreai-samples.DemoApp/ModelLibrary` (kept, 2.5 GB). Hashing the 37 installed files afterwards with an independent script matched every size and SHA-256 of `distribution.json`, with no other file. A second run with `--model-library-selftest=selftest.json --acquire=all` ([report](ios-models-distribution-20261019/selftest-macos-rerun.json)) reported all three as `installed`, requested nothing and took 4.5 s.
- iPhone ([report](ios-models-distribution-20261019/selftest-iphone.json)): exit 0 in 42.4 s. The refresh (2.0 s) migrated both EmbeddingGemma 2 variants from the existing `Documents/embeddinggemma2` and `Documents/embeddinggemma2-int4-int8`; GLiNER2.5-multi had no legacy folder and was downloaded. The 47 entries of those Documents folders kept their sizes and modification times, and the managed root holds exactly the listed files. The report also shows no failures, the five registered demos matching the catalog, camera and network `available`, microphone and speech recognition `undetermined`, and the managed root excluded from backups.

## Device checks

Status: the iPhone self-test with the acquisition was performed on 2026-10-19 (above). The other checks are **not performed**. The distribution is published (#210), so they wait only for the user's own check on the target iPhone.

| ID | Check | Status | Restart condition |
|---|---|---|---|
| Self-test | `--model-library-selftest` on the iPhone: no failures, the five registered demos match the catalog, the capability states as granted, and the managed root exists and is excluded from backups | Performed 2026-10-19 with `--acquire all` ([Distribution adoption](#distribution-adoption-210)) | — |
| E2E-03 | Acquisition, inspection, all five demos, switching while work runs, recovery from a failed stop, deletion and relaunch, with the artifact in use and the release of resources confirmed | Not performed. Acquisition and inspection through the self-test are recorded above | User |
| E2E-04 | Legacy assets and saved data kept; variants, publication conditions, corruption, interruption, lack of space and denied permissions never turned into a successful run | Not performed | User |
| E2E-05 | MF-09 and MF-12 to MF-17 on the device: language, descriptions, search, text size and VoiceOver keep data and operability | Not performed | User |
| DL-07 | First download, cancel, retry, completion, relaunch and deletion; spoken state, size and progress; originals kept | Not performed | User |
| COMPARE-05 | Models, comparison demo, real extraction, comparison, back and reopen, without choosing the model again | Not performed | User |
| EXTRACT-05 | Models, extraction, input, run, result, back and reopen; the artifact used, spans and scores against the existing baseline | Not performed | User |
| INSPECT-06 | Text and voice inspection, confirmation and sharing; back, interruption and reopen while recording or speaking; no audio resource running afterwards | Not performed | User |
| IMAGE-06 | Models, add images, index, search, original image, back and reopen, for each adopted variant against the existing reference conditions | Not performed | User |
| CAMERA-06 | Models, words, start, change of words, rotation, back, re-entry and a camera interruption; no capture, inference or notification after leaving | Not performed | User |

### Procedure

Build and install the signed app as described in [the iPhone app](../demo-app.md#iphone-app). Run the self-test first; it loads no model.

```bash
BID=com.coreai-samples.DemoApp.ios
xcrun devicectl device process launch --console --terminate-existing --device <UDID> $BID \
  --model-library-selftest selftest.json
xcrun devicectl device copy from --device <UDID> --domain-type appDataContainer --domain-identifier $BID \
  --source Documents/selftest.json --destination ./selftest.json
```

Add `--acquire all` (or one artifact ID) to download the catalog's artifacts through the library store; the report records each route, size, time and inspection. Then launch normally and follow the child checks above with the heavy-job lock (`CLAUDE.md`), recording each run. E2E-05 covers EN and JP, opening and closing descriptions, search with clear, cancel and no results, returning with the query and position kept, Dynamic Type at the largest sizes, 44 × 44 point targets and VoiceOver names, values and order. Also watch the points the child reviews left open:

- Comparison: after background, foreground and adding a card, the content appears again for the new session (the lease and reactivation depend on it).
- Inspection: the end of the OS recognition task, audio session interruptions, and a share sheet opened from the demo while Models shows the stopping state.
- Image search: physical memory after the release; it was not measured.
- Inspection by the managed store hashes every file on each check; record how long an EmbeddingGemma 2 FP16 check takes.
- Camera search: a stop requested while a permission dialog is shown waits for the answer.
- Transfers report progress per completed file; there is no background transfer or byte-range resume.

### Record

Add one row per run. Do not record tokens, personal inputs, photos or audio, or local absolute paths; use the artifact identity (`model/artifact@revision`) and the self-test's home-relative root.

| Date | Commit | Catalog and distribution revision | Upstream revision | OS and device | Operation | Result | Notes |
|---|---|---|---|---|---|---|---|
| 2026-10-19 | PR #212 on `6499acd` | `gliner2.5-multi/gliner2.5-multi-fp32-t256-t512@043e738`, `embeddinggemma2/embeddinggemma2-fp16-safe-s128-v70@daab6db`, `embeddinggemma2/embeddinggemma2-int4-int8-ptq-s128-v70@daab6db` | `2ca71aa` (GLiNER2.5-multi), `914f7f8` (EmbeddingGemma 2) | macOS 27.0 (26A428), Apple M4 Max | `--model-library-selftest selftest.json --acquire all` | PASS | Three downloads, 109.1 s |
| 2026-10-19 | PR #212 on `6499acd` | Same | Same | macOS 27.0 (26A428), Apple M4 Max | `--model-library-selftest=selftest.json --acquire=all` | PASS | All `installed`, nothing requested, 4.5 s |
| 2026-10-19 | PR #212 on `6499acd` | Same | Same | iOS 27.0.1 (24A446), iPhone 17 Pro | `--model-library-selftest selftest.json --acquire all` | PASS | GLiNER2.5-multi downloaded; both EmbeddingGemma 2 variants migrated; legacy folders unchanged; 42.4 s |
