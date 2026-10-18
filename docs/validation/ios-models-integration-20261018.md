# iOS Models integration

2026-10-18 JST. Issue #198 (parent #199), PR #209. This record checks the ordinary `CoreAISamples-iOS` assembly, with the real runner, the catalog and the five demo providers, from acquisition through every demo. Automated results and device checks are kept apart. A build or a fake-based test does not stand in for a device check, and nothing below that is marked pending has been performed.

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
| Format | `python3 Tools/Format/check_format.py --base origin/main` | PASS (9 changed Swift files) |
| Irodori inputs | `./Tools/Irodori/build-inputs.sh` | exit 0 |
| CPU tests | `swift test --package-path CoreAISamplesKit` | 681 tests: 649 passed, 32 skipped, 0 failed. The skipped tests need model bundles, fixtures or environment variables (`CLAUDE.md`, heavy jobs) |
| macOS Release | `xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples -configuration Release -destination 'platform=macOS' build` | BUILD SUCCEEDED |
| iPhoneOS Release | `xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples-iOS -configuration Release -destination 'generic/platform=iOS' build` | BUILD SUCCEEDED, signed with the local development team |
| Native preview | `xcodegen generate --spec Examples/ModelLibraryPreview/project.yml`, then a simulator build of `ModelLibraryPreview` | BUILD SUCCEEDED (build only; the preview was not driven) |
| String Catalogs | `python3 Tools/Localization/check-catalogs.py` | 0 errors |

E2E-01 is covered by `ModelLibraryAssemblyTests` in `ModelLibraryRuntimeTests`, which assembles `ModelLibraryAssembly` from the real `IOSDemoCatalog` with five fake providers that carry its demo IDs:

- "E2E-01 The app assembly connects one provider to each demo of the iOS catalog, and browsing, refreshing and download requests for its unconfigured artifacts create no session, call no provider and leave legacy folders unchanged"
- "E2E-01 Direct operations of the app assembly with another owner's ticket, an artifact that was never acquired, or a session that is stopping or stopped take no lease and start no provider work"
- "The model library self-test reports the iOS catalog, its five providers, the capability states and missing unconfigured artifacts, fails when a catalog demo has no provider, and leaves the iOS-only checks out without a registry"

Each test failed when its behavior was broken on purpose: one adapter left unregistered, an empty connected-demo set, a lease kept for an identity that is not installed, a self-test that ignores a registry mismatch, and the self-test destination checked before the evaluations (`IOSDemoLaunchTests`). The sources were restored afterwards.

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

## Device checks

Status: **not performed**. The restart conditions are:

- **Distribution published**: #188 adopted, the catalog updated with the adopted repositories and revisions in a separate change, and the files downloadable.
- **User**: the user's own check on the target iPhone.

| ID | Check | Status | Restart condition |
|---|---|---|---|
| Self-test | `--model-library-selftest` on the iPhone: no failures, the five registered demos match the catalog, the capability states as granted, and the managed root exists and is excluded from backups | Not performed | User (possible now) |
| E2E-03 | Acquisition, inspection, all five demos, switching while work runs, recovery from a failed stop, deletion and relaunch, with the artifact in use and the release of resources confirmed | Not performed | Distribution published, user |
| E2E-04 | Legacy assets and saved data kept; variants, publication conditions, corruption, interruption, lack of space and denied permissions never turned into a successful run | Not performed | Distribution published, user |
| E2E-05 | MF-09 and MF-12 to MF-17 on the device: language, descriptions, search, text size and VoiceOver keep data and operability | Not performed | User. The description-only screens can be checked now; inputs and results need a published distribution |
| DL-07 | First download, cancel, retry, completion, relaunch and deletion; spoken state, size and progress; originals kept | Not performed | Distribution published, user |
| COMPARE-05 | Models, comparison demo, real extraction, comparison, back and reopen, without choosing the model again | Not performed | Distribution published, user |
| EXTRACT-05 | Models, extraction, input, run, result, back and reopen; the artifact used, spans and scores against the existing baseline | Not performed | Distribution published, user |
| INSPECT-06 | Text and voice inspection, confirmation and sharing; back, interruption and reopen while recording or speaking; no audio resource running afterwards | Not performed | Distribution published, user |
| IMAGE-06 | Models, add images, index, search, original image, back and reopen, for each adopted variant against the existing reference conditions | Not performed | Distribution published, user |
| CAMERA-06 | Models, words, start, change of words, rotation, back, re-entry and a camera interruption; no capture, inference or notification after leaving | Not performed | Distribution published, user |

### Procedure

Build and install the signed app as described in [the iPhone app](../demo-app.md#iphone-app). Run the self-test first; it loads no model.

```bash
BID=com.coreai-samples.DemoApp.ios
xcrun devicectl device process launch --console --terminate-existing --device <UDID> $BID \
  --model-library-selftest selftest.json
xcrun devicectl device copy from --device <UDID> --domain-type appDataContainer --domain-identifier $BID \
  --source Documents/selftest.json --destination ./selftest.json
```

Then launch normally. Before a distribution is published, the description-only parts of E2E-05 apply: EN and JP, opening and closing descriptions, search with clear, cancel and no results, returning with the query and position kept, Dynamic Type at the largest sizes, 44 × 44 point targets and VoiceOver names, values and order. After a distribution is published, follow the child checks above with the heavy-job lock (`CLAUDE.md`) and record each run. Also watch the points the child reviews left open:

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
| | | | | | | | |
