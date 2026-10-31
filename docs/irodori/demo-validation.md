# Speech demo validation — 2026-10-13

This unit implements [#92](https://github.com/oka-yuji/coreai-samples/issues/92) on runtime commit `45936ba8f50e9d59182dae5d5948c232ba218fee` ([#91](https://github.com/oka-yuji/coreai-samples/issues/91)). The criterion IDs and original 2026-10-13 issue bodies are unchanged.

## Acceptance mapping

| Condition | Evidence | Status |
|---|---|---|
| U1: valid model/text/seed, no duplicate starts | `SpeechFeatureTests.inputGuards`, `modelSelection`; `SpeechServiceTests.persistenceAndInput` | CPU pass |
| U2: generation results, failure, stop acknowledgement and old sessions | `stopGeneration`, `screenLifetime`, `playbackAndFailure`; runner/store lifecycle tests | CPU pass |
| U3: playback and save outcomes belong to their original operation | Four `AudioPlaybackTests`, `stopRelease`, `runnerOutcomes`, `exportSessions`, `exportTickets` | CPU pass; real player/panel behavior belongs to U6 |
| U4: real service wired into Speech and Models; tests and build | Package tests, format, lint, Release app build, project regenerated from `project.yml` | Pass |
| U5: local FP32 folder and instructions | Bundle preparation verified source and output SHA values; [manual instructions](demo.md) | Pass |
| U6: actual app/GPU/playback/save and restart | [Validation limits](demo.md#validation-limits) | **Not measured by this validation** |

The app links `SpeechFeature`, `SpeechRuntime` and the pinned native tokenizer; it does not invoke Python or the lab CLI. At this historical revision, iOS was Extract-only; the [current iPhone guide](iphone.md) describes the later speech integration. All changed source and test hashes are in [the manifest](demo-validation/manifest.json). The model folder contains the four FP32 assets (3,368,343,098 bytes), tokenizer and `irodori.json`; preparation executes no model. Three examples and the runtime's checksum resource bundle are present in the built app. The arm64 executable passes `codesign --verify --deep --strict`.

## Automated verification

Environment: Apple M4 Max, macOS 27.0 (26A428), Xcode 27.0 (27A266a), Swift 6.4. `COREAI_`, `DECIDE_`, `GLINER_` and `CLEF_` environment variables were removed from the child processes. Asset-dependent existing tests were skipped; this is not a model/GPU test result.

- Speech CPU tests: **27 passed**, comprising the prior 11 runtime/core tests and 16 feature/playback/service tests. [Log](demo-validation/speech-cpu.log).
- Full package CPU suite: **exit 0** with the expected asset-dependent skips. This ran before the final stop-notification refinement; all affected Speech tests and the sanitizer tests were rerun after that change. [Log](demo-validation/all-cpu.log).
- Audio and service tests with Thread Sanitizer: **9 passed**, no sanitizer reports. [Log](demo-validation/speech-tsan.log).
- Strict SwiftLint with the existing baseline: **0 violations**. Swift format: **14 changed Swift files passed**. [Lint log](demo-validation/swiftlint.log).
- Xcode Release build: **succeeded** after `xcodegen generate`. [Summary](demo-validation/xcode-build-summary.log). The AppIntents metadata tool reports that extraction is skipped because this app has no AppIntents dependency. The package's unchanged `ReloadTestEngine.swift` also reports an unused `withLock` result while compiling tests; neither warning is new Speech code.

Representative commands, after clearing the model environment variables:

```sh
./Tools/Irodori/build-inputs.sh
swift test --package-path CoreAISamplesKit --filter 'Speech|Irodori|AudioPlayback'
swift test --package-path CoreAISamplesKit
swift test --package-path CoreAISamplesKit --sanitize=thread --filter 'AudioPlaybackTests|SpeechServiceTests'
swiftlint lint --config .swiftlint.yml --baseline .swiftlint-baseline.json --strict --no-cache
python3 Tools/Format/check_format.py --base 45936ba8f50e9d59182dae5d5948c232ba218fee
xcodegen generate
xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples -configuration Release \
  -destination 'platform=macOS' -derivedDataPath .build/SpeechDemo build
```

The root `check.sh` was not executed because it launches the app. Its build/test stages were invoked individually. No Core AI model execution, app launch, speaker output or panel interaction was performed in this unit.

## Review and remaining work

Two independent code reviews are complete, with no unresolved confirmed or plausible findings. [UI/build review](demo-validation/ui-review.json), [playback/runtime review](demo-validation/playback-review.json).

- The UI reviewer independently ran the seven pure feature tests and structurally compared the generated project: Mac adds the four Speech sources, examples and three products; existing Mac products remain and the iOS target is unchanged.
- The playback reviewer reproduced **R92-1 (P3)** against the original Runner: cancelling a stop task still delivered its acknowledgement after cleanup. The existing Store already rejected that old task token, so the current UI was protected. The Runner now checks cancellation after awaited cleanup; `cancelledStop` keeps cleanup mandatory while suppressing the obsolete acknowledgement. The parent inspected the before/after source, regression and recorded proof.
- The independent Release harness passed **31 CPU tests**, including four additional playback/lifetime probes. Delayed old cancellation dispatch, duplicate decode-failure callbacks and Store destruction were exercised. The tested source copies match the final runtime, feature, core and Store source. [After log](demo-validation/playback-after-tests.log), [additional tests](demo-validation/independent-playback-tests.swift.txt).
- The initial independent failure log contained interleaved UTF-8 output. The reviewer retained the raw bytes and a replacement-decoded readable copy; this was a recovered log-decoding problem, separate from the two expected regression failures against the old Runner.

The source audit covers the changed Feature, runtime services, playback, views, navigation, build settings, examples and tests, plus their Store/runtime boundaries. This review did not measure real panels, audio devices, keyboard/focus, visual layout or GPU execution. The prior runtime and tokenizer review is in [runtime-review.md](runtime-review.md).

## Reflection and document alignment

The implementation follows the planned two units, #91 then #92. The UI unit adds 587 Swift source lines plus navigation/configuration and tests, within the planned 600–950 source/configuration range. The provisional 8 SP estimate is retained; no time or line-count conversion is applied. Contract details settled within the approved scope: explicit reopening of the last folder, separate export tickets for late panel callbacks, and a Mac-only Cargo pre-build phase also represented in XcodeGen's source configuration.

The [current Speech instructions](demo.md) describe the implemented paths and measurement limits. This historical record covers the source and automated checks listed above.
