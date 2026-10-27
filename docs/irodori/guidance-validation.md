# Irodori first-use guidance

Scope: #353 / #354, October 28, 2026, based on `98d74083a5a8e7bf90e3aa3bb088cdfc358f8e01`. Xcode 27.0 (27A266a), Swift 6 language mode, macOS/iOS 27 deployment. This change improves the existing text-only demo's preparation and operation instructions. It does not complete the separately requested model download or reference-voice recording/import and conditioning.

## Acceptance evidence

| Condition | Evidence | Result |
|---|---|---|
| GUIDE-01: Preparation, input, generation, playback/save and reference-voice scope are explained. | `LocalSpeechCatalogTests.firstUseGuidance` opens the speech destination without a model, verifies preview access without creating a session, and checks that preparation and operation instructions remain in the catalog. | Passed |
| GUIDE-02: English/Japanese preparation instructions and a guide link are available; an example can be used without changing the seed. | The preparation text, labeled import button and guide link are outside the disabled preview. `PhoneSpeechContent` offers the first example directly and retains seed 1708 inside optional voice settings. Compiled catalog lookup checks passed for the Mac app (1,692), iOS app (1,692) and iOS ModelLibraryUI bundle (118), with zero mismatches. | Host/source checks passed; physical-device layout remains unverified. |
| GUIDE-03: Regression tests, lint/format and Release builds pass. | Commands below. | Passed |

## Commands

```sh
./Tools/Irodori/build-inputs.sh
swift test --package-path CoreAISamplesKit
swiftlint lint --config .swiftlint.yml --baseline .swiftlint-baseline.json --strict --no-cache
python3 Tools/Format/check_format.py --base origin/main
python3 Tools/Localization/check-catalogs.py
xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples -configuration Release -destination 'platform=macOS' -derivedDataPath .build/IrodoriGuidance build
xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples-iOS -configuration Release -destination 'generic/platform=iOS' -derivedDataPath .build/IrodoriGuidancePhone -allowProvisioningUpdates build
xcodebuild -project CoreAISamples.xcodeproj -scheme SpeechPhone -configuration Release -destination 'generic/platform=iOS' -derivedDataPath .build/IrodoriGuidancePhone -allowProvisioningUpdates build
```

The package command exited zero and reported 946 tests across 36 test-product runs, with 45 conditional skips. No learned model was executed. Lint reported zero violations in 496 files; format passed for the five changed Swift files. All three app builds succeeded. The signed iOS products contain both file-sharing keys and the increased-memory-limit entitlement; the normal iOS app passed `codesign --verify --deep --strict`.

Local logs for this run are `/tmp/irodori353-{all-tests,focused,lint,format,mac-build,ios-build,speechphone-build,bundle-translations,ios-translations,library-translations}.log`. The Swift lookup tool was run with the actual `.xcstrings` file and corresponding app/resource bundle. An initial invocation with a directory instead of a catalog file failed, then the corrected invocations above passed. The existing source-key checker passed against the Mac output; it assumes the Mac target name and does not discover the iOS target output, so compiled iOS bundle lookups were checked directly.

## Review and remaining work

An independent source reviewer examined all eight implementation/test/translation/guide files and their callers. No confirmed defect remained. The parent reviewed the same diff, test/build results and compiled translations. The final review scope token matched the reviewed files. Changes retain the existing generation, playback, cancellation and export session contracts. Physical-device layout, tap targets and audible output were not exercised.

The initial usability issue came from instructions that started after the undocumented model-preparation step and exposed seed configuration as a first-use requirement. The revised screen places preparation beside its action and makes the seed optional. The operation guide in `iphone.md` now matches the UI. The model/runtime support record remains unchanged because these checks do not establish new model or device support.

The user's recording/file-import request remains required work. Upstream reference conditioning requires a DACVAE audio encoder, speaker conditioning, a compatible duration predictor and DiT inputs. The present four-component bundle uses no reference voice. At this check, the existing model-asset volume `/Volumes/HDD` was unavailable and the shared heavy-job lock was held by the active TeleOCR evaluation. Those conditions do not make a recorder-only UI a working voice-cloning implementation. Restore or identify the model assets, settle the reference-voice intent, then prepare and validate the necessary assets and connect both recording and file import to generation.
