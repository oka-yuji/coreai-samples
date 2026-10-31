# iPhone Speech implementation validation — 2026-10-13

Scope: #95 on #94 commit `0894efabd9b793efc9db0a57b581db93966a6511`; parent #96. No Mac GPU inference was performed.

## Acceptance mapping

| Criterion | Evidence | Result |
|---|---|---|
| M1: model list lifecycle and validation | `SpeechPhoneFeatureTests.librarySessions`, `SpeechLibraryTests`, the reused bundle SHA tests and output-collision regression | CPU pass |
| M2: operation sessions and background/cancellation cleanup | Existing SpeechFeature tests; `PhoneSpeechCheckTests.stopWaits`, interruption handling and independent multiple-stop checks; SwiftUI lifecycle wiring review | CPU/source pass; physical UI behavior remains M5 |
| M3: isolated iOS app, CPU/lint/format and build compatibility | Signed SpeechPhone Release build; existing Mac and Extract iOS builds; full package CPU suite; inspected plist/signature/profile | Pass |
| M4: installation, actual generation/playback/output/time/memory on iPhone 17 Pro | Install attempted; CoreDevice error 4016; tunnel unavailable and DDI services absent | **Waiting for device connection; not passed** |
| M5: user listening and screen acceptance | [Validation limits](iphone.md#validation-limits) | **Not performed** |

The generated app's identifier is `com.coreai-samples.SpeechPhone`, minimum OS is 27.0, and both Files-sharing keys are true. Its executable and all changed sources are hashed in [the manifest](iphone-validation/manifest.json). Host `codesign --verify --strict` passed with unrestricted read access; the actual signature and embedded profile contain the increased-memory-limit entitlement. These checks do not establish device acceptance.

## Verification

Environment: Apple silicon Mac, macOS27/Xcode27/Swift6.4. Core AI model-test environment variables were removed from CPU child processes. Existing asset-dependent tests were skipped, not counted as device/model verification.

- **46 Speech CPU tests** and the full package CPU suite passed. [Full log](iphone-validation/cpu-all.log).
- **9 diagnostic/library tests** passed with Thread Sanitizer. [Log](iphone-validation/tsan.log).
- **52 independent Release CPU tests** passed after the namespace fix. [Log](iphone-validation/diagnostic-after.log).
- **8 independent UI/feature/library CPU checks** passed. These were pure logic checks, not a physical screen test. [Review](iphone-validation/ui-review.json).
- Strict SwiftLint: zero violations. Changed Swift format: 14 files passed.
- Signed SpeechPhone, Mac regression and existing iOS Extract builds succeeded. The only retained build warning is skipped AppIntents extraction for targets without that dependency. [Build summaries](iphone-validation/signed-build.log).

## Independent review and resolved defects

Two independent reviews report no unresolved confirmed or plausible findings. Their complete candidate ledgers and coverage are retained: [UI/build](iphone-validation/ui-review.json), [diagnostic/persistence/lifetime](iphone-validation/diagnostic-review.json).

- **P95-01/P2:** `INFOPLIST_KEY_UIFileSharingEnabled=YES` did not appear in the first generated app's plist. A dedicated Info.plist now explicitly supplies both sharing keys, and the rebuilt signed product was checked. The Info.plist is not copied a second time as a resource.
- **R95-1/P2:** a model named `SpeechChecks` could receive diagnostic subdirectories before its input validation. The before-run independently added a result directory below the model. The fix rejects a model marker in the output root before mkdir and rejects the reserved input name, case variants and resolved aliases. Regression checks verify the original directory contents and bytes stay unchanged.
- The UI reviewer could not verify signing trust from its sandbox. XML/DER signature and profile structure were independently inspected; the parent then verified entitlements and the signature successfully outside the sandbox. Device installation remains independently blocked by connectivity.
- One additional independent multiple-stop fake used a mutable call counter after suspension and timed out. Only its own identified CPU helper was stopped; the fixture was corrected to preserve the call number before suspension, and the complete suite then passed. This recovered test-fixture failure is retained separately from production findings.

## Device boundary and resumption

The installation command targeted the requested iPhone 17 Pro, but returned `com.apple.dt.CoreDeviceError` 4016: the required trusted connection, CoreDevice services and power assertion were unavailable. A subsequent inventory reported `tunnelState=unavailable`, `ddiServicesAvailable=false`, and last connection at 2026-10-13 03:55 UTC. Its cached OS field still held 26.6.1 and did not establish the device's current OS.

This attempt produced no app installation, model transfer, model execution, audible playback, device WAV or device memory measurement. Later device generation and WAV results are in [reference-input validation](reference-input-validation.md); current operation is described in [the setup guide](iphone.md).

## Scope and reflection

The app unit changed 796 source/configuration lines versus the estimated 510–770. The additional scope within the same approved behavior is the independently testable diagnostic environment, stop/resource coordination and output-namespace protection. Tests (342 lines), XcodeGen output (278 lines) and documents/evidence are counted separately. The runtime/app split remains appropriate: the native portability boundary is #94, while model placement, mobile UI and its diagnostic entry point share this unit's complete usage path. The original provisional 8–13 SP is retained; no time conversion is made.

This historical record distinguishes host build results from device measurements. Current behavior and measurement limits are in [the iPhone guide](iphone.md).

The local worktree has the existing development team configured in git-ignored `Config/Local.xcconfig` for iPhoneOS only; the Mac signing configuration is not changed.

## Integration check — 2026-10-16

The latest-main integration passed the complete CPU suite, strict lint, formatting, signed SpeechPhone Release build, macOS Release build and the existing CoreAISamples-iOS Release build. The generated SpeechPhone app passes codesign verification; both file-sharing keys and the increased-memory-limit entitlement are present. A fresh read of the requested iPhone reported iOS 27.0.1 with a connected tunnel and Developer Mode enabled. The earlier connection blocker is no longer the observed state. No installation, model execution, audible playback or device export was performed in this integration check; M4 and M5 remain unchecked.
