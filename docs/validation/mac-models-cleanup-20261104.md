# Mac model-management cleanup (2026-11-04)

Issue #466, PR #494, depending on #465 / PR #493. Application cleanup: `1f4f3d6`. F was squash-merged as `732b1bd`, and G was retargeted to `feature/mac-models` and revalidated. Integration remains #493 → #494 → the user-controlled PR #467.

| Condition | Evidence | Status |
|---|---|---|
| CLEAN-01 | The six legacy sources and `Demo.models` are removed. No surviving Swift or project reference names their types. Strict SwiftLint passed with seven existing exceptions removed and none added. | Passed |
| CLEAN-02 | Replacement library-selection self-tests, Mac Release, the package suite and runtime localization passed. The initial and final library-route GUI runs both passed 7/7; the final verification uses a separately named executable. | Passed |
| CLEAN-03 | Current README/docs guide users to the sidebar headings, with separate optional CLI downloads. Privacy checks and ModelReports generation/check/tests passed. The owner specified full-payload local inspection before each card push and anonymous hub inspection afterwards on 2026-11-05. Both full-payload local audits passed before the card updates, and both anonymous complete-history hub audits passed afterwards; the payload manifests match. See [the distribution record](mac-models-distribution-20261104.md#quick-start-card-update-2026-11-05). The original acceptance text and IDs remain unchanged, with this method addendum recorded in #466. | Passed |
| CLEAN-04 | The user confirms parent MAC-01–03 on main after integration. | Pending |
| Parent MAC-05 | The iOS catalog/source sets remain unchanged; ModelLibrary tests and the unsigned iPhoneOS Release build passed. | Build/CPU passed |

The package run reported 1,192 tests across 39 runs with no failures; 64 tests were skipped (51 conditional and 13 without the optional fixture/model inputs). `--agent-selftest --offline` passed, including the pinned library catalog, absence of an uninstalled menu candidate, busy-switch refusal, conversation preparation and refusal of an unmanaged runtime path. Runtime localization checked 100 keys with no missing Japanese translations; all 912 source keys in the application catalog have the configured Japanese translation where translation is required.

The baseline changed from 38 to 31 entries: six belonged to deleted legacy sources, and the direct GUI diagnostic load now uses a structured child instead of its old unstructured Task. The remaining heartbeat exception is unchanged. Size-lint observations are advisory rather than acceptance gates.

The default GUI check now validates the acquired Qwen artifact and uses the normal usage/selection path. `--model <folder>` remains an explicit direct diagnostic override and is identified as such in its result. The first step fails if the required acquired artifact is absent; a diagnostic override does not establish ordinary library acquisition or UI acceptance.

Independent review found and corrected stale current instructions and duplicate CLI/app downloads. The model-report notes now record successful library acquisition and retain pending Mac UI acceptance. Historical numerical evidence, support marks and iOS instructions were preserved. The 3 SP estimate and deletion/diagnostic/documentation scope are unchanged. The automated checks do not replace the owner’s ordinary UI or physical-device acceptance.

## Initial native GUI check

The [seven-step report](mac-models-cleanup-20261104/gui-initial-results.json) was produced at 2026-11-04T23:48:03.499117+09:00 on an Apple M5 Max / macOS 27.2, using the acquired Qwen through the default library route with no `--model` override. The application showed preparation, loaded the model, generated a reply, stopped another reply and started an empty conversation. The observed stop acknowledgement was 0.219 seconds. These are diagnostic observations, not a controlled model-performance benchmark or user acceptance.

ScreenCaptureKit was denied by TCC; the existing diagnostic used its own view's `cacheDisplay` fallback. The direct computer-use tool also denied app access, so no independent visual acceptance is claimed. The final verification copy additionally uses a distinct `CFBundleExecutable` as required by the 2026-11-05 handoff.

## Final native GUI check

After retargeting to `feature/mac-models` and merging F (`aaa989f`), the final [seven-step report](mac-models-cleanup-20261104/gui-final-results.json) again passed through the normal library route with no `--model` override. The copy used bundle ID `com.coreai.samples.validation465` and the separate executable `CoreAISamplesCheck466`; its acquired models and conversations are isolated from the owner's application. The run loaded the acquired Qwen, generated a reply, stopped another reply and cleared the test conversation. Stop acknowledgement was 1.007 seconds. Model preparation reused this validation copy's cache.

ScreenCaptureKit did not list the validation window; the existing diagnostic used the view's `cacheDisplay` fallback. No independent visual acceptance is claimed. The same named copy passed `--agent-selftest --offline` and `--check-localization` (100 runtime keys, zero missing Japanese translations). The retargeted source passed all eight repository checks, with the build's explicit DerivedData directory supplied to the dependency-pin checker.
