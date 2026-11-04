# Mac model-management cleanup (2026-11-04)

Issue #466, PR #494, depending on #465 / PR #493. Application cleanup: `1f4f3d6`. The implementation is staged on the dependency branch while native acquisition is awaiting exclusive use of the machine. Integration remains #493 → #494 → the user-controlled PR #467.

| Condition | Evidence | Status |
|---|---|---|
| CLEAN-01 | The six legacy sources and `Demo.models` are removed. No surviving Swift or project reference names their types. Strict SwiftLint passed with seven existing exceptions removed and none added. | Passed |
| CLEAN-02 | Replacement library-selection self-tests, Mac Release, the package suite and runtime localization passed. The full native `--gui-e2e` run is pending. | Partial |
| CLEAN-03 | Current README/docs guide users to the sidebar headings, with separate optional CLI downloads. Privacy checks and ModelReports generation/check/tests passed. Two HF card drafts are prepared, but their full payload/history/gate audits and remote updates are pending. | Partial |
| CLEAN-04 | The user confirms parent MAC-01–03 on main after integration. | Pending |
| Parent MAC-05 | The iOS catalog/source sets remain unchanged; ModelLibrary tests and the unsigned iPhoneOS Release build passed. | Build/CPU passed |

The package run reported 1,192 tests across 39 runs with no failures; 64 tests were skipped (51 conditional and 13 without the optional fixture/model inputs). `--agent-selftest --offline` passed, including the pinned library catalog, absence of an uninstalled menu candidate, busy-switch refusal, conversation preparation and refusal of an unmanaged runtime path. Runtime localization checked 100 keys with no missing Japanese translations; all 912 source keys in the application catalog have the configured Japanese translation where translation is required.

The baseline changed from 38 to 31 entries: six belonged to deleted legacy sources, and the direct GUI diagnostic load now uses a structured child instead of its old unstructured Task. The remaining heartbeat exception is unchanged. Size-lint observations are advisory rather than acceptance gates.

The default GUI check now validates the acquired Qwen artifact and uses the normal usage/selection path. `--model <folder>` remains an explicit direct diagnostic override and is identified as such in its result. The first step fails if the required acquired artifact is absent; a diagnostic override does not establish ordinary library acquisition or UI acceptance.

Independent review found and corrected stale current instructions and duplicate CLI/app downloads. The model-report notes now explicitly retain pending Mac acquisition and UI checks. Historical numerical evidence, support marks and iOS instructions were preserved. The 3 SP estimate and deletion/diagnostic/documentation scope are unchanged. This record is not an acceptance of model generation, physical-device behavior or HF publication.
