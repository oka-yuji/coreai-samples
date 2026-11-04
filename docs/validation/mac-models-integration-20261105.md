# Mac Models integration verification (2026-11-05)

Parent #458 / draft PR #467 integrates A through H. F #493 was squash-merged as `732b1bd`, G #494 as `1fc3d09`, and H #499 as `e8e9452`. The final candidate combines `e8e9452` with main `806ac64`. The main update affects JEV release documentation and model-report data; the application, package sources and configuration remain identical to the verified H implementation. README conflict resolution preserves both the sidebar acquisition instructions and the latest JEV public-release information. Model tables were regenerated from records.

The [check summary](mac-models-integration-20261105/checks.json) records all eight repository checks, runtime localization, the offline agent self-test and the final [GUI result](mac-models-integration-20261105/gui-results.json). All passed. The package suite reported 1,199 tests across 39 runs, no issues and 64 optional skips. The unsigned iPhoneOS Release passed for the same H source; this is build evidence, not physical-device acceptance.

The final GUI check used a copy with bundle ID `com.coreai.samples.validation465` and executable `CoreAISamplesCheck467`, under the shared heavy-job lock. It used the acquired Qwen through the ordinary library route with no `--model` override, checked preparation/loading, generated a reply, stopped another reply and cleared the test conversation. The owner's application, data, settings and Keychain were not used for this verification. The diagnostic used its own view capture fallback when ScreenCaptureKit did not return the window; independent visual acceptance is not claimed.

The [F record](mac-language-models-20261104.md) includes real anonymous library acquisitions of Qwen's 12 files and Clef's 25 files with bytes/SHA-256, ready inspections and zero failures. The [G record](mac-models-cleanup-20261104.md) and [distribution record](mac-models-distribution-20261104.md#quick-start-card-update-2026-11-05) cover legacy removal and audited HF card updates. The [H record](mac-models-light-inspection-20261105.md) reports refresh median 7.8 → 0.002089 seconds while retaining full acquisition/import/migration/diagnostic verification.

## Owner acceptance remains pending

PR #467 remains draft. Its readiness transition and merge into main require the owner's instruction. SIDE-03, HIST-02, LIB-06, USE-05, WIRE-02, LLM-06, CLEAN-04 and LIGHT-05, together with parent MAC-01–03, remain for the owner on main. Automated GUI observations do not check all model-specific ordinary workflows. The already approved PUB-02 is not requested again. Issue #464 can close on main integration; the other child issues remain open until their manual conditions are accepted.

The final integration adds no new product scope. The verification boundary is explicit: a build is not a device check, and a structural ready result is not a new full-content guarantee. No additional implementation candidates were introduced by this final documentation merge.
