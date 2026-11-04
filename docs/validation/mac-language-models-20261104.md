# Mac language-model library integration (2026-11-04)

Issue #465, integration PR #467. Implementation: `8c7a012` (catalog, storage, migration) and `dd02aa8` (application and runtime lifecycle), with termination-cancellation corrections in `e79bbc3`.

| Condition | Evidence | Status |
|---|---|---|
| LLM-01 | `LanguageModelArtifactTests.publishedInventories`; the catalog pins Qwen `8e24aebe036eae577d4a3bd019ba728a73358318` and Clef `12e30be4e391310377c534524df8962359a08b0f`, matching the distribution record | CPU passed |
| LLM-02 | `LanguageModelArtifactTests.manifests` accepts both fixed manifests and rejects altered bytes even when the fixture inventory is updated | CPU passed |
| LLM-03 | `movesCompleteFolder`, `preservesRejectedSources`, `resumesCleanup`; `MigrationCleanupTests.readyDespiteCleanupFailure` covers first migration and cleanup resumption without making the verified managed copy unavailable | CPU passed |
| LLM-04 | `conversationLocations` and `conversation/legacyModelKeepsRestoredHistory` restore a saved legacy location and preserve the conversation when selecting the managed location | CPU passed |
| LLM-05 | Mac Release, strict lint, formatting, package pins and package tests passed; anonymous acquisition through the application is still pending | Partial |
| LLM-06 | Normal UI generation, restored conversation continuation, Decide and Recipes with real models, and the visible deletion controls | User acceptance pending |

The package run reported 1,192 tests with no failures across 39 runs; 64 tests were skipped (51 conditional and 13 without the optional fixture/model inputs). The macOS application passed `--agent-selftest --offline` and `--check-localization` (102 runtime keys, no missing Japanese translations). The iPhoneOS Release build passed with signing disabled; this is a build result, not an iPhone execution result. Logs are retained outside the repository.

Runtime lease tests cover a path reconstructed using `URL(filePath:)`, refusal before installation and for a different directory, independent holders and deletion after the last release. Decide and Recipe tests cover cancellation that finishes late, closing and reopening, rejection of old release acknowledgements and loading again after release. Independent reviews corrected directory-URL comparison, pending selection lifetime, cleanup failure handling and termination cancellation. Their final inspected scope had no remaining confirmed findings; model execution and user acceptance remain separate.

The original 5 SP estimate and the parent split remain unchanged. The additional lifecycle code completes the planned stop/release contract for the three newly connected demos; keeping it with their connection avoids a merge point where deletion is possible while native work still owns the model. The later removal of the old management screen and documentation migration belong to #466.

Post-review regression checks include `MacModelSelectionTests` (resetting a pending selection, rejecting old load/unload completions and distinct task identities), the monotonic model-release marker in both Features, and application self-tests for cleanup/resume order and keeping hidden demos stopped after a failed close. A CPU SwiftUI probe reproduced omission of a transient `closing` notification; the monotonic marker recovered the second load. The real-model and normal-UI checks remain pending.
