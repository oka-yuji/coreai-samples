# Lightweight model-library inspection (2026-11-05)

Issue #495 / PR #499 follows F #493 and G #494 on `feature/mac-models`. Implementation: `eda5a4f`.

| Condition | Evidence | Result |
|---|---|---|
| LIGHT-01 | `RecordedInspectionTests.sameSizeChange` changes model and manifest bytes without changing size: ordinary inspection stays ready, explicit full verification rejects the checksum. | Passed |
| LIGHT-02 | `untrustedLedger` and `structureChange` cover absent/corrupt/mismatched ledgers, size, missing/extra files and symlinks; existing STORE-05 and local-import rejection tests remain green. | Passed |
| LIGHT-03 | `promotionVerifiesContent`, existing import/migration tests, `MigrationCleanupTests` and `LocalImageCatalogTests.verifiesContents` retain complete verification and protect source/journal on failure. | Passed |
| LIGHT-04 | [Three before/after measurements](mac-models-light-inspection-20261105/refresh.json), eight repository checks and Mac/iOS Release builds. | Subsecond target passed; minutes-scale baseline not reproduced |
| LIGHT-05 | The owner checks ordinary library responsiveness and model use on main. | Pending |

## Measurements

The isolated validation app held Qwen (12 files) and Clef (25 files), totaling 24,131,136,176 bytes. All six runs used the shared heavy-job lock, ordinary library `.refresh` inside `--model-library-selftest ... --acquire qwen3.8-27b-coreai-4bit`, and no pending migration journal. The target was already installed; no downloads or model inference occurred. Bundle identity and data stayed isolated from the owner's app, and each version had a distinct executable name.

On Apple M5 Max / macOS 27.2, the before values were [7.8, 7.8, 7.8] seconds and the after values were [0.002163, 0.002089, 0.002049] seconds. Median: **7.8 → 0.002089 seconds**. The original diagnostic rounded to tenths; the new diagnostic records microsecond precision. The acquisition check measures dispatch through library idle. Its full initial inventory and full post-acquisition inspection are outside that interval and remain enabled. Both models were ready after full initial checks in every run, with no acquisition failures.

The OS file cache was not purged. Both versions run full inspection before measuring refresh, so these are comparable warm-cache measurements, not cold-disk bounds. The Issue's parenthetical minutes-scale estimate was not observed on this machine; it is retained in the acceptance text and not represented as measured evidence.

## Validation and review

The package suite reported 1,199 tests across 39 runs, zero issues and 64 skipped tests (51 conditional, 13 lacking optional inputs). Strict SwiftLint, swift-format against `origin/feature/mac-models`, working-tree and commit privacy, Irodori input build, Mac Release and dependency pins passed. The unsigned iPhoneOS Release build also passed.

Independent review covered Storage, all six production inspection call sites, migration cleanup, explicit diagnostics, tests and documentation. One confirmed regression in the cleanup namespace failure path was corrected and covered by `inaccessibleJournal`. A fixture selection error in the new diagnostic test was corrected to select the artifact by ID. Remaining confirmed findings: zero. Full verification leaves the installation ledger available for deletion; ordinary ready still expresses structural consistency, as documented in [the inspection contract](../model-library-ui.md#recorded-artifact-inspection). Persistent content-failure invalidation is not part of this contract.

The 2 SP estimate is retained. The implementation needs migration result handling and explicit diagnostic call sites in addition to the initially estimated inspection branch; these preserve the existing full-verification boundary in one acceptance unit. No refresh trigger, ledger format or UI control changed.
