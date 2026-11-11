# Chronos-2 Models verification

Issue [#544](https://github.com/oka-yuji/coreai-samples/issues/544), 2026-11-12 (JST). [Structured scope](results.json).

The trusted nine-file FP32 bundle imports through the existing Models store, passes full inspection, becomes ready, and respects leases. Five rejected replacement cases preserve the prior installation and input folder. The original nine source hashes match before and after the opt-in test. The test uses a private temporary managed directory and does not run inference.

The normal Mac application was also opened through its usual window and sidebar. Chronos-2, its Japanese category and guidance, and the import action were visible. Selecting the pinned source folder produced the imported indicator and reimport/delete controls. The normal managed copy and the unchanged source were independently checked against all nine pinned hashes after the app exited. The managed Chronos copy was retained for subsequent demo work. No prediction adapter is connected in this unit.

## Evidence boundaries

- Storage test: `COREAI_CHRONOS2_IMPORT_BUNDLE=/path/to/bundle swift test --package-path CoreAISamplesKit --filter ChronosImportTests.knownBundle`, under the shared heavy-job lock. The default suite skips it explicitly.
- CPU synthetic installation tests exercise bookkeeping and rejection paths; they do not establish that synthetic model payloads are valid. Full inspection rejects those payloads.
- The Mac app's `--check-localization` checks actual `DemoModel`/`Demo` mappings against the catalog, Chronos's preview route, and the main app bundle's English/Japanese category presentation. All checks passed, with 102 runtime UI keys and no missing Japanese translations.
- All nine String Catalogs passed: 1336 keys, English source and Japanese translation, no missing translations or placeholder mismatches. Existing entries in the main catalog were unchanged; nine keys were added.
- Mac Release and iOS Release builds passed. The final iOS artifact passed `codesign --verify --deep --strict`; its actual Info.plist enables file sharing and in-place document opening. No iPhone installation or inference was performed.
- The first iOS build was unsigned because the isolated worktree lacked the existing ignored `Config/Local.xcconfig`. That build is retained as an unsigned result, not a signing pass. The existing configuration was copied unchanged, kept ignored, and a separate signed build was verified.
- Host: Apple M5 Max, macOS27.2.1 (26B131), Xcode27.2 (27B52), Swift6.4.2, SDK27.2. This is functionality validation, not a latency/memory benchmark; power and thermal conditions were not sampled.

## Integration findings

The first SF candidate registered the catalog but did not reach the Mac sidebar, which uses a separate `DemoModel.allCases` list. The fixed route now exposes the model's existing acquisition controls and explicit preview. A headless check compares the actual sidebar mappings, so catalog count alone is no longer the evidence.

String Catalog completeness also missed that classification was rendered verbatim. The shared UI presentation helper now resolves only the category through the main app bundle and the explicit locale. Architecture, model identity, variant and catalog metadata remain unchanged. CPU tests exercise locale switching, fallback and formatting; the actual main-bundle check and Japanese Mac preview confirm the presentation boundary.

Forecast execution, the NYC map, API selection and saved predictions remain in the following Issues. These import/build checks do not establish iPhone or prediction-demo support or replace user acceptance.
