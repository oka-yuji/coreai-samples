# iPhone Duo toolbar preparation

Issue [#532](https://github.com/oka-yuji/coreai-samples/issues/532), based on `9011899`. This is the first implementation unit of [#536](https://github.com/oka-yuji/coreai-samples/issues/536).

## Implemented scope

The normal iOS Models entry uses a named globe menu for the existing language preference and exposes the current EN/JP choice as its accessibility value. Mac and legacy diagnostic entries retain their segmented picker. About and model information expose both a name and a symbol, without custom toolbar sizing. Completion and cancellation controls use the system button roles with the existing localized labels and explicit symbols, preserving the app's selected presentation language. The reference-image confirmation retains **Use this crop** and adds its confirmation symbol. The two sheets shared with Mac keep their previous Mac toolbar branches.

This unit preserves the existing navigation, model ownership, demo operations, language preference, input and results. Two-column navigation, camera rotation, and fold-specific content arrangements belong to the subsequent Issues.

## Environment and automated evidence

- Xcode 27.2 (`27B52`), Apple Swift 6.4.2, device SDK iOS 27.2, deployment target iOS 27.0.
- The user updated the selected Xcode. The implementation did not change `xcode-select`.
- The checked iOS 27.2 Simulator SDK does not contain `CoreAI.framework`; the separate Models preview is the simulator entry for later layout checks.
- Repository privacy checks, strict SwiftLint, strict changed-file formatting, and all nine String Catalogs passed.
- The Rust input adapter, default `CoreAISamplesKit` tests, macOS Release build, and unsigned iOS Release build passed. Optional model/fixture tests were not enabled. No model inference or physical-device acceptance was performed.
- Dependency selections matched the committed pins. The first pin check could not locate the custom DerivedData folder; passing that build folder explicitly completed the check without changing dependencies.

Commands use the repository's required checks. The isolated builds additionally select their own output directories:

```sh
python3 -I Tools/Privacy/check-local-paths.py
python3 -I Tools/Privacy/check-local-paths.py --commits origin/main..HEAD
swiftlint lint --config .swiftlint.yml --baseline .swiftlint-baseline.json --strict --no-cache
python3 Tools/Format/check_format.py --base origin/main
python3 Tools/Localization/check-catalogs.py
./Tools/Irodori/build-inputs.sh
swift test --package-path CoreAISamplesKit --force-resolved-versions
xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples -configuration Release -destination 'platform=macOS' -derivedDataPath .local/build/mac -onlyUsePackageVersionsFromResolvedFile build
python3 -I Tools/Packages/check-pins.py .local/build/mac
xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples-iOS -configuration Release -destination 'generic/platform=iOS' -derivedDataPath .local/build/ios -onlyUsePackageVersionsFromResolvedFile CODE_SIGNING_ALLOWED=NO build
```

## Acceptance boundary

| Condition | Evidence and remaining work |
|---|---|
| BAR-01 | Repository checks and String Catalog validation passed with the selected Xcode 27.2. |
| BAR-02 | The unsigned device Release build passed with Xcode 27.2 and the iOS 27.2 SDK. |
| BAR-03 | Pending owner verification of edge-to-edge display, safe areas and camera/status regions on Duo. |
| BAR-04 | Pending owner verification of vertical placement and named overflow actions on the outer display. |
| BAR-05 | Pending owner verification of language, information, sheet actions, crop confirmation and search in all requested poses. |
| BAR-06 | Pending VoiceOver verification on a physical Duo. |
| BAR-07 | Mac branches are preserved and the Mac build passed; owner visual and interaction verification remains pending. |

The manual conditions remain unchecked. This record does not mark Duo device support or the parent Issue accepted.
