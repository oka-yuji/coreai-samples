# iOS model library UI

`ModelLibraryUI` consumes `Store<ModelLibraryFeature>` and a `ModelDemoRendering` provider. It provides the model sections, shared demo tiles, system search, model status/actions, descriptions, and model information. The regular iOS app opens this view on ordinary launch through `PhoneModelLibraryScreen`.

The renderer has two distinct inputs. `preview(for:)` receives only a metadata selection and must create display-only content without model, camera, or audio resources. `content(for:)` receives the active validated session. The host calls it only when the library permits that session. Renderer content is embedded in the host's scrolling content and should not create a second vertically scrolling screen or another navigation stack.

Acquisition controls, sizes, and failures use the selected variant's installation, and Model information shows a variant picker when the model has more than one visible variant or its selection is no longer visible. A demo without a connected adapter keeps its description and shows that execution is not connected. The description host uses the same deletion confirmation as Models; after deletion it stays on the demo and keeps its description and inputs. When the runner finds that a running demo's model was deleted elsewhere, the demo stops and returns to its description with the download action, keeping its inputs and results.

The renderer must retain each demo's input/result state across view reconstruction. It must route operations through the execution gate and let the library runner handle stop-and-release acknowledgements. A disabled SwiftUI view is not the sole execution boundary.

The controller and real model runner are supplied by app integration; the iOS app uses `ModelLibraryRendering` and `ModelLibraryRunner` from `ModelLibraryRuntime` (see [Runtime](model-library-contracts.md#runtime)). Published model definitions require verified distribution metadata; this component supplies no invented repositories or model revisions. Individual demo redesigns remain separate from this shared host. Dynamic demo text currently resolves through the host app bundle; app integration must supply and verify the corresponding translations and searchable labels in both English and Japanese. The synthetic preview uses English demo names.

## Validation

The reusable UI compiles with the iOS 27 device SDK and through the independent [native preview](../Examples/ModelLibraryPreview/README.md) with the iOS 27 simulator SDK. Its String Catalog contains English source strings and complete Japanese translations. Compiled Japanese strings are present in the generated resource bundle.

The project generation manifest now retains the existing ObjectFinder dependencies, examples, and camera usage text that were present in the Xcode project but absent from `project.yml`. Regenerating the project must not remove those existing app capabilities.

An isolated iPhone 17 / iOS 27 simulator fixture confirmed navigation while stopping, a failed stop followed by retry, and return to Models after acknowledgement. The deletion alert was checked in English and Japanese for the model name, deletion explanation, and size. Acquisition buttons exposed 44 × 44 point accessibility frames and accepted a tap 20 points from their center after the explicit label hit-region update. These checks used synthetic resources, not real models.

The package test runs discovered 539 tests: 507 passed and 32 optional model/fixture-dependent tests were skipped. Strict lint and format checks passed. Both the main macOS Release and iPhoneOS Release builds passed; the latter used unsigned device SDK compilation. The main simulator build cannot link Core AI in the installed simulator SDK, so the standalone preview supplies the simulator UI checks.

Full navigation coverage, Dynamic Type, VoiceOver, and device behavior still require the manual checks listed in the preview. HTML approval and successful compilation do not complete these checks. The current functional-screen HTML proposals are linked from the design review index.

## Review follow-up

The review found and corrected missing deletion-alert details and a preview host localization setting. It also made the preview retain inputs/results by model and demo instead of tying them to a temporary view. XcodeGen now generates an explicit preview Info.plist with mixed localizations enabled. Runtime validation, transfers, and the stop/release adapters of the five demos have since been connected (#190–#197). The remaining production work is the distribution configuration (#188) and the device acceptance of #198. Redesigns from the functional-screen proposals are separate.

## Regular iOS launch

Build and run the **CoreAISamples-iOS** scheme without diagnostic arguments to open Models. `IOSDemoLaunch` selects the launch destination before constructing legacy demo stores. `IOSDemoCatalog` lists GLiNER2.5-multi (comparison, extraction, inspection) and EmbeddingGemma 2 (image search, camera search). Search aliases include the app's English and Japanese labels without rebuilding the Store when the display language changes.

The catalog has no configured model distribution. The normal app therefore displays **Description only**, hides acquisition actions, and keeps demo inputs disabled. Since #192, `PhoneModelLibraryScreen` assembles the store through `IOSModelLibraryAssembly` with the managed store, the transfer service, and `ModelLibraryRunner` instead of `ModelLibraryBrowsingRunner`. With no distribution, inspection reports every artifact as missing without reading model files, migration candidates in Documents are only listed and are neither copied nor changed, and downloads are refused before any request. All five demos have registered adapters (#193–#197), but until a distribution definition is adopted (#188) no session is created, so no adapter content or work starts. The `unconfigured` artifact contracts are internal placeholders, not validated model formats. `ModelLibraryBrowsingRunner` remains for hosts and tests that only browse.

Explicit `--image-search`, `--object-finder`, `--truck-inspection`, and `--legacy-demos` use the original demos. `--extract-eval` and `--image-search-eval` retain the evaluation routes, and `--model-library-selftest` writes the self-test report described below. Extract evaluation resolves relative gold/form/asset/output paths against Documents on iOS; macOS retains its current-directory behavior. These explicit diagnostic routes are separate from ordinary Models browsing.

`IOSDemoLaunchTests` covers destination precedence, including the self-test, catalog ownership, translated search aliases, descriptor decoding, and refusal to execute through the browsing runner. Production iPhoneOS and macOS builds and CPU regression checks accompany the startup change. Physical-device UI acceptance remains a user check after merging.

An isolated iPhone 17 / iOS 27 simulator target compiled the actual `PhoneDemoApp`, `PhoneModelLibraryScreen`, `DemoLanguage`, app strings, and shared Feature/UI, with only diagnostic/legacy view types substituted. It verified ordinary Models startup, EN/JP switching, Japanese demo search, description-only controls, and returning with the search query preserved. No model or physical-device operation was part of that check.

### Self-test

`--model-library-selftest <output.json>` checks the assembly without loading a model, writes a JSON report, and exits with status 0 when every check passes and 1 otherwise. The iPhone app resolves a relative path against Documents; the Mac app resolves it against the current directory. The report records:

- that `IOSDemoCatalog` passed catalog validation (every link names a catalog demo, and every artifact has well-formed files and root), its demo IDs, and the number of models and demos visible on iOS for release;
- each artifact's contract, distribution (`repository@revision`), upstream revision, file count, and inspection result. An artifact without a distribution must be `missing`; inspection reads no model files in that case;
- the managed root relative to the home directory, whether it exists, and whether it is excluded from backups;
- on iPhone only, the registered demos with whether they match the catalog, and the capability states. The Mac app has no demo adapters and does not read device permissions, so its report omits both and says so in `scope`.

`ModelLibraryAssemblyTests` checks the report with the real catalog and fake adapters. The run on a device is part of the [integration record](validation/ios-models-integration-20261018.md).
