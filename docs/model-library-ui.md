# iOS model library UI

`ModelLibraryUI` consumes `Store<ModelLibraryFeature>` and a `ModelDemoRendering` provider. It provides the model sections, shared demo tiles, system search, model status/actions, descriptions, and model information. The existing app entry point is not switched by this component change.

The renderer has two distinct inputs. `preview(for:)` receives only a metadata selection and must create display-only content without model, camera, or audio resources. `content(for:)` receives the active validated session. The host calls it only when the library permits that session. Renderer content is embedded in the host's scrolling content and should not create a second vertically scrolling screen or another navigation stack.

The renderer must retain each demo's input/result state across view reconstruction. It must route operations through the execution gate and let the library runner handle stop-and-release acknowledgements. A disabled SwiftUI view is not the sole execution boundary.

The controller and real model runner are supplied by app integration. Published model definitions require verified distribution metadata; this component supplies no invented repositories or model revisions. Individual demo redesigns remain separate from this shared host. Dynamic demo text currently resolves through the host app bundle; app integration must supply and verify the corresponding translations and searchable labels in both English and Japanese. The synthetic preview uses English demo names.

## Validation

The reusable UI compiles with the iOS 27 device SDK and through the independent [native preview](../Examples/ModelLibraryPreview/README.md) with the iOS 27 simulator SDK. Its String Catalog contains English source strings and complete Japanese translations. Compiled Japanese strings are present in the generated resource bundle.

The project generation manifest now retains the existing ObjectFinder dependencies, examples, and camera usage text that were present in the Xcode project but absent from `project.yml`. Regenerating the project must not remove those existing app capabilities.

An isolated iPhone 17 / iOS 27 simulator fixture confirmed navigation while stopping, a failed stop followed by retry, and return to Models after acknowledgement. The deletion alert was checked in English and Japanese for the model name, deletion explanation, and size. Acquisition buttons exposed 44 × 44 point accessibility frames and accepted a tap 20 points from their center after the explicit label hit-region update. These checks used synthetic resources, not real models.

The package test runs discovered 539 tests: 507 passed and 32 optional model/fixture-dependent tests were skipped. Strict lint and format checks passed. Both the main macOS Release and iPhoneOS Release builds passed; the latter used unsigned device SDK compilation. The main simulator build cannot link Core AI in the installed simulator SDK, so the standalone preview supplies the simulator UI checks.

Full navigation coverage, Dynamic Type, VoiceOver, and device behavior still require the manual checks listed in the preview. HTML approval and successful compilation do not complete these checks. The current functional-screen HTML proposals are linked from the design review index.

## Review follow-up

The review found and corrected missing deletion-alert details and a preview host localization setting. It also made the preview retain inputs/results by model and demo instead of tying them to a temporary view. XcodeGen now generates an explicit preview Info.plist with mixed localizations enabled. The remaining production work is model distribution configuration, runtime validation and stop/release adapters, and connections to the approved functional screens.
