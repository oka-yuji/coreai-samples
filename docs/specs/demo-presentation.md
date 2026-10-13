# Demo language and recipe inspection

The user's 2026-10-09 feedback requests a consistent EN/JP control, String Catalog localization, clearer Clef model readiness, and a small set of useful decision diagnostics. The screenshot shows Stopped with no loaded model; the previous footer did not name the required bundle in that state.

## Scope and contracts

- Mac Chat, Agent, Decide, Recipes, Extract, Models and the sidebar share one trailing toolbar language control. The preference is persisted. Selecting a language updates presentation without rebuilding Stores or re-running inference.
- English remains the code and protocol language. Japanese translations live in String Catalog resources. The previous English-only UI convention is superseded by the user's explicit request.
- The app's Localizable catalog holds controls, statuses and explanations. The Ingredients catalog holds known ingredient display names, preserving distinctions such as chicken versus chicken thigh. Unknown labels fall back to their original spelling.
- Recipe source text, documents, chat/agent responses, filenames and identifiers remain original data. Unrecognized runtime messages are identified as original details. Localization does not rewrite the recipe, prompts, options or confirmed inventory.
- A display-only ingredient update is ignored by RecipeFeature; an actual name edit uses the existing invalidation/confirmation rules. Quantity input syntax remains the existing recipe-unit format and `enough`.
- Recipes will always identify the expected Clef bundle and distinguish unloaded, loading and loaded state. Choosing a folder and explicitly retrying the remembered path use the existing effects and cancellation barrier.
- Developer details are collapsed by default. They will project actual photo timing and candidate counts, recipe counts, decision input/reused tokens, elapsed time, question options with probabilities/logits, the existing review threshold and deterministic stock/quantity results. Missing measurements stay unrecorded. Model scores remain uncalibrated.
- Optional diagnostics preserve decoding of existing report JSON. No new inference endpoint or shared task owner is introduced.

## Work and validation

[Issue 52](https://github.com/oka-yuji/coreai-samples/issues/52) owns language presentation and catalog coverage, including source-value retention and six-screen GUI checks. Its original estimate is 4,000–6,500 body lines including authored translation resources, provisional 8 SP against EX-5 v1. The common/Recipes versus other-screen split was considered; the global preference and complete call-site/catalog coverage are accepted as one atomic UI contract. Review partitions cover the language owner, editable recipe inputs, other screens and catalogs.

[Issue 53](https://github.com/oka-yuji/coreai-samples/issues/53) owns model readiness and diagnostics, depends on 52, and verifies the combined live-model flow. Original estimate 1,300–2,300 body lines including translations, provisional 5 SP against EX-5 v1. The existing result projection and model-operation state are kept together. Integrate 52 before 53 and revert in reverse order. PR creation, merging and release remain outside this work.

Acceptance IDs and original text remain in the issues. Evidence must include compiled catalog lookup in both languages, placeholder integrity, the changed reducer contract, full required repository checks, language switching with a current recipe result, and actual diagnostics versus the model's returned distribution. User manual acceptance is kept distinct from agent verification.

## Verified API choices

The installed Foundation SDK declares `Bundle.localizedString(forKey:value:table:localizations:)` available from macOS 15.4 / iOS 18.4. A local compiled Swift probe returned English, Japanese, then English in one process with explicit `Locale.Language` values. `xcstringstool compile` generated both en.lproj and ja.lproj from a catalog with sourceLanguage=en and a Japanese translation. SwiftUI's environment locale handles localized view literals; explicit bundle lookup handles dynamic presentation strings. No global language preference override or Bundle swizzling is used.

References: [Apple's String Catalog guide](https://developer.apple.com/documentation/xcode/localizing-and-varying-text-with-a-string-catalog), [SwiftUI localization](https://developer.apple.com/documentation/swiftui/preparing-views-for-localization), the installed Foundation.swiftinterface, and local HIG pages for segmented controls, toolbars and disclosure controls.

## Current state

Issue 52 implementation and automatic verification passed: catalog coverage and placeholders, 912 compiled lookups, 54 actual runtime UI keys, 10 recipe feature tests and the full ordinary suite, strict lint, format and Mac Release build. Independent review corrected general ingredient labels that became too specific, dynamic-string lookup gaps and a stale Binding snapshot that could drop a second name edit. Source hashes and review history are in `../research/demo-localization/validation.json`. Combined GUI/IME checks and issue 53 diagnostics remain in progress; user manual acceptance is not inferred from these results.
