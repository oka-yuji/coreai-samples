# Demo language and recipe inspection

The user's 2026-10-09 feedback requests a consistent EN/JP control, String Catalog localization, clearer Clef model readiness, and a small set of useful decision diagnostics. The screenshot shows Stopped with no loaded model; the previous footer did not name the required bundle in that state.

## Scope and contracts

- Mac Chat, Agent, Decide, Recipes, Extract, Models and the sidebar share one trailing toolbar language control. The preference is persisted. Selecting a language updates presentation without rebuilding Stores or re-running inference.
- English remains the code and protocol language. Japanese translations live in String Catalog resources. The previous English-only UI convention is superseded by the user's explicit request.
- The app's Localizable catalog holds controls, statuses and explanations. The Ingredients catalog holds known ingredient display names, preserving distinctions such as chicken versus chicken thigh. Unknown labels fall back to their original spelling.
- Recipe source text, documents, chat/agent responses, filenames and identifiers remain original data. Unrecognized runtime messages are identified as original details. Localization does not rewrite the recipe, prompts, options or confirmed inventory.
- A display-only ingredient update is ignored by RecipeFeature; an actual name edit uses the existing invalidation/confirmation rules. Quantity input syntax remains the existing recipe-unit format and `enough`.
- Recipes always identifies the expected Clef bundle and distinguishes unloaded, loading and loaded state. Choosing a folder and explicitly retrying the remembered path use the existing effects and cancellation barrier.
- Developer details are collapsed by default. They project actual photo timing and candidate counts, recipe counts, decision input/reused tokens, elapsed time, question options with probabilities/logits and the existing review threshold. Deterministic stock/quantity results remain in the ordinary result card. Missing measurements stay unrecorded. Model scores remain uncalibrated.
- Optional diagnostics preserve decoding of existing report JSON. No new inference endpoint or shared task owner is introduced.

## Work and validation

[Issue 52](https://github.com/oka-yuji/coreai-samples/issues/52) owns language presentation and catalog coverage, including source-value retention and six-screen GUI checks. Its original estimate is 4,000–6,500 body lines including authored translation resources, provisional 8 SP against EX-5 v1. The common/Recipes versus other-screen split was considered; the global preference and complete call-site/catalog coverage are accepted as one atomic UI contract. Review partitions cover the language owner, editable recipe inputs, other screens and catalogs.

[Issue 53](https://github.com/oka-yuji/coreai-samples/issues/53) owns model readiness and diagnostics, depends on 52, and verifies the combined live-model flow. Original estimate 1,300–2,300 body lines including translations, provisional 5 SP against EX-5 v1. The existing result projection and model-operation state are kept together. Integrate 52 before 53 and revert in reverse order. PR creation, merging and release remain outside this work.

Acceptance IDs and original text remain in the issues. Evidence must include compiled catalog lookup in both languages, placeholder integrity, the changed reducer contract, full required repository checks, language switching with a current recipe result, and actual diagnostics versus the model's returned distribution. User manual acceptance is kept distinct from agent verification.

## Verified API choices

The installed Foundation SDK declares `Bundle.localizedString(forKey:value:table:localizations:)` available from macOS 15.4 / iOS 18.4. A local compiled Swift probe returned English, Japanese, then English in one process with explicit `Locale.Language` values. `xcstringstool compile` generated both en.lproj and ja.lproj from a catalog with sourceLanguage=en and a Japanese translation. SwiftUI's environment locale handles localized view literals; explicit bundle lookup handles dynamic presentation strings. No global language preference override or Bundle swizzling is used.

References: [Apple's String Catalog guide](https://developer.apple.com/documentation/xcode/localizing-and-varying-text-with-a-string-catalog), [SwiftUI localization](https://developer.apple.com/documentation/swiftui/preparing-views-for-localization), the installed Foundation.swiftinterface, and local HIG pages for segmented controls, toolbars and disclosure controls.

## Current state

Issue 52 is saved through `ca8a947`. Its initial automatic evidence remains in `../research/demo-localization/validation.json`. Independent review corrected general ingredient labels that became too specific, dynamic-string lookup gaps and a stale Binding snapshot that could drop a second name edit. GUI verification then found that the navigation title could retain the initial language while the body switched; explicit selected-locale lookup in a fixed ViewModifier corrected this without recreating Stores. The reasoning control now uses the existing "Reasoning effort" key, and the native Extract editor updates its accessibility label while preserving its marked-text guard.

Issue 53 adds an optional `RecipeDiagnostics` projection of the successful `DecisionResult`. The service validates question IDs and copies raw choice order, selected IDs, probabilities, logits and total rendered input tokens. Existing report JSON decodes with diagnostics absent. The check duration retains its previous boundary (after queue/model validation through document preparation and result mapping); internal runtime preparation is included, but the explicit bundle-opening action is excluded. Cross-check reuse remains zero for the first check. The unchanged `0.6` review boundary is shared by the rule and its display. Tests cover raw projection including a tied distribution, legacy JSON, roundtrip and independent ingredient/equipment threshold boundaries.

The ordinary suite, strict lint, format and normal Mac Release build passed. Combined resources pass 994 compiled lookups and 60 dynamic UI keys. A separate application ID was used to verify all six screens in English/Japanese while preserving the user's running original app. The selected language survived relaunch; a focused, confirmed egg entry kept its amount and confirmation across EN/JP, while a real edit cleared confirmation. A live fetched omelette recipe and its URL remained intact across language switching. The inspector's empty/count/threshold states render in both languages. Detailed scope, source hashes and unresolved checks are in [the combined verification record](../research/demo-presentation/validation.json) and [GUI record](../research/demo-presentation/gui-verification.json).

On 2026-10-09 at 14:22:47–14:23:37 UTC, the Release app built from code `8f7fcf8` (SHA-256 `430250eae1632b45f4aef8625bd2d19dcaded896ba66e7ce9c552656124cd4d6`) completed both functional paths with the HF copy at revision `58e654c86485b29a47fef820226977742ff6868a`. On M4 Max 128 GB, macOS 27.0 (26A428), Xcode 27.0 (27A266a), the five recipe cases and changed-toaster case retained actual diagnostics: ginger missing, two eggs short, unavailable equipment changing to available with 403 reused tokens, and unconfirmed mozzarella remaining uncertain. The maintenance example used 2,645 input tokens for five questions and 17 options; answers matched the reference and maximum probability/logit differences were zero. This satisfies INSPECT-04 for the unchanged code and expectations. The [sanitized execution record](../research/demo-presentation/hf-download-20261009/summary.json) identifies the source, bundle and canonical private lab record (`CoreLLM` commit `f8a9a492`). This is a single functional verification, not a performance or accuracy benchmark.

The earlier app-close approval refusal and GPU wait are retained as history. They are not the current blocker: the automated run completed under the shared lock and released it. Loaded/loading/retry/cancellation GUI, retention of a loaded photo/report while switching languages, narrow-window and marked-text checks remain separate. User manual checkboxes have not been marked by this run.

## Reflection and next work

The original 8 SP and 5 SP estimates are retained. Rework came from dynamic localization call sites, preserving the meaning of general ingredient names, a stale editable Binding snapshot, and navigation-title behavior found only in the GUI. Compiler catalog coverage alone did not cover these contracts. The successful fixes and reproduction boundaries are recorded in the project evidence and the isolated `coreai-guide` branch (`efff1ba`, `c4ce9a2`).

The same Store/Runner/Service five-example scenario and changed-toaster result now have the recorded downloaded-copy execution above. Remaining GUI work is EN/JP retention with a loaded photo/report, model readiness transitions, narrow-window use and marked-text input. The attempted resize remains unverified. The manual checkboxes in issues 52 and 53 remain user-owned. This documentation follow-up changes no source, prompt, threshold or expectation.

## Historical verification boundary

The following paragraph described the earlier verification boundary before the downloaded-copy execution. It is retained as history and does not describe the current start status:

> Live-model diagnostics, loaded/loading/retry/cancellation GUI and retention of an actual photo/report are not claimed as verified on this version. Automatic approval review rejected quitting the user's original app because it contains unsaved inventory/photo state. The user was asked whether it may close with the recorded entries restored; no response has been received. No simultaneous model/GPU job was started. User manual acceptance remains pending; the feature cycle is not complete, merged or released.
