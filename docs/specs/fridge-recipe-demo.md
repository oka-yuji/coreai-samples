# Fridge recipe demo

Contract v1, agreed in the conversation on 2026-10-08; implementation refinement recorded on 2026-10-09.

The Mac demo accepts a fridge photograph, proposes inventory items, lets the person edit and confirm them, loads a recipe URL, and checks the recipe against that reviewed inventory and kitchen equipment. The user supplied five synthetic photographs. They are demo fixtures, not evidence of accuracy on real fridges.

## Scope and ownership

- Photos: the macOS 27 on-device SystemLanguageModel, explicitly selected and checked for availability and vision support. No Private Cloud Compute or remote photo API.
- Recipe retrieval: public HTTPS Kikkoman recipe pages with an eight-digit recipe ID. Every redirect stays on the allowlisted origin and path. Failed or incomplete pages never silently become sample recipes.
- Clef-Flash: interprets whether each recipe ingredient is required, optional or unclear, and whether the stated equipment conditions are met.
- Deterministic code: matches confirmed ingredient names using explicit aliases and compares quantities in matching units. Known branded ingredients have exact aliases; unrelated foods containing a similar word do not match.
- Photo suggestions are not inventory facts until confirmed. Unseen items remain unknown until the person confirms the complete inventory. Unknown ingredient names do not prove absence. Opaque containers do not prove an ingredient is present.
- Amounts from photos are suggestions for countable whole items only. Grams and liquid volume are never inferred. The person may enter the recipe's units or explicitly enter `enough` for the whole recipe. A changed recipe invalidates confirmation of `enough`, including changed content at the same URL; numeric stock remains confirmed. No portion scaling is performed.
- Repeated required quantities consume the stock once, in recipe order. Optional needs do not create shortages. Uncertain needs do not consume stock belonging to confirmed requirements and remain review items.
- Model probabilities are uncalibrated. The original recipe remains available for review. This feature does not generate recipes, substitutions, freshness or allergy judgments.

RecipeCore owns immutable data, parsing and quantity/presence rules. RecipeRuntime exposes a MainActor service, serializes resource use, and owns photo, HTTP and decision-model resources. RecipeFeature/Store owns UI state. The runner in RecipeRuntime translates effects to service calls and actions; the app injects model-path persistence. Session/revision guards reject late completions. Stop cancels the operation and waits for the service to become idle; closing the screen cancels its effects, and reopening preserves an unfinished idle barrier.

## Implementation units and acceptance

The accepted implementation request and the user's five supplied images are the authorization for this scope. No new approval is required for implementation choices inside it. PR creation, merging and publishing are outside the current work.

- [Issue 49](https://github.com/oka-yuji/coreai-samples/issues/49): recipe/photo services and deterministic checks. Estimated 750–1,050 body lines, provisional 8 SP against EX-8 v1.
- [Issue 50](https://github.com/oka-yuji/coreai-samples/issues/50): Recipes view, reducer, runner and full headless/GUI verification. Estimated 650–950 body lines, provisional 8 SP against EX-8 v1.
- [Parent 51](https://github.com/oka-yuji/coreai-samples/issues/51): tracks the full workflow. Implement/integrate 49 before 50; revert in reverse order. The owner of 50 verifies the combined path.

A combined implementation was considered. The separate service/data contract permits unit and live-model validation before a UI is added; the view and reducer stay together to preserve the same state/cancellation rules. The estimates include tests, integration, review and corrections, and are not time estimates.

The exact acceptance IDs and text are in the issues. The five scenarios are: ingredients available after explicit review; ginger missing; one egg for a three-egg recipe; neither oven nor toaster available (then toaster available); and mozzarella unconfirmed behind an opaque container.

## Refinement after the first integration run

The first version asked Clef to infer ingredient presence and supplied quantities in the same question. It classified one available egg as missing in a three-egg recipe and misread some generic stock names against branded recipe names. The raw failed run is retained in the research record. The v2 implementation keeps confirmed inventory and arithmetic in code, and asks Clef only about recipe requirements and equipment. The user-facing expected results are unchanged; no acceptance threshold was relaxed.

## State of verification

Service/Core unit and live-model tests have run. Independent review found and verified fixes for optional quantities, missing method steps, overly broad aliases, repeated stock use, uncertain demand, URLSession lifetime and cancellation ordering. The actual Store/runner/live-service headless path passed all five scenarios. The UI review additionally corrected recipe-relative `enough` confirmations and the idle barrier across closing and reopening. Controlled tests use the actual Store and runner for success, failure, cancellation, stopping and retry.

| Acceptance | Evidence and current status |
|---|---|
| RC-01, RC-03 v1 | RecipeCoreTests: parsing, complete methods, URL rules, quantities, optional and repeated demand passed. |
| RC-02, RC-04 v1 | Five real photo/live URL/model cases passed; raw suggestions and scripted inventory are separated in the research data. |
| RC-05, RC-06 v1 | RecipeRuntimeTests / RecipeServiceTests: HTTP bounds, redirect, failure and cancellation barriers passed. URLSession probes are in the service review. |
| RF-02, RF-03 v1 | RecipeFeatureTests / RecipeRunnerTests passed, including invalidation, late results, recipe changes, actual Store cancellation and a close/reopen before idle. |
| RF-04 v1 | `--recipe-e2e` passed all five cases and the toaster alternative; source/verification details are in `ui-verification.json`. |
| RF-06 v1 | Full ordinary tests, strict lint, changed-file format and Mac Release build passed. New Swift comments/CJK: zero. Existing demo source changes are limited to the new sidebar destination and headless dispatch. |
| RF-01, RF-05 v1 | Agent GUI verification passed on `64e138b` after user-authorized launch: all five photo/recipe paths, name/amount edits, direct file import and typed URL, two eggs short, appliance changes and unconfirmed mozzarella. Details: `gui-approved-verification.json`. User manual acceptance remains pending. |

No manual acceptance box has been checked on the user's behalf. Implementation and automatic verification are distinct from full acceptance, merging and release. The user steps are in [the demo guide](../recipes-demo.md).

## Reflection and next action

The original 8 + 8 SP estimates are retained. The service body was 883 lines; the UI/state/runner/headless body was about 955 lines, slightly above its initial 650–950 estimate. Extra work came from model-role separation and the confirmation/cancellation boundaries found in review. These are corrections within the original outcomes, not a relaxed acceptance contract. Future uses should keep recognized candidates, human-confirmed facts, recipe-relative claims and deterministic arithmetic separate.

The user authorized launch on 2026-10-09 and the agent completed the stated GUI flow checks without changing application code. The remaining acceptance action is the user trying the documented examples; launch approval is not a substitute for that manual acceptance. A rapid live Stop was not observed before the decision finished, so cancellation evidence remains the controlled actual Store/Runner and Service tests. Neither issue nor parent is fully accepted merely because the automatic tests pass. Broader recipe-site support, real-fridge evaluation, automatic substitutions and iPhone execution remain outside this implementation; no additional feature work is started.

## References

Recipe URLs: https://www.kikkoman.co.jp/homecook/search/recipe/00006446/ , 00056818, 00004002, 00004825 and 00001454 on the same recipe path.

Images, live results, environment and failures are documented in `../research/fridge-photo-recognition.md` and its data directory. The ordinary tests do not load a model. Live tests require RECIPE_MODEL_TEST=1, FRIDGE_PHOTOS_DIR and COREAI_DECISION_BUNDLE under the repository GPU lock.
