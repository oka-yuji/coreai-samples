# Fridge photograph and recipe integration

Measured 2026-10-08–09 on M4 Max 128 GB, macOS 27.0 (26A428), Xcode 27 / Swift 6.4. Five user-provided synthetic photographs, matching the five selected recipe URLs. Original hashes are in `fridge-photo-recognition/images.json`.

## Photo recognition

Vision revision 2 was tested with the whole photo, objectness-saliency region and a fixed 3×3 grid (11–13 regions per photo). It found several vegetables but missed the eggs in the first photo's top-15 regional results, while cheese and mushrooms in the gratin photo scored at most 0.122 and 0.112. It was not selected as the inventory reader.

SystemLanguageModel.default reported available with vision capability. Attachment(imageURL:) with guided generation ran locally. The first prompt took 1.34–2.15 seconds per image and distinguished one egg from three, but omitted some visible meat. A revised instruction explicitly separated visibility through clear packaging from uncertainty about the species. It took 1.17–2.18 seconds on these five images. Counts and food types are still suggestions: the herb in the first photo is not treated as confirmed mitsuba, and an opaque white container does not become mozzarella. The production reader uses an oriented thumbnail and the same explicit on-device model; production evidence is captured by the integration path, not substituted by these exploratory results.

## Recipe interpretation and arithmetic

The first service integration run failed the one-egg case: the model combined quantity and presence and returned missing. It also reported false missing/optional labels for some branded names. That result is preserved in `integration-v1-failed.json`.

The v2 path asks Clef for required/optional/uncertain and equipment, then matches confirmed inventory and computes quantities in code. All five target outcomes passed, including two eggs short and the toaster alternative. Initial check calls took about 2.82–4.63 seconds in this run; this is a five-case functional check, not a latency benchmark. The reviewed inventories in the test are explicitly scripted test inputs, not claims about all food in each photograph. Their explicit `enough` values do not validate numeric parsing; fractions, unknown units, optional quantities, repeated demand and uncertain demand have separate unit/probe coverage.

## Review and limits

Independent review reproduced and checked corrections for optional stock causing false shortage, partially discarded method steps, soy-sauce-flavored dressing treated as soy sauce, repeated rows spending the same stock twice, and uncertain demand consuming mandatory stock. The quantity allocation probe covered 250 combinations. A delegate-based URLSession retained its delegate without explicit invalidation; the client now invalidates it in deinit. Real URLSession/URLProtocol probes showed cancellation and oversize failure ended the transfer. Model publication/name retrieval now precedes the final cancellation check and shared-state update.

The full-worktree SwiftLint baseline initially missed unchanged entries because diagnostics used `/tmp` while the worktree was rooted at `/private/tmp`. The unchanged dedicated main clone passed its ordinary strict lint. A temporary baseline with only paths expanded to the diagnostic prefix passed strict lint for the worktree; the repository baseline was not changed.

Source pages were fetched with HTTP 200 and one Recipe JSON-LD object each. The five saved pages were independently checked for complete ingredients, instructions and servings. Full source-page prose is not copied into the demo's source fixtures. Network errors remain errors. Generic recipes outside the supported site, real-fridge accuracy, quantities hidden by packaging, ingredient substitutions and iPhone execution are outside this result.

## Reproduction

Acquire CoreLLM/.locks/heavy-job and ensure no other model process is running. Use the existing clef-flash-9b-decide-v1 bundle.

```sh
RECIPE_MODEL_TEST=1 FRIDGE_PHOTOS_DIR=/path/to/five-images COREAI_DECISION_BUNDLE=/path/to/bundle RECIPE_METRICS_DIR=/tmp/recipe-evidence swift test --package-path CoreAISamplesKit -c release --filter RecipeIntegrationTests
```

Ordinary `swift test --package-path CoreAISamplesKit` skips this live integration test. The Recipes UI and its same-Store headless check are documented with Issue 50 when implemented.
