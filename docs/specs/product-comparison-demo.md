# Product specification comparison

Specification v1, 2026-10-14. The user selected “product descriptions to specification notes”, with Apple/Microsoft URLs or pasted descriptions and comparable cards. Tracking: [#101](https://github.com/oka-yuji/coreai-samples/issues/101); source/extraction [#97](https://github.com/oka-yuji/coreai-samples/issues/97), UI/integration [#98](https://github.com/oka-yuji/coreai-samples/issues/98). The acceptance conditions remain P1–P5 / U1–U6 in those issues.

## Behavior

The product comparison demo is titled **GLiNER2.5-multi** on Mac and iPhone. The Mac sidebar description identifies the product comparison workflow. **Add a product** accepts an official HTTPS URL or pasted description, and creates one card from the captured page/text. A model folder must be selected first. Cards are kept in memory, up to four, with the same attribute order: chip/processor, memory, storage, display, weight, battery life and price. The page title or supplied title identifies the card; the model's product span is a fallback.

A card represents values mentioned on a page. It is not a single purchasable configuration: a page can describe several screen sizes, processor options, amounts and prices. Multiple values remain candidates. An empty field means no candidate was found, which can also be a model miss. The app does not infer missing values, normalize currencies, compare superiority or buy anything.

Each candidate must exactly match its Unicode-scalar range in the captured text, including the UTF-8 sequence. A value opens its containing paragraph, full captured text, final URL when present, and capture time. Qualifiers and configuration context remain in that source; a short extracted span alone is not a confirmed specification. The source URL is an external link, not a continuously refreshed value.

The input is a snapshot at submission. Editing it, changing the model, stopping or leaving cancels the current operation; generation checks reject late output. A canceled model operation may finish cooperatively, so another request waits for the service to become idle. Errors retain earlier cards and the draft for retry. Removing a card only removes that card. Language selection changes interface text, not input, model labels or source content.

## Boundaries

- Source addresses: exact HTTPS hosts `apple.com`, `www.apple.com`, `support.apple.com`, `microsoft.com`, `www.microsoft.com`, `support.microsoft.com`, `learn.microsoft.com`; no credentials or nonstandard port. Validate each redirect before contact; at most four requests.
- Fetch at most 2 MiB, UTF-8 HTML/XHTML/plain text. Per-request timeout 20 s, resource timeout 30 s. No JavaScript rendering, browser cookies, login or protection bypass. HTML extraction excludes scripts, styles, comments and nested template/navigation/footer sections before selecting main/body. It is not a general browser DOM renderer.
- Readable text is limited to 12,000 Unicode scalars. Reject larger input without silently cutting it. Pages requiring scripts, refusing requests or containing no usable body direct the user to paste a product section instead.
- The built-in links point to MacBook Air specifications and **Surface Laptop 5** specifications. The latter is an older model with a usable support specification page, not a claim to be the newest Surface.
- Reuse the existing GLiNER2.5-multi asset. Eight fixed Japanese schema labels live in a JSON resource. No training, conversion or threshold tuning is part of this change.
- Each chunk starts at up to 600 scalars and shrinks until the actual schema-plus-text tokenizer fits the existing 512-token limit. Overlap is up to 64 scalars. Translate candidates back to the full source, and deduplicate only matching field/range. This does not establish general long-document extraction quality or cross-chunk entity resolution.
- Cards and input are session state. There is no database, saved comparison, price monitor or cloud inference.

## Design and ownership

`ProductCore` owns URL checks, HTML-to-text processing, chunk ranges and evidence validation. It has no model, I/O or asynchronous state. `ProductFeature` owns the draft, cards, phases, generation and commands. SwiftUI only presents that state and sends actions.

`ProductRunner` calls `ProductService` on MainActor. `LiveProductService` serializes source and extraction operations and exposes a drain for cancellation. `ProductPageClient` handles bounded HTTP requests; `ProductExtractor` owns the cached GLiNER resource and refuses overlapping inference. The generic `Store` is the task/cancellation owner. Mac and iOS share this implementation.

The existing label/inspection UI remains under **Extraction lab**. Its headless `--extract-eval` and Mac `--gui-e2e ... --extract ...` routes remain separate; their success does not validate product extraction. B3 inspection-memo findings and the stopped M1 training decision remain unchanged.

## Verification

Code snapshot: A `0153016`; B `9b8a542`, accessibility fix `ccf505a`. Xcode 27.0 / Swift 6.4.0.34.1, macOS 27.0. Evidence is in the local scratch directory `/private/tmp/gliner-product-cards-20261014/` and the private knowledge record `projects/coreai-samples/gliner-product-cards-20261014/実装と検証.md`.

| Acceptance | Evidence and current result |
|---|---|
| #97 source/evidence contracts | ProductCore 12 + ProductRuntime 6 tests passed. Hidden template and comment/script boundary reproductions fixed and independently checked. |
| U1–U3 state, snapshot, error/retry | ProductFeature 8 tests passed; real runner exercised with a stub service. Real model integration is separate below. |
| U4 build/regression | Full default SwiftPM run passed; model-dependent tests remained disabled. Mac and generic iOS Release builds passed. SwiftLint: zero violations. swift-format: 18 changed Swift files passed. Existing headless entry routing retained by source inspection. |
| U5 source acquisition | Native `--product-source-check` passed for two official pages and their pasted text. MacBook Air: 9,864 scalars; Surface Laptop 5: 8,038. Both reports have `modelRan: false`. |
| U5 inference | Deferred to post-merge validation at the user's explicit request. Source retrieval and fabricated preview cards do not substitute for model inference. |
| U6 | Agent checked two aligned cards, Japanese source context, English/Japanese switching and missing price on Mac. The final build exposes raw values and localized missing text through accessibility values. VoiceOver audio and narrow-window layout were not verified. User confirmation remains pending. iPhone hardware rendering/inference is unverified. |

The source-only and model paths are explicit:

```sh
APP=/path/to/CoreAISamples.app/Contents/MacOS/CoreAISamples
"$APP" --product-source-check /path/to/private/source-output
"$APP" --product-e2e /path/to/private/model-output --extract /path/to/gliner-bundle
open /path/to/CoreAISamples.app --args --product-preview
```

The second command reads the model and requires the process check and heavy-job lock in `CLAUDE.md`. It drives the real Store/Runner/Service through two URLs and pasted text, then checks that a rejected URL preserves prior cards and allows retry. The diagnostic output includes text hashes, sources and raw candidates. It is an integration check, not an independently labeled quality benchmark. Keep complete third-party descriptions in private/local evidence, not in distributed fixtures. `--product-preview` is explicitly fictional and performs no model inference.

## Remaining acceptance and next work

Independent source/UI reviews and correction checks are complete: four HTML/context findings and one accessibility finding were fixed. Remaining acceptance is real-model integration and user confirmation of the changed screen. Do not mark the whole feature accepted until those results are recorded. The original inspection-memo M1 run is not automatically resumed for this new topic. Preserve its data and failed S0 evidence; any later training needs a product-specific data/evaluation decision. On 2026-10-14 the user explicitly requested PR creation and merging to main before their later validation. Keep P5/U5 and U6 open; merging does not establish model quality or complete acceptance. Model publication and purchases remain outside this change.

The original estimates remain 8 SP for #97 and 8–13 SP for #98, provisional against EX-8 v1. The observed HTML/evidence review fixes are implementation rework, not a change to quality thresholds. Reusable lesson: validate the source before judging extraction, and test conditions on adjacent lines as well as exact spans.
