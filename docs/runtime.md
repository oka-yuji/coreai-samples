# Language-generation runtime and Swift integration

[Home](../README.md) · [日本語](runtime.ja.md) · [Verification](verification.md)

Developer guide for the runtime used by Chat, Work and `coreai-chat`. For Qwen3.8 27B setup, measurements and app-specific limitations, see the [model guide](qwen3.8-27b.md). Run commands from the repository root unless a different directory is shown.

## Convert your own bundle

Chat and Work also run other language-generation bundles made by Apple's converter in [apple/coreai-models](https://github.com/apple/coreai-models). Their runtime works as a Foundation Models `LanguageModel`, so an app talks to it through an ordinary `LanguageModelSession`.

**1. Make a bundle** with Apple's converter. Setup (it uses `uv`) and the supported models are in [its models README](https://github.com/apple/coreai-models/blob/1.0.0/models/README.md). This is the command used here for Qwen3-0.6B:

```bash
git clone https://github.com/apple/coreai-models.git && cd coreai-models && git checkout 1.0.0
uv run coreai.llm.export Qwen/Qwen3-0.6B --output-dir ~/coreai-bundles
```

This writes `~/coreai-bundles/qwen3_0_6b_4bit_dynamic`. The runtime has been run with that bundle, with Qwen3.5-0.8B and Qwen3.5-9B, and with Qwen3.8-27B in 4-bit and 8-bit. Those Qwen3.5 and Qwen3.8 exports used the converter revision from [pull request #321](https://github.com/apple/coreai-models/pull/321), rather than the 1.0.0 tag.

**2. Chat** from the `CoreAISamplesKit` folder:

```bash
cd CoreAISamplesKit
swift build -c release
.build/release/coreai-chat --model ~/coreai-bundles/qwen3_0_6b_4bit_dynamic --system "You are a concise assistant." \
  --prompt "My name is Taro. Remember it." --prompt "What is the capital of France? Answer with one word." \
  --prompt "What is my name?" --max-tokens 1024 --stats
```

Each `--prompt` is one turn of the same conversation. When the model thinks, the thinking prints under `[reasoning]` and the answer under `[response]`. `--stats` writes JSON lines to stderr: one for the load (`cached`, `specialize_s`, `load_s`) and one per turn, where `cached_tokens` is the part of the prompt reused from earlier turns.

**3. Run the demo app.** Install Rust/Cargo for the speech input adapter (Xcode builds it automatically). Open `CoreAISamples.xcodeproj` in Xcode, choose the `CoreAISamples` scheme and `My Mac`, and press Run. Xcode resolves the package dependencies the first time. The `CoreAISamples` scheme runs a Release build, and `CoreAISamples-Debug` runs a Debug build for debugging. In the app, **Models** opens the bundle folder from step 1. The separate [Irodori screen](irodori/demo.md) accepts the local Irodori FP32 folder for text-to-speech; its app/GPU/audio acceptance is pending. An [iPhone speech app](irodori/iphone.md) also builds for iOS 27; device execution remains pending.

## Use it from Swift

```swift
import CoreAIBackend
import Foundation
import FoundationModels

let model = try await LocalLanguageModel(bundle: URL(filePath: "/path/to/qwen3_0_6b_4bit_dynamic"))
let session = LanguageModelSession(model: model, instructions: "You are a concise assistant.")
for try await snapshot in session.streamResponse(
    to: "What is the capital of France? Answer with one word.",
    options: GenerationOptions(maximumResponseTokens: 1024)
) {
    print(snapshot.content)
}
```

This is the core of `CoreAISamplesKit/Sources/coreai-chat/ChatCommand.swift`, cut down to the calls that matter; each snapshot contains the whole answer so far. The CLI sends one turn per `--prompt` to the same session and prints only the new part. The thinking arrives as `.reasoning` entries in `session.transcript`, the token counts in `session.usage`, and the stop reason in the response metadata (`stopReason`). `LocalLanguageModel(bundle:contextLimit:overflow:)` also takes a context limit and an `OverflowPolicy` (below). The demo app and `Tools/Verify` use `CoreAIBackend` from the `CoreAISamplesKit` package in this repository as a local path dependency. Adding this repository's URL in Xcode's Add Package Dependencies is not supported, because the package lives in a folder of the repository. To use it in your own app, clone the repository, add the `CoreAISamplesKit` folder as a local package (File > Add Package Dependencies > Add Local) and link `CoreAIBackend`; updates come through the clone.

## What the runtime does

- **Keeps input text from inserting control tokens.** Literal strings listed in the bundle's `added_tokens` are removed from user prompts (including RAG sources) and tool results before rendering. Removal also handles nested markers and markers formed by joining the remaining text. System instructions and model-generated history are unchanged. Text without those strings is preserved.

- **Reuses what it has already read.** Each request is compared with the tokens the engine already holds, and only the part after the first difference is read again (in hybrid models, from the nearest saved point; next item): across the turns of a conversation and across the steps of a tool loop. The reused count is reported as `cachedTokenCount` in the session's usage. Sessions created from the same `LocalLanguageModel` share one engine, and their requests run one at a time.
- **Returns to a saved point in hybrid models.** Bundles with fixed-size recurrent state (Qwen3.5 and Qwen3.8) cannot be rewound by trimming the cache, so the runtime saves that state where the user's turn ends, where generation starts and where two prompts part, and restores the nearest point instead of reading from the start (`Engine.checkpoint()`, `Engine.restore(_:)`).
- **Separates thinking from the answer.** Text between `<think>` and `</think>` streams as `.reasoning` entries, the rest as the response. `ContextOptions.reasoningLevel` reaches the chat template as `reasoning_effort` (`.custom("off")` sets `enable_thinking` to false); whether it has an effect depends on the template.
- **Calls tools.** Tool definitions go into the chat template, and calls are parsed while streaming from `<tool_call>` blocks in either form, JSON (`{"name": ..., "arguments": ...}`) or `<function=...>` with `<parameter=...>`. A malformed block stays in the answer as text. With `toolCallingMode` `.disallowed` the tools are left out of the prompt. Foundation Models runs the tools and asks the model again.
- **Handles a full context in one of two ways** (`OverflowPolicy`). `.error`, the default, throws `LanguageModelError.contextSizeExceeded` only when the prompt itself no longer fits; otherwise the reply gets what is left. `.dropOldToolOutputs` first replaces the oldest tool results with `[truncated]` (the newest two groups are kept) and then halves the newest ones, to leave room for the reply; if that is not enough, it falls back to `.error`.
- **Samples as asked.** With no sampling options it decodes greedily. The `GenerationOptions` sampling modes `.random(top:seed:)` and `.random(probabilityThreshold:seed:)` and its `temperature` are honoured, and the same seed gives the same text.
- **Stops and cancels cleanly.** Generation ends at the end-of-sequence token or `<|endoftext|>` (the stop token is not fed back to the model) or at `maximumResponseTokens`, and the response metadata says which (`stopReason`: `stopToken` or `maxTokens`). A reply that runs out of tokens while still thinking ends with an empty answer, not an error. After a cancelled request, the next one on the session works.

The runtime reloads after ten completed read requests at the next new conversation. A request counts if prefill starts, including a failed or cancelled prefill; reuse without a read does not count. Agent tool steps and later turns remain in the same conversation. A configured process-footprint limit can also reload before the next read within a conversation.

Reloading releases the engine and saved states, preserves CPU history, and reads the complete prompt into a new engine. `sharedTokens` retains the logical shared prefix, while cached input is zero and prefill covers the complete prompt. Only a request which loads a replacement records `reloadSeconds`. Reloading adds a model load; interrupting a conversation also adds a full-prompt prefill. In the calibration run described below, periodic reloads took 4.19–4.23 seconds for 27B and 0.145–0.153 seconds for 0.8B. Forced reloads within a 15,797-token 27B conversation took 92.23–92.89 seconds to the first response entry, including the reload and full prefill. Long histories therefore cost more to resume.

## Supported requests

The executor validates requests before acquiring or changing the inference state. The table covers the public request options in the macOS 27.0 SDK. The current request controls generation; settings stored on earlier prompts describe those earlier turns and are not replayed as new instructions.

| Field or case | Behavior |
|---|---|
| `LanguageModelExecutorGenerationRequest.id` | Request identity; it does not alter rendering or sampling. |
| `transcript` | Text instructions, prompts, reasoning, responses, tool calls and tool outputs are rendered. Unknown entry kinds are refused. |
| `enabledToolDefinitions` | Rendered when tool calling is allowed; omitted when it is disallowed. |
| `schema` | Refused with `unsupportedCapability(.guidedGeneration)`; request plain text instead. |
| `generationOptions` | The fields below control this generation. |
| `contextOptions` | The fields below control this rendering. |
| `metadata` | Nonempty custom metadata is refused with `unsupportedGenerationGuide`; model-specific passthrough settings are not implemented. Key names, not values, appear in the error. |
| `GenerationOptions.samplingMode` / deprecated `sampling` | The two names refer to the same setting. Omitted means greedy. |
| `.greedy` | Selects the first maximum. |
| `.random(top:seed:)` | Top-k sampling; the seed is used. |
| `.random(probabilityThreshold:seed:)` | Top-p sampling; the seed is used. |
| `temperature` | Passed to the sampler; omitted means 1. Nonpositive values select greedily. |
| `maximumResponseTokens` | Caps the response at the remaining context space. |
| `toolCallingMode` omitted / `.allowed` | Advertises the enabled tools without forcing a call. |
| `.disallowed` | Omits tools from the template. |
| `.required` | Refused with `unsupportedCapability(.guidedGeneration)` because forced tool calls require constrained decoding. |
| Future sampling or tool-calling kinds | Refused instead of falling back to another setting. |
| `ContextOptions.includeSchemaInPrompt` | Has no effect without a schema; a request with a schema is refused. |
| `reasoningLevel` omitted | Uses the template's default. |
| `.light` / `.moderate` / `.deep` | Maps to `low` / `medium` / `xhigh` template values. |
| `.custom("off")` | Sets `enable_thinking` to false. |
| `.custom("low")` / `.custom("medium")` / `.custom("xhigh")` | Uses the same template values as the named levels. |
| Other custom reasoning values | Refused with `unsupportedCapability(.reasoning)` and the rejected value. |
| `Transcript.Entry.instructions` | Preserves system text and tool definitions. |
| `.prompt` | Sanitizes user text before rendering. |
| `.reasoning` / `.response` | Preserves model-generated history. |
| `.toolCalls` | Rebuilds calls and their arguments in the recorded order. |
| `.toolOutput` | Applies context truncation, then sanitizes the remaining text. |
| `Transcript.Segment.text` | Supported. |
| `.structure` / `.attachment` and future segment kinds | Refused with `unsupportedTranscriptContent` before inference. |
| `Transcript.Prompt.id` | Identifies the stored entry; not model input. |
| `segments` | Only text is accepted. |
| `options` / `contextOptions` / `metadata` | Stored turn information. The current request's corresponding fields are authoritative. |
| `responseFormat` | Refused on the current prompt; historical formats do not constrain a new plain-text turn. |

Reasoning strength depends on the bundle. Qwen3.8 uses the three effort values; Qwen3.5's template uses the thinking switch and does not distinguish those effort values. The runtime does not add a thinking-token budget. Custom reasoning names are limited to the four documented values so an unrecognized name is never silently forwarded.

## Operational notes

- **The first load prepares the model with automatic compute-unit selection (no preferred compute unit, with frequent reshapes expected).** The setting is `SpecializationOptions.default` with `expectFrequentReshapes = true`. Core AI keeps the prepared copy in `~/Library/Caches/coreai-cache/<OS build>/<name>/`, where `<name>` is the process name of a command-line tool (`coreai-chat`, `coreai-verify`, `swiftpm-testing-helper` for `swift test`) or the bundle ID of an app (`com.coreai-samples.DemoApp`). Later loads by the same program reuse it; another program prepares its own copy, which takes about as much disk as the model (more for an unquantized bundle). `coreai-chat --stats` shows whether the copy was reused (`cached`) and how long preparing took (`specialize_s`). After an update from an earlier version, which prepared the model for the GPU, the first load specializes again; the copy prepared with the earlier setting stays until the system removes it, for example when storage runs low, or until the prepared copy is removed as described below, which removes both the new and the old copy of that bundle for that program.

- **To remove a prepared copy**, call `Engine.deleteSpecializationCache(bundle:)`, which removes the calling program's copy of that bundle, use **Clear Prepared Copy** in the demo app's Models screen (`--clear-prepared` in its screen check), or delete the folder.

Configure the library with `LocalLanguageModel.ReloadPolicy(readInterval:footprintLimitBytes:)`, passed as `reloadPolicy:` to the model initializer. An interval of zero disables periodic reloads; a footprint limit of zero disables the memory trigger.

The defaults are ten completed reads and a process-footprint trigger of 18,000,000,000 bytes (18 decimal GB). The threshold was calibrated on October 9, 2026 on an M4 Max with 128 GiB, macOS 27.0 (26A428) and Xcode 27, using the Qwen3.8-27B 4bit dynamic `gdnfp32` variant. The `M4-h36-memory-v2` run starting at 2026-10-09 13:03:12 UTC (run ID `3cc72f3295ab`) used thirty prompts, a new conversation every ten turns, context limit 16,384, up to 64 generated tokens per turn, interval ten and no footprint trigger. Its maximum of all thirty immediate post-turn footprints was 11,937,577,240 bytes; multiplying by 1.5 and rounding up to a decimal GB gives 18,000,000,000 bytes. The delayed samples were not used in that calculation. This is a measured reload trigger, not a safe memory budget for every device.

## Limitations

- The language-generation runtime has been run with the converter's macOS exports (`*_dynamic` folders) for Qwen3, Qwen3.5 and Qwen3.8. The stop tokens, the `<think>` tags and the tool-call forms above are the ones those templates use; other model families may need more. Clef uses the separate `DecisionModel` API.

- The language-generation runtime accepts text: a transcript segment that is not text throws `LanguageModelError.unsupportedTranscriptContent`. Recipes uses a separate Apple on-device model for its photo input; the Clef bundle itself is text only.

- No guided generation: a request with a response schema throws `LanguageModelError.unsupportedCapability`.

- The demo uses a context limit of 16,384 tokens. Contexts beyond that limit are outside the verification coverage.

- Use one active session per model unless the simultaneous-session checks pass for the deployed bundle and runtime.

- Cancellation is checked before the prompt is read and between generated tokens, not while a long prompt is being read.

- Decoding is synchronous, one token at a time, like the official `sequential` engine; the official default engine pipelines the work and decodes faster, most of all on small models (`coreai-verify speed` runs the three side by side).

- The separately distributed Clef bundle (downloading it requires access) works with Decide and Recipes. Chat and Work require separate language-generation bundles. The converter revision used for the Qwen3.5 and Qwen3.8 measurements is linked in [Convert your own bundle](#convert-your-own-bundle); it comes from a pull request that is not part of the 1.0.0 tag.

## Repository layout

```
CoreAISamples.xcodeproj  the demo app project: open it and press Run
CoreAISamples/           macOS SwiftUI demo app: Models, Chat (Chat and Work, with RAG), Decide, Recipes, Extract
CoreAISamplesKit/        the Swift package the app is built on
  Sources/CoreAIBackend/ the runtime: Engine (Core AI calls), LocalLanguageModel (Foundation Models), ModelBundle (metadata.json)
  Sources/RAGContext/    passages, ranking, token budget, prompt and dates for answers from retrieved sources; a document-folder reader
  Sources/coreai-chat/   the command-line chat (ChatCommand.swift)
  Tests/                 CoreAIBackendTests and RAGContextTests (swift test)
Config/                  the app's signing settings: Signing.xcconfig and Local.xcconfig.template
project.yml              the XcodeGen spec of CoreAISamples.xcodeproj
Tools/Verify/            separate package: coreai-verify, which compares this runtime with apple/coreai-models 1.0.0
docs/demo-app.md         the demo app: building it, headless self-tests, screen checks and logs
LICENSE                  MIT, for the code
```

## Demo app

`CoreAISamples` is a macOS SwiftUI app built on this repository's `CoreAIBackend` and `RAGContext`, without the official `coreai-models` package.

- **Models** opens a bundle folder on this Mac, loads it, and clears the prepared copy. Its catalog lists Qwen3.8 27B (4bit), but the private repository currently prevents anonymous download; open an existing bundle folder instead. The catalog's memory figure, ~32 GiB peak RSS while loading, is the peak resident memory of the app process while it loaded that bundle on the Mac in the [Qwen measurements](qwen3.8-27b.md#key-numbers), measured with `/usr/bin/time -l`: 31.51 GiB from the prepared copy (median of 48 loads) and 31.88 GiB at the first load. It is not a minimum memory requirement.
- **Chat** is one conversation with an independent Chat | Work picker. **Search and sources** starts off: answers use the model and existing conversation without new searches or document reads. Turn it on and select Web, Docs, or both to collect sources before answering. Work can save notes with either setting. Retrieved material and stable source IDs such as `[S1]` remain in the same saved conversation. [Access settings](specs/conversation-search-access.md) explain cancellation, continuation and archive compatibility.
- **Decide** scores a document against editable choices with Clef-Flash, reuses the document, and keeps human confirmation separate from model candidates.
- **Recipes** checks a recipe against reviewed fridge inventory and kitchen equipment. Five synthetic photos and their recipe URLs are bundled, and developer details show the actual decision scores and timings.
- **Extract** turns official Apple/Microsoft product descriptions or pasted text into up to four source-linked specification cards. Open a candidate to inspect its original conditions; multiple configurations and missing fields stay visible. The existing label and inspection-memo experiment remains under **Extraction lab**, and its B3 finding is unchanged. Product extraction is a separate use case; see the [specification and verification status](specs/product-comparison-demo.md).

The demo app's Chat and Work handle Qwen3, Qwen3.5 and Qwen3.8 bundles. Adding another family's Hugging Face model ID prefix, such as `"google/gemma-"`, to `BundleSupport.supportedModelIDPrefixes` in `CoreAISamples/Demos/ModelManager/BundleSupport.swift` lets the app open that family's bundles but does not make them work: the stop tokens, the `<think>` tags and the tool-call forms the runtime handles stay the Qwen ones (see Limitations), so such a bundle can load and still answer with broken text. The list is the only family check. When you open or load a bundle folder, the app reads `source.hf_model_id` from the bundle's `metadata.json` and refuses the bundle, with the reason on the Models screen or in Chat, if the field is missing or does not start with one of the prefixes. It does not check how the bundle was converted.

To try it, open `CoreAISamples.xcodeproj` and press Run. Building it from the command line, the headless self-tests, the screen checks and the logs are in [docs/demo-app.md](demo-app.md).
