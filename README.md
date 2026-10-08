# Core AI Samples

Swift code for running open language models such as Qwen on a Mac with Apple's Core AI framework: a runtime library, a command-line chat, a demo app and a verification tool.
The runtime reads the model bundles made by Apple's converter in [apple/coreai-models](https://github.com/apple/coreai-models) and works as a Foundation Models `LanguageModel`, so an app talks to it through an ordinary `LanguageModelSession`.

> **Status:** not released yet. No model bundle has been published, so you convert one yourself (step 1 below). The first release is planned to be Qwen3.8 27B (4-bit) for macOS.

## Who this is for

- **For** Swift developers who want an open model inside a Mac app through the Foundation Models API, with the runtime's source at hand.
- **For** anyone who wants to check such a runtime against Apple's own: `Tools/Verify` compares the two token by token.
- **Not for**, yet: iOS apps, or ready-made model downloads (none are published). For macOS 26 / iOS 26, see [coreml-llm-samples](https://github.com/oka-yuji/coreml-llm-samples).

Requirements: an Apple silicon Mac with macOS 27, and Xcode 27 with the Metal Toolchain component. `CoreAISamplesKit/Package.swift` also declares iOS 27, but nothing has been run on iOS.

## Quick start

**1. Make a bundle** with Apple's converter. Setup (it uses `uv`) and the supported models are in [its models README](https://github.com/apple/coreai-models/blob/1.0.0/models/README.md). This is the command used here for Qwen3-0.6B:

```bash
git clone https://github.com/apple/coreai-models.git && cd coreai-models && git checkout 1.0.0
uv run coreai.llm.export Qwen/Qwen3-0.6B --output-dir ~/coreai-bundles   # writes ~/coreai-bundles/qwen3_0_6b_4bit_dynamic
```

The runtime has been run with that bundle, with bundles of Qwen3.5-0.8B and Qwen3.5-9B, and with Qwen3.8-27B in 4-bit and 8-bit. The Qwen3.5 and Qwen3.8 exports come from [pull request #321](https://github.com/apple/coreai-models/pull/321) of apple/coreai-models, which is still open (checked 2026-10-03), so they are not in the 1.0.0 tag.

**2. Chat** from the `CoreAISamplesKit` folder:

```bash
cd CoreAISamplesKit
swift build -c release
.build/release/coreai-chat --model ~/coreai-bundles/qwen3_0_6b_4bit_dynamic --system "You are a concise assistant." \
  --prompt "My name is Taro. Remember it." --prompt "What is the capital of France? Answer with one word." \
  --prompt "What is my name?" --max-tokens 1024 --stats
```

Each `--prompt` is one turn of the same conversation. When the model thinks, the thinking prints under `[reasoning]` and the answer under `[response]`. `--stats` writes JSON lines to stderr: one for the load (`cached`, `specialize_s`, `load_s`) and one per turn, where `cached_tokens` is the part of the prompt reused from earlier turns.

**3. Run the demo app.** Open `CoreAISamples.xcodeproj` in Xcode, choose the `CoreAISamples` scheme and `My Mac`, and press Run. Xcode resolves the package dependencies the first time. The `CoreAISamples` scheme runs a Release build, and `CoreAISamples-Debug` runs a Debug build for debugging. In the app, **Models** opens the bundle folder from step 1.

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
    print(snapshot.content)  // the whole answer so far, growing with each snapshot
}
```

This is the core of `CoreAISamplesKit/Sources/coreai-chat/ChatCommand.swift`, cut down to the calls that matter; the CLI sends one turn per `--prompt` to the same session and prints only the new part of each snapshot. The thinking arrives as `.reasoning` entries in `session.transcript`, the token counts in `session.usage`, and the stop reason in the response metadata (`stopReason`). `LocalLanguageModel(bundle:contextLimit:overflow:)` also takes a context limit and an `OverflowPolicy` (below). The demo app and `Tools/Verify` use `CoreAIBackend` from the `CoreAISamplesKit` package in this repository as a local path dependency; there is no tagged release to depend on yet. Adding this repository's URL in Xcode's Add Package Dependencies is not supported, because the package lives in a folder of the repository. To use it in your own app, clone the repository, add the `CoreAISamplesKit` folder as a local package (File > Add Package Dependencies > Add Local) and link `CoreAIBackend`; updates come through the clone.

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

Reloading releases the engine and saved states, preserves CPU history, and reads the complete prompt into a new engine. `sharedTokens` retains the logical shared prefix, while cached input is zero and prefill covers the complete prompt. Only a request which loads a replacement records `reloadSeconds`. Reloading adds a model load; interrupting a conversation also adds a full-prompt prefill. Long histories therefore cost more to resume.

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

## Repository layout

```
CoreAISamples.xcodeproj  the demo app project: open it and press Run
CoreAISamples/           macOS SwiftUI demo app: Models, Chat (with RAG), Agent, Extract
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

- **Models** opens a bundle folder on this Mac, loads it, and clears the copy prepared for the GPU.
- **Chat** streams the reply and the thinking, and with RAG answers from web pages and local documents with numbered sources.
- **Agent** searches the web, reads pages and saves notes through the Foundation Models tool loop.
- **Extract** shows the spans GLiNER2.5-multi (Core AI, FP32) finds in a text for the labels you type, with their scores; mapping an inspection memo onto a form with option suggestions is kept as an experiment. Technical verification of a Core AI conversion. Zero-shot GLiNER2.5-multi did not reach the pre-registered bar for Japanese inspection memos (outcome B3). The evaluated version is the default candidate filter, R1 + N1: on a second sealed set of 200 memos, fabricated candidates 13/200, primary-span precision 0.788, recall 0.822 (the first evaluation, rule R1 on the first sealed set: 9/199, 0.791, 0.874). Suggestions are uncalibrated scores.

To try it, open `CoreAISamples.xcodeproj` and press Run. Building it from the command line, the headless self-tests, the screen checks and the logs are in [docs/demo-app.md](docs/demo-app.md).

## Verification

`Tools/Verify` is a separate Swift package and the only place that depends on Apple's `coreai-models` package, pinned to 1.0.0; the library, the CLI and the demo app do not. Its `coreai-verify` tool drives Apple's runtime (`--runtime official`, with `--engine default` or `sequential`) and this one (`--runtime own`) and prints one JSON line per measurement. In the table, "own" is this repository's runtime.

| Subcommand | What it does |
|---|---|
| `chat` | Stream several turns through one LanguageModelSession. |
| `tools` | Ask questions that may need tools; print one record per model step and one per question. |
| `bench` | Time a reload, then stream one fixed prompt three times, each in a new session. |
| `compare` | Feed the same tokens to the own engine and the official sequential engine; compare greedy tokens and logits. |
| `rewind` | Save a point after a shared preamble, generate, restore, and generate again; time a full re-read for comparison. |
| `speed` | Alternate the own engine, the official sequential engine, and the official default engine, one process each. |

`chat`, `tools` and `bench` take `--runtime` and `--engine`, plus `--context-limit` and `--drop-old-tool-outputs` for `--runtime own`.

```bash
cd Tools/Verify
swift build -c release
.build/release/coreai-verify compare --model <bundle> --max-tokens 64 --prompt "en=List three fruits, one per line."
.build/release/coreai-verify chat --model <bundle> --runtime own --max-tokens 64 --prompt "What is the capital of France? Answer with one word."
```

From the `CoreAISamplesKit` folder, `swift test` runs the tests that need no model, such as the reuse plan, the parsers, the sampler, the context fitting and the RAG passages and dates. The tests that load a model run only when `COREAI_TEST_BUNDLES` lists bundle folders, separated by `:`.

```bash
cd CoreAISamplesKit
swift test
COREAI_TEST_BUNDLES=/path/to/bundle-a:/path/to/bundle-b swift test
```

The model-loading tests share one serialized suite; tokenizer-only checks remain independent. Acquire the heavy-job lock described in `CLAUDE.md` before running them. `seededSamplingRepeats` compares complete generated token sequences for top-k (40) and top-p (0.95), temperature 0.8 and seed 7, with state reuse disabled. The bundle, prompt, sampling settings and prefill segmentation must match for exact reproducibility. `COREAI_SEED_REPETITIONS` changes the default 20 repetitions; use the same test with 5 repetitions for a release candidate's larger bundle.

The boundary tests render exactly 16,382, 16,383, 16,384 and 16,385 input tokens against a 16,384-token context. With either overflow policy, the first two inputs leave room for at most two and one generated tokens; the other two must throw `contextSizeExceeded` with the limit and input count. Separate histories check tool-output truncation, its metadata and irreducible overflow. State checks compare against a full reread using the eight-tick near-tie rule.

`twoSessionsShareOneModel` submits 50 pairs through one model, records overlapping executor calls, checks for the other session's marker and cancels one request after its first streamed fragment in every fifth pair. It also checks recovery and the final state. `COREAI_CONCURRENCY_PAIRS=1` selects one pair for a Thread Sanitizer run. These are opt-in checks; a skipped model test does not validate a bundle.

```bash
COREAI_TEST_BUNDLES=/path/to/bundle COREAI_SEED_REPETITIONS=5 swift test --filter ModelIntegrationTests.seededSamplingRepeats
COREAI_TEST_BUNDLES=/path/to/bundle COREAI_CONCURRENCY_PAIRS=1 swift test --sanitize=thread --filter ModelIntegrationTests.twoSessionsShareOneModel
```

The own runtime's `chat` and `tools` commands accept `--reload-interval` and `--reload-footprint-bytes`; zero disables the respective trigger. `chat --new-conversation-every 10` starts a new session after every ten prompts. Own-runtime chat records include `generated_tokens`, the complete sampled output IDs without the stop token, including the last unprocessed ID at a response limit. Failed CLI turns omit this field. The Verification SPI returns the most recent request that entered the Runtime: empty before decoding, complete on success, and the current partial samples on decode failure or cancellation. Framework rejection before the executor runs does not update the SPI snapshot. Turn and tool-step records include `reload_seconds` when present, and memory records include graphics footprint bytes (the graphics footprint ledger plus its compressed ledger). `chat --check-state` checks each reloaded turn and the final state.

```bash
COREAI_TEST_BUNDLES=/path/to/bundle swift test --filter ModelIntegrationTests
COREAI_TEST_BUNDLES=/path/to/bundle COREAI_TEST_RELOAD_INTERVAL=0 COREAI_TEST_RELOAD_FOOTPRINT_BYTES=1 swift test --filter ModelIntegrationTests.twoSessionsShareOneModel
```

The low-threshold concurrency run requires at least one observed reload. Model-enabled results are required in addition to the CPU policy and fake-engine tests.

## Operational notes

- **The first load prepares the model for the GPU.** Core AI keeps the prepared copy in `~/Library/Caches/coreai-cache/<OS build>/<name>/`, where `<name>` is the process name of a command-line tool (`coreai-chat`, `coreai-verify`, `swiftpm-testing-helper` for `swift test`) or the bundle ID of an app (`com.coreai-samples.DemoApp`). Later loads by the same program reuse it; another program prepares its own copy, which takes about as much disk as the model (more for an unquantized bundle). `coreai-chat --stats` shows whether the copy was reused (`cached`) and how long preparing took (`specialize_s`).
- **To remove a prepared copy**, call `Engine.deleteSpecializationCache(bundle:)`, which removes the calling program's copy of that bundle, use **Clear Prepared Copy** in the demo app's Models screen (`--clear-prepared` in its screen check), or delete the folder.

Configure the library with `LocalLanguageModel.ReloadPolicy(readInterval:footprintLimitBytes:)`, passed as `reloadPolicy:` to the model initializer. An interval of zero disables periodic reloads; a footprint limit of zero disables the memory trigger.

The footprint default is zero in this private draft and has not been calibrated. The release threshold must be the maximum post-turn footprint from the approved variant 27B thirty-turn run with interval ten and no threshold, multiplied by 1.5 and rounded up to a decimal GB. The measured threshold, source run and integrated reload costs must be recorded before release; it does not represent a safe memory limit for every device.

## Limitations

- Run on macOS only (Apple silicon, macOS 27.0). `CoreAISamplesKit/Package.swift` declares iOS 27, but nothing has been run on iOS. The demo app's iPhone target (`CoreAISamples-iOS`, Extract only, see [docs/demo-app.md](docs/demo-app.md#extract-on-iphone)) builds but has not been run on a device yet.
- Only bundles from the converter's macOS export (the `*_dynamic` folders) have been run, all of them Qwen models (Qwen3, Qwen3.5, Qwen3.8). The stop tokens, the `<think>` tags and the tool-call forms above are the ones those templates use; other model families may need more.
- Text in, text out: a transcript segment that is not text throws `LanguageModelError.unsupportedTranscriptContent`, and there is no image or audio input.
- No guided generation: a request with a response schema throws `LanguageModelError.unsupportedCapability`.
- The demo uses a context limit of 16,384 tokens. Contexts beyond that limit are outside the verification coverage.
- Use one active session per model unless the simultaneous-session checks pass for the deployed bundle and runtime.
- Cancellation is checked before the prompt is read and between generated tokens, not while a long prompt is being read.
- Decoding is synchronous, one token at a time, like the official `sequential` engine; the official default engine pipelines the work and decodes faster, most of all on small models (`coreai-verify speed` runs the three side by side).
- No published bundles and no tagged release yet, and the Qwen3.5 and Qwen3.8 exports need the unmerged branch of apple/coreai-models (Quick start, step 1).

## Related

- [coreml-llm-samples](https://github.com/oka-yuji/coreml-llm-samples) — the Core ML line, for macOS 26 / iOS 26.

## License

Code: MIT — see [LICENSE](LICENSE). Model weights are distributed separately under their own licenses.
