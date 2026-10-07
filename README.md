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

- **Reuses what it has already read.** Each request is compared with the tokens the engine already holds, and only the part after the first difference is read again (in hybrid models, from the nearest saved point; next item): across the turns of a conversation and across the steps of a tool loop. The reused count is reported as `cachedTokenCount` in the session's usage. Sessions created from the same `LocalLanguageModel` share one engine, and their requests run one at a time.
- **Returns to a saved point in hybrid models.** Bundles with fixed-size recurrent state (Qwen3.5 and Qwen3.8) cannot be rewound by trimming the cache, so the runtime saves that state where the user's turn ends, where generation starts and where two prompts part, and restores the nearest point instead of reading from the start (`Engine.checkpoint()`, `Engine.restore(_:)`).
- **Separates thinking from the answer.** Text between `<think>` and `</think>` streams as `.reasoning` entries, the rest as the response. `ContextOptions.reasoningLevel` reaches the chat template as `reasoning_effort` (`.custom("off")` sets `enable_thinking` to false); whether it has an effect depends on the template.
- **Calls tools.** Tool definitions go into the chat template, and calls are parsed while streaming from `<tool_call>` blocks in either form, JSON (`{"name": ..., "arguments": ...}`) or `<function=...>` with `<parameter=...>`. A malformed block stays in the answer as text. With `toolCallingMode` `.disallowed` the tools are left out of the prompt. Foundation Models runs the tools and asks the model again.
- **Handles a full context in one of two ways** (`OverflowPolicy`). `.error`, the default, throws `LanguageModelError.contextSizeExceeded` only when the prompt itself no longer fits; otherwise the reply gets what is left. `.dropOldToolOutputs` first replaces the oldest tool results with `[truncated]` (the newest two groups are kept) and then halves the newest ones, to leave room for the reply; if that is not enough, it falls back to `.error`.
- **Samples as asked.** With no sampling options it decodes greedily. The `GenerationOptions` sampling modes `.random(top:seed:)` and `.random(probabilityThreshold:seed:)` and its `temperature` are honoured, and the same seed gives the same text.
- **Stops and cancels cleanly.** Generation ends at the end-of-sequence token or `<|endoftext|>` (the stop token is not fed back to the model) or at `maximumResponseTokens`, and the response metadata says which (`stopReason`: `stopToken` or `maxTokens`). A reply that runs out of tokens while still thinking ends with an empty answer, not an error. After a cancelled request, the next one on the session works.

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

## Operational notes

- **The first load prepares the model for the GPU.** Core AI keeps the prepared copy in `~/Library/Caches/coreai-cache/<OS build>/<name>/`, where `<name>` is the process name of a command-line tool (`coreai-chat`, `coreai-verify`, `swiftpm-testing-helper` for `swift test`) or the bundle ID of an app (`com.coreai-samples.DemoApp`). Later loads by the same program reuse it; another program prepares its own copy, which takes about as much disk as the model (more for an unquantized bundle). `coreai-chat --stats` shows whether the copy was reused (`cached`) and how long preparing took (`specialize_s`).
- **To remove a prepared copy**, call `Engine.deleteSpecializationCache(bundle:)`, which removes the calling program's copy of that bundle, use **Clear Prepared Copy** in the demo app's Models screen (`--clear-prepared` in its screen check), or delete the folder.

## Limitations

- Run on macOS only (Apple silicon, macOS 27.0). `CoreAISamplesKit/Package.swift` declares iOS 27, but nothing has been run on iOS. The demo app's iPhone target (`CoreAISamples-iOS`, Extract only, see [docs/demo-app.md](docs/demo-app.md#extract-on-iphone)) builds but has not been run on a device yet.
- Only bundles from the converter's macOS export (the `*_dynamic` folders) have been run, all of them Qwen models (Qwen3, Qwen3.5, Qwen3.8). The stop tokens, the `<think>` tags and the tool-call forms above are the ones those templates use; other model families may need more.
- Text in, text out: a transcript segment that is not text throws `LanguageModelError.unsupportedTranscriptContent`, and there is no image or audio input.
- No guided generation: a request with a response schema throws `LanguageModelError.unsupportedCapability`.
- Cancellation is checked before the prompt is read and between generated tokens, not while a long prompt is being read.
- Decoding is synchronous, one token at a time, like the official `sequential` engine; the official default engine pipelines the work and decodes faster, most of all on small models (`coreai-verify speed` runs the three side by side).
- No published bundles and no tagged release yet, and the Qwen3.5 and Qwen3.8 exports need the unmerged branch of apple/coreai-models (Quick start, step 1).

## Related

- [coreml-llm-samples](https://github.com/oka-yuji/coreml-llm-samples) — the Core ML line, for macOS 26 / iOS 26.

## License

Code: MIT — see [LICENSE](LICENSE). Model weights are distributed separately under their own licenses.
