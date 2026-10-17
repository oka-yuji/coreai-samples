# Core AI Samples

[日本語](README.ja.md)

Swift samples that run open models on a Mac through Apple's Core AI and Foundation Models. **Chat** and **Agent** run Qwen3.8 27B; the agent searches the web, reads pages and saves notes. **Recipes** compares a fridge photo's ingredients with a recipe URL and **Decide** scores a document against an editable checklist, both with Clef-Flash 9B. **Extract** turns product descriptions into source-linked specification cards with GLiNER2.5-multi. `Tools/Verify` compares this repository's runtime with Apple's own, token by token.

Demo titles use model names: **Clef-Flash 9B**, **GLiNER2.5-multi**, **EmbeddingGemma 2**, and **Irodori**. The conversation title shows its selected or saved model. Sidebar descriptions and the iPhone menu's model sections distinguish each model's demos; Chat and Work remain conversation modes.

## Who this is for

- **For** Swift developers who want a 27B chat model and a web-browsing agent running locally on a Mac through `LanguageModelSession`, with the measurements behind them.
- **For** Swift developers who want to try local recipe checks or document decisions, inspect the scores, and reuse the code.
- **For** developers integrating open language-generation models through Foundation Models; Chat and Work use a separate runtime and bundle from Recipes and Decide.
- **For** anyone who wants to check such a runtime against Apple's own: `Tools/Verify` compares the two token by token.
- **Not for** iPhone or iPad, contexts beyond the demo's 16,384 tokens, using Clef as a chat model, or treating a model probability as a measured chance of correctness.

Requirements: an Apple silicon Mac with macOS 27 and Xcode 27 with the Metal Toolchain component. Qwen3.8 27B takes 16.85 GB to download plus a prepared copy of about the same size for each program that loads it, and its process reached 34.2 GB (31.9 GiB) of resident memory while loading; it was measured on an M4 Max with 128 GiB, and a minimum-memory configuration has not been established. Recipes photo reading also needs Apple Intelligence with vision available on the Mac. Clef was measured on an M4 Max with 128 GiB; a minimum-memory configuration has not been established. Clef and the Recipes demo have not been verified on iOS.

## Models

| Bundle | Used by | Download | Guide |
|---|---|---|---|
| Qwen3.8 27B, 4-bit, text only | Chat, Agent, `coreai-chat` | [Hugging Face](https://huggingface.co/okayuji/Qwen3.8-27B-coreai-4bit) | [Quick start](#quick-start-chat-and-agent-with-qwen38-27b) below |
| Clef-Flash 9B, text only, uint4 backbone and fp16 head | Recipes, Decide | [Hugging Face](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide) (requires access) | [English](docs/clef-flash.md) · [日本語](docs/clef-flash.ja.md) |

## Quick start: Chat and Work with Qwen3.8 27B

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), then get the app and the bundle (12 files, 16.85 GB):

```sh
git clone https://github.com/oka-yuji/coreai-samples.git
cd coreai-samples
hf download okayuji/Qwen3.8-27B-coreai-4bit --revision 8e24aebe036eae577d4a3bd019ba728a73358318 \
  --local-dir ./models/qwen3.8-27b-coreai-4bit
```

Open `CoreAISamples.xcodeproj`, choose the **CoreAISamples** scheme and **My Mac**, and run; the scheme builds Release. In **Models**, choose **Open Folder…**, select `models/qwen3.8-27b-coreai-4bit` and press **Load**. The first load took 86.87 s on an M4 Max, 82.84 s of it preparing the model for the GPU; later loads took 3.6–10.3 s. The conversation title shows the model name; use **Chat** mode for conversation or **Work** mode for a task that needs the web. **Models** can also download the same pinned revision itself: **Download** next to *Qwen3.8 27B (4bit)* checks every file against the Hub.

Chat defaults to 16 generation steps per reply; Work keeps its separate default of eight. In the conversation settings, **Max steps per reply** changes the selected mode's limit from 1 to 64. Each new question starts a fresh budget. If research exhausts it, the app makes one extra answer attempt with tools and reasoning disabled, using the information already collected. **Stop** still cancels the request. See the [budget contract and checks](docs/specs/conversation-step-budget.md).

From the command line, in the `CoreAISamplesKit` folder:

```sh
cd CoreAISamplesKit
swift build -c release
.build/release/coreai-chat --model ../models/qwen3.8-27b-coreai-4bit --max-tokens 2048 --stats \
  --prompt "What is the capital of France? Answer with one word."
```

The model thinks before it answers: the thinking prints under `[reasoning]` and the answer under `[response]`, and `--stats` reports the load and each turn on stderr.

| The bundle | |
|---|---|
| Repository and revision | [`okayuji/Qwen3.8-27B-coreai-4bit`](https://huggingface.co/okayuji/Qwen3.8-27B-coreai-4bit) at `8e24aebe036eae577d4a3bd019ba728a73358318`, the revision the app's Catalog pins; its tag is `v1.0.0` |
| Size | 12 files, 16,852,543,780 bytes (16.85 GB, 15.70 GiB); the model file `main.mlirb` is 16,839,684,298 bytes |
| Source | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, text only (the vision encoder is not included) |
| Conversion | Apple's converter at commit `36e9350` of [pull request #321](https://github.com/apple/coreai-models/pull/321), which is not part of the 1.0.0 tag; preset `qwen3_5_cfg` (4-bit weights, bfloat16 compute) with the gated-delta state update computed in fp32; macOS export with dynamic shapes |
| Tokenizer | `tokenizer/tokenizer.json` is the original Qwen3.8-27B file, in place of the copy the converter re-saved; on the 2,257 strings used in the measurements, Swift produced the same token IDs with either file |
| Memory | the process reached 33.8 GB (31.5 GiB) of resident memory when loading from the prepared copy and 34.2 GB (31.9 GiB) at the first load; 16.0–16.7 GiB stayed resident during Agent steps |
| Platform | macOS 27 on Apple silicon, GPU; not iPhone or iPad |

### Sample output

An Agent task in the app's headless driver (`--agent-e2e`), reasoning off, on the live web, October 12, 2026. This run used `--instructions cite-fetched --citation-guard on`; the app's default is the same instructions with the guard off. The answer and the two notes under it are verbatim:

```
Read https://developer.apple.com/documentation/createml and list three key points.
```

```
Here are three key points from Apple's Create ML documentation:

1. **Train custom ML models on your Mac with familiar tools.** Create ML lets you use Swift and macOS playgrounds to create and train machine learning models locally. You train a model by showing it representative samples (e.g., many images of dogs to recognize dogs), then test it on unseen data and evaluate performance before integrating it into your app via Core ML.

2. **Broad model types are supported.** The framework covers image models (image classifier, object detector, hand pose classifier), video models (action classifier, hand action classifier, style transfer), text models (text classifier, word tagger, word embedding, gazetteer), sound classifiers, motion/activity classifiers, and tabular models (classifiers, regressors, recommenders).

3. **It leverages Apple's built-in ML infrastructure.** Because Create ML builds on the machine learning infrastructure used in Apple products like Photos and Siri, your image classification and natural language models are smaller and train much faster. It also provides built-in tools for evaluating model accuracy (metrics for classifiers, regressors, word taggers, recommenders, and object detectors) and controlling training sessions (MLJob, MLTrainingSession, checkpoints).

**Platform availability:** iOS 15.0+, iPadOS 15.0+, Mac Catalyst 15.0+, macOS 10.14+, tvOS 16.0+, visionOS 1.0+.

Sources:
https://developer.apple.com/documentation/createml
--- note Citations
1/1 cited URLs were retrieved in this task · mean 8-gram overlap 0.38
--- note Citation guard
Cited sources fetched · 1/1 cited URLs were fetched in this task
```

The agent called `fetch_page` once (6.3 s, 5,993 characters) and answered in the second step. The whole task took 33.3 s after a 4.5 s load from the prepared copy.

| Step | Prompt tokens | Reused | Generated | First token | Decode |
|---|---|---|---|---|---|
| 1 (tool call) | 877 | 0 | 34 | 7.97 s | 23.8 tok/s |
| 2 (answer) | 2,189 | 911 | 317 | 4.95 s | 23.0 tok/s |

### Key numbers

MacBook Pro (Mac16,6) with an M4 Max and 128 GiB, macOS 27.0 (26A428), the bundle above, greedy decoding, one model process at a time, October 11–12, 2026. The runtime and the Agent code were the same as in this repository; the Agent's instructions and citation guard were set by the flags shown.

| Measurement | Result | Conditions |
|---|---|---|
| Decode speed | 22.74 tok/s, median of 144 generation steps | Agent, 12 frozen tasks × 4 rounds (48 runs) on the live web, `--instructions frozen --citation-guard off`, reasoning low (36 runs) and off (12) |
| Time to first token | 5.54 s to read the prompt, 6.46 s to the first text on screen (medians of the same steps) | the same 48 runs |
| Agent tasks | 48/48 passed the frozen grader, 48/48 on answer quality; no unintended re-render, no run that failed to stop | the same 48 runs |
| Agent with the citation guard on | 41/42 tasks achieved and 1 held; the guard verified 35, repaired 6 and held 1; decode 22.90 tok/s, first token 5.33 s / 6.46 s (151 steps) | 21 tasks × reasoning low and off on the live web, `--instructions cite-fetched --citation-guard on` |
| Load | 86.87 s the first time, 82.84 s of it preparing for the GPU; 3.6–10.3 s afterwards (median 4.7 s, 48 runs) | the app |
| Resident memory | 33.8 GB (31.5 GiB) peak per process from the prepared copy, 34.2 GB (31.9 GiB) at the first load; 16.0–16.7 GiB during Agent steps | `/usr/bin/time -l` and the app's `metrics.jsonl` |
| RAG | first text on screen after 15.53 s (median of 3) with 2,950 source tokens, 3,120 input tokens in all | the app's `--selftest --rag`, 3,000-token source budget, local documents only |

The default Agent settings (cite-fetched instructions, guard off) have not been measured with this bundle. Correctness checks, including agreement with a reference implementation, are under [Verification](#verification); constraints are under [Limitations](#limitations).

## Quick start: Recipes or Decide

Can you make a recipe with what is in your fridge? **Recipes** reads a fridge photo, lets you check the proposed ingredients, and compares your inventory with a recipe URL. It uses Apple's on-device model for the photo, **Clef-Flash 9B through Core AI** for recipe requirements and equipment, and Swift code for quantities. **Decide** uses the same Clef bundle to score a document against editable questions. Both expose the model's uncalibrated probabilities.

**Requires access to the separately distributed Clef bundle.** The `hf download` line below works only with a Hugging Face account that has access to that repository. Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), then get the app and decision bundle:

```sh
git clone https://github.com/oka-yuji/coreai-samples.git
cd coreai-samples
hf download okayuji/clef-flash-9b-coreai-decide --local-dir ./models/clef-flash-9b-decide-v1
```

Open `CoreAISamples.xcodeproj`, choose the **CoreAISamples** scheme and **My Mac**, set the Run build configuration to **Release**, and run. The **EN / JP** control at the top right switches every Mac screen.

In the **Clef-Flash 9B** sidebar entry for recipe checks, choose **Open Clef-Flash bundle** and select `models/clef-flash-9b-decide-v1`. Try **Examples → Three-egg omelette**, review the suggested egg as `1 pcs`, confirm it, then **Check recipe**. The recipe needs three eggs, so the confirmed inventory is two short. Follow the [Recipes walkthrough](docs/recipes-demo.md) for all five bundled photos and recipe URLs. Photos and inventory stay on the Mac; fetching a recipe URL needs an internet connection.

In the **Clef-Flash 9B** sidebar entry for document checklists, open the same bundle, choose **Examples → Maintenance agreement**, then **Run checklist**. Edit a question and run again to reuse the document. The [Decide guide](docs/decide.md) covers review, JSON export and the Swift API.

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

**3. Run the demo app.** Install Rust/Cargo for the speech input adapter (Xcode builds it automatically). Open `CoreAISamples.xcodeproj` in Xcode, choose the `CoreAISamples` scheme and `My Mac`, and press Run. Xcode resolves the package dependencies the first time. The `CoreAISamples` scheme runs a Release build, and `CoreAISamples-Debug` runs a Debug build for debugging. In the app, **Models** opens the bundle folder from step 1. The separate [Irodori screen](docs/irodori/demo.md) accepts the local Irodori FP32 folder for text-to-speech; its app/GPU/audio acceptance is pending. An [iPhone speech app](docs/irodori/iphone.md) also builds for iOS 27; device execution remains pending.

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

- **Models** opens a bundle folder on this Mac, loads it, and clears the copy prepared for the GPU. Its catalog lists Qwen3.8 27B (4bit) for download. The catalog's memory figure, ~32 GiB peak RSS while loading, is the peak resident memory of the app process while it loaded that bundle on the Mac in the measurement table below, measured with `/usr/bin/time -l`: 31.51 GiB from the prepared copy (median of 48 loads) and 31.88 GiB at the first load. It is not a minimum memory requirement.
- **Chat** is one conversation with a Chat | Work picker: Chat answers and searches the web when it needs facts, Work searches, reads pages and saves notes through the Foundation Models tool loop, and with RAG the app answers from web pages and local documents with persistent source IDs such as `[S1]` in the same conversation. Saved conversations reopen with their source ledger. Its citation guard is off by default and can check fetched sources before accepting an answer.
- **Decide** scores a document against editable choices with Clef-Flash, reuses the document, and keeps human confirmation separate from model candidates.
- **Recipes** checks a recipe against reviewed fridge inventory and kitchen equipment. Five synthetic photos and their recipe URLs are bundled, and developer details show the actual decision scores and timings.
- **Extract** turns official Apple/Microsoft product descriptions or pasted text into up to four source-linked specification cards. Open a candidate to inspect its original conditions; multiple configurations and missing fields stay visible. The existing label and inspection-memo experiment remains under **Extraction lab**, and its B3 finding is unchanged. Product extraction is a separate use case; see the [specification and verification status](docs/specs/product-comparison-demo.md).

The demo app's Chat and Work handle Qwen3, Qwen3.5 and Qwen3.8 bundles. Adding another family's Hugging Face model ID prefix, such as `"google/gemma-"`, to `BundleSupport.supportedModelIDPrefixes` in `CoreAISamples/Demos/ModelManager/BundleSupport.swift` lets the app open that family's bundles but does not make them work: the stop tokens, the `<think>` tags and the tool-call forms the runtime handles stay the Qwen ones (see Limitations), so such a bundle can load and still answer with broken text. The list is the only family check. When you open or load a bundle folder, the app reads `source.hf_model_id` from the bundle's `metadata.json` and refuses the bundle, with the reason on the Models screen or in Chat, if the field is missing or does not start with one of the prefixes. It does not check how the bundle was converted.

To try it, open `CoreAISamples.xcodeproj` and press Run. Building it from the command line, the headless self-tests, the screen checks and the logs are in [docs/demo-app.md](docs/demo-app.md).

## Verification

Run the complete build and model-free checks from the repository root. These need an Apple silicon Mac with macOS 27 and Xcode 27 selected with `xcode-select`.

```bash
./check.sh
```

The script builds and tests `CoreAISamplesKit`, builds `Tools/Verify` and runs its CPU tests in Release, builds the macOS app in Release, and runs the Agent offline self-test. It clears inherited `COREAI_*`, `GLINER_*`, `DECIDE_*` and `CLEF_*` controls before these stages, so an unrelated model-test setting cannot start inference. The last line is `CHECK PASS` with exit code 0, or `CHECK FAIL: <stage>` with exit code 1.

To add language-model checks, install Python 3 for JSON result validation and supply bundle folders separated by `:`. The first bundle is also used for the official-runtime comparison and the state check; a small bundle such as Qwen3.5-0.8B keeps this check short.

```bash
COREAI_TEST_BUNDLES=/path/to/bundle-a:/path/to/bundle-b ./check.sh
```

This adds the model-enabled tests, two fixed prompts compared against Apple's sequential engine, and two own-runtime chat turns followed by a processed-state check. Both prompts must match every generated token and every step's logits on the matching `prefill-function` path. The two `main-only` records are retained as diagnostics because they use a different prefill path; their agreement is not a pass criterion. Missing records, reported errors and a failed state check fail the script even if the verification command exits successfully. The JSON records are saved under `.build/check/`.

The runtime and demo-app measurements use this environment:

| Component | Measured configuration |
|---|---|
| macOS | 27.0, build 26A428 |
| Xcode | 27.0, build 27A266a |
| Mac | MacBook Pro (Mac16,6), Apple M4 Max |
| Unified memory | 128 GiB (137,438,953,472 bytes) |

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

The own runtime's `chat` and `tools` commands accept `--reload-interval` and `--reload-footprint-bytes`; zero disables the respective trigger. `chat --new-conversation-every 10` starts a new session after every ten prompts. Own-runtime chat records include `generated_tokens`, the complete sampled output IDs without the stop token, including the last unprocessed ID at a response limit. Failed CLI turns omit this field. The Verification SPI returns the last request admitted to execution after request-only validation: empty before decoding, complete on success, and the current partial samples on decode failure or cancellation. Request-only validation failures are rejected before the execution lock and leave the previous snapshot, processed state and conversation tracking unchanged. Framework rejection before the executor runs also leaves the snapshot unchanged. Turn and tool-step records include `reload_seconds` when present, and memory records include graphics footprint bytes (the graphics footprint ledger plus its compressed ledger). `chat --check-state` checks each reloaded turn and the final state.

```bash
COREAI_TEST_BUNDLES=/path/to/bundle swift test --filter ModelIntegrationTests
COREAI_TEST_BUNDLES=/path/to/bundle COREAI_TEST_RELOAD_INTERVAL=0 COREAI_TEST_RELOAD_FOOTPRINT_BYTES=1 swift test --filter ModelIntegrationTests.twoSessionsShareOneModel
```

The low-threshold concurrency run requires at least one observed reload. Model-enabled results are required in addition to the CPU policy and fake-engine tests.

For a memory-growth measurement, add `--memory-sample-turn 10 --memory-sample-turn 20 --memory-sample-turn 30` to an own-runtime `chat` command with thirty prompts. The option is off by default. At each selected turn, the original immediate `turn` record is emitted first; the same model is then kept alive without further inference for five seconds, followed by a separate `memory-sample` record with the turn number, `wait_seconds`, actual `elapsed_seconds` through the memory capture, and the usual memory fields. Any requested state check runs afterward, including the final check. Duplicate or nonexistent turn numbers are rejected before loading.

The five-second interval is a measurement convention, not a guarantee that memory has stabilized or reclamation has completed on every device. Keep the immediate values and peak alongside the delayed samples. The footprint threshold calibration still uses the maximum immediate post-turn footprint, multiplied by 1.5 and rounded up to a decimal GB. CPU-only checks for the observation option and its wait/capture behavior run with `swift test --package-path Tools/Verify`.

### Qwen3.8 27B bundle

These checks used the model file of the bundle in [Models](#models), in the environment above, on October 7–12, 2026.

- **Same output as Apple's runtime.** `coreai-verify compare --max-tokens 1024` with four prompts (Japanese, English, code, and a long generation) produced the same token at every step as Apple's sequential engine on the matching `prefill-function` path, 43, 41, 293 and 1,024 steps, with a maximum logit difference of 0 and no non-finite values. Both runtimes ran the same converted bundle, so this checks the runtime, not the conversion.
- **Agreement with the original weights did not meet the bar set before measuring.** The reference was PyTorch running the same 4-bit weights in fp32 on the CPU. On 40 inputs not used before (Japanese, English, Hindi, Thai and code; prose, and chat renders shaped like Agent and RAG prompts), 512 positions each, Core AI ranked the reference's top token first at 99.21% of 20,480 positions, and the probability given to the actual next token differed by 0.002435 on average, 0.02796 at the 99th percentile and 0.778 at most; 182 positions (0.8887%) differed by more than 0.03, 26 of them with a different top token. No value was NaN or infinite. The mean and the top-1 rate met their criteria (at most 0.01, at least 99%); the 99th percentile (at most 0.02), the share above 0.03 (at most 0.5%, keeping the top token at each) and the maximum (at most 0.0875) did not. The larger differences sit in English prose and in the Agent- and RAG-shaped renders, at copied URLs, version numbers and the probability of ending a message; Japanese, Hindi and Thai prose had 1 position above 0.03 in 9,216. Core AI's probabilities also depend on how the prompt is split into calls: reading 10 of these inputs in 128-token calls instead of one 512-token call changed the top token at 77 of 5,120 positions. Greedy generation continued from 31 outlying positions stayed coherent in the assistant's own turns, while 4–5 continuations of pasted source text went wrong: a stray Latin letter in Japanese text, broken code, an end of message inside code.
- **The effect of 4-bit quantization was measured on token probabilities only.** On five earlier inputs (the Universal Declaration of Human Rights in Japanese, English, Hindi and Thai, and Swift source; 512 positions each, October 7, 2026), the original bfloat16 model run in PyTorch on the GPU served as the reference. This bundle's model file in Core AI ranked the same top token at 91.48% of the 2,560 positions, with a mean difference of 0.03806 (maximum 0.9990) in the probability of the actual next token. The same 4-bit weights run in PyTorch bfloat16 gave 91.17%, a mean of 0.03862 and a mean KL divergence of 0.0739. On the same inputs, Core AI and PyTorch running those 4-bit weights in fp32 differed by a mean of 0.0019. These are reference values with no pass/fail criterion. Task quality with and without quantization was not compared.
- **Task results are a separate check.** The 48/48 Agent runs and the guard-on runs in [Key numbers](#key-numbers) measure finished tasks; they do not show numerical agreement with the original model. Two graders (the same AI model with the same written guide, not independent people) read the 48 frozen-instruction answers against the text the agent had received: 22 supported, 15 partly supported, 11 not supported, 48 meeting the request. Ten of the 11 unsupported answers relied on search snippets without opening a page.
- **State and sampling.** The processed-state check passed in all 9 checked tool-loop runs with this bundle, and seeded top-k and top-p sampling repeated the same tokens 5 of 5 times, with a different seed changing them.

## Operational notes

- **The first load prepares the model for the GPU.** Core AI keeps the prepared copy in `~/Library/Caches/coreai-cache/<OS build>/<name>/`, where `<name>` is the process name of a command-line tool (`coreai-chat`, `coreai-verify`, `swiftpm-testing-helper` for `swift test`) or the bundle ID of an app (`com.coreai-samples.DemoApp`). Later loads by the same program reuse it; another program prepares its own copy, which takes about as much disk as the model (more for an unquantized bundle). `coreai-chat --stats` shows whether the copy was reused (`cached`) and how long preparing took (`specialize_s`).
- **To remove a prepared copy**, call `Engine.deleteSpecializationCache(bundle:)`, which removes the calling program's copy of that bundle, use **Clear Prepared Copy** in the demo app's Models screen (`--clear-prepared` in its screen check), or delete the folder.
- **Qwen3.8 27B, first load.** The app prepared the bundle in 82.84 s and finished loading in 86.87 s the first time; loads from the prepared copy took 3.6–10.3 s (October 11, 2026). `coreai-chat` makes its own prepared copy of 16.86 GB (15.70 GiB), which took 105.6 s while other builds kept the Mac busy (1-minute load average 20–29).
- **Slow downloads.** If `hf download` stays below 1 MB/s, raising hf-xet's download concurrency helped in one measurement: with `HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY=32 HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY=32`, this bundle downloaded at 18.91 MB/s (891 s), where the default settings had run at 0.2–0.5 MB/s (hf-xet 1.5.2, huggingface_hub 0.36.2, October 11, 2026).
- **Long sessions.** In 100 consecutive new conversations through `coreai-verify chat` with Qwen3.8 27B (up to 256 tokens each), decode speed stayed at 20.11–21.62 tok/s from the 41st conversation on, down from 22.33–23.31 tok/s in the first ten, except for the first question after each reload. Resident memory grew 0.28% from the 10th to the 100th conversation, and each periodic reload took 4.31–4.77 s. The cause of the slowdown has not been identified.

Configure the library with `LocalLanguageModel.ReloadPolicy(readInterval:footprintLimitBytes:)`, passed as `reloadPolicy:` to the model initializer. An interval of zero disables periodic reloads; a footprint limit of zero disables the memory trigger.

The defaults are ten completed reads and a process-footprint trigger of 18,000,000,000 bytes (18 decimal GB). The threshold was calibrated on October 9, 2026 on an M4 Max with 128 GiB, macOS 27.0 (26A428) and Xcode 27, using the Qwen3.8-27B 4bit dynamic `gdnfp32` variant. The `M4-h36-memory-v2` run starting at 2026-10-09 13:03:12 UTC (run ID `3cc72f3295ab`) used thirty prompts, a new conversation every ten turns, context limit 16,384, up to 64 generated tokens per turn, interval ten and no footprint trigger. Its maximum of all thirty immediate post-turn footprints was 11,937,577,240 bytes; multiplying by 1.5 and rounding up to a decimal GB gives 18,000,000,000 bytes. The delayed samples were not used in that calculation. This is a measured reload trigger, not a safe memory budget for every device.

## Limitations

- Agent page fetches and Web RAG require every resolved address to be public. They refuse loopback, private, link-local, unspecified, shared, multicast and reserved addresses, including IPv4-mapped IPv6 and legacy numeric IPv4 forms. Checks run before the first request, every URLSession redirect, every main-frame or subframe WebView navigation, and on the final URL; WebView content rules also block local names and non-public literal addresses in subresources. Failed resolution refuses the fetch. DNS rebinding between validation and connection remains possible, as do JavaScript or subresource requests to names that resolve to private addresses. The app runs without the App Sandbox.

- Run on macOS only (Apple silicon, macOS 27.0). `CoreAISamplesKit/Package.swift` declares iOS 27, but nothing has been run on iOS. The demo app's iPhone target (`CoreAISamples-iOS`, Extract only, see [docs/demo-app.md](docs/demo-app.md#extract-on-iphone)) builds but has not been run on a device yet.
- The language-generation runtime has been run with the converter's macOS exports (`*_dynamic` folders) for Qwen3, Qwen3.5 and Qwen3.8. The stop tokens, the `<think>` tags and the tool-call forms above are the ones those templates use; other model families may need more. Clef uses the separate `DecisionModel` API.
- The language-generation runtime accepts text: a transcript segment that is not text throws `LanguageModelError.unsupportedTranscriptContent`. Recipes uses a separate Apple on-device model for its photo input; the Clef bundle itself is text only.
- No guided generation: a request with a response schema throws `LanguageModelError.unsupportedCapability`.
- The demo uses a context limit of 16,384 tokens. Contexts beyond that limit are outside the verification coverage.
- Use one active session per model unless the simultaneous-session checks pass for the deployed bundle and runtime.
- Cancellation is checked before the prompt is read and between generated tokens, not while a long prompt is being read.
- Decoding is synchronous, one token at a time, like the official `sequential` engine; the official default engine pipelines the work and decodes faster, most of all on small models (`coreai-verify speed` runs the three side by side).
- The separately distributed Clef bundle (downloading it requires access) works with Decide and Recipes. Chat and Work require separate language-generation bundles. The converter revision used for the Qwen3.5 and Qwen3.8 measurements is linked in [Convert your own bundle](#convert-your-own-bundle); it comes from a pull request that is not part of the 1.0.0 tag.
- The Qwen3.8 27B conversion did not meet the agreement criteria fixed before it was measured ([Verification](#qwen38-27b-bundle)): at a small share of positions, Core AI's probabilities differ from the same weights in PyTorch, most of all in English prose and in chat-rendered tool results and sources.
- Under an Agent answer, the **Citations** note counts a cited URL as retrieved when it appeared in the task's search results or in a fetched page, so `4/4 cited URLs were retrieved in this task` was shown for a task that opened no page. The citation guard checks against the pages `fetch_page` delivered instead.
- The citation guard, off by default, checks that the cited pages were fetched in the task, not that the answer agrees with them. URLs whose fetch failed are reported only as a count. Characters that follow a URL without a space, such as Japanese text, are read as part of that URL.
- With the guard on, 1 of 42 live-web runs ended held. Its repair turn fetched the page that the first answer had cited without opening, but the repaired answer stopped at the token limit, so the app kept the first answer and its check, which lists that page as not fetched, and added a note saying so. The check fixed beforehand for making the guard the default counts such a run as citing an unfetched page and allowed none, so the default is off.
- `fetch_page` fails with `error: fetch_page needs an absolute http or https URL` on some pages whose embedded frames, often advertisements, navigate to a non-web address, even when the page URL is correct (4 of the 48 frozen benchmark runs).
- The Agent can follow instructions written in a page it reads. Of 10 runs over 5 pages written to test this (citation guard on), 2 saved a note a page asked for; none fetched the URL a page asked for or wrote outside the notes folder. The tools write only into the notes folder unless **Allow file operations** is on, which adds `move_note`.
- `move_note` also accepts the notes folder as a destination, while its description to the model and its error message name only Desktop, Documents and Downloads.
- When the research step limit is reached, Chat and Work make one extra answer attempt without tools. The attempt uses the existing output-token limit and can still fail or exhaust the context; the app retains the research and shows the actual outcome. This path has CPU integration coverage with Foundation Models; its answer quality with Qwen3.8 27B remains unmeasured.
- Not measured with Qwen3.8 27B: the default Agent settings (cite-fetched instructions, guard off), task quality with and without 4-bit quantization, Macs other than an M4 Max with 128 GiB, macOS builds other than 27.0 (26A428), and contexts beyond 16,384 tokens.

## Related

- [coreml-llm-samples](https://github.com/oka-yuji/coreml-llm-samples) — the Core ML line, for macOS 26 / iOS 26.

## License

Code: MIT — see [LICENSE](LICENSE). Model weights are distributed separately under their own licenses: the Qwen3.8 27B and Clef-Flash 9B bundles keep the Apache License 2.0 of their source models, and each Hugging Face repository includes the `LICENSE` and a `NOTICE.md` that lists the changes.
