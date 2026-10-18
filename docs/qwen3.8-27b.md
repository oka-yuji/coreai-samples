# Chat and Work with Qwen3.8 27B

[Models and roadmap](../README.md#models-and-roadmap) · [日本語](qwen3.8-27b.ja.md)

Run local chat, web research, document retrieval and note-taking on a Mac with the 4-bit, text-only Qwen3.8 27B bundle. Recorded measurements use **Agent** for Work's tool-running path.

Requirements: Apple silicon Mac, macOS 27, and Xcode 27 with the Metal Toolchain component. Allow 16.85 GB for the download and about the same again for each program's prepared copy. Loading reached 34.2 GB (31.9 GiB) peak RSS on an M4 Max with 128 GiB; a minimum-memory configuration has not been established. This bundle is not for iPhone or iPad.

**Distribution status (2026-10-19):** the bundle exists on Hugging Face but is currently private. The download below requires `hf auth login` with an account that has access. The app's anonymous download cannot access it; open an already downloaded folder instead.

Before building the app, complete the [common setup](../README.md#get-started), including Rust/Cargo.


<!-- model-report:start -->
<a id="model-report"></a>

## Evaluation summary

[Measurement protocol](benchmarking.md) · [Source record](model-reports/records/qwen38-27b.json)

Results below retain their original dates and workloads. Different rows may use different runs; they are not a controlled comparison. Legacy records explicitly retain unrecorded conditions. Missing values are not zero or a pass.

### Demo support

| Platform | State | Verified scope / remaining work | Evidence |
|---|---|---|---|
| macos | ✅ | Recorded Mac app Chat/Work route, 4-bit text bundle; broader quality remains limited. | [E1](qwen3.8-27b.md#key-numbers) |
| ios | — | This 27B bundle is not an iPhone/iPad demo. | — |

### Performance and quality

| Metric | Result | Statistic / samples | Scope and evidence |
|---|---|---|---|
| Bundle / payload size | 16,852,543,780 B | single; n=1 | [E3](qwen3.8-27b.md#bundle) · Fixed workload in the linked evidence. |
| Prepared cache size | 16.86 GB | single; n=1 | [E4](qwen3.8-27b.md#operational-notes) · Rounded CLI prepared-copy size; each executable prepares its own copy. |
| First load, including specialization | 86.87 s | single; n=1 | [E1](qwen3.8-27b.md#key-numbers) · One first app load; includes 82.84 s of specialization. |
| Load from prepared cache | 4.7 s | median; n=48 | [E1](qwen3.8-27b.md#key-numbers) · Prepared-cache app loads; observed range 3.6–10.3 s. |
| Request / pipeline completion time | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Peak RSS | 34.2 GB | single; n=1 | [E1](qwen3.8-27b.md#key-numbers) · Rounded first-load process RSS; not a minimum device RAM requirement. |
| Peak physical footprint | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Resident memory during inference | 16 GiB | min; n=unrecorded | [E1](qwen3.8-27b.md#key-numbers) · Reported resident-memory range during Agent steps; original sample count not stated. |
| Resident memory during inference | 16.7 GiB | max; n=unrecorded | [E1](qwen3.8-27b.md#key-numbers) · Reported resident-memory range during Agent steps; original sample count not stated. |
| Memory after release minus baseline | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Task success fraction | 1 ratio | aggregate; n=48 | [E1](qwen3.8-27b.md#key-numbers) · 48/48 frozen-grader task successes; separate support audit: 22 supported, 15 partly, 11 unsupported. |
| Reference error (definition in evidence) | 0.778 error | max; n=20480 | [E2](qwen3.8-27b.md#verification) · Maximum next-token probability difference; predeclared maximum 0.0875 was not met. |
| Stop acknowledgement time | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Time to first output token (TTFT) | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Time to first visible text | 6.46 s | median; n=144 | [E1](qwen3.8-27b.md#key-numbers) · First displayed text per generation step, including reasoning; not first final answer. |
| Time to first answer after reasoning | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Decode throughput | 22.74 tokens/s | median; n=144 | [E1](qwen3.8-27b.md#key-numbers) · Fixed workload in the linked evidence. |

### Verification coverage

| Check | State | Scope / remaining work | Evidence |
|---|---|---|---|
| Conversion / reference fidelity | Failed (stated criterion) | Some original-weight agreement criteria failed; same-bundle runtime agreement is a separate pass. | [E2](qwen3.8-27b.md#verification) |
| Task quality | Not measured | No completed check for this scope under the common protocol. | — |
| Repeated use / resource release | Not measured | No completed check for this scope under the common protocol. | — |
| Stop / retry / recovery | Not measured | No completed check for this scope under the common protocol. | — |

### Evidence and conditions

| Evidence | Date / record / route level | Device / OS / tools | Workload and recorded conditions |
|---|---|---|---|
| [E1](qwen3.8-27b.md#key-numbers) | 2026-10-12 / legacy / native_app | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | 12 frozen live-web Agent tasks × 4 rounds; 48 runs / 144 generation steps; greedy; reasoning low/off; cite-fetched default not measured. Recorded measurement window: 2026-10-11–12. · variant=4-bit text; backend=Core AI; GPU preferred; route=macOS app --agent-e2e; source_revision=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0; artifact_revision=8e24aebe036eae577d4a3bd019ba728a73358318; batch_size=1; concurrency=1; unrecorded: code_revision, warmups, power, thermal, plan_sha256 |
| [E2](qwen3.8-27b.md#verification) | 2026-10-12 / legacy / diagnostic | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | 40 held-out inputs × 512 positions; same 4-bit weights in CPU PyTorch FP32 vs Core AI. Recorded measurement window: 2026-10-07–12. · variant=4-bit text; backend=Core AI; GPU preferred; route=diagnostic; batch_size=1; concurrency=1; unrecorded: source_revision, artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |
| [E3](qwen3.8-27b.md#bundle) | 2026-10-12 / legacy / conversion | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | See fixed cases in the evidence. · variant=4-bit text; backend=Core AI; GPU preferred; route=conversion; batch_size=1; concurrency=1; unrecorded: source_revision, artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |
| [E4](qwen3.8-27b.md#operational-notes) | 2026-10-12 / legacy / diagnostic | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | First CLI preparation while other builds ran (load average 20–29); separate prepared copy from the app. · variant=4-bit text; backend=Core AI; GPU preferred; route=coreai-chat first preparation; source_revision=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0; artifact_revision=8e24aebe036eae577d4a3bd019ba728a73358318; batch_size=1; concurrency=1; unrecorded: code_revision, warmups, power, thermal, plan_sha256 |

### Setup, limitations and progress

- [Chat and Work](qwen3.8-27b.md)
<!-- model-report:end -->

## Quick start

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), then get the app and the bundle (12 files, 16.85 GB):

```sh
git clone https://github.com/oka-yuji/coreai-samples.git
cd coreai-samples
hf download okayuji/Qwen3.8-27B-coreai-4bit --revision 8e24aebe036eae577d4a3bd019ba728a73358318 \
  --local-dir ./models/qwen3.8-27b-coreai-4bit
```

Open `CoreAISamples.xcodeproj`, choose the **CoreAISamples** scheme and **My Mac**, and run; the scheme builds Release. In **Models**, choose **Open Folder…**, select `models/qwen3.8-27b-coreai-4bit` and press **Load**. The first load took 86.87 s on an M4 Max, 82.84 s of it preparing the model for the GPU; later loads took 3.6–10.3 s. The conversation title shows the model name; use **Chat** mode for conversation or **Work** mode for tasks and notes. Enable **Search and sources → Web** in the conversation settings when you want web research. **Models** has a downloader for the same pinned revision, but it requires anonymous access to the repository; it is unavailable while the bundle is private.

Chat defaults to 16 generation steps per reply; Work keeps its separate default of eight. In the conversation settings, **Max steps per reply** changes the selected mode's limit from 1 to 64. Each new question starts a fresh budget. If research exhausts it, the app makes one extra answer attempt with tools and reasoning disabled, using the information already collected. **Stop** still cancels the request. See the [budget contract and checks](specs/conversation-step-budget.md).

From the command line, in the `CoreAISamplesKit` folder:

```sh
cd CoreAISamplesKit
swift build -c release
.build/release/coreai-chat --model ../models/qwen3.8-27b-coreai-4bit --max-tokens 2048 --stats \
  --prompt "What is the capital of France? Answer with one word."
```

The model thinks before it answers: the thinking prints under `[reasoning]` and the answer under `[response]`, and `--stats` reports the load and each turn on stderr.

### Bundle

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

## Verification

These checks used the model file of the bundle in [Bundle](#bundle), in the environment above, on October 7–12, 2026.

- **Same output as Apple's runtime.** `coreai-verify compare --max-tokens 1024` with four prompts (Japanese, English, code, and a long generation) produced the same token at every step as Apple's sequential engine on the matching `prefill-function` path, 43, 41, 293 and 1,024 steps, with a maximum logit difference of 0 and no non-finite values. Both runtimes ran the same converted bundle, so this checks the runtime, not the conversion.
- **Agreement with the original weights did not meet the bar set before measuring.** The reference was PyTorch running the same 4-bit weights in fp32 on the CPU. On 40 inputs not used before (Japanese, English, Hindi, Thai and code; prose, and chat renders shaped like Agent and RAG prompts), 512 positions each, Core AI ranked the reference's top token first at 99.21% of 20,480 positions, and the probability given to the actual next token differed by 0.002435 on average, 0.02796 at the 99th percentile and 0.778 at most; 182 positions (0.8887%) differed by more than 0.03, 26 of them with a different top token. No value was NaN or infinite. The mean and the top-1 rate met their criteria (at most 0.01, at least 99%); the 99th percentile (at most 0.02), the share above 0.03 (at most 0.5%, keeping the top token at each) and the maximum (at most 0.0875) did not. The larger differences sit in English prose and in the Agent- and RAG-shaped renders, at copied URLs, version numbers and the probability of ending a message; Japanese, Hindi and Thai prose had 1 position above 0.03 in 9,216. Core AI's probabilities also depend on how the prompt is split into calls: reading 10 of these inputs in 128-token calls instead of one 512-token call changed the top token at 77 of 5,120 positions. Greedy generation continued from 31 outlying positions stayed coherent in the assistant's own turns, while 4–5 continuations of pasted source text went wrong: a stray Latin letter in Japanese text, broken code, an end of message inside code.
- **The effect of 4-bit quantization was measured on token probabilities only.** On five earlier inputs (the Universal Declaration of Human Rights in Japanese, English, Hindi and Thai, and Swift source; 512 positions each, October 7, 2026), the original bfloat16 model run in PyTorch on the GPU served as the reference. This bundle's model file in Core AI ranked the same top token at 91.48% of the 2,560 positions, with a mean difference of 0.03806 (maximum 0.9990) in the probability of the actual next token. The same 4-bit weights run in PyTorch bfloat16 gave 91.17%, a mean of 0.03862 and a mean KL divergence of 0.0739. On the same inputs, Core AI and PyTorch running those 4-bit weights in fp32 differed by a mean of 0.0019. These are reference values with no pass/fail criterion. Task quality with and without quantization was not compared.
- **Task results are a separate check.** The 48/48 Agent runs and the guard-on runs in [Key numbers](#key-numbers) measure finished tasks; they do not show numerical agreement with the original model. Two graders (the same AI model with the same written guide, not independent people) read the 48 frozen-instruction answers against the text the agent had received: 22 supported, 15 partly supported, 11 not supported, 48 meeting the request. Ten of the 11 unsupported answers relied on search snippets without opening a page.
- **State and sampling.** The processed-state check passed in all 9 checked tool-loop runs with this bundle, and seeded top-k and top-p sampling repeated the same tokens 5 of 5 times, with a different seed changing them.

## Operational notes

- **Qwen3.8 27B, first load.** The app prepared the bundle in 82.84 s and finished loading in 86.87 s the first time; loads from the prepared copy took 3.6–10.3 s (October 11, 2026). `coreai-chat` makes its own prepared copy of 16.86 GB (15.70 GiB), which took 105.6 s while other builds kept the Mac busy (1-minute load average 20–29).

- **Slow downloads.** If `hf download` stays below 1 MB/s, raising hf-xet's download concurrency helped in one measurement: with `HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY=32 HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY=32`, this bundle downloaded at 18.91 MB/s (891 s), where the default settings had run at 0.2–0.5 MB/s (hf-xet 1.5.2, huggingface_hub 0.36.2, October 11, 2026).

- **Long sessions.** In 100 consecutive new conversations through `coreai-verify chat` with Qwen3.8 27B (up to 256 tokens each), decode speed stayed at 20.11–21.62 tok/s from the 41st conversation on, down from 22.33–23.31 tok/s in the first ten, except for the first question after each reload. Resident memory grew 0.28% from the 10th to the 100th conversation, and each periodic reload took 4.31–4.77 s. The cause of the slowdown has not been identified.

## Limitations

The runtime-wide [input, context and cancellation limits](runtime.md#limitations) also apply.

- Agent page fetches and Web RAG require every resolved address to be public. They refuse loopback, private, link-local, unspecified, shared, multicast and reserved addresses, including IPv4-mapped IPv6 and legacy numeric IPv4 forms. Checks run before the first request, every URLSession redirect, every main-frame or subframe WebView navigation, and on the final URL; WebView content rules also block local names and non-public literal addresses in subresources. Failed resolution refuses the fetch. DNS rebinding between validation and connection remains possible, as do JavaScript or subresource requests to names that resolve to private addresses. The app runs without the App Sandbox.

- This Qwen bundle and Chat/Work were verified on Apple silicon with macOS 27.0. iOS support for other models is separate.

- The Qwen3.8 27B conversion did not meet the agreement criteria fixed before it was measured ([Verification](#verification)): at a small share of positions, Core AI's probabilities differ from the same weights in PyTorch, most of all in English prose and in chat-rendered tool results and sources.

- Under an Agent answer, the **Citations** note counts a cited URL as retrieved when it appeared in the task's search results or in a fetched page, so `4/4 cited URLs were retrieved in this task` was shown for a task that opened no page. The citation guard checks against the pages `fetch_page` delivered instead.

- The citation guard, off by default, checks that the cited pages were fetched in the task, not that the answer agrees with them. URLs whose fetch failed are reported only as a count. Characters that follow a URL without a space, such as Japanese text, are read as part of that URL.

- With the guard on, 1 of 42 live-web runs ended held. Its repair turn fetched the page that the first answer had cited without opening, but the repaired answer stopped at the token limit, so the app kept the first answer and its check, which lists that page as not fetched, and added a note saying so. The check fixed beforehand for making the guard the default counts such a run as citing an unfetched page and allowed none, so the default is off.

- `fetch_page` fails with `error: fetch_page needs an absolute http or https URL` on some pages whose embedded frames, often advertisements, navigate to a non-web address, even when the page URL is correct (4 of the 48 frozen benchmark runs).

- The Agent can follow instructions written in a page it reads. Of 10 runs over 5 pages written to test this (citation guard on), 2 saved a note a page asked for; none fetched the URL a page asked for or wrote outside the notes folder. The tools write only into the notes folder unless **Allow file operations** is on, which adds `move_note`.

- `move_note` also accepts the notes folder as a destination, while its description to the model and its error message name only Desktop, Documents and Downloads.

- When the research step limit is reached, Chat and Work make one extra answer attempt without tools. The attempt uses the existing output-token limit and can still fail or exhaust the context; the app retains the research and shows the actual outcome. This path has CPU integration coverage with Foundation Models; its answer quality with Qwen3.8 27B remains unmeasured.

- Not measured with Qwen3.8 27B: the default Agent settings (cite-fetched instructions, guard off), task quality with and without 4-bit quantization, Macs other than an M4 Max with 128 GiB, macOS builds other than 27.0 (26A428), and contexts beyond 16,384 tokens.
