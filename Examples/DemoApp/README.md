# Demo app (macOS)

`Examples/DemoApp` is a SwiftUI app that runs a local model bundle through Foundation Models with the runtime in this repository (`CoreAIBackend`). It does not use the official `coreai-models` package.

- **Models**: open a bundle folder on this Mac (the app remembers it), load it, and clear the copy the app prepared for the GPU. The download catalog for Hugging Face is in place but stays empty until bundles are published.
- **Chat**: a `LanguageModelSession` on `LocalLanguageModel`. Replies stream, thinking streams into a collapsed section, and the status line shows the time to the first token, the speed, the input, reused and output tokens, and why the reply stopped. With **RAG** on, the app first collects sources from the web and from `~/Documents/rag/` and the model answers from them with numbered citations. See [RAG](#rag) below.
- **Agent**: searches the web, reads pages, saves notes and answers with a `Sources:` section, on the model that Chat has loaded. See [Agent](#agent) below.

## Requirements

- [XcodeGen](https://github.com/yonaskolb/XcodeGen), only to regenerate the demo app's project

## Build and run

```bash
cd Examples/DemoApp
xcodegen generate   # only needed after editing project.yml; the generated project is committed
xcodebuild -scheme DemoApp -configuration Release -derivedDataPath .build build
open .build/Build/Products/Release/DemoApp.app
```

The app is signed ad hoc and runs without the App Sandbox. To sign with your team, copy `Local.xcconfig.template` to `Local.xcconfig` and set `DEVELOPMENT_TEAM`. The context window is 16,384 tokens (or the bundle's maximum, if smaller) for Chat, RAG and Agent; launch with `--context-limit N` to change it, for example `open .build/Build/Products/Release/DemoApp.app --args --context-limit 1024`.

The first load of a model prepares it for the GPU (about 21 s for Qwen3.5-9B on an M4 Max); later loads reuse the prepared copy in `~/Library/Caches/coreai-cache/<OS build>/com.coreai-samples.DemoApp/`.

## Headless self-test

```bash
APP=.build/Build/Products/Release/DemoApp.app/Contents/MacOS/DemoApp
$APP --selftest --model /path/to/bundle --prompts-file prompts.txt [--max-tokens 2048] [--context-limit 4096]
```

- `prompts.txt` holds the turns of one conversation, separated by lines that read `===PROMPT===`.
- Each turn prints one JSON line: `turn`, `answer`, `inputTokens`, `cachedTokens` (reused), `outputTokens`, `ttftSeconds`, `tokensPerSecond` (output tokens divided by the decode time) and `stopReason`. The last line is `SELFTEST PASS` or `SELFTEST FAIL`, and the exit status is 0 or 1.
- A pass means every turn finished without an error, generated tokens, stopped on a stop token or the token limit, and every turn after the first reused tokens. No window opens.

## Screen check

```bash
$APP --gui-e2e /path/to/output --model /path/to/bundle [--clear-prepared] [--context-limit 1024] [--max-tokens N]
```

The app opens the bundle in Models, loads it, chats one turn, starts a long reply, stops it and starts a new chat, saving a screenshot after each step with `results.json`, `driver.log` and the metrics rows of the run. `--clear-prepared` removes the prepared copy first, so the load shows the preparing state. With `--context-limit`, it then keeps chatting until the context window is full.

## Metrics

Every generation appends one JSON line to `~/Library/Application Support/com.coreai-samples.DemoApp/metrics.jsonl`: the model, the turn, the input, reused and output tokens, the time to the first token, the speed, the stop reason, the resident memory and the app build. RAG answers also carry `sources`, `sourceTokens`, `ragPromptTokens` and `retrievalSeconds`.

## RAG

Turn on **RAG** in the Chat status bar, then pick **Web**, **Docs** or both. The app retrieves; the model never calls a tool.

1. **Web**: search (DuckDuckGo, top 8; Brave Search when the Agent settings hold an API key), then read the top 5 pages in parallel and keep up to 3,000 characters of each, the part that matches the question.
2. **Docs**: read every `.txt`, `.md` (also `.text`, `.markdown`) and `.pdf` file in `~/Documents/rag/`. The folder is created on first use and read again on every question, so there is no index to rebuild. PDFs are read with PDFKit.
3. Split pages and documents into passages of about 500 characters that repeat the last 100 characters of the previous one, rank them against the question with SQLite FTS5 (BM25), and keep the best ones up to 6,000 tokens, so that the whole prompt stays within 8,000 tokens (counted with the model's tokenizer, `LocalLanguageModel.countTokens(_:)`).
4. Answer in a new `LanguageModelSession` per question: the instructions are a fixed Japanese instruction with today's date, and the prompt is the sources as `資料[n] title / url / date` plus the passage, then the question. RAG turns are not added to the chat history, and the reasoning setting is Chat's.

The answer is followed by its sources, numbered like the `[n]` marks in the text (`[n] title — url (date)`; `日付不明` when the page gives no date), and one line for each source that failed (no network, no search results, a page that would not load). With no sources at all, the reply is `資料が取得できませんでした。` and the model does not run. The status line adds `sources N | source tok N | prompt(rag) N tok | retrieval N.Ns`.

A page's date comes from its `article:published_time` and similar meta tags, then a `<time datetime>` tag, then the search snippet, then the first 600 characters of the page text. The splitting, ranking, budget, prompt and dates are in the `RAGContext` library of this package, with no network, files or UI (`swift test --filter RAG`).

Headless, with no window:

```bash
$APP --selftest --model /path/to/bundle --rag "question 1|question 2" [--rag-web off] [--rag-local off] \
  [--rag-docs /path/to/folder] [--max-tokens N]
```

Each question prints `[rag] turn=N search N / pages N / passages N → N sources, N tokens`, one `[rag] turn=N source [k] title — url (date)` and one `[rag] turn=N passage [k] …` (the text the model saw) line per source, `[rag] turn=N note …` for each failure and `[rag] turn=N answer=…` on stderr, then its JSON line and `SELFTEST PASS` or `SELFTEST FAIL` on stdout. A question with sources passes when the model answers without an error; a question without sources passes when the reply is `資料が取得できませんでした。` and the model did not run. `--rag-docs` reads another folder instead of `~/Documents/rag/`. `$APP --gui-e2e /path/to/output --model /path/to/bundle --rag "question"` turns RAG on and answers one question on the screen, saving screenshots of the switches and of the answer with its sources and the status line.

## Agent

The loop belongs to Foundation Models: the screen runs `LanguageModelSession(model:tools:instructions:)` with four tools and streams the response, and Foundation Models calls the tools and generates again until the model answers. The app counts the steps (one step is one generation) by watching the transcript.

| Tool | Arguments | Returns |
|---|---|---|
| `web_search` | `query`, `max_results` (1–10, default 5) | title, url and snippet per result (DuckDuckGo, or Brave Search with an API key) |
| `fetch_page` | `url`, `query` (optional) | the page's readable text from a `WKWebView`, or the passages that match `query`, up to **Page budget** characters |
| `write_note` | `title`, `content` | `saved to <path>` (Markdown in the notes folder; a taken name gets ` 2`, ` 3`, …) |
| `move_note` | `title`, `destination` | `moved to <path>`, only into Desktop, Documents, Downloads or a folder under them |

- A tool never throws: a failure comes back to the model as one line that starts with `error: `. Foundation Models ends the whole turn when a tool throws.
- Foundation Models runs the calls of one response at the same time and adds the results in call order. There is no limit on the number of calls per response.
- `move_note` is offered only when **Allow file operations** was on when the task started; the tools of a task are fixed when its session starts.
- The instructions carry today's date as `yyyy-MM-dd (EEE)` and ask the model to retry a failed call once in a different way and to end the answer with `Sources:`. **Max steps** (default 8) cancels the task with "Step limit reached". When the context fills up, the runtime drops the oldest tool results (the newest two are kept) or shortens the newest one, and the screen shows a "Context full" note; if the prompt still does not fit, the task ends with "Context window is full".
- Settings, kept in `UserDefaults`: reasoning effort (Off, Low, Medium, Xhigh; default Low), max steps (1–16), max tokens per step (default 1,536), page budget (500–20,000 characters, default 10,000), Brave Search API key (stored in the clear; empty means DuckDuckGo), allow file operations, notes folder.

Headless, with no window:

```bash
$APP --agent-e2e --model /path/to/bundle --task "…" [--effort off|low|medium|xhigh] [--max-steps N] \
  [--page-budget N] [--max-tokens N] [--deadline SECONDS] [--file-ops] [--notes-dir /path] [--context-limit N]
$APP --agent-selftest [--offline]
```

- `--agent-e2e` prints each entry as `--- <kind> <title>` and its text, `[e2e] …` progress and status lines, `[e2e] outcome: <answered|step-limit|overflow|cap|stopped|failed>`, and `AGENT E2E COMPLETED  diverges=N  capSteps=N` (exit 0) or `AGENT E2E INCOMPLETE …` (exit 1). `diverges` counts steps whose prompt did not extend the previous step's prompt and output (the line `[agent] re-render diverges` goes to stderr); `capSteps` counts steps that stopped at the token limit.
- `--agent-selftest` checks the instructions, the tool definitions and arguments, the note and move rules, the page passage selection, the search and HTML parsers, the citation check, the verdict table and the transcript format. `--offline` skips the network cases.
- `$APP --gui-e2e /path/to/output --model /path/to/bundle --agent` drives the Agent screen instead of Chat: it sends a task, waits for the tool calls and results and the answer with `Sources:`, stops a second task, and starts a new task, saving a screenshot after each step.

Logs: `metrics.jsonl` gets one `message` line per step (prompt, reused and generated tokens, time to the first token, prefill time, speed, finish reason, resident memory, the step's tool calls, whether it diverged, citations on the answer) and one `agent-tool` line per tool call (tool, argument, start and end time, seconds, result size, error). Each task is saved as Markdown in `~/Library/Application Support/com.coreai-samples.DemoApp/agent-transcripts/<session>.md`; notes go to `…/agent-notes/` unless another folder is set.
