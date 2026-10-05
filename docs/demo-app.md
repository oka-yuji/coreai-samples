# Demo app (macOS)

`CoreAISamples` (`CoreAISamples.xcodeproj` and the `CoreAISamples` folder at the repository root) is a SwiftUI app that runs a local model bundle through Foundation Models with the runtime in this repository (`CoreAIBackend`). It does not use the official `coreai-models` package.

- **Models**: open a bundle folder on this Mac (the app remembers it), load it, and clear the copy the app prepared for the GPU. The download catalog for Hugging Face is in place but stays empty until bundles are published.
- **Chat**: a `LanguageModelSession` on `LocalLanguageModel`. Replies stream, thinking streams into a collapsed section, and the status line shows the time to the first token, the speed, the input, reused and output tokens, and why the reply stopped. With **RAG** on, the app first collects sources from the web and from `~/Documents/rag/` and the model answers from them with numbered citations. See [RAG](#rag) below.
- **Agent**: searches the web, reads pages, saves notes and answers with a `Sources:` section, on the model that Chat has loaded. See [Agent](#agent) below.

## Requirements

- [XcodeGen](https://github.com/yonaskolb/XcodeGen), only to regenerate the demo app's project

## Build and run

In Xcode, open `CoreAISamples.xcodeproj` at the repository root, choose the `CoreAISamples` scheme and `My Mac`, and press Run. The `CoreAISamples` scheme runs a Release build and `CoreAISamples-Debug` a Debug build. From the command line, at the repository root:

```bash
xcodegen generate   # only needed after editing project.yml; the generated project is committed
xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples -configuration Release -derivedDataPath .build build
open .build/Build/Products/Release/CoreAISamples.app
```

The app is signed ad hoc and runs without the App Sandbox. To sign with your team, copy `Config/Local.xcconfig.template` to `Config/Local.xcconfig` and set `DEVELOPMENT_TEAM`; the template also switches to automatic signing with an Apple Development certificate. Choosing a team in Xcode's Signing & Capabilities instead writes it into the shared project file. A `Local.xcconfig` made in an earlier checkout under `Examples/DemoApp/` is not moved by git; move it to `Config/Local.xcconfig`. The context window is 16,384 tokens (or the bundle's maximum, if smaller) for Chat, RAG and Agent; launch with `--context-limit N` to change it, for example `open .build/Build/Products/Release/CoreAISamples.app --args --context-limit 1024`.

The first load of a model prepares it for the GPU (about 21 s for Qwen3.5-9B on an M4 Max); later loads reuse the prepared copy in `~/Library/Caches/coreai-cache/<OS build>/com.coreai-samples.DemoApp/`.

## Headless self-test

```bash
APP=.build/Build/Products/Release/CoreAISamples.app/Contents/MacOS/CoreAISamples
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
3. Split pages and documents into passages of about 500 characters that repeat the last 100 characters of the previous one, rank them against the question with SQLite FTS5 (BM25), and keep the best ones up to the source budget, so that the whole prompt stays within 8,000 tokens (counted with the model's tokenizer, `LocalLanguageModel.countTokens(_:)`). The source budget is 3,000 tokens by default (`RAGContext.sourceTokenBudget`); the **Sources** menu next to the RAG switches also offers 6,000, and `--rag-budget 3000|6000` sets it at launch.
4. Answer in a new `LanguageModelSession` per question: the instructions are a fixed Japanese instruction with today's date, and the prompt is the sources as `資料[n] title / url / date` plus the passage, then the question. RAG turns are not added to the chat history, and the reasoning setting is Chat's.

The answer is followed by its sources, numbered like the `[n]` marks in the text (`[n] title — url (date)`; `日付不明` when the page gives no date), and one line for each source that failed (no network, no search results, a page that would not load). With no sources at all, the reply is `資料が取得できませんでした。` and the model does not run. The status line adds `sources N | source tok N | prompt(rag) N tok | retrieval N.Ns`.

A page's date comes from its `article:published_time` and similar meta tags, then a `<time datetime>` tag, then the search snippet, then the first 600 characters of the page text. The splitting, the packing to the token budget, the prompt and its citations, the date parsing, the reading of a documents folder (the app supplies the PDF text) and a term-match ranking are in the `RAGContext` library of the `CoreAISamplesKit` package, with no network or UI, and `swift test --package-path CoreAISamplesKit --filter RAG` tests them; the SQLite FTS5 (BM25) ranking is in the app, which uses the term-match ranking only when the FTS5 ranking returns nothing.

Headless, with no window:

```bash
$APP --selftest --model /path/to/bundle --rag "question 1|question 2" [--rag-web off] [--rag-local off] \
  [--rag-docs /path/to/folder] [--rag-budget 3000|6000] [--max-tokens N]
```

Each question prints `[rag] turn=N search N / pages N / passages N → N sources, N tokens`, one `[rag] turn=N source [k] title — url (date)` and one `[rag] turn=N passage [k] …` (the text the model saw) line per source, `[rag] turn=N note …` for each failure and `[rag] turn=N answer=…` on stderr, then its JSON line and `SELFTEST PASS` or `SELFTEST FAIL` on stdout. A question with sources passes when the model answers without an error; a question without sources passes when the reply is `資料が取得できませんでした。` and the model did not run. `--rag-docs` reads another folder instead of `~/Documents/rag/`. `$APP --gui-e2e /path/to/output --model /path/to/bundle --rag "question"` turns RAG on and answers one question on the screen, saving screenshots of the switches and of the answer with its sources and the status line.

## Agent

The loop belongs to Foundation Models: the screen runs `LanguageModelSession(model:tools:instructions:)` with three tools (four when **Allow file operations** is on, below) and streams the response, and Foundation Models calls the tools and generates again until the model answers. The app counts the steps (one step is one generation) by watching the transcript.

| Tool | Arguments | Returns |
|---|---|---|
| `web_search` | `query`, `max_results` (1–10, default 5) | title, url and snippet per result (DuckDuckGo, or Brave Search with an API key) |
| `fetch_page` | `url`, `query` (optional) | the page's readable text from a `WKWebView`, or the passages that match `query`, up to **Page budget** characters |
| `write_note` | `title`, `content` | `saved to <path>` (Markdown in the notes folder; a taken name gets ` 2`, ` 3`, …) |
| `move_note` | `title`, `destination` | `moved to <path>`, only into Desktop, Documents, Downloads or the notes folder, or a folder under one of them (never the home folder or a folder above it) |

- A tool never throws: a failure comes back to the model as one line that starts with `error: `. Foundation Models ends the whole turn when a tool throws.
- Foundation Models runs the calls of one response at the same time and adds the results in call order. There is no limit on the number of calls per response.
- `move_note` is offered only when **Allow file operations** was on when the task started; the tools of a task are fixed when its session starts.
- The instructions carry today's date as `yyyy-MM-dd (EEE)` and ask the model to retry a failed call once in a different way and to end the answer with `Sources:`. **Max steps** (default 8) cancels the task with "Step limit reached". When the context fills up, the runtime drops the oldest tool results (the newest two are kept) or shortens the newest one, and the screen shows a "Context full" note; if the prompt still does not fit, the task ends with "Context window is full".
- Settings, kept in `UserDefaults`: reasoning effort (Off, Low, Medium, Xhigh; default Low), max steps (1–16), max tokens per step (default 1,536), page budget (500–20,000 characters, default 10,000), Brave Search API key (stored in the clear; empty means DuckDuckGo), allow file operations, citation guard (default off), notes folder.
- Chat and Agent use the same loaded model, so one does not start while the other is generating: the other screen's Send button is off and a line under it says why.

Headless, with no window:

```bash
$APP --agent-e2e --model /path/to/bundle --task "…" [--effort off|low|medium|xhigh] [--max-steps N] \
  [--page-budget N] [--max-tokens N] [--deadline SECONDS] [--file-ops] [--notes-dir /path] [--context-limit N] \
  [--instructions frozen|cite-fetched] [--citation-guard on|off] [--sources-required yes|no|auto] \
  [--date "yyyy-MM-dd (EEE)"] [--stable-note-paths] \
  [--tool-cassette /path/cassette.jsonl --cassette-mode record|replay|replay-or-record [--cassette-latency replay|none]]
$APP --agent-selftest [--offline]
```

- `--agent-e2e` prints each entry as `--- <kind> <title>` and its text, `[e2e] …` progress and status lines, `[e2e] outcome: <answered|held|step-limit|overflow|cap|stopped|failed>`, and `AGENT E2E COMPLETED  diverges=N  capSteps=N` (exit 0) or `AGENT E2E INCOMPLETE …` (exit 1). `diverges` counts steps whose prompt did not extend the previous step's prompt and output (the line `[agent] re-render diverges` goes to stderr); `capSteps` counts steps that stopped at the token limit.
- After the outcome line, every run prints the citation check, `[guard] check stage=answer citations_total=N citations_fetched=N citations_failed=N citations_unfetched=[…] note_citations_unfetched=[…] sources_required=yes|no sources_missing=true|false`, whether the guard is on or off (a second line, `stage=repaired`, after a repair turn). The answer's URLs split into fetched, failed (`fetch_page` tried them and failed) and unfetched (never tried, listed); `note_citations_unfetched` lists the notes' never-tried URLs. With the guard on, `[guard] outcome=verified|repaired|held|skipped reason=… repair_turns=N repair_fetches=N repair_seconds=N repair_prompt_tokens=N repair_generated_tokens=N drafts=N` follows; `skipped` means the task ended without a complete answer. After a repair turn, `[e2e] answer before repair (N chars):` and that answer come before `[e2e] final answer`. A held run prints `[e2e] outcome: held` and still ends with `AGENT E2E COMPLETED` (exit 0).
- `--instructions cite-fetched` adds two lines after the three above: cite only pages opened with `fetch_page` in the task (a search snippet is not a source), and check the row and column headers when quoting a number from a table. `frozen`, the default, keeps the instructions as they are.
- `--citation-guard on` turns the guard on for the run. `--sources-required yes` requires at least one fetched URL in the answer, `no` requires none (a never-tried URL still fails the check), and `auto`, the default, decides as on screen. `--date` replaces today's date in the instructions. `--stable-note-paths` makes `write_note` answer `saved to agent-notes/<file>.md` whatever `--notes-dir` is, so that the model reads the same text in every run; the note is still saved in `--notes-dir`.
- `--tool-cassette` with `--cassette-mode` replays and records `web_search` and `fetch_page` results in a file with one JSON line per result: the key (the tool and its arguments as the tool reads them: the query and the number of results for `web_search`, the url and the query for `fetch_page`), the text, the pages it carries, whether the call succeeded, the final URL, and the milliseconds it took. A recorded key is replayed, the first line of a key winning, after waiting the recorded time (`--cassette-latency none` skips the wait). An unrecorded key is fetched live and appended (`record`, `replay-or-record`), or answered with `error: cassette miss` (`replay`, which never writes). A call that was stopped is not recorded. `write_note` and `move_note` never use the cassette. stderr gets `[cassette] hit|miss|record tool=… key=[…] ms=N`, and the run ends with `[cassette] summary hits=N misses=N records=N`.
- `--agent-selftest` checks the instructions, the tool definitions and arguments, the note and move rules, the page passage selection, the search and HTML parsers, the citation check, the verdict table and the transcript format, and, with no model and no network, the cite-fetched instructions, the guard's URL reader and ledger check, the cassette, the repair fetch slots, the drafts, the stable note paths, the RAG source budget and the bundle errors on the Models screen. `--offline` skips the network cases.
- `$APP --gui-e2e /path/to/output --model /path/to/bundle --agent` drives the Agent screen instead of Chat: it sends a task, waits for the tool calls and results and the answer with `Sources:`, stops a second task, and starts a new task, saving a screenshot after each step.

Logs: `metrics.jsonl` gets one `message` line per step (prompt, reused and generated tokens, time to the first token, prefill time, speed, finish reason, resident memory, the step's tool calls, whether it diverged, `splicedGenerations`, citations on the answer), one `agent-tool` line per tool call (tool, argument, start and end time, seconds, result size, error), and one `agent-run` line per task. `splicedGenerations` counts the earlier generations of the session that the runtime put back token for token into the history it rebuilt for that step (the text was the same but the tokenizer split it differently). The `agent-run` line has the instructions (`agentInstructions`: `frozen` or `cite-fetched`), whether the guard was on (`citationGuard`), the checks above (`ledgerChecks`), the ledger (`fetchLedger`), every `fetch_page` call with whether it succeeded (`fetchAttempts`), `guardOutcome` and `guardReason`, after a repair turn `repairTurns`, `repairFetches`, `repairSeconds`, `repairPromptTokens` and `repairGeneratedTokens`, and with a cassette `cassetteHits`, `cassetteMisses` and `cassetteRecords`. Each task is saved as Markdown in `~/Library/Application Support/com.coreai-samples.DemoApp/agent-transcripts/<session>.md`; notes go to `…/agent-notes/` unless another folder is set.

### Citation guard

With **Citation guard** on, the app checks the sources of the answer before it accepts it. It keeps a ledger of the pages `fetch_page` handed to the model in the session: the URL asked for, the URL the page came from after redirects, and the SHA-256 and length of the text the model got. Search results and failed fetches (an HTTP error, a refused PDF or local address, a name that does not resolve) are not in it; next to it the app keeps every URL `fetch_page` was called with and whether the call succeeded. When the task ends with a complete answer, the app reads every URL in the answer (the text and `Sources:`) and in the notes saved during the task: a URL in the ledger is fetched, a URL `fetch_page` tried and failed on is failed, and any other URL was never tried. Scheme and host are compared in lower case, path and query as written but with their percent-escapes decoded (`Metal_%28API%29` is `Metal_(API)`; `%2F`, `%3F`, `%23` and `%25` stay), without the fragment and trailing slashes; the URL before and after a recorded redirect count as the same. A bracket that opens inside a URL may close inside it (`https://ja.wikipedia.org/wiki/Metal_(API)`). The check fails when the answer or a note has a URL that was never tried, or when sources are required and the answer has no fetched URL. A failed URL is not a failure of the check: saying that a page could not be read is a right answer. Sources are required when the task called a web tool or the answer has a URL, unless every `fetch_page` of the session failed.

- When the check fails, the app sends one more turn in the same session. It lists the URLs that were never tried (or `none listed`) and asks the model to open the pages it needs with `fetch_page` (at most 3), answer again with only fetched pages under `Sources:`, and save its note again under the same title. The turn counts toward **Max steps** together with the task. Each `fetch_page` of the turn takes one of three slots before it starts (a failure or a replay takes one too), so a fourth call returns `error: citation repair fetch limit reached` even when the calls run at the same time. The answer of the turn is checked again.
- With the guard on, `write_note` saves to `.draft/` in the notes folder (it tells the model the path the note will have) and the note moves to the notes folder only when the answer and the notes pass; the repair turn saving the same title rewrites the draft.
- When the check fails again, no step is left, the task is stopped, the repair turn does not end with an answer, or (headless) the `--deadline` has passed, the answer stays on screen but is held: the notes stay in `.draft/` with the reason in `.draft/HELD.txt`. The guard does not judge a task that ends without a complete answer (`skipped`: step limit, token limit, stop, error), and its notes stay in `.draft/` as well.
- The transcript ends with `Citation guard: 引用元取得確認` and the counts (passed, with or without the repair turn, and every URL the answer cites was fetched), only the counts (passed, but a cited URL could not be fetched or none was, such as `3/4 cited URLs were fetched in this task · 1 could not be fetched`), or `Citation guard: 引用元の取得を確認できませんでした` and the reason. It says the cited pages were fetched in the task, not that the answer is true.
- `move_note` cannot move a draft, so with the guard on it finds a note only after a task has passed.
