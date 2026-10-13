# Demo app (macOS, and Extract on iPhone)

`CoreAISamples` (`CoreAISamples.xcodeproj` and the `CoreAISamples` folder at the repository root) is a SwiftUI app that runs a local model bundle through Foundation Models with the runtime in this repository (`CoreAIBackend`). It does not use the official `coreai-models` package.

- **Models**: open a bundle folder on this Mac (the app remembers it), load it, and clear the copy the app prepared for the GPU. The Hugging Face download catalog lists Qwen3.8 27B (4bit). Downloads require a pinned 40-character commit revision. Every file is checked against the Hub listing: SHA-256 for LFS files, and git blob SHA-1 (including the `blob <size>\0` header) for other files. A missing checksum or mismatched download fails the operation. Restarting resumes by file: only files whose size and checksum match are skipped; partial or corrupt files are downloaded again. The completion marker is written only after all files pass. The app opens a bundle only if `source.hf_model_id` in its `metadata.json` starts with one of `BundleSupport.supportedModelIDPrefixes` (Qwen3, Qwen3.5 and Qwen3.8). That prefix check is the only family check: it does not verify how the bundle was converted, and adding a prefix does not make another family work (see the README). The catalog's memory figure, ~32 GiB peak RSS while loading, is the peak resident memory of the app process while it loaded the 27B bundle on an M4 Max Mac with 128 GiB of unified memory, measured with `/usr/bin/time -l`: 31.51 GiB from the prepared copy (median of 48 loads) and 31.88 GiB at the first load. It is not a minimum memory requirement.
- **Chat**: a `LanguageModelSession` on `LocalLanguageModel`. Replies stream, thinking streams into a collapsed section, and the status line shows the time to the first token, the speed, the input, reused and output tokens, and why the reply stopped. With **RAG** on, the app first collects sources from the web and from `~/Documents/rag/` and the model answers from them with numbered citations. See [RAG](#rag) below.
- **Agent**: searches the web, reads pages, saves notes and answers with a `Sources:` section, on the model that Chat has loaded. See [Agent](#agent) below.
- **Extract**: a technical verification of a Core AI conversion. Type labels and GLiNER2.5-multi (Core AI, FP32) marks the spans it finds for each in a text, with their scores; mapping an inspection memo onto a form with option suggestions is kept as an experiment. The result exports as JSON. See [Extract](#extract) below.

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

The first load of a model prepares it for the GPU (20.72 s for Qwen3.5-9B on October 3, 2026, and 82.84 s for the Qwen3.8 27B 4-bit bundle on October 11, 2026, both on an M4 Max with macOS 27.0 (26A428)); later loads reuse the prepared copy in `~/Library/Caches/coreai-cache/<OS build>/com.coreai-samples.DemoApp/`.


## Extract on iPhone

`CoreAISamples-iOS` is an iPhone app with [Extract](#extract) only: a screen that extracts one note, and the headless `--extract-eval`. Chat, Agent and Models stay on the Mac. The bundle ID is `com.coreai-samples.DemoApp.ios` and it needs iOS 27.0. It runs on a device only: the iOS Simulator SDK (`iPhoneSimulator27.0.sdk`) has no `CoreAI.framework`, so a simulator build stops at `Unable to resolve module dependency: 'CoreAI'`. The Extract service, runner, headless evaluation and bundled JSON files are the Mac app's; the entry point and the screen are in `CoreAISamples/iOS/`. It has not been run on a device yet.

**Build and install.** Without a team, the device build is left unsigned: it shows that the app compiles, but it cannot be installed.

```bash
xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples-iOS -configuration Release -destination 'generic/platform=iOS' build
```

To install it, copy `Config/Local.xcconfig.template` to `Config/Local.xcconfig` and set `DEVELOPMENT_TEAM` (the template signs the iPhone build too), turn on Developer Mode on the iPhone and connect it. In Xcode, choose the `CoreAISamples-iOS` scheme and the iPhone and press Run (a Release build; the first Run also registers the iPhone with the team), or, once the iPhone is registered, from the command line:

```bash
xcrun devicectl list devices    # the iPhone's identifier, <UDID> below
xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples-iOS -configuration Release \
  -destination 'generic/platform=iOS' -derivedDataPath .build -allowProvisioningUpdates build
xcrun devicectl device install app --device <UDID> .build/Build/Products/Release-iphoneos/CoreAISamples-iOS.app
```

**Copy the asset and the gold file into the app's Documents.** The asset folder is the one the Mac reads (`contract.json`, one `.aimodel` and `tokenizer.json` in `tokenizer/` or in the folder itself), made of real files: `cp -RL` resolves symbolic links. The FP32 (1.23 GB) and FP16 (0.62 GB) folders can sit side by side, and the app opens either.

```bash
BID=com.coreai-samples.DemoApp.ios
mkdir -p /tmp/gliner-fp32/tokenizer
cp -RL /path/to/export/contract.json /path/to/export/*.aimodel /tmp/gliner-fp32/
cp -L /path/to/tokenizer.json /tmp/gliner-fp32/tokenizer/
xcrun devicectl device copy to --device <UDID> --domain-type appDataContainer --domain-identifier $BID \
  --source /tmp/gliner-fp32 --destination Documents/gliner-fp32
xcrun devicectl device copy to --device <UDID> --domain-type appDataContainer --domain-identifier $BID \
  --source /path/to/gold.json --destination Documents/gold.json
```

- Copy before you launch the app, and let the copy finish: a model read while its files are still arriving has been reported to leave a broken entry in the specialization cache.
- Leave out `--remove-existing-content true`, which has been reported to empty the whole data container.

**The screen** shows the same note as the Mac's Extract screen and lists the folders in Documents that hold a `contract.json`, as they were at launch; relaunch the app after copying another one. Tap a folder to open it (no folder can be tapped while one opens): the status shows the load time and the `revision`. Edit the note (the first example is filled in) and press **Extract**, which extracts again even when the note has not changed: each example label of `extract_examples.json` lists the spans the model returns for it, as written in the note, with their scores, and a line shows the time of each stage. The labels cannot be edited here, and the form experiment is on the Mac only. The model is specialized for the GPU; launch with `--compute-units default` to let Core AI place it (the footer shows which).

**Headless evaluation.** Launch with the arguments the Mac takes (see [Extract](#extract)). Relative paths are read in Documents: the app makes Documents its working directory. The app keeps the screen from locking during the run; unlock the iPhone before you launch, and keep the app in the foreground, since an app in the background cannot use the GPU. The app exits when the run ends, which ends `--console`, and stdout and stderr come back through it as on the Mac.

```bash
xcrun devicectl device process launch --console --terminate-existing --device <UDID> $BID \
  --extract-eval gold.json --asset gliner-fp32 --out result-fp32-gpu.json
xcrun devicectl device copy from --device <UDID> --domain-type appDataContainer --domain-identifier $BID \
  --source Documents/result-fp32-gpu.json --destination ./result-fp32-gpu.json
```

- `--compute-units gpu`, the default, prefers the GPU; `--compute-units default` leaves the choice among the CPU, the GPU and the Neural Engine to Core AI. Try the default placement in a launch of its own. Any other value is an argument error (exit status 2).
- The output JSON also has `device` (the hardware model identifier, `hw.machine`), `osVersion` (the OS version with its build) and `computeUnits` next to the asset's `revision`. The Mac writes the same fields (`device` is `hw.model` there) and takes `--compute-units` too.

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
$APP --gui-e2e /path/to/output --model /path/to/bundle [--context-limit 1024] [--max-tokens N] [--clear-prepared]
```

The app opens the bundle in Models, loads it, chats one turn, starts a long reply, stops it and starts a new chat, saving a screenshot after each step with `results.json`, `driver.log` and the metrics rows of the run. `--clear-prepared` removes the prepared copy first, so the load shows the preparing state. With `--context-limit`, it then keeps chatting until the context window is full. Put options without a value, such as `--clear-prepared` and `--agent`, after the options that take one: on October 14, 2026, `--clear-prepared` placed before `--max-tokens 2048` left the app without a window and the check never started, while the same options with `--clear-prepared` last ran to the end.

## Metrics

Every generation appends one JSON line to `~/Library/Application Support/com.coreai-samples.DemoApp/metrics.jsonl`: the model, the turn, the input, reused and output tokens, the time to the first token, the speed, the stop reason, the resident memory and the app build. RAG answers also carry `sources`, `sourceTokens`, `ragPromptTokens` and `retrievalSeconds`.

## RAG

Web retrieval uses the same address checks and WebView content rules as Agent page fetches, including validation before requests and redirects. A refused target is not included as a source. DNS rebinding and private addresses hidden behind subresource or JavaScript hostnames are outside these checks; App Sandbox is off.

Turn on **RAG** in the Chat status bar, then pick **Web**, **Docs** or both. The app retrieves; the model never calls a tool.

1. **Web**: search (DuckDuckGo, top 8; Brave Search when the Agent settings hold an API key), then read the top 5 pages in parallel and keep up to 3,000 characters of each, the part that matches the question.
2. **Docs**: read every `.txt`, `.md` (also `.text`, `.markdown`) and `.pdf` file in `~/Documents/rag/`. The folder is created on first use and read again on every question, so there is no index to rebuild. PDFs are read with PDFKit.
3. Split pages and documents into passages of about 500 characters that repeat the last 100 characters of the previous one, rank them against the question with SQLite FTS5 (BM25), and keep the best ones up to the source budget, so that the whole prompt stays within 8,000 tokens (counted with the model's tokenizer, `LocalLanguageModel.countTokens(_:)`). The source budget is 3,000 tokens by default (`RAGContext.sourceTokenBudget`); the **Sources** menu next to the RAG switches also offers 6,000, and `--rag-budget 3000|6000` sets it at launch.
4. Answer in a new `LanguageModelSession` per question: the instructions are a fixed Japanese instruction with today's date, and the prompt is the sources as `資料[n] title / url / date` plus the passage, then the question. RAG turns are not added to the chat history, and the reasoning setting is Chat's.

The answer is followed by its sources, numbered like the `[n]` marks in the text (`[n] title — url (date)`; `日付不明` when the page gives no date), and one line for each source that failed (no network, no search results, a page that would not load). With no sources at all, the reply is `Could not retrieve any sources.` and the model does not run. The status line adds `sources N | source tok N | prompt(rag) N tok | retrieval N.Ns`.

A page's date comes from its `article:published_time` and similar meta tags, then a `<time datetime>` tag, then the search snippet, then the first 600 characters of the page text. The splitting, the packing to the token budget, the prompt and its citations, the date parsing, the reading of a documents folder (the app supplies the PDF text) and a term-match ranking are in the `RAGContext` library of the `CoreAISamplesKit` package, with no network or UI, and `swift test --package-path CoreAISamplesKit --filter RAG` tests them; the SQLite FTS5 (BM25) ranking is in the app, which uses the term-match ranking only when the FTS5 ranking returns nothing.

Headless, with no window:

```bash
$APP --selftest --model /path/to/bundle --rag "question 1|question 2" [--rag-web off] [--rag-local off] \
  [--rag-docs /path/to/folder] [--rag-budget 3000|6000] [--max-tokens N]
```

Each question prints `[rag] turn=N search N / pages N / passages N → N sources, N tokens`, one `[rag] turn=N source [k] title — url (date)` and one `[rag] turn=N passage [k] …` (the text the model saw) line per source, `[rag] turn=N note …` for each failure and `[rag] turn=N answer=…` on stderr, then its JSON line and `SELFTEST PASS` or `SELFTEST FAIL` on stdout. A question with sources passes when the model answers without an error; a question without sources passes when the reply is `Could not retrieve any sources.` and the model did not run. `--rag-docs` reads another folder instead of `~/Documents/rag/`. `$APP --gui-e2e /path/to/output --model /path/to/bundle --rag "question"` turns RAG on and answers one question on the screen, saving screenshots of the switches and of the answer with its sources and the status line.

## Agent

`fetch_page` resolves its target before fetching and permits it only when every address is public. It rejects loopback, private, link-local, unspecified, shared, multicast and reserved ranges, including mapped IPv4 and legacy numeric forms. URLSession redirects are checked before following them, and a refused redirect returns the refusal message without its response body. Main-frame and subframe WebView navigations and final URLs are also checked. WebView content rules block local hostname suffixes and non-public literal addresses in subresources. These checks do not pin DNS answers to connections or inspect hostname resolution for JavaScript and other subresources, and the app runs without the App Sandbox.

`--agent-selftest --offline` checks address classification, navigation decisions, redirect refusal and installed content rules. A live test can add `--fetch-redirect-test-url <public-redirect-to-loopback>` while a local listener records connections; Agent and RAG must both refuse it and the listener must receive no requests. `--fetch-capture-dir <directory>` captures three public pages for a before/after comparison.

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
- After an answer with a `Sources:` section, a **Citations** note reports `n/m cited URLs were retrieved in this task` and the mean 8-gram overlap between the answer and the text of each cited URL. A URL counts as retrieved when it appeared in the task's search results or in a fetched page, so a URL seen only as a search result counts; the [citation guard](#citation-guard) uses fetched pages only.
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
- `--instructions cite-fetched` adds two lines after the three above: cite only pages opened with `fetch_page` in the task (a search snippet is not a source), and check the row and column headers when quoting a number from a table. `cite-fetched` is the default. `--instructions frozen --citation-guard off` selects the frozen comparison conditions.
- `--citation-guard on` or `off` sets the guard for the run without saving it; without the flag, the run uses the setting saved by the app, and off when none is saved. `--sources-required yes` requires at least one fetched URL in the answer, `no` requires none (a never-tried URL still fails the check), and `auto`, the default, decides as on screen. `--date` replaces today's date in the instructions. `--stable-note-paths` makes `write_note` answer `saved to agent-notes/<file>.md` whatever `--notes-dir` is, so that the model reads the same text in every run; the note is still saved in `--notes-dir`.
- `--tool-cassette` with `--cassette-mode` replays and records `web_search` and `fetch_page` results in a file with one JSON line per result: the key (the tool and its arguments as the tool reads them: the query and the number of results for `web_search`, the url and the query for `fetch_page`), the text, the pages it carries, whether the call succeeded, the final URL, and the milliseconds it took. A recorded key is replayed, the first line of a key winning, after waiting the recorded time (`--cassette-latency none` skips the wait). An unrecorded key is fetched live and appended (`record`, `replay-or-record`), or answered with `error: cassette miss` (`replay`, which never writes). A hand-written `fetch_page` entry whose query is `*` matches any query for that exact URL; an exact key wins, and this does not apply to `web_search`. Live calls with the reserved query `*` are returned without being written to the cassette. A call that was stopped is not recorded. `write_note` and `move_note` never use the cassette. stderr gets `[cassette] hit|miss|record tool=… key=[…] ms=N`, and the run ends with `[cassette] summary hits=N misses=N records=N`.
- `--agent-selftest` checks the instructions, the tool definitions and arguments, the note and move rules, the page passage selection, the search and HTML parsers, the citation check, the verdict table and the transcript format, and, with no model and no network, the cite-fetched instructions, the guard's URL reader and ledger check, the cassette, the repair fetch slots, the drafts, the stable note paths, the RAG source budget and the bundle errors on the Models screen. `--offline` skips the network cases.
- `$APP --gui-e2e /path/to/output --model /path/to/bundle --agent` drives the Agent screen instead of Chat: it sends a task, waits for the tool calls and results and the answer with `Sources:`, stops a second task, and starts a new task, saving a screenshot after each step.

Logs: `metrics.jsonl` gets one `message` line per step (prompt, reused and generated tokens, time to the first token, prefill time, speed, finish reason, resident memory, the step's tool calls, whether it diverged, `splicedGenerations`, citations on the answer), one `agent-tool` line per tool call (tool, argument, start and end time, seconds, result size, error), and one `agent-run` line per task. `splicedGenerations` counts the earlier generations of the session that the runtime put back token for token into the history it rebuilt for that step (the text was the same but the tokenizer split it differently). The `agent-run` line has the instructions (`agentInstructions`: `frozen` or `cite-fetched`), whether the guard was on (`citationGuard`), the checks above (`ledgerChecks`), the ledger (`fetchLedger`), every `fetch_page` call with whether it succeeded (`fetchAttempts`), `guardOutcome` and `guardReason`, after a repair turn `repairTurns`, `repairFetches`, `repairSeconds`, `repairPromptTokens` and `repairGeneratedTokens`, and with a cassette `cassetteHits`, `cassetteMisses` and `cassetteRecords`. Each task is saved as Markdown in `~/Library/Application Support/com.coreai-samples.DemoApp/agent-transcripts/<session>.md`; notes go to `…/agent-notes/` unless another folder is set.

### Citation guard

With **Citation guard** on, the app checks the sources of the answer before it accepts it. It keeps a ledger of the pages `fetch_page` handed to the model in the session: the URL asked for, the URL the page came from after redirects, and the SHA-256 and length of the text the model got. Search results and failed fetches (an HTTP error, a refused PDF or local address, a name that does not resolve) are not in it; next to it the app keeps every URL `fetch_page` was called with and whether the call succeeded. When the task ends with a complete answer, the app reads every URL in the answer (the text and `Sources:`) and in the notes saved during the task: a URL in the ledger is fetched, a URL `fetch_page` tried and failed on is failed, and any other URL was never tried. Scheme and host are compared in lower case, path and query as written but with their percent-escapes decoded (`Metal_%28API%29` is `Metal_(API)`; `%2F`, `%3F`, `%23` and `%25` stay), without the fragment and trailing slashes; the URL before and after a recorded redirect count as the same. A bracket that opens inside a URL may close inside it (`https://ja.wikipedia.org/wiki/Metal_(API)`). Reading stops at white space, at ``<>"'`{}|\^`` and at non-ASCII punctuation and symbols, so letters written right after a URL without a space, such as Japanese text, are read as part of it. The check fails when the answer or a note has a URL that was never tried, or when sources are required and the answer has no fetched URL. A failed URL is not a failure of the check: saying that a page could not be read is a right answer. Sources are required when the task called a web tool or the answer has a URL, unless every `fetch_page` of the session failed.

- When the check fails, the app sends one more turn in the same session. It lists the URLs that were never tried (or `none listed`) and asks the model to open the pages it needs with `fetch_page` (at most 3), answer again with only fetched pages under `Sources:`, and save its note again under the same title. The turn counts toward **Max steps** together with the task. Each `fetch_page` of the turn takes one of three slots before it starts (a failure or a replay takes one too), so a fourth call returns `error: citation repair fetch limit reached` even when the calls run at the same time. The answer of the turn is checked again.
- With the guard on, `write_note` saves to `.draft/` in the notes folder (it tells the model the path the note will have) and the note moves to the notes folder only when the answer and the notes pass; the repair turn saving the same title rewrites the draft.
- When the check fails again, no step is left, the task is stopped, the repair turn does not end with an answer, or (headless) the `--deadline` has passed, the answer stays on screen but is held: the notes stay in `.draft/` with the reason in `.draft/HELD.txt`. The guard does not judge a task that ends without a complete answer (`skipped`: step limit, token limit, stop, error), and its notes stay in `.draft/` as well.
- The transcript ends with `Citation guard: Cited sources fetched` and the counts (passed, with or without the repair turn, and every URL the answer cites was fetched), only the counts (passed, but a cited URL could not be fetched or none was, such as `3/4 cited URLs were fetched in this task · 1 could not be fetched`), or `Citation guard: Could not confirm that the cited sources were fetched` and the reason. It says the cited pages were fetched in the task, not that the answer is true.
- `move_note` cannot move a draft, so with the guard on it finds a note only after a task has passed.

When a guarded task ends without an answer, notes with no URLs or only URLs whose retrieval was attempted are saved. Notes with an untried URL remain drafts with the reason in `HELD.txt`; the guard result stays `skipped`.

## Extract

> Technical verification of a Core AI conversion. Zero-shot GLiNER2.5-multi did not reach the pre-registered bar for Japanese inspection memos (outcome B3). The evaluated version is the default candidate filter, R1 + N1: on a second sealed set of 200 memos, fabricated candidates 13/200, primary-span precision 0.788, recall 0.822. Suggestions are uncalibrated scores.

Extract runs GLiNER2.5-multi, converted to Core AI (FP32), on a text you paste or type. By default you type labels and see the spans the model returns for each label, with their scores. Mapping an inspection memo onto a form, where each span's clause is classified over the options of its field, is kept as an experiment behind **Experiment: inspection form**. Nothing is filled in automatically.

### Evaluation

Before each sealed set of 200 synthetic Japanese inspection memos was opened, once, two bars were fixed: B1 for suggesting the values of a form (span precision 0.95, recall 0.90, no fabricated candidate) and B2 for a span-extraction sample (0.90, 0.80, at most 2). Missing both is outcome B3, which limits the release to a technical verification of the conversion. Each memo leaves one chosen field of form U1 unmentioned, and a candidate on that field is a fabricated candidate. The memos were written for the test and their answers checked independently, not by domain experts.

The evaluated version is the default candidate filter, R1 + N1. Rule N1 was chosen on a development set, and then this version was scored once on a second sealed set of 200 new memos, none excluded. It missed both bars: outcome B3. Rule R1 alone was scored on the same model output as a control, without the bars.

| Second sealed set: 200 memos, Core AI FP32 | R1 + N1 (default) | R1 alone | B2 |
|---|---|---|---|
| Fabricated candidates on the field each memo leaves out | 13/200 (6.5%) | 16/200 (8.0%) | at most 2 |
| Primary-span precision (each field's most probable candidate) | 0.788 (449/570) | 0.775 (449/579) | 0.90 |
| Span recall, any candidate | 0.822 (470/572) | 0.822 (470/572) | 0.80 |
| List fields with a mention whose primary span's top suggestion is right | 277/387 | 277/387 | |
| Memos with all six fields right | 63/200 | 61/200 | |
| "No matching option" fields left without an option | 4/32 | 4/32 | |
| Mention precision / recall | 0.930 / 0.886 | 0.917 / 0.888 | |

- N1 cut the fabricated candidates on the field each memo leaves out by 19%, from 16 to 13; the three it removed were number fields whose every candidate was a number with an equipment marker. On the development set it was chosen on, where half of the memos name equipment as `P-` and a number, it had cut them by 47% (17 to 9 of 120). It also removed every candidate of one mentioned field, all of them equipment numbers, which added one miss.
- The 13 left are list fields (8: an equipment name as the next action 4 times, a phrase outside every answer span 3 times, another field's value once) and number fields on another field's value (5).
- The options are skewed: oil leak none 70 / present 18, abnormal sound normal 74 / abnormal 17. In both fields the right top suggestions (70 and 74) are as many as always choosing the more common option would give, so 277/387 does not show that the model tells the options apart.
- Core AI made the same decisions as PyTorch FP32 on all 200 memos (1,200 fields), with probabilities within 2.93e-5. Four span ends differ by one character, as on the first sealed set (below), which puts Core AI's span F1 0.70 points above PyTorch's (0.804 against 0.797).

The first evaluation scored rule R1 alone, the default then, on the first sealed set (one memo is excluded, so 199 count). It missed both bars as well:

| First sealed set: 199 memos, Core AI FP32, rule R1 | Result | B2 |
|---|---|---|
| Fabricated candidates on the field each memo leaves out | 9/199 (4.5%) | at most 2 |
| Primary-span precision (each field's most probable candidate) | 0.791 (462/584) | 0.90 |
| Span recall, any candidate | 0.874 (500/572) | 0.80 |
| List fields with a mention whose primary span's top suggestion is right | 296/394 | |
| Memos with all six fields right | 57/199 | |
| "No matching option" fields left without an option | 2/24 | |
| Mention precision / recall | 0.942 / 0.918 | |

- Core AI made the same decisions as PyTorch FP32 on all 249 sealed texts (the 199 memos and 50 delivery reports), with probabilities within 2.2e-5. Four span ends differ by one character: gliner2 adds a period to a text that does not end with `.`, `!` or `?`, and four of its spans ended on that period, past the end of the text.
- The 50 delivery reports, on a form that was not tuned, are reported apart and not added in: mention precision 0.959, recall 0.784, fabricated candidates 1/50.
- Warm, one memo took 27 ms (p50) and 43 ms (p95) from preprocessing to the mapping on an M4 Max, without drawing.
- Rule R1 (the candidate filter below) did not reduce the fabricated candidates on the sealed set, and it added 6 misses.
- Run again with R1 + N1, the first sealed set gives fabricated candidates 7/199 and primary-span precision 0.797 (462/580), with recall unchanged, and the 50 delivery reports do not change. This is a regression check, not an independent evaluation: the set had been opened.

The evaluation does not show:

- that the model can fill in inspection forms for people, or that it never makes up a candidate;
- quality on Japanese text in general, or on forms it was not tried with;
- that further training will bring the fabricated candidates under a given rate;
- that a person's confirmation makes the output safe, or that corrections in a memo are handled (the correction words only mark the place);
- that a score is the probability of being right;
- anything on iPhone, or that it is ready to ship;
- that 43 ms includes drawing, or that the speed target for 512-token inputs is met (not measured; the sealed memos are short);
- the same output as PyTorch in every detail (four span ends differ).

### The screen

**The asset folder** holds `contract.json`, one `.aimodel` (`encode_t256`, `pair_t256`, `encode_t512`, `pair_t512`, in FP32 or FP16: an FP16 export takes and returns Float16 values, which the app reads into Float) and the tokenizer's `tokenizer.json` in a `tokenizer/` folder or in the folder itself. Choose it with **Open Asset Folder…**; the app remembers it and opens it again when the Extract screen appears. Opening specializes the model for the GPU, and the status line shows how long it took and the `revision` in `contract.json`.

- **Note**: paste or type the text, or press **Paste Example** (examples in `CoreAISamples/Demos/Extract/Resources/extract_examples.json`). The note is extracted 300 ms after the last change, and never while an input method is composing a word. A change of the text, of the labels or form, of the mode or of the candidate filter starts a new version: a result that arrives for an older version is dropped, and the confirmations are cleared.
- **Evidence**: the note with each label's (or field's) spans in its color. Select a label (its row or its name under the note) to make its spans stronger and underlined. Correction words found by rule (`correction_words.json`) are red; they mark the place only.
- **Labels**, the default: one row per label with its description, each span the model returns for it as written in the note, and the span's score. **Add Label** and **Edit** take the label, which the model reads, and an optional description; the screen starts with the example labels of `extract_examples.json`. A label is kept as a number field without a unit, so no clause is classified, and nothing is confirmed. The candidate filter below is for the form only: no span of a label is dropped.
- **Experiment: inspection form** maps the note onto form U1 (`Resources/forms/inspection_u1.json`) instead. Switching between the labels and the form clears the result and the choices and extracts again.
  - **Form**: one row per field with its state (`Candidate`, `Confirmed` or `Not detected`; not detected can also mean the model missed it), each candidate as written in the note with its span score, the number and unit of a number field, the two best options of a list field with `score (uncalibrated)`, and `Correction cue (rule)` with the words found in the sentence of a candidate. **Confirm** offers every option (or every candidate of a number field), `No matching option` (`None of these`) and `On hold`; nothing is chosen by default, even when a field has two values. **Clear** removes the choice. A new result for the same text and form keeps the choices.
  - **Candidate filter**: `R1+N1` (`clauseFieldNameAndEquipmentNumber`), the default and the evaluated version, applies rule R1 to list fields and rule N1 to number fields. R1 drops a candidate of a list field when its clause holds the name of another list field and not its own; N1 drops a candidate of a number field when every numeral in it has an equipment marker (`No.3`, `P-3`; the full rule is under [Headless](#headless)). `R1` (`clauseFieldName`) applies R1 alone and `None` keeps every candidate; both stay for experiments. Dropped candidates are not shown, and the export keeps them apart. The filter starts as R1+N1 at every launch.
  - **Add Field** and **Edit** change the form: name, the label the model reads (the name when empty), type (number with a unit, or list with comma-separated options) and description. A field keeps its option labels and its task while its name and option names stay. **Reset Form** goes back to form U1. The model takes up to 16 labels or fields and 16 options.
- The status line shows the time of the last extraction: preprocessing, `encode`, `pair` (with the candidates and the decoding), the clause classifications (count and time), the mapping and the total.
- **Export JSON** writes `~/Library/Application Support/com.coreai-samples.DemoApp/extract-exports/extract-<time>.json` with separate sections: `input` (text and version), `mode` (`labels` or `form`), `form` (the labels or the form), `model` (the candidates per label or field with probabilities, the suggestions, the dropped candidates, `probabilitiesCalibrated: false`, `candidateFilter` (`clauseFieldNameAndEquipmentNumber`, `clauseFieldName` or `none`; always `none` for the labels), the asset's `assetRevision`, the app's `appVersion` and the timing), `rules` (the correction cues) and `confirmations`. Every position is given in Unicode scalars and in UTF-16.

The state, the actions and `reduce` are in the `ExtractFeature` library of `CoreAISamplesKit` (no UI, no model; `swift test --package-path CoreAISamplesKit --filter ExtractFeature`), run by the generic `Store` of `FeatureStore`.

### Headless

With no window:

```bash
$APP --extract-eval /path/to/gold.json --asset /path/to/asset-folder --out /path/to/result.json [--form /path/to/form.json] [--candidate-filter r1|r1n1|none]
```

- The gold file is a JSON array of sentences in the GL2 format (`id`, `text`, `main_unmentioned_field`, `features`, and per field `mentioned`, `explicit_empty`, `statement_span`, `value_span`, `value`, `unit`, `option`; positions in Unicode scalars). Sentences with `excluded: true` are left out of every count. `--form` replaces form U1; the correction words stay the bundled ones.
- Every sentence is extracted with the service the screen uses, after one warm-up run on the first sentence (reported as `cold`).
- `--candidate-filter` picks the rule that drops candidates: `r1n1` (the default and the screen's), which adds rule N1 to R1 for number fields, `r1` (rule R1 alone) or `none`. Any other value is an argument error (exit status 2). N1 drops a candidate that overlaps one or more numeral runs of the note when an equipment marker touches every one of them (`No.3`, `P-3`): a run is half- or full-width digits and numeral words in kana and kanji, the longest word first, ended by anything else (a `.` too; the decimal point written in kana or kanji is one of the numeral words); a marker ends right before the run or starts right after it, compared with full-width ASCII (U+FF01...U+FF5E) as ASCII and in lower case. A candidate with an unmarked run, or with no run, is kept. The markers and numeral words are in `CoreAISamplesKit/Sources/GLiNERCore/Resources/equipment_numbers.json`.
- Scoring, per field, per `features` tag and overall: **mention** is found when a candidate overlaps the gold statement span, fabricated when the field is not mentioned but has candidates (`mainFabrication` counts the sentence's `main_unmentioned_field` on its own), missed when it is mentioned with no candidates, and misplaced when it is mentioned and no candidate overlaps its statement span (a false positive and a false negative). **Span match**: a candidate overlaps the field's gold value span, stays in the same sentence (ended by `。`) and overlaps no other field's value span; two fields that share a predicate have the same value span, which does not count (precision over all candidates, each matching candidate counting, recall over gold value spans; exact ranges are given for reference). The **policy** takes the candidate with the highest probability: its top suggestion for a list field (`option`; a null gold option never matches, and `optionNull` counts the null ones the policy leaves without an option; `optionTopTwo` checks the two best) and its number and unit for a number field (`number`; the form's unit when none is written; `numberInCandidates` checks every candidate). The number is the one `ParsedNumber` reads, digits only: `numberWithoutDigits` counts the gold values written in kana or kanji, which cannot match. Explicitly empty fields are not in the option, top-two and number scores. The miss rate counts mentioned fields with no candidate; a misplaced field is counted on its own. `agreement` counts fields that agree with the gold under the policy (an empty gold agrees with no candidates) and `allFields` the sentences whose fields all agree. `correctionCues` compares the sentences with a cue with those the gold tags `correction`.
- The output has the gold file's path and SHA-256, the form, the asset (`path`, and `floatType`: `float32` or `float16`) and `revision`, `candidateFilter` (`clauseFieldName`, `clauseFieldNameAndEquipmentNumber` or `none`), `droppedCandidates` (every candidate the rule dropped, with the sentence `id`, `field`, `start`, `end`, `text` and `probability`; they are not scored) and `appVersion`, the asset load time, the timing (`cold`, and the `p50` and `p95` of each stage of the warm runs, nearest rank, in milliseconds; `total` is from the start of preprocessing to the end of the mapping), the summaries and every sentence with its judgments and candidates. stdout gets one line for the overall summary and one per field, a timing line, then `EXTRACT EVAL DONE <out>` (the summaries per tag are in the JSON only); the exit status is 0, 1 when the run fails, or 2 when an argument is missing or has a value it does not know.

`$APP --gui-e2e /path/to/output --extract /path/to/asset-folder` drives the Extract screen: it opens the asset in the label mode, pastes the first example, switches to the form experiment, adds the example's extra field and selects it, confirms one number field and one list field and exports the JSON (with its mode, candidate filter, asset revision and app version), saving a screenshot after each step with `results.json`, `driver.log` and `export.json`. Give the asset folder as the value of `--extract`: AppKit reads the arguments as `-key value` pairs and opens a path left on its own as a document, and SwiftUI then opens no window. If no window appears for another reason (a saved window state that AppKit cannot restore also leaves the app without one), put `-ApplePersistenceIgnoreState YES` before the other arguments.
