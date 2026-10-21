# Verification and measurements

[Home](../README.md) · [日本語](verification.ja.md) · [Runtime](runtime.md)

## Verification

For Swift formatting, run `python3 Tools/Format/check_format.py --base origin/main`.
The [format checker guide](../Tools/Format/README.md) describes prerequisites,
file selection, and regression tests. The checker runs entirely from this
repository and does not need a personal configuration directory.

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

## Measurement fields and boundaries

The per-request measurement is opt-in. `chat --runtime own --measure summary` adds a `measurement` object to each `turn` record, and `--measure trace` also adds one row per sample. `--measure off`, the default, leaves the record without this field, and generation, stops, errors, metadata and usage are the same as without the option. `tools` rejects `--measure`, and `bench` has no such option. `CoreAIBackend` collects the values and `coreai-verify` prints them. The Verification SPI `lastRequestMeasurement()` returns the measurement of the last request admitted to execution, like `lastGeneratedTokens()`, and nothing when that request ended before generation, for example on a context overflow or a load failure. A turn record omits `measurement` when its own request has none; a request rejected before execution leaves the previous request's measurement in the SPI, so the CLI compares `request_id` with the previous turn's. Times are seconds from one monotonic clock and counts are integers.

The runtime's generation starts after request validation, template rendering, context fitting and any reload, which is reported separately as `reload_seconds`. The stages after the read are contiguous: for a completed request, `read_seconds` plus the five stage totals equals `total_seconds` up to rounding. When an error ends the request, the stage in progress counts only toward `total_seconds`.

| Field | Start → end | Counted | Unit |
|---|---|---|---|
| `Engine.Stats.decodeSeconds` | each `step`: start of its forward, input preparation included → end of the forward after `function.run` | completed steps (`decodeTokens`) | s |
| `Engine.Stats.prepareSeconds` | each forward of a prefill chunk or a step: start → just before `function.run`. Added only while `measuresPreparation` is true; the runtime turns it on for measured requests, so an unmeasured request reads no clock here | forwards | s |
| metadata `prefillSeconds` = `read_seconds` | generation start → end of the read (prefill or restore); on a read failure, → the failure | tokens read, `prefilled_tokens` (metadata `prefillTokens`) | s |
| metadata `decodeSeconds` | end of the read → end of the drain (sampling, text, sending and steps together) | — | s |
| `first_token_seconds`, `last_token_seconds` | generation start → return of the first and last `sampler.next`, including the stop token's sample | — | s |
| `first_send_seconds` | generation start → return of the first `output.send` that sent at least one event | — | s |
| `sample_seconds` | end of the previous stage (the read for the first sample) → return of `sampler.next`, including the cancellation check | samples, including the stop token | s |
| `text_seconds` | end of the sample → end of `tokenizer.decode` and `parser.feed` | samples other than the stop token | s |
| `send_seconds` | end of the text → return of `output.send`, including calls with nothing to send | send calls | s |
| `step_seconds` | end of the send → return of the shared-compute `engine.step` (includes its preparation and forward) | `steps`, completed steps | s |
| `prepare_seconds` | change in `Engine.Stats.prepareSeconds` over the decode loop; a part of `step_seconds` | step forwards | s |
| `drain_seconds` | end of the last stage → return of the final text conversion and `output.send(last: true)`; 0 when an error ended the request | — | s |
| `total_seconds` | generation start → end of the drain, before the metadata is sent; on an error, → the error | — | s |
| speed `load_s` | start → end of `Engine.init` (own) or `EngineFactory.createEngine` (official, no longer including `warmup(queryLength: 1)`) | — | s |
| speed `warmup_s` | end of the load → start of the measured run. Own: 8 tokens after a 16-token prefix, then `reset`. Official: `warmup(queryLength: 1)`, an 8-token drain and `reset(to: 0)` | — | s |
| speed `specialize_s`, `cached` | own `Engine.LoadStats`: specialization time, 0 for a cached model | — | s |
| speed `max_tokens`, `generated_tokens` | requested limit (formerly `decode_tokens`); tokens actually received by `onToken` or from the official sequence | tokens | tokens |
| speed `prefill_s`, `decode_s` | run start → first token; first token → last token | — | s |
| speed `decode_tok_s` | `(generated_tokens − 1) / decode_s`; absent below two tokens | tokens | tokens/s |

The counts are `prompt_tokens`, `prefilled_tokens` (tokens the engine read for this request, counted per finished chunk, so a read that fails partway keeps the chunks before the failure; 0 if the prompt was fully reused or the read failed before any chunk), `sampled_tokens` (the same list as `generated_tokens`: no stop token, the last unprocessed token at the limit included), `steps` and `sends` (sends with at least one event, the drain included). `outcome` is `stopToken`, `maxTokens`, `cancelled` or `failed`; the last two add `error`. A speed record's `stop_reason` follows the count for the own engine, which ignores stop tokens there and so ends at the limit (`maxTokens`, or `stopToken` if it ended earlier). For an official engine it is the reason its iterator ended (`InferenceOutputSequence.stopReason`): `maxTokens`, `stopToken` (EOS), `stopSequence`, `cancelled`, `error` or `unknown`. The sequential engine shortens the requested count to the remaining context, so `maxTokens` can come with fewer than `max_tokens` tokens. A `trace` row exists for each `sampler.next`, so a `stopToken` request has `sampled_tokens + 1` rows. The stop token's row has only `sample_seconds`, and `step_seconds` is absent when no step followed.

The fields map to the [common record contract](benchmarking.md) as follows.

| Common metric | Source in these records |
|---|---|
| `ttft_s` | `measurement.first_send_seconds`, the first output handed to Foundation Models. The difference from `first_token_seconds` is the text conversion and the send. It starts at the generation start; the turn record's `ttft_s` is end to end from the session call. |
| `decode_tps` | `(samples − 1) / (last_token_seconds − first_token_seconds)`, where `samples` is `sampled_tokens + 1` for `stopToken` and `sampled_tokens` otherwise (the number of trace rows). For speed runs, `decode_tok_s`. |
| `latency_s` | `measurement.total_seconds` |
| `first_load_s`, `warm_load_s` | speed `load_s` (or the chat `load` record's `init_s`) with `cached` false or true. Own runtime only. |
| `peak_rss_bytes` | speed `peak_rss_bytes`, the child process's lifetime peak from `getrusage` |
| `resident_bytes` | `memory.rss_bytes`, a sample at `captured_at` |

These items cannot be observed or are not measured by these commands:

| Item | Status |
|---|---|
| GPU execution time | Not collected. Add it to a common record only when observed with Instruments; do not derive it from wall time. |
| Actual compute placement | No public API reports it. `observed_compute` is always `unknown`; `requested_compute` and `expect_frequent_reshapes` in own load and speed records are only the specialization request. |
| Official specialization time and cache hit | Not observable through `EngineFactory`; official speed records omit `specialize_s`, `cached` and `requested_compute`. Fix the cache state by procedure. |
| `peak_footprint_bytes` | Not measured. `footprint_bytes` is a sample, not a peak. |
| `os_proc_available_memory` | Unavailable in the macOS SDK; not recorded. |

Every `memory` object has `status` (`ok` or `failed`), `captured_at` (ISO 8601 wall clock) and `unit` (`B`). `error` (for example `task_info returned 5`) appears only on failure, and `rss_bytes`, `footprint_bytes` and `graphics_footprint_bytes` appear only on success. A speed record omits `peak_rss_bytes` when `getrusage` fails. The summary reports the maximum over successful captures with their count `peak_rss_runs`, and the median decode rate over runs that have one with their count `decode_runs`. Never count a missing value as 0.

`summary` collects totals and counts. Measurement reads the clock up to six times per sample, four in the loop and two in the step's forward, and `trace` also stores a row. Before comparing results, run an A/A check: alternate the same command with `--measure off` and `--measure trace` and compare the turn records' `total_s` and `decode_tok_s`.

```bash
.build/release/coreai-verify chat --model <bundle> --runtime own --reasoning off --measure trace --max-tokens 32 --prompt "List three fruits, one per line."
```

## Model-specific results

Qwen3.8 27B reference comparisons and unmet criteria are in the [model verification](qwen3.8-27b.md#verification); speed and memory are in [Key numbers](qwen3.8-27b.md#key-numbers). Find other model guides in [Models and roadmap](../README.md#models-and-roadmap).
