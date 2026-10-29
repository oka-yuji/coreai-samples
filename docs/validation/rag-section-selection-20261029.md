# Section selection validation — 2026-10-29

Issue #376 / parent #378 plan v1.1. Base `feat/375-structured-bm25` at `ebe96c6`. macOS 27 / Xcode 27 / Swift 6.4. CPU/native fake execution only; Qwen weights have not been used.

| ID | Evidence | Status |
|---|---|---|
| RG-L01 | `RAGSectionSelectionTests` (6 functions), `ToolFreeTokenCountTests` (2 functions), `rag-section/valid`, `inputBudget` | Verified known aliases, limits, deduplication, original excerpts and exact backend input counting |
| RG-L02 | Strict parser tests; native `invalid`, `timeout`, join-before-answer | Verified reasoned fallback without retry or extra URL/tool requests |
| RG-L03 | `rag-section/none`, `rag-eval/none-no-answer/structure-llm/none` | Zero answer calls; diagnostic completes with `selectionNone`, retaining selection usage |
| RG-L04 | `rag-section/stop`, `earlyStop`, join-before-answer, restart-after-stop | Stop waits for execution to settle; no answer/tools; next request succeeds |
| RG-L05 | Native probe `maximumActive`, `historyClean`; Runtime lock fence | At most one selection/answer executor active; selection JSON absent from conversation history |
| RG-L06 | `rag-section/cost/*`, `rag-eval/selection-costs/structure-llm`, `unknown-cost` | Selection usage retained and added to totals, answer usage remains separate |
| RG-L07 | Native fake C and CLI implementation available | **Pending:** existing Qwen bundle path/hash unavailable; real-model smoke is not claimed |
| RG-L08 | Required checks below, A/B/C native answer/note/permission regression | Automated checks recorded separately from RG-L07 |

## Commands and evidence

Required commands: strict SwiftLint using the repository baseline; `python3 Tools/Format/check_format.py --base origin/main`; `./Tools/Irodori/build-inputs.sh`; `swift test --package-path CoreAISamplesKit`; Mac Release `xcodebuild` using `/private/tmp/rag374-derived`; built app `--agent-selftest --offline`.

Logs: `/private/tmp/rag376-{lint,format,irodori,full-tests,build-final,offline-final}.log`. Python driver: `python3 -m unittest discover -s Tools/RAGEvaluation/tests` (8 tests). Native selftests use synthetic Foundation Models executors with hand-controlled gates, not Qwen or a timed sleep to guess completion order. Live Web/WebView and absent-weight tests retain their explicit skips.

A final measurement audit also found that native usage may remain zero when the backend throws before its terminal usage event. A measured handle sharing the same Runtime now records failed partial prefill/sampled counts; CPU `measuredFailure` confirms one token read from a four-token input, no extra engine load, and unchanged subsequent unmeasured behavior (`partial-cost-tests.log`). Missing measurements remain explicitly unknown; `partial-usage-proof.log` directly confirms native input usage 0 alongside one completed prefill token. Counts exclude work inside a failed kernel and EOS sampling.

Independent review found CRLF ID rejection and loss of a later evidence fragment after choosing the correct section. Both reproduce in `/private/tmp/rag376-review-before.log` and pass in `review-after.log`. The input counter matches native usage and causes no additional fake-engine read/load (`token-preview.log`). `offline-join.log` records timeout/Stop settlement, restart, NONE and fatal behavior; the final log also adds early Stop, total usage, C notes and the NONE result contract.

The first stop probe intentionally held the native executor before its first channel event but waited for the executor's cancellation callback before releasing it. It reached the outer process timeout with both response APIs. This is not accepted as a successful stop check. Request and completion gates were separated; model-free success does not imply a bound on real Qwen stop latency.

## Remaining acceptance

RG-L07 and #373 RG-E06 require the verified existing Qwen3.8 27B 4bit bundle. #377 owns the real Work 72-case comparison, bounded repetitions, Chat compatibility sample, factual-support grading and adoption decision. The unresolved model path is not replaced with another model, synthetic scores or a download. The manifest keeps `expectedModelHash` unset and native evaluation rejects it before loading.

The specification is `docs/specs/rag-section-selection.md`. Confirmed findings are appended immediately to the Core AI guide and consolidated there. No default strategy, archive schema, UI setting, model conversion or merge is included in this change.

Chat compatibility also passes the same native evaluation path in A/B/C, including source IDs, saved completion and separate C usage. Logs: `/private/tmp/rag376-chat-{build,offline,lint,format}.log`. Work remains the primary target. This follow-up changes tests only; runtime implementation stays at `73ad3f6`.
