# Final-answer stop ordering

Issue #425. Runtime source at `95dc1ea`; macOS 27.2 (26B123), Swift 6.4, Xcode 27.0. All observations use native Foundation Models with a synthetic LanguageModel executor and fake tools, without trained models or GPU inference.

## Reproduction

The #423 validation ran ten pairs of offline self-tests. The note checks passed in all 20 executions, while one execution failed `conversation/budget-toolInFinal` with `final=2`, `requests=5` and `finish=stopped`. This separates the final-answer issue from note-folder collisions.

The deterministic probe retains the expected final-request count of one. Only in a new `toolInFinalBeforeStop` scenario, its Store runner holds the final-answer stopForBudget effect until either a second final request begins or the generation returns. The fake executor and runner release an AsyncStream gate; no sleep controls the ordering.

Three independent executions produced the same result:

| Scenario | Final requests | Total requests | Outcome |
|---|---:|---:|---|
| Existing toolInFinal scheduling | 1 | Existing scenario | Pass |
| Final stop delivery held by the gate | 2 | 4 | Fail in all 3 runs |

The failing order was `final-1 → stop-wait → final-2 → stop-deliver → final-return`. Only web_search from the research portion was executed; the later tool was refused. The temporary probe source and raw logs are retained in the ignored local validation directory.

## Cause

ConversationFeature.decide returns a refusal and a budget stop as independent effects at the step limit (`ConversationFeature.swift:429-441`). ToolRunner returns the refusal as an ordinary string (`ToolRunner.swift:156-169`). Foundation Models can consume that tool result and start another model request before the generation task receives cancellation. Merely testing the current cancellation flag at executor entry cannot cover a cancellation that has not arrived yet.

This is an observable ordering in the product's native session/tool path, not a failure of the assertion's final count. The fake intentionally violates the final answer's toolCallingMode.disallowed setting, so the evidence concerns handling that unsupported response. It does not measure how often a real model emits such a response.

## Adopted follow-up: #522

Keep the existing one-final-answer contract. Distinguish a terminal budget refusal from ordinary tool errors and permission refusals. Feature chooses the terminal effect; Runner maps its call ID to a terminal ToolRunner decision; the native Tool boundary ends that generation instead of returning a result that can invite another model request. Ordinary network and permission errors remain text results.

The existing preserveTranscript setting, stream join, budgetStopped transition, unfinished-call closure and one answer-only generation remain required. Verification must include ordinary success/error/empty output, Stop, notes/drafts, restore/next turn and the delayed-stop scenario. The fixed expectation stays at one; it is not relaxed to tolerate duplicates.

TIF-01 has both success and deterministic failure orders. TIF-02 identifies the native product boundary and its race. TIF-03 is the scoped #522 implementation contract and its regression checks. The fix itself is tracked separately and is not claimed implemented by this research record.
