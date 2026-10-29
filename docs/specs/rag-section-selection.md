# Existing-model section selection

Issue #376 / parent #378 plan v1.1. `--rag-strategy structure-llm` selects C. The primary target remains the existing Qwen3.8 27B 4bit bundle in Mac Work, with Chat as a compatibility path. Normal startup remains flat. This is an application-level experiment, not a reproduction of STAIR training.

## Candidate and budget contract

C uses B's candidate pool after the same page budget, including flat candidates for documents with unavailable structure. BM25 shortlists distinct sections; an unmatched pool retains the existing order. Each section is shown with a short alias (`C1`...), its heading path (at most 200 Characters) and the first 240 Characters of its best-ranked original fragment. The alias maps to a known document/version/position section ID. It is not a citation or a generated URL.

Initial limits are fixed in the evaluation manifest: 12 candidate sections, 3 selected sections, 2,048 input tokens, 128 output tokens and 30 seconds before requesting cancellation. `RAGSectionLimits` rejects increases beyond these caps. Greedy sampling, reasoning off, no tools and one call without retry are fixed by the implementation. Candidate entries are removed from the end until the input fits. The input counter renders the same tool-free transcript, sanitizer and reasoning-off template as the loaded backend; it does not load another model or run the engine. The output reserve must also fit the context.

Instructions contain only the selection policy. Question, headings and excerpts are encoded as prompt data. Responses accept known aliases one per line, with CRLF normalized; duplicate aliases are removed. Unknown IDs, extra prose, blank interior lines, more than three distinct IDs and output overflow cause a reasoned fallback to B's result (including B's own A fallback). A valid `NONE` supplies no sources and generates no answer. Fatal model errors use the existing failure path.

Chosen sections keep the same candidate-set BM25 order within each section. This prevents a correct section choice from losing its later evidence fragment when the budget only fits one fragment. Original headings and bodies still pass through B's atomic group packing and the final conversation prompt budget. Generated text is never registered as source evidence.

## Execution and stop

Runner owns acquisition and selection in the existing retrieval run. Selection uses a separate tool-free native session with the same loaded `LocalLanguageModel` value. Its transcript never enters the answer session. The selector copies the model handle with request-summary diagnostics enabled; both handles share the same Runtime and loaded engine. No additional weights are loaded by the selector. The answer handle retains its prior diagnostics setting and the backend memory/reload policy is unchanged.

The 30-second timer starts at the native request after prompt preparation, capped by the overall request's remaining deadline. It requests cancellation; it is not a guarantee that native execution ends within 30 seconds. The structured task group joins, the SDK session settles, and the existing Runtime lock is acquired through `processedTokens()` before fallback/answer starts. User Stop or an expired overall deadline stops the whole run, with no answer or fallback execution. A subsequent request can start after settlement.

Native fake checks separately gate cancellation request and executor completion. Before the first channel event, waiting for the executor's cancellation callback before releasing a shielded test gate caused a circular wait with both stream and single-response APIs. The regression checks instead prove that native remains active and answers remain zero after cancellation is requested, then releases execution and verifies settlement. Actual Qwen cancellation latency remains unmeasured.

## Measurements

`selection` records status/reason, menu section IDs, selected IDs, raw response, prepared input count, native input/reused/output usage, elapsed time and cancellation-to-settlement time. Selection time includes preparation and cleanup. Existing Runtime measurements additionally retain actual prefilled and sampled tokens on partial failure/cancellation; a prior request measurement is rejected by request ID. Runtime prefill counts completed forward chunks, not partial work inside a failing kernel; sampled counts exclude EOS. This supplemental measurement applies to the selector, while answer usage remains native-reported. `usageStatus` distinguishes native-reported usage, runtime partial measurements, unavailable usage and no request. A partial measurement's prompt count is the full prepared input, while `prefilledTokens` is the portion actually read; unread tokens are not labeled as reused. `candidateMetrics` measures recall over the original ranges belonging to sections actually included in the final menu, not just their displayed excerpts.

Evaluation `answerInputTokens`, `answerReusedTokens` and `answerOutputTokens` remain separate from selection usage. Their difference gives uncached answer input tokens. The existing top-level token totals include both calls, including failed/fallback selections. An attempted selection with unavailable usage makes the corresponding aggregate unknown instead of adding zero or reporting answer-only costs. Overall answer TTFT and total time start before retrieval; process lifetime peak includes the whole run. These fields do not establish factual correctness.

Model-free manifest validation accepts C only when selection limits are frozen. `--retrieval-output` refuses C because it cannot perform LLM selection without a native model. Native execution also requires the verified model hash; absent assets or unset hashes cannot be counted as successful cases.
