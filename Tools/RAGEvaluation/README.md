# Fixed RAG evaluation

This diagnostic follows issue #373 and the Qwen3.8 Work comparison in #378, plan v1.1.
It uses the existing conversation runner and source ledger. Normal app launches retain their existing behavior.

Only `flat` is implemented here. `structure-bm25` and `structure-llm` fail validation until their implementations arrive.

## Corpus and model identity

`corpus.json` contains self-authored, fictional manuals and glossary distractors: 12 development questions and 24 evaluation questions. Each evaluation category has three Japanese questions and one English question. Documents and some facts are shared between splits; this is a question split, not an unseen-document generalization test. Do not use the evaluation questions to tune retrieval or prompts.

The categories cover heading context, repeated headings, paraphrases, multiple evidence ranges, text without headings, and absent answers. The glossaries deliberately contain query vocabulary without answers. Heading-to-body distance and distractors make returning every candidate within the source budget less likely. Actual token counts and comparison sensitivity still require the pinned tokenizer and model.

The manifest pins normalized UTF-8 text by SHA-256 and gold evidence by document identity, version, Swift Character offset and exact UTF-8 content. Offsets are not Python code-point indices or UTF-16 offsets. The native validator checks raw HTML/text normalization as well. Any corpus or setting change changes the recorded manifest hash.

`settings.expectedModelHash` starts unresolved. Before model execution, independently identify the approved Qwen3.8 27B 4bit bundle and freeze its `ModelBundle.mainHash` value here. Do not copy the hash from an arbitrary bundle just to satisfy validation. Model execution rejects an unresolved or different hash before loading the model. The driver additionally records all bundle file hashes. No model download, conversion or training is performed.

## Validate without a model

```sh
CoreAISamples --rag-eval Tools/RAGEvaluation/corpus.json --validate-only
swift test --package-path CoreAISamplesKit --filter RAGEvaluationTests
python3 -m unittest discover -s Tools/RAGEvaluation/tests -v
CoreAISamples --agent-selftest --offline
```

Validation checks the corpus, not model quality. The offline app tests use a fake Foundation Models executor and do not load learned weights.

## Run a comparison input

After freezing the verified model hash, use a new output directory:

```sh
python3 -m Tools.RAGEvaluation \
  --app /path/to/CoreAISamples.app/Contents/MacOS/CoreAISamples \
  --manifest Tools/RAGEvaluation/corpus.json \
  --model /path/to/approved-qwen38-4bit \
  --strategy flat --mode work --split development \
  --output .local/rag-runs/new-development-run
```

The driver checks for existing model processes, owns the shared heavy-job lock, runs one case per fresh process and archives the lock only after confirmed process cleanup. An unconfirmed child retains the lock; do not delete it to continue. `COREAI_HEAVY_JOB_LOCK`, when used, must match all other model jobs on the machine.

The native single-case entry point is:

```sh
CoreAISamples --rag-eval Tools/RAGEvaluation/corpus.json \
  --case heading-evaluation-1 --mode work --rag-strategy flat \
  --model /path/to/approved-qwen38-4bit --output /new/case-directory
```

Acquire the same heavy-job lock before invoking a model case directly. The driver is the preferred entry point. The old `--agent-e2e` contract remains unchanged; it skips precollection and does not substitute for this evaluation.

All retrieval is replayed. Unknown search queries and URLs produce recorded errors; neither precollection nor later Work tools fall back to the network. Treat a replay miss as an incomplete experiment, not as evidence of a model's factual error. If additional request keys are needed, prepare and freeze them before rerunning every affected comparison pair.

## Records and interpretation

- `plan.json` fixes the requested case identities, manifest hash, mode, strategy, settings and executable/bundle identity.
- `manifest.json` retains the exact input bytes in the run directory and in each native case directory.
- `case-NNN/result.json` contains native results, candidate and supplied original passages, source validity, usage, timing, memory and replay errors.
- `case-NNN.control.json` separately records parent-side launch and confirmed termination. A native completion cannot override unconfirmed child termination.
- Each case retains its conversation JSON/Markdown and isolated notes. The result only counts as completed when its conversation was saved.
- `summary.json` keeps every requested case. Missing results are `notRun`; attempted cases with unconfirmed termination are `running`. Native `cancelled` results remain cancelled even when the process exits nonzero.

`completed` is an execution state, not a correct-answer grade. Answer correctness, factual support and correct abstention remain ungraded for human review against the gold and the actual supplied evidence. UI placeholders are not model answers.

Retrieval recall counts fully covered gold ranges of the correct document/version across all supplied passages, including original headings. Hit@1/@3 use the first one/three supplied body passages in retrieval order; heading-only passages do not consume body ranks. Absent-answer cases have no recall score; do not count them as perfect recall.

These retrieval metrics describe precollected RAG sources. Later Work tool responses are retained in the conversation archive and their requests in the result; use those actual responses too when judging the final answer's support. Do not treat precollection recall as end-to-end agent recall. A missing driver control record is unconfirmed termination, even if the native record says completed.

`answerTTFTSeconds` starts immediately before the Host request, including retrieval, and ends when the first nonempty displayed answer segment is observed. Thinking, tool calls and the no-answer placeholder do not count. It is an observed response time, not GPU prefill time or a guarantee that the final answer is already complete. `totalSeconds` includes request completion and conversation saving; model loading is outside this interval. Usage comes from the native session, including Work's multiple steps. `selectionSeconds` is zero for A because A performs no extra LLM selection; lexical ranking is included in retrieval time.

`lifetimePeakFootprintBytes` comes from the kernel's process-lifetime physical-footprint peak after the run, including model loading. It is not RSS or a difference between sampled memory values. Each model case uses a fresh process. An unavailable value remains unavailable.

To recheck the result identities and regenerate the full-denominator summary:

```sh
python3 -m Tools.RAGEvaluation --summary-only --output .local/rag-runs/new-development-run
```

Keep model paths, conversation data and raw logs local. Publish reviewed summaries with their conditions and limitations. Issue #377 owns the real-model A/B/C comparison and adoption decision.
