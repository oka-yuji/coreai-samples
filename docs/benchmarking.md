# Model evaluation and reporting

[日本語](benchmarking.ja.md) · [Models](../README.md#models-and-roadmap)

Protocol **1**, adopted 2026-10-19. Use one documented workload and model variant per measurement condition. The same records generate the README's macOS/iOS marks and each model's evaluation tables. A documentation check validates structure and evidence links; model quality and device operation still require the recorded experiments and review.

## What readers see

Every model guide has the same five sections: demo support, performance and quality, verification coverage, evidence/conditions, and setup/progress. A planned model gets the same fields with explicit gaps, so missing work remains visible.

| Display | Meaning |
|---|---|
| ✅ | A native app executed the model on a physical device of that OS, for the variant and route named in its evidence |
| △ | Implementation exists, an older/diagnostic route ran, or device acceptance is pending |
| — | No native app demo is provided on that OS; a conversion CLI may still work |
| Released | The converted bundle is publicly downloadable; separate from demo support |
| Passed / Failed | A stated check and workload, with an evidence reference |
| Not recorded here | The common record has no value; check historical evidence before concluding it was never measured |
| Not measured | No result is available for this scope; never interpreted as zero or a pass |
| Not applicable | Explicitly justified for this model/workload in the source record |

A successful Mac conversion does not set the macOS demo mark. A build, simulator, fictional preview, or fake-based test does not set either mark. An old iPhone diagnostic does not establish the current Models entry. The mark covers only the variants and operations named beside it; user acceptance, broader OS/device coverage and production readiness remain separately recorded.

## Selected metrics

MLPerf Client uses workload-specific quality checks with TTFT, token generation speed, and complete-task duration. We use those measurement ideas in this repository's own protocol. The profile, models and scores here are our own. [MLPerf Client](https://mlcommons.org/benchmarks/client/)

| Area | Fields to collect | Boundary / reason |
|---|---|---|
| Identity and workload | Source revision, converted artifact revision/hash, code revision, precision, tokenizer/preprocessor, device/RAM, OS build, SDK/tools, backend, input/output sizes, batch, concurrency, dataset/fixture revision, seed, generation settings | Conditions that make a comparison interpretable |
| Startup | First load including specialization; load in a new process with an existing prepared cache; prepared-cache bytes | Separate installation-like work from ordinary use |
| Response | Request/pipeline completion, median and p95 with sample count; first meaningful output for streaming models | State exactly when timing starts and stops, including host work and synchronization |
| Resources | Bundle/payload bytes, prepared bytes, peak RSS, peak physical footprint, resident bytes during inference, memory after release relative to baseline | Different measurements; retain units and measurement scope |
| Quality | Fixed held-out task success and modality metrics | Independent from runtime correctness; keep failures and denominator |
| Fidelity | Original precision → quantized reference, matched reference → converted runtime, and original → final output as separate comparisons | State max/mean error, normalization, cosine/top-1 as applicable, and frozen criteria |
| Reliability | Repeated requests, cancellation/retry, release/reload, invalid inputs, corruption/interrupted download, offline behavior | Check the actual app/asset path, not only a fake |

Memory counters are separate quantities. Record raw bytes for new runs, identify process-lifetime versus sampled peaks, and do not add RSS to footprint. Include system pressure/swap/wired observations when relevant. A memory peak is not a proven minimum-RAM requirement. These conventions follow this project's existing Core AI measurement rules.

| Model kind | Additional fields shown | Workload to freeze |
|---|---|---|
| LLM / VLM | TTFT, first visible text, first answer after reasoning (LLM), decode tokens/s; ITL can be added | Prompt tokens, actual output tokens, reasoning, tool calls, cache reuse, context limit; VLM image/frame count and shape |
| Decision | Complete read and reused-document latency | Document tokens, question/option count; held-out labeled answers and calibration if probabilities will be interpreted |
| Extraction | Precision, recall, F1 | Entity/span matching policy, schema, text length, chunking, language and held-out set |
| Embedding/search | Index/query latency, Recall@k, nDCG@k, reference cosine | Corpus size, dimensions, k, relevance labels, threshold, text/image preprocessing |
| TTS | First playable audio, RTF, CER, human listening score | Text/language/voice, seed, audio duration/sample rate, streaming and included stages |
| Image generation | Image completion time, CLIP score, human quality rubric | Resolution, steps, scheduler, seed, guidance, reference images, editing constraints |
| OCR | CER, WER, structure/layout F1 | Normalization, reading order, language, document types and image dimensions |
| Depth | AbsRel, metric-depth RMSE, delta1 | Ground-truth scale, scale/shift alignment, valid-pixel mask and dataset |
| Video | Complete generation time, generated frames/s, human temporal/content rubric | Frame count, FPS, resolution, steps, seed, audio and temporal consistency |

TTFT measures request start to the first output token/chunk; first *visible text* and first *final answer* can be later. Preserve that distinction for reasoning/tool models. `prefillSeconds` alone is not a measured TTFT. Define decode speed and ITL over tokens after the first chunk, including the actual chunk token count; token and chunk counts are not interchangeable. [NVIDIA AIPerf metric definitions](https://docs.nvidia.com/aiperf/reference/ai-perf-metrics-reference)

Retrieval uses ranked relevance measures such as nDCG and Recall; a tiny thresholded example pass rate is not automatically Recall@k. [MTEB paper and evaluation framework](https://arxiv.org/abs/2210.07316)

RTF is generation time divided by generated audio duration; declare whether it includes preprocessing, model loads and saving. It is a speed metric. ASR-based CER/WER and listening answer different quality questions, and CER/WER can exceed 1 when insertions occur. [Coqui synthesis implementation](https://github.com/coqui-ai/TTS/blob/dev/TTS/utils/synthesizer.py), [Hugging Face CER metric](https://huggingface.co/spaces/evaluate-metric/cer/blob/main/README.md)

CLIP scores can support image/text alignment assessment. FID/KID need a suitable reference dataset, sample size and fixed implementation; they are optional here rather than a score computed from a few demo images. Keep image/editing human checks alongside automated metrics. [Diffusers evaluation guidance](https://huggingface.co/docs/diffusers/main/conceptual/evaluation)

AbsRel/RMSE/delta1 are depth metrics only with the stated valid pixels and scale convention. [Depth Anything evaluation implementation](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/util/metric.py)

Energy per request/token and sustained thermal performance are recommended extensions when tools can measure them. Retain power mode, thermal state, sustained-load duration and background activity even when energy is unavailable. Add a new metric with its definition/unit and validation tests before using it.

## Repeatable measurement flow

1. Select the model variant, OS/device and exact app or CLI route. Define the primary metric, quality threshold, workload/fixture revision, input/output shape, seed/settings and timing boundaries **before** measuring. Calibration and held-out evaluation use separate inputs. Changing a threshold creates a new plan version.
2. Generate a plan, fill all fields and freeze its revision/hash:

   ```sh
   mkdir -p .local
   python3 -m Tools.ModelReports template --kind llm --model qwen38-27b > .local/qwen-plan.json
   shasum -a 256 .local/qwen-plan.json
   ```

3. Acquire the shared heavy-job lock from `CLAUDE.md` before loading any model. Use a Release build, one model process and concurrency 1 for the baseline. Record tool versions, power/thermal state, other GPU activity and whether specialization was already cached. Compare variants interleaved under the same conditions.
4. Separate first-ever specialization, new-process cached loading, first inference, and steady inference. The initial project default is one warmup and 20 measured runs per fixed condition. More warmup may be necessary; record the actual count and reason. Also retain a continuous-use run and a controlled release/retry check. Do not fabricate repetitions for old records.
5. Capture run-level data using the selected existing harness. Each JSONL row has `run_id`, `context`, `status` (`warmup`, `passed`, `failed`) and `metrics`. All rows share model, variant, device, OS, workload, cache condition and the frozen `plan_sha256`; each metric has a numeric `value` and a registered `unit`. This collector format does not execute a model. See [the format example](model-reports/run-format.example.jsonl); its values are synthetic and are never imported into model results.
6. Summarize the captured records:

   ```sh
   python3 -m Tools.ModelReports summarize --input .local/qwen-runs.jsonl > .local/qwen-summary.json
   ```

   The collector separates successful, failed and warmup counts. Numeric latency statistics cover successful measured runs, with failures kept visible. Median uses the middle value/mean of the middle pair; p95 uses nearest rank `ceil(0.95*n)`. With fewer than 20 successful samples, p95 is `null`. This is a conservative display rule for this project, not a statistical guarantee: tail claims need enough repetitions and uncertainty analysis for their purpose. Never call p95 the worst case; retain max and raw samples.
7. Review raw evidence, quality/fidelity and operational checks separately. Sanitize public records: no credentials, personal inputs, complete scraped pages, local absolute paths or device identifiers. Existing conversion reports can first pass through `python3 -m Tools.Conversion.summary`; their verification duration is not inference latency.
8. Update the model record, render, check and review. Keep the old condition/result when the model, precision, workload, runtime or OS changes. The production release decision still uses the established ten-item Core AI checklist; generated tables are evidence navigation.

## Records and updating a model

`model-reports/catalog.json` fixes the order/date; `model-reports/metrics.json` owns labels, units and per-kind fields; `model-reports/records/<id>.json` owns each model's results. The docs data is not the app's download catalog.

An evidence entry records a pinned source location, scope (`native_app`, `diagnostic`, `conversion`, `plan`), whether the model actually ran, outcome and context. `standard` requires all context fields and pinned revisions. Dates identify the original result/report; a multi-day window also stays in the workload. Dates for plans/status reviews are labeled separately from execution dates. `legacy` preserves `null` for fields the original report did not record; every missing field is displayed. A historical run is not upgraded by filling guessed values.

A measurement names its metric, value/unit, statistic, sample count, evidence ID and bilingual scope note. `source_value` optionally points to an exact local JSON value, which the checker compares. Derived values such as RTF state their formula and source quantities. Non-comparable inputs remain separate rows. A failed quality gate remains failed even when conversion fidelity passes.

To add a model, copy a planned record such as `model-reports/records/teleocr.json`, change its ID/kind/names/use/overview/release notes/pages/links, and append its ID to `model-reports/catalog.json`. Start with unmeasured checks and no measurements. Create both linked Markdown pages with exactly one `<!-- model-report:start -->` / `<!-- model-report:end -->` pair; add setup text around that block. Fill actual evidence before changing a support mark. Existing narrative guides and historical records remain outside the generated blocks.

```sh
python3 -m Tools.ModelReports render
python3 -m Tools.ModelReports check
python3 -m unittest discover -s Tools/ModelReports/tests -v
```

`render` replaces only marked blocks. `check` rejects stale tables, broken local links/anchors, invalid values/units, missing evidence, and a checked OS supported only by a plan/build/diagnostic. It succeeds with clearly displayed measurement gaps. `check --require-complete <model-id>` additionally requires every common/profile metric to have current-protocol evidence or a justified not-applicable entry, and checks to pass or be justified as not applicable. Completeness must be covered by one pinned variant/device/OS/plan group; measurements from different variants cannot fill each other’s gaps. This checks **report completeness**, not scientific validity or release approval.

`./check.sh` runs the report checker and its tests before the normal builds. Add/update records in the same PR as model or verification changes; review the generated English and Japanese tables, the source measurements, failed criteria and applicable demo route together. Readability and factual evidence require review even when all validators pass.

Structured metrics with task/data/source references also match the model-card reporting approach on the Hub; this tool currently generates repository Markdown. [Hugging Face evaluation metadata](https://huggingface.co/docs/hub/model-cards#evaluation-results)
