# TeleOCR fixed precision evaluation — 2026-10-28

This records the macOS CLI page pipeline. It does not establish native-app or iPhone acceptance. [Machine-readable evidence](summary.json) contains the unrounded values, asset hashes, comparison counts and canonical performance runs.

## Protocol

M4 Max (Mac16,6), 137,438,953,472 bytes RAM; macOS 27.0 (26A428), Xcode 27.0 (27A266a), Swift 6.4. Core AI used `SpecializationOptions.default` with `expectFrequentReshapes=true`; the actual compute unit was not instrumented. PyTorch references used MPS. The source model is pinned at `70a19ec6a3b036311af19ae5e762e650f41a5980`.

The frozen plan hash is `eadd399a8942f11d6e907df4ed86a05bffa9c6ba4ad8f43e58387508ad758b1b`. The numerical CLI/reference path is from `ea722fca17aaf0a469d9d0cbf8271b7c3d348e8d`. The later SIGHUP supervision fix changes cleanup, not the inference code or the measured executable.

- 35 synthetic pages and 24 selected OmniDocBench annotation crops are separate sets. The latter is an element-crop subset, not a full-page OmniDocBench benchmark. Input images and ground truth were frozen by hash; private raw images and text are not redistributed here.
- Every condition uses layout at 1036 square, then native-resolution crops with text/table prompts, greedy decoding, repetition penalty 1.05 and a 4096-token generation cap per request.
- Original BF16, uncompressed FP32 control, matching compressed FP32 references and Core AI are separate conditions. Original-to-compressed quality and reference-to-Core-AI output agreement remain separate axes. FP32 reference scales and exported FP16 scales are distinct numerical conditions.
- 531 quality processes and 88 performance processes completed: 615 passed and 4 warmups, no process failure or non-EOS result. Each Core AI condition has one specialization-cold load, one warmup and 20 interleaved fresh-process measurements of F4. Cold p95 is not estimated from its single observation.
- `pipeline_s` includes image decoding, layout/crops/recognition and OTSL/Markdown assembly. Loading, file I/O and JSON serialization are outside it. The old `generate_s` keeps its narrower runtime interval.
- Only the selected specialization entries were cleared. OS file and Metal caches were not flushed. Power/thermal-warning observations were recorded before and after performance runs. No temperature or GPU-utilization measurement was made.

## Performance

| Condition | First load s (n=1) | Cached load s (median, n=20) | Page s (median / p95, n=20) | Output tokens (median) |
|---|---:|---:|---:|---:|
| FP16 | 42.509 | 1.255 | 6.144 / 6.781 | 288 |
| Decoder INT8 | 42.537 | 0.969 | 6.098 / 6.558 | 288 |
| Decoder INT4 | 42.834 | 0.949 | 6.007 / 6.493 | 287 |
| Vision INT8 + decoder FP16 | 41.546 | 1.155 | 8.588 / 10.652 | 288 |

GB = 10^9 bytes, GiB = 2^30 bytes. File sizes are logical bytes; shared directories are not a total for installing every condition. Cold memory is one process-lifetime peak; prepared memory is the maximum of 20 process-lifetime peaks. Both include load. RSS and footprint are different measures and are not added. These peaks do not specify a minimum device RAM.

| Condition | Bundle GB | Cache GB | Cold RSS GiB (n=1) | Prepared RSS GiB (max, n=20) | Cold footprint GiB (n=1) | Prepared footprint GiB (max, n=20) |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 2.847 | 6.686 | 9.368 | 8.453 | 3.521 | 3.051 |
| Decoder INT8 | 2.421 | 4.498 | 6.262 | 5.371 | 3.562 | 3.108 |
| Decoder INT4 | 2.201 | 4.278 | 5.913 | 4.983 | 3.744 | 3.126 |
| Vision INT8 + decoder FP16 | 2.187 | 4.703 | 7.666 | 6.606 | 3.609 | 3.126 |

## Quality and reference agreement

All four Core AI conditions retain the same per-sample scored judgments: synthetic 33 good + 2 conditional; OmniDocBench 18 good + 1 conditional + 4 insufficient + 1 unscored. The unscored case, `OD-table-en-02`, was classified as text by layout, so the page path did not emit table OTSL. It remains in the denominator and is not counted as a successful table.

| Set | Condition | Exact page prediction + EOS | Equal joined text | Scored judgments changed vs matching FP32 reference |
|---|---|---:|---:|---:|
| synthetic | FP16 | 33/35 | 35/35 | 0 (0 unscored) |
| synthetic | Decoder INT8 | 31/35 | 34/35 | 0 (0 unscored) |
| synthetic | Decoder INT4 | 33/35 | 34/35 | 0 (0 unscored) |
| synthetic | Vision INT8 + decoder FP16 | 33/35 | 34/35 | 0 (0 unscored) |
| omnidocbench | FP16 | 23/24 | 23/24 | 0 (1 unscored) |
| omnidocbench | Decoder INT8 | 23/24 | 23/24 | 0 (1 unscored) |
| omnidocbench | Decoder INT4 | 20/24 | 22/24 | 0 (1 unscored) |
| omnidocbench | Vision INT8 + decoder FP16 | 23/24 | 24/24 | 0 (1 unscored) |

Exact page prediction includes layout, boxes, labels, prompts, text and OTSL. A layout/box difference with identical text still fails that criterion. The mismatch fraction in the common model report is this binary page criterion, not CER or logit nRMSE. Output differences have not been relabeled as numerical-tolerance passes. Conversely, these observations alone do not identify a converter defect.

The uncompressed BF16-to-FP32 control is itself exactly equal in only 23/35 synthetic page predictions, with unchanged judgments. Dtype effects must not be attributed automatically to compression. Component logit/vision checks remain separate evidence in [the quantization validation](../../teleocr-quantization/2026-10-27.json).

## Prior predictions and decision

- FP16 preserved the previous 35/35 judgments. Decoder INT8 had zero synthetic judgment drops, within the prior expectation of at most a few.
- The prediction of increased INT4 degradation in numbers and tables was not borne out on the fixed synthetic set. On the numbers stratum, CER rose from 0.0193101 to 0.0206901, while numeric recall/precision rose from 0.9742063 to 0.9861111. Table cell agreement stayed 0.9847222, numeric agreement 1.0 and strict shape agreement 1.0. This is not a claim of general INT4 improvement.
- Vision INT8 is not selected for the macOS distribution. Its F4 median page time is about 43% longer than decoder INT4 with a similar bundle size, higher RSS/cache use and the same fixed-set judgment categories. It remains an independently reported condition. This does not decide an iOS configuration.

## Limits

The scoring thresholds and input sets were not changed after seeing results. Small selected crops and synthetic pages are not proof of held-out real-document quality. The OmniDocBench subset uses the research-only data policy described in [the evaluation tool](../../../../Tools/OCREval/README.md). Real-photo review, normal Models UI, interactive real-model stop/retry, long-lived release/residency, iPhone and distribution validation remain separate tasks. CLI execution does not change the native-demo support marks.

The common report retains insufficient/unscored quality results and non-exact reference criteria as failures of those stated criteria. Successful process execution is a separate observation.
