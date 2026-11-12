# Chronos-2 product sales verification

Issue [#529](https://github.com/oka-yuji/coreai-samples/issues/529), 2026-11-13 JST. [Structured results](results.json). [Usage and source terms](../../chronos2-sales.md).

The production Models, Store, service and session path passed 16 fixed SKU/date cases and the same cases after release/reopen: **10,752 raw quantile comparisons**. Every scalar met the fixed `abs(error) <= 1e-4 + 1e-4*abs(reference)` threshold; maximum absolute difference was `8.392333984375e-05`. Both closes left zero model leases. The test replaces rendering and supplies already acquired original source bytes, so normal Mac UI was checked separately.

## Inputs and independent checks

Four SKUs were fixed before model evaluation: B1_1, B2_1, B3_1 and B4_1, at March 26, 2016 and June 1, September 1 and December 1, 2018. Each request uses the preceding 365 calendar days, with three left-padding NaNs. The source's 28 missing dates remain missing. Official `Chronos2Pipeline.predict` FP32/CPU references use the pinned model revision, context length 8192, one series/batch, 16 output steps and `cross_learning=False`. All 21 quantiles are retained.

The original [Mendeley v1 dataset](https://doi.org/10.17632/njdkntcpc9.1) is credited to Paolo Mancuso, Veronica Piccialli and Antonio M. Sudoso (2021). Its CC BY-NC 3.0 terms differ from UCI's CC BY 4.0 listing. Original observations, derived time-series fixtures and raw native reports are not distributed here. The app acquires the pinned original only after an explicit action. Aggregate validation statistics are recorded below; they do not reconstruct the source time series.

An independent Python parser reproduced every context, actual and repeated seven-day baseline. The native JSON was then decoded back to Float32 and checked using exact rational arithmetic, with final metric doubles agreeing within `1e-10`. The second pass is a stability check, not additional quality samples.

| Fixed first-pass diagnostic | Chronos median | Seven-day seasonal naive |
| --- | ---: | ---: |
| MAE, count | 4.44884534 | 5.1509434 |
| WAPE, ratio | 0.901678022 | 1.04397706 |

There are **212 comparable pairs out of 224**. Both methods use the same pairs and denominator. Missing actuals/baselines are excluded, not zero-filled. Negative predictions are clipped for the chart and metrics, while raw quantiles are unchanged. These are fixed retrospective demo diagnostics; no held-out quality, calibration or pretraining-overlap claim follows from them.

## Normal Mac application

The ordinary sidebar reached the interactive preview; prediction stayed disabled until opening the prepared model. Source loading, product/date/horizon changes and language changes did not start inference. B1_1 at December 1, 2018 for 14 days and B2_1 for 7 days produced real native results and correctly rounded metrics. December 8 remained a missing actual. EN/JP switching preserved selections, result and runtime value. Leaving for the existing NYC preview and returning started a fresh sales preview. The original Japanese setting was restored and the app process exited.

Initial inspection found that global numeric day indices compressed the chart at its right edge. Commit `1954168e1dcfd8dd8f7af438c0370d080b0b5a31` sets the visible first-to-last domain; the resulting chart spans the history and forecast correctly. The accessibility dates were already correct, illustrating why source/AX checks do not replace rendered inspection. Real sales screenshots remain private.

The native comparison uses `55df3378a40a98f8e34c6fafa8e8a33bb1001e71`; the later change affects chart presentation only. The normal Mac check uses `1954168e1dcfd8dd8f7af438c0370d080b0b5a31`.

## Timing, memory and limits

The two new-session runtime requests took 1.714539 and 1.328287 seconds. Thirty already-loaded requests ranged from 0.004353 to 0.007274 seconds. These are Debug functional-test intervals, not a controlled benchmark: no warmups, isolated specialization-cache experiment or twenty-repeat protocol was performed. Source download, Models import, UI and metric calculation are outside the runtime interval.

The whole test process's kernel lifetime physical-footprint peak was **212,411,176 bytes**. Before import and after final close, point footprints were 11,322,088 and 66,159,552 bytes. Point RSS/footprint and lifetime peak are distinct; none is a GPU allocation sum, model-only residency or iPhone budget. Returned SDK field counts were checked. The native test logged two nonfatal MPS-import messages; actual delegate placement was not profiled.

Host: Apple M5 Max, 137,438,953,472 bytes RAM, macOS 27.2.1 (26B131), Xcode 27.2 (27B52), Swift 6.4.2 and SDK 27.2. Power was AC before/after; `pmset` reported no thermal/performance warning. Temperature, GPU utilization, other host activity and a benchmark-plan hash were not measured. This remains a legacy functional record.

Required repository checks, full CPU tests, Mac Release and iOS Release passed. The isolated checkout initially lacked its ignored local signing override and produced an unsigned iOS app; that result was retained and corrected. Host `codesign` verification then passed, with both sharing keys and the increased-memory entitlement in the actual artifact. A sandbox-only trust error was not treated as signature failure or bypassed by changing trust settings. String Catalog translations, compiler call-site keys and explicit English compiled values were checked.

S1–S6 have the stated source, CPU, native, normal Mac and build evidence. S7 remains for the user's Mac/iPhone interaction. Physical iPhone inference, native mid-request cancellation and general forecasting quality remain unverified. Stop/join, delayed completion and permission changes have CPU fake coverage. The ordinary host network was not disabled; offline cache behavior has a separate deterministic test.
