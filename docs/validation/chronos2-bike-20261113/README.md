# Chronos-2 Seoul Bike verification

Issue [#530](https://github.com/oka-yuji/coreai-samples/issues/530), 2026-11-13 JST. [Structured results](results.json). [Demo and source terms](../../chronos2-bike.md).

The production Models/Store/Service/Sessions path passed five fixed origins twice: **3,360 raw scalar comparisons**, all with `abs(error) <= 1e-4 + 1e-4*abs(reference)`. Both closes left zero leases. Seoul Bike requests **Core AI CPU-only execution**, EFR=false, using the unchanged FP32 model files. Product sales and NYC retain GPU preference.

## Numerical boundary and retained failure

The five origins are February 15, May 17, August 15, October 5 and November 8, 2018, all at 08:00 Asia/Seoul. Each input contains exactly the preceding 336 rental counts; all 21 quantiles and 16 output hours are compared. Weather and future rentals are excluded from model input. The original UCI CSV, all 8,760 timestamps and source-derived windows were checked independently; native metrics were recomputed from Float32-restored values with exact rational arithmetic, agreeing within `1e-10`.

GPU preference with EFR=false missed the frozen tolerance at May 17, quantile index 0 / step 0: reference -1.236328125, native -1.23602294921875, error 0.00030517578125 versus limit 0.0002236328125. The full production run retained this failure on both passes. EFR=true missed a different November scalar. CPU-only passed the same frozen five windows without changing inputs, quantiles, dtype or thresholds. Official CPU direct forward reproduced the saved pipeline reference exactly. The responsible numerical operation was not identified, and this is not a claim that CPU is universally more accurate.

Maximum absolute difference after the CPU selection was 0.001708984375; the tolerance includes its relative term. Native source is `0008c7d7a74693172a93eb2a1b0de8908e91fb3e`. The later locale-only change and new UI tests are `7251127ee8e49ff0370299818b31be4817487930`; numerical source modules and fixed inputs were compared unchanged, and full checks/normal GUI were rerun.

## Fixed-window diagnostics

Only the first pass contributes quality diagnostics: **80 paired hours**, using the same actuals and denominator for the model and 168-hour seasonal naive. The repeat pass is not another independent sample. All outputs retain raw values; display and metrics alone clip negative forecasts to zero. Quantiles are not sorted and do not guarantee calibrated coverage.

| Origin date, 08:00 Seoul | Model MAE | Weekly naive MAE | Model WAPE ratio | Naive WAPE ratio |
| --- | ---: | ---: | ---: | ---: |
| 2018-02-15 | 28.91 | 120.06 | 0.16144 | 0.67051 |
| 2018-05-17 | 358.37 | 180.56 | 1.98472 | 1.00000 |
| 2018-08-15 | 93.97 | 336.56 | 0.10574 | 0.37872 |
| 2018-10-05 | 1133.75 | 44.56 | 25.44177 | 1.00000 |
| 2018-11-08 | 634.78 | 1181.50 | 18.16905 | 33.81753 |
| Aggregate | 449.95 | 372.65 | 1.69434 | 1.40325 |

The model is worse than the naive baseline on the aggregate of these five windows. Per-hour results for 08:00 through 23:00, five dates per hour, are in `results.json`. These retrospective examples do not establish performance on unseen pretraining data, general demand accuracy or the benefit of weather conditioning. Weather-conditioned and 24-hour forecasts are unsupported.

## Normal Mac application

The ordinary sidebar reached a model-free preview with explicit data loading and disabled prediction. On the last source date, switching 23:00 / 1 hour to 16 hours clamps the origin to 08:00. Opening the prepared model starts a fresh session; loading, selections and language changes do not infer automatically.

May 17/08:00 / 16 hours produced the expected rounded raw-derived values and metrics. EN/JP switching preserved date, results and displayed runtime. The first GUI attempt exposed Japanese AX labels under English body text: `String(localized:bundle:locale:)` did not select the requested bundle language in the observed environment. Explicit `Bundle.localizedString(...localizations:)` fixed this; eight label cases and unavailable metrics went from 10 issues to passing, and normal GUI confirmed both languages.

Temperature and rainfall plots were inspected separately, including negative winter temperature and an explicit last observation before the origin. April 11 was selected from the source as the earliest valid 08:00 window with 16 zero actuals; WAPE displays Unavailable/算出不可 for both methods, with 16/16 and 1/1 pair counts. This additional GUI case is not part of the five-window quality aggregate. The initial GUI plan incorrectly assumed October 5 had zero actuals; source and UI show a total 713, and that assumption was not used as a passing check.

Product sales B1_1 / December 1 / 14 days preserved 13 valid pairs and the missing December 8 actual, including English AX labels. The existing NYC on-device route produced Times Square median 120 and P10–P90=102–146 at January 24, 2025, 17:00 New York, and saved its normal forecast record. Returning to Seoul Bike showed a fresh preview without old data or results. Japanese was restored and the application process exited.

## Timing, memory and validation limits

The two new-session requests took 1.042557 and 0.469694 s. Eight already-loaded requests ranged 0.014244–0.014527 s. These are functional Debug intervals with no warmup or twenty-repeat/cache-isolation protocol, not a controlled performance comparison. Runtime includes bundle validation, verified copying and model loading when needed; source loading, Models import, UI and metrics are outside it.

The process kernel lifetime footprint peak was **56,198,560 bytes**, sampled RSS maximum **643,416,064 bytes**, and sampled footprint maximum **51,889,568 bytes**. Returned SDK field counts were 93. The lifetime includes import and both passes; these are not model-only memory, total GPU allocations, per-request peaks or an iPhone budget.

Host: Apple M5 Max, 137,438,953,472 bytes RAM, macOS 27.2.1 (26B131), Xcode 27.2 (27B52), Swift 6.4.2 and SDK 27.2. AC power and no `pmset` thermal/performance warning were observed before/after. Temperature, utilization and a benchmark-plan hash are unmeasured. The record remains a legacy functional record.

Full repository checks, source reproduction tests, Mac Release and signed iOS Release passed. Actual app bundles were checked for all three source resources, compiled translations, both iOS file-sharing keys and the increased-memory entitlement. Sales passed another 10,752 native scalar comparisons with independent metrics. CPU fakes cover cancellation/stop joins; native in-flight cancellation, physical iPhone operation, spoken VoiceOver and human acceptance remain unverified. The host network was not disabled for the GUI checks.

B1–B6 have the stated evidence. B7 remains for the user's Mac/iPhone interaction. TukuruSF schema 1 worker verification, parent corrections, raw/native/GUI checks, independent AI review and merge are separate stages. Earlier numerical and setup failures remain in the structured record and private logs.

## Integration with public acquisition support

After main added public Models download/import support in PR #563, commit `daf3ea1f2807f6a30121c62f452d8a954e4519a7` repeated both production native tests: 3,360 bike and 10,752 sales scalar comparisons passed, both closes returned zero leases, and independent metrics remained correct. Every Float32 raw output was identical to the earlier CPU-fix run. These are integration repeats, not new quality samples.

Full checks and signed Mac/iOS builds passed again. The acquisition test's CPU catalog check includes all three Chronos demo links while connecting only its taxi adapter. Normal Mac GUI retained the previously imported model without reacquisition, opened Seoul Bike, predicted the fixed May 17 window and preserved values and runtime across correctly translated EN/JP AX labels. The app was closed with Japanese restored. This follow-up does not claim a fresh GUI network download or physical iPhone acceptance.
