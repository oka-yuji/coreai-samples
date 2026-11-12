# Product sales with Chronos-2

[日本語](chronos2-sales.ja.md)

The Mac sidebar and iOS Models offer **Product sales forecast** under Chronos-2. Both use the existing managed FP32 artifact (`chronos2-fp32`), Models preparation, owner permissions and resource leases. Select **Load public sales data**, choose a product, a Europe/Rome start date and 7 or 14 days, then explicitly select **Predict on this device**. Opening, importing, preparing and changing selections never starts prediction or data acquisition. A model-free preview can acquire and inspect data; it cannot produce predictions or substitute cached model results after an error.

## Source and terms

[Hierarchical Sales Data, Mendeley Data v1](https://doi.org/10.17632/njdkntcpc9.1), dataset `njdkntcpc9`, is the source. The dataset creators are Paolo Mancuso, Veronica Piccialli and Antonio M. Sudoso (2021). [UCI's listing](https://archive.ics.uci.edu/dataset/611/hierarchical+sales+data) labels CC BY 4.0; the original Mendeley distribution specifies **CC BY-NC 3.0**. Use the original noncommercial terms. The app displays the source link and this distinction beside the data action. No original or derived real sales values are bundled in source, tests or resources.

The only acquisition URL is:

```text
https://data.mendeley.com/public-files/datasets/njdkntcpc9/files/08bb4f43-6dfa-4995-b268-42fa0690ba6b/file_downloaded
```

Admission requires 888,985 bytes and SHA-256 `0dfd4a5bf801bf79ddb5fb1f6bc8d023487c36dd6a2a70b312cdfa9fa6568d83`, before parsing or caching. Responses have a 2 MiB hard limit. Standard provider redirects are permitted; content identity is still mandatory. Actor-owned I/O reads and atomically replaces only this app's pinned sales cache. Every read revalidates content, including offline reopen. An invalid cache is reported; another explicit load retries acquisition. [Reacquisition script](../Tools/Forecasting/README.md).

## Input and comparison

The strict quoted-CSV parser validates all 237 columns: DATE, the 118 QTY columns and their corresponding binary PROMO columns. It rejects malformed quoting or row/header widths, duplicate or unordered dates, invalid calendar dates, nonfinite/overflowing numbers, negative or fractional counts and nonbinary flags. Four SKUs are fixed without reference to prediction performance: B1_1, B2_1, B3_1, B4_1. Promotion flags are validated but never used as model input.

The declared daily Europe/Rome grid is 2014-01-01 through 2018-12-31: 1,826 dates from 1,798 source rows. All 28 missing dates, including the first date, remain missing. Missing does not mean zero or a known closure. Calendar arithmetic advances days through DST; it does not add 86,400 seconds. Chart dates and per-date accessibility labels explicitly use Europe/Rome, independently of the host timezone and display language.

Each request captures the preceding 365 calendar days strictly before the selected origin, plus held-out actuals and a baseline for the selected horizon. Selectable origins keep the entire horizon within the source range. The last seven calendar days before origin form the seasonal naive baseline; a 14-day forecast repeats that same week twice. No second-week actual becomes a baseline input. Missing baseline and actual observations remain nil; missing context becomes NaN for `ChronosContext`, which adds three left-padding NaNs to this 365-point input.

The existing model contract is single-series FP32, 21 quantiles × 16 steps, official revision `29ec3766d36d6f73f0696f85560a422f50e8498c`. All 336 raw scalars, actual bundle identity, captured request, runtime seconds and timing boundary are retained. Display selects P10/index 2, median/index 10 and P90/index 18, then the first 7 or 14 steps. Negative model values are clamped only for presentation and metrics. History uses points and actuals use squares, so missing observations are never connected. Baseline lines break at missing dates.

MAE and WAPE use the same valid actual/median/baseline triples. Both errors share a comparison count and denominator. WAPE is the sum of absolute errors divided by the sum of actuals; a zero denominator is unavailable, while MAE may still be defined. No valid triples yields no MAE/WAPE. Observed sales are not latent demand, and P10–P90 is not a calibrated coverage guarantee. The model does not support 28 days, covariates or recursive extension in this demo.

## Ownership and verification

`ForecastFeature` contains pure dataset, request, metrics and ticket state. Shared Store owns tasks; ForecastRunner joins in-flight operations even when cancellation is ignored. The MainActor service authorizes the request against the loaded dataset and checks native identity against the managed resource. Each Models session has a fresh service and ChronosRuntime. Stop revokes permission before cancellation, joins work, releases the runtime and returns to Models for lease release. Old completions and old closes cannot revive or release a later session.

Default CPU tests cover synthetic parsing, calendar/DST, missing dates, no future leakage, common-pair metrics, explicit actions, stale completions, stop/retry, repository admission/cache and managed session ownership. They make no network requests and execute no model. Native tests are disabled unless **all four** variables below are set; use the repository's shared heavy-job lock:

```sh
COREAI_FORECAST_BUNDLE=/path/to/chronos2-bundle \
COREAI_SALES_CSV=/path/to/private/sales.csv \
COREAI_FORECAST_REFERENCE=/path/to/private/reference.json \
COREAI_FORECAST_REPORT=/new/private/report.json \
swift test --package-path CoreAISamplesKit --filter ForecastNativeTests.sales
```

Reference shape (placeholders describe arrays; no real observations are redistributed):

```text
{"cases":[{"id":"unique-case-id","series_id":"B1_1","origin_index":<checked index>,"horizon":14,"context":[<365 floats or null>],"quantiles":[<21 arrays of 16 floats>]}]}
```

Use multiple SKU/date cases generated independently from the original source and official FP32 model. The test exercises production Models/Store/Service/Sessions, replaces rendering only, and supplies already acquired source bytes through the repository's I/O boundary. It checks every context (nil to NaN), raw scalar with `abs(error) <= 1e-4 + 1e-4 * abs(reference)`, identity, zero leases after close, and repeats after reopen. Existing report paths are refused. The report contains private source-derived inputs and raw results, source/reference hashes, common-pair metrics, runtime seconds, OS/machine/precision and memory samples. Keep it private.

Runtime time covers the runtime request through validated output, including bundle hashing, snapshot copying and loading when needed. It excludes sales acquisition, Models import, UI rendering and metric calculation. RSS and physical footprint are point observations; the kernel peak covers the whole test process lifetime, not a per-request peak or GPU allocation sum. Diagnostics check SDK-returned field counts and encode unsupported memory fields as null.

[Recorded verification](validation/chronos2-sales-20261113/README.md) covers all 16 fixed cases twice, independent metric arithmetic, normal Mac interaction and signed host builds. Physical iPhone inference and user acceptance remain open. The fixed backtest and functional timing do not establish general forecasting quality or a controlled performance benchmark.
