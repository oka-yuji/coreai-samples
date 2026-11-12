# Seoul Bike forecast

[日本語](chronos2-bike.ja.md)

The Seoul Bike demo uses the existing Chronos-2 FP32 artifact to compare hourly rental predictions with held-out observations and a weekly seasonal baseline. It shares the Forecast Store, runner, service, sessions and model ownership with product sales. NYC taxi and product sales remain separate destinations.

## Use the demo

1. Open **Chronos-2 → Seoul Bike forecast** in the Mac sidebar, or select the Seoul Bike demo in iOS Models. Prepare the same Chronos-2 artifact through the available Models acquisition controls.
2. Choose **Load bundled bike data**. The verified sample is bundled; loading needs no network. The model-free preview can show history and past weather, but cannot predict.
3. Choose a date and whole hour in **Asia/Seoul**, then a horizon of **1, 4, 8 or 16 hours**. Dates and hours are limited to those with enough history and held-out observations. Increasing the horizon clamps a late origin to the final valid hour.
4. Choose **Predict on this device**. Opening the screen, preparing the model, loading data, changing selections or changing the display language never starts inference. Stop and close wait for in-flight operations and model release before Models returns its lease.

## Input and display

The single series, `seoul-bike`, contains 8,760 hourly rental counts from 2017-12-01 00:00 through 2018-11-30 23:00 in Asia/Seoul. Input is exactly the **336 observations before the origin**. The weekly baseline uses the corresponding hours **168 hours before the origin**. Neither input includes future rentals, weather, holidays or operating flags.

The chart shows the last 72 hours, native median and P10–P90, held-out actuals and seasonal naive. The original 21 × 16 native quantiles, request timestamps, actuals, baseline and model identity remain intact. Display and metrics clamp negative predictions to zero. MAE and WAPE use the same valid actual/median/baseline triples; compared and possible hour counts are shown. WAPE stays unavailable when actuals sum to zero and may exceed 100%. P10–P90 is not a coverage guarantee.

Separate temperature (°C) and rainfall (mm) charts show only the 72 hours **before** the selected origin, plus the last observed holiday and operating state. These are historical measurements, not a weather forecast or a model input. Weather-conditioned, 24-hour and recursive forecasting are unsupported. The 295 non-functioning rows contain observed zero rentals during shutdown; they do not establish zero latent demand.

## Source and reproducibility

Seoul Bike Sharing Demand [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5F62R.

License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). The source lists no individual creators.

Modification: Encoding only, from Windows-1252 (CP1252) to UTF-8. CRLF line endings, headers, rows, units and values are unchanged. No imputation or aggregation was performed. The original notice is bundled as `ForecastRuntime/Resources/SeoulBike/NOTICE.txt`; [source preparation](../Tools/Forecasting/bike-data.md) records the original archive, hashes and reproduction commands.

SwiftPM copies the SeoulBike directory without processing its CSV. Each load checks the exact CSV, manifest and notice hashes before parsing. `BikeDataset.parse` independently validates headers, hourly order, complete dates, integer counts, finite weather, categories and shutdown zeros. `pastWeather(originIndex:count:)` accepts only valid bike origins and 1–336 preceding rows. Sales do not fabricate weather, and their pinned download/cache and Rome calendar-day semantics are retained.

## Verification boundaries

CPU checks exercise bundled source admission, malformed data, history and horizon bounds, weather isolation, kind isolation, explicit actions, delayed stop and fresh sessions. The supervised independent verifier compares all 8,760 hours and five input/actual/baseline/weather windows with a separate source oracle. Ordinary tests do not load a model or use the network.

```sh
swift test --package-path CoreAISamplesKit --force-resolved-versions --filter 'Bike|Forecast|Chronos'
python3 -B -I -m unittest discover -s Tools/Forecasting -p test_bike_source.py -v
```

The native harness is opt-in and requires the shared heavy-job lock:

```sh
COREAI_FORECAST_BUNDLE=/path/to/chronos2-bundle COREAI_BIKE_REFERENCE=/path/to/bike-reference.json COREAI_BIKE_REPORT=/new/private/bike-report.json swift test --package-path CoreAISamplesKit --force-resolved-versions --filter ForecastNativeTests.bike
```

The fixed reference supplies five distinct origins with `series_id=seoul-bike`, `horizon=16`, 336 context values and all 21 × 16 quantiles. The test uses production Models/Store/Service/Sessions, checks every context value and raw scalar (atol=rtol=1e-4), closes with zero leases, then reopens and repeats. The new, non-overwriting private report retains raw predictions, actuals, baseline, paired metrics, timing, model identity and memory boundaries. Its normalized source SHA-256, original CSV SHA-256 and source manifest SHA-256 are distinct fields. Current memory samples and process-lifetime kernel peak are not interchangeable.

The version 2 report records `scalarFailures` for each case and the complete run. Scalar mismatches keep the test failed while allowing the remaining outputs and lease checks to be recorded. A report's existence does not mean the comparison passed: require a successful test exit and zero scalar failures. Invalid inputs, identities, shapes or lifecycle conditions still stop the test immediately.

Native execution, normal Mac/iPhone UI, accessibility, signed builds and user acceptance are supervisor checks. Their outcomes are not inferred from CPU success or from this harness being present.
