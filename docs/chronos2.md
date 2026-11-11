# Chronos-2

[Home](../README.md) · [日本語](chronos2.ja.md)

A reproducible FP32 Core AI conversion of Amazon Chronos-2 for one time series and a 16-step quantile forecast. The Mac Python Core AI path passes 18 fixed synthetic cases. Swift runtime, Models and the NYC sample are tracked in [#548](https://github.com/oka-yuji/coreai-samples/issues/548).

## Conversion contract

Use [the maintainer recipe](../Tools/Conversion/recipes/chronos2/README.md) with the pinned official snapshot. Inputs contain 1-8192 observations; missing values and left padding are NaN. The native input is Float32[1,T], T a multiple of 16. The output is Float32[1,21,16], with no state. This profile does not expose covariates, cross-learning, quantization or longer forecast horizons.

No weights are hosted by this repository. The recipe creates a local bundle and inventory. A fresh conversion can have a different asset/manifest hash; update the trusted catalog only after verifying that exact bundle.

## Conversion verification

[The recorded conditions and failures](validation/chronos2-conversion-20261111/README.md) distinguish conversion fidelity from forecasting quality. All 6,048 native output values pass `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`. This does not establish taxi-demand accuracy, performance or iPhone application support.

## Swift runtime

`ChronosCore` validates raw contexts and all 21 quantiles. `ChronosRuntime` owns one request at a time and waits for load/inference completion before releasing its model. The [Mac Swift verification](validation/chronos2-swift-20261112/README.md) passes the same 18 cases and a release/reload repeat. The fixed NYC route also passes production-service comparison and normal Mac UI checks; physical iPhone inference remains unverified.

```sh
swift run --package-path CoreAISamplesKit chronos-verify --help
```

Explicit model commands require the shared lock in `AGENTS.md`. JSON input is `{"context":[1,2,null,4]}`; `null` is missing. Results preserve the full raw `[1,21,16]` tensor and identity. Output files must be new and have an existing parent directory.

## Models import and preview

Models on Mac and iOS lists Chronos-2 with the fixed NYC taxi pickup forecast. Select **Import model** and choose the complete prepared Chronos-2 FP32 folder. The nine files use 478,864,956 bytes (about 479 MB); allow additional space for the managed copy. This build provides no download URL. Opening Models, importing or inspecting the bundle does not start inference. The interactive precomputed preview works without a model; a prepared model enables explicit on-device prediction. A native failure never switches modes automatically.

The trusted artifact is `chronos2-fp32`, contract `chronos2-fp32-v1`, from `amazon/chronos-2` revision `29ec3766d36d6f73f0696f85560a422f50e8498c`. Its `chronos2.json` SHA-256 is `786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572`. Models requires the exact original manifest bytes and all nine pinned file sizes and hashes. That byte identity fixes the metadata contract, including the state-free Float32 `[1,T]` to `[1,21,16]` ABI, without loading the model or creating a runtime snapshot during import. Even reformatting the manifest requires a new trusted identity.

The existing Models store copies and checks the folder before promotion. Altered, missing, extra or symbolic-link entries are rejected; a rejected replacement preserves the previous registration and input folder. Active leases prevent replacement and deletion until the last lease is released. The source folder is retained when the managed copy is deleted.

CPU checks cover catalog/preview behavior, history windows, stop/owner boundaries and synthetic storage. The opt-in real-bundle import check is `COREAI_CHRONOS2_IMPORT_BUNDLE=/path/to/chronos2-bundle swift test --package-path CoreAISamplesKit --filter ChronosImportTests.knownBundle`, under the shared heavy-job lock. It checks import, full inspection, leases, rejected replacements and source hashes without inference. Its default skip is not an import pass; native import, app builds, iPhone inference and normal UI acceptance need separate evidence.

## Fixed NYC demo

Choose one of three January 2025 scenes, a Manhattan taxi zone and a half-hour forecast interval. Advancing reveals only the bin that just elapsed. The forecast keeps its original start time; another explicit prediction appends only elapsed truth to the last 1,008 observations. Forecasts can start at steps 0…12, with four displayed steps and 16 available truth bins. Reset clears the forecast. Changing the display language, zone or interval does not run the model or change its inputs.

The map, numbers and chart share one Feature state. The model reads 21 days, while the chart shows the last four hours and the next two hours. Counts mean recorded pickups, not passengers or unmet demand. The shaded 10–90% quantile range is not a calibrated coverage guarantee. Apple basemap tiles require network access; fixed data and inference do not. Sources, hashes, aggregation, attribution and offline reproduction are in [Tools/NYC](../Tools/NYC/README.md).

Native results retain the complete 21×16 raw tensor and the actual verified bundle identity for every zone. Only presentation clips negative counts. Preview results are explicitly precomputed, retain the supplied 3×4 projection and have no device processing time. The native loader checks the managed manifest identity on its verified snapshot before constructing a Core AI model.

`ChronosFeature` is pure; FeatureStore owns operation tasks, the runner joins the complete 20-series operation, and sessions revoke permission before stopping. Models returns the lease only after session stop and runtime release. Each session gets a fresh service. Preview also uses a Store and stops when hidden or backgrounded.

The opt-in test below requires the shared heavy-job lock, a prepared model and the pinned official fixed-scene reference. It drives production Models/Store/Service/Sessions, compares all 20,160 raw values (60 series) at frozen `atol=rtol=1e-4`, closes the demo and requires zero leases. The reference SHA-256 is fixed in the test. Missing prerequisites are failures when opted in. Default CPU runs skip it; worker CPU success does not establish D5, app/device builds or D7.

```sh
COREAI_CHRONOS2_DEMO_BUNDLE=/path/to/chronos2-bundle \
COREAI_CHRONOS2_DEMO_REFERENCE=/path/to/fixed-scenes-reference.json \
COREAI_CHRONOS2_DEMO_DATA=/path/to/NYCData \
COREAI_CHRONOS2_DEMO_REPORT=/new/report.json \
swift test --package-path CoreAISamplesKit --filter ChronosDemoNativeTests.fixedScenes
```

`COREAI_CHRONOS2_DEMO_DATA` is optional and defaults to the bundled data; the report is optional and never overwrites. [The recorded verification](validation/chronos2-demo-20261112/README.md) covers raw-source reproduction, native comparison, normal Mac UI, localization and host builds. Physical iPhone inference and user acceptance remain open.

<!-- model-report:start -->
<a id="model-report"></a>

## Evaluation summary

[Measurement protocol](benchmarking.md) · [Source record](model-reports/records/chronos2.json)

Results below retain their original dates and workloads. Different rows may use different runs; they are not a controlled comparison. Legacy records explicitly retain unrecorded conditions. Missing values are not zero or a pass.

### Demo support

| Platform | State | Verified scope / remaining work | Evidence |
|---|---|---|---|
| macos | ✅ | Normal Models route, native fixed-scene prediction, interval/clock changes and reopen checked. User acceptance remains open. | [E4](validation/chronos2-demo-20261112/README.md) |
| ios | △ | Interactive fixed preview and native adapter are connected; signed build passed. Physical iPhone inference and UI acceptance remain open. | [E4](validation/chronos2-demo-20261112/README.md) |

### Performance and quality

| Metric | Result | Statistic / samples | Scope and evidence |
|---|---|---|---|
| Bundle / payload size | 478,864,956 B | single; n=1 | [E1](validation/chronos2-conversion-20261111/README.md) · Nine-file local bundle, including reference vectors and metadata; not runtime memory. |
| Prepared cache size | Not measured | — | Not measured in this conversion-fidelity run; app behavior, task quality and performance require separate evidence. |
| First load, including specialization | Not measured | — | Not measured in this conversion-fidelity run; app behavior, task quality and performance require separate evidence. |
| Load from prepared cache | Not measured | — | Not measured in this conversion-fidelity run; app behavior, task quality and performance require separate evidence. |
| Request / pipeline completion time | Not measured | — | Not measured in this conversion-fidelity run; app behavior, task quality and performance require separate evidence. |
| Peak RSS | Not measured | — | Not measured in this conversion-fidelity run; app behavior, task quality and performance require separate evidence. |
| Peak physical footprint | Not measured | — | Not measured in this conversion-fidelity run; app behavior, task quality and performance require separate evidence. |
| Resident memory during inference | Not measured | — | Not measured in this conversion-fidelity run; app behavior, task quality and performance require separate evidence. |
| Memory after release minus baseline | Not measured | — | Not measured in this conversion-fidelity run; app behavior, task quality and performance require separate evidence. |
| Task success fraction | Not measured | — | Not measured in this conversion-fidelity run; app behavior, task quality and performance require separate evidence. |
| Reference error (definition in evidence) | 0.00152587891 error | max; n=6048 | [E1](validation/chronos2-conversion-20261111/README.md) · Maximum absolute error over all scalars; each value passed atol 1e-4 + rtol 1e-4. n counts scalar comparisons, not independent runs. |
| Stop acknowledgement time | Not measured | — | Not measured in this conversion-fidelity run; app behavior, task quality and performance require separate evidence. |
| Forecast MAE | Not measured | — | Not measured in this conversion-fidelity run; app behavior, task quality and performance require separate evidence. |
| Forecast WAPE | Not measured | — | Not measured in this conversion-fidelity run; app behavior, task quality and performance require separate evidence. |

### Verification coverage

| Check | State | Scope / remaining work | Evidence |
|---|---|---|---|
| Conversion / reference fidelity | Passed (stated scope) | 18 synthetic cases and 60 fixed NYC scene/zone cases pass frozen raw-tensor comparisons. | [E1](validation/chronos2-conversion-20261111/README.md) · [E2](validation/chronos2-swift-20261112/README.md) · [E4](validation/chronos2-demo-20261112/README.md) |
| Task quality | Not measured | Numerical fidelity and fixed-scene operation do not establish held-out forecasting quality. | — |
| Repeated use / resource release | Passed (stated scope) | Release/reload, 60-series production-service comparison with zero leases, and normal Mac leave/reopen checked. No memory measurement or iPhone acceptance. | [E2](validation/chronos2-swift-20261112/README.md) · [E4](validation/chronos2-demo-20261112/README.md) |
| Stop / retry / recovery | Not measured | CPU fakes cover cancellation and stop joins; cancellation of the native model and normal app remain unmeasured. | — |

### Evidence and conditions

| Evidence | Date / record / route level | Device / OS / tools | Workload and recorded conditions |
|---|---|---|---|
| [E1](validation/chronos2-conversion-20261111/README.md) | 2026-11-11 / standard / conversion | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 Build version 27B52; Swift 6.4.2; macOS SDK 27.2; Python 3.11.9; chronos-forecasting 2.3.2; torch 2.9.0; coreai-torch 0.4.3; coreai-core 1.0.0b3 | 18 fixed synthetic series, raw length 1-8192, missing values and constant/periodic/trend inputs; all 21 quantiles x 16 forecast steps. One inference per case, no warmup. This measures numerical fidelity only. · variant=coreai-fp32-univariate; backend=Python Core AI GPU preference; actual delegate not profiled; route=Chronos-2 recipe: independent official CPU reference -> Core AI native verification -> bundle; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=e749b5653298b826770f57ac6a814967fad7162a; batch_size=1; concurrency=1; warmups=0; power=AC Power sampled before and after the run; thermal=Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; same after the run; temperature and GPU utilization not measured |
| [E2](validation/chronos2-swift-20261112/README.md) | 2026-11-12 / legacy / diagnostic | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; macOS/iOS SDK 27.2; Debug package | 18 fixed cases / 6048 raw values, then release/reload and the first 336 values again. Numerical validation, no warmup or performance benchmark. Power/thermal and a benchmark-plan hash were not recorded. · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; actual delegate not profiled; route=chronos-verify --verify; separate opt-in native test; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=25903277e60e99bfd85e217a96d69b18df82852c; batch_size=1; concurrency=1; warmups=0; unrecorded: power, thermal, plan_sha256 |
| [E3](validation/chronos2-models-20261112/README.md) | 2026-11-12 / legacy / diagnostic | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package tests; Release Mac/iOS applications | Pinned nine-file bundle import, five rejected replacements, lease gating and source-hash preservation; normal Mac import/preview and signed iOS build. No inference or performance measurement. · variant=coreai-fp32-univariate; backend=ManagedModelStore and normal Mac Models UI; no inference; route=Local bundle import -> inspect -> ready -> lease/release; explicit preview; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=d68828d601fea9d8b6756cc7fe2b2200749d7fa4; unrecorded: batch_size, concurrency, warmups, power, thermal, plan_sha256 |
| [E4](validation/chronos2-demo-20261112/README.md) | 2026-11-12 / legacy / native_app | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package and Release Mac application | 60 fixed scene/zone cases / 20160 raw values; native Mac GUI prediction and comparison, then close. CPU fakes cover stop/owner boundaries. No iPhone inference, user acceptance or performance measurement. Power/thermal and a benchmark-plan hash were not recorded. · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; actual delegate not profiled; route=Normal Mac Models -> explicit on-device prediction -> interval and clock changes -> leave/reopen; separate production-service native comparison; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=06e010133afd10d0d31532cfa804b97818944b30; batch_size=1; concurrency=1; warmups=0; unrecorded: power, thermal, plan_sha256 |

### Setup, limitations and progress

- [Conversion recipe](../Tools/Conversion/recipes/chronos2/README.md)

- [Conversion evidence](validation/chronos2-conversion-20261111/README.md)

- [Swift runtime evidence](validation/chronos2-swift-20261112/README.md)

- [Models import evidence](validation/chronos2-models-20261112/README.md)

- [Fixed NYC demo evidence](validation/chronos2-demo-20261112/README.md)

- [Implementation plan #548](https://github.com/oka-yuji/coreai-samples/issues/548)
<!-- model-report:end -->

## Official NYC API dates

The date source supplements the three bundled scenes. Select **Official NYC API dates**, check published dates, choose a New York date and half-hour, and fetch the 21-day history. A prepared Models resource and explicit **On-device prediction** are required to predict an API date. Fetch actual pickups separately after a successful prediction. Entering Models or switching the source starts neither a request nor inference.

The API publishes historical annual datasets; TLC monthly Parquet files are a different publication. The catalog's modification date is not trip-data freshness. Closed years from 2017 onward are discovered from the official domain, owner, provenance and exact annual name. Before filling absent rows with zero, the selected dataset's metadata must confirm the same identity/revision, published historical annual scope and `calendar_date`/`number` column types. The catalog revision must be a complete canonical UTC timestamp and match `rowsUpdatedAt`; data must have been updated after the annual period closed. `publicationDate` can precede that boundary, as in the captured 2019–2021 metadata. The 2021 `As needed` frequency is accepted only with the provider’s explicit historical-publication/corrections declaration. A missing year invalidates a 21-day history even when dates on both sides are listed. Requests and date selection share this restriction.

Intervals advance by 1,800 seconds in UTC. New York controls omit spring's nonexistent times. Both occurrences of autumn's repeated floating timestamps are missing; their combined provider counts are never assigned to either occurrence. Published zero is observed zero. Unpublished, ambiguous and not-yet-fetched intervals remain distinct. Failures retain the previous valid display and mode. Advancing uses only the elapsed prefix of explicitly fetched actuals, including its missing mask.

`NYCAPIClient` owns coverage generations, transport and disposable cache. Requests use the two fixed HTTPS hosts, reject redirects and credentials, stop receiving above 8 MiB and accept fewer than 50,000 aggregate rows. Queries split at year boundaries and at seven UTC days. Only an explicit transient error receives one retry. Each window retains original response bytes and SHA-256, query URL/bounds, status, acquisition time and catalog/dataset metadata. Cache reads check file type and size before allocation, then revalidate hashes and rebuild counts from original bytes. After a successful save, committed cache files are limited to 32 windows and 64 MiB. Each window has at most 32 MiB of raw evidence and a 48 MiB encoded file. One exclusive lock covers recovery of abandoned UUID staging files, writing and pruning. An active save can temporarily add one encoded file before pruning.

An API forecast retains all native 21×16 values per zone plus its immutable input snapshot. Its SHA-256 covers deterministic sorted-key JSON with UTC start/origin, nullable input counts, dataset revisions, preprocessing version and acquisition evidence. The snapshot's `series` is the exact model input; any `actuals` member is source evidence for the elapsed prefix used by that later prediction. Fetching future actuals never changes an earlier snapshot. The cache is not a saved forecast journal.

CPU tests include fault injection and an optional replay of supplied original responses:

```sh
COREAI_CHRONOS2_API_FIXTURES=/path/to/fixtures swift test --package-path CoreAISamplesKit --filter ChronosAPIReplayTests
```

The fixtures directory contains `api-baseline`, `api-fixed-history` and `api-fixed-actuals`. A separate pinned nine-year metadata replay uses `COREAI_CHRONOS2_API_ANNUAL_METADATA=/path/to/capture` with `ChronosAPIReplayTests.annualMetadata`; that capture contains `api-baseline` and `api-annual-metadata`. The fixed actuals fixture covers two hours; replay does not infer additional observed hours. Under the shared heavy-job lock, the parent can run the live API/native acceptance check:

```sh
COREAI_CHRONOS2_API_BUNDLE=/path/to/chronos2-bundle COREAI_CHRONOS2_API_REFERENCE=/path/to/fixed-scenes-reference.json COREAI_CHRONOS2_API_REPORT=/new/private/report.json swift test --package-path CoreAISamplesKit --filter ChronosAPINativeTests.liveHistoryAndNative
```

It checks Friday 2025-01-24 17:00 New York history against all 20,160 pinned input values, then compares all 6,720 native values through production Models/Store/Service at fixed `atol = rtol = 1e-4` and checks zero leases after closing. It writes `report.history.json` before inference and `report.json` after successful comparison; both must be new private paths. Default tests skip this check. Offline replay and CPU fakes do not establish live API, native, Mac/iPhone UI, signing or user acceptance. The existing generated validation tables above retain their original scope.
