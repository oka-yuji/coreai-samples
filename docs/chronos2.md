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
| macos | ✅ | Normal Models route supports fixed scenes and API histories, native prediction, explicit actual comparison and time changes. User acceptance remains open. | [E4](validation/chronos2-demo-20261112/README.md) · [E5](validation/chronos2-api-20261112/README.md) |
| ios | △ | Fixed/API UI and native adapter are connected; signed build passed. Physical iPhone inference and UI acceptance remain open. | [E4](validation/chronos2-demo-20261112/README.md) · [E5](validation/chronos2-api-20261112/README.md) |

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
| Conversion / reference fidelity | Passed (stated scope) | 18 synthetic cases, 60 fixed NYC scene/zone cases and 20 live API series pass frozen raw-tensor comparisons. The API inputs overlap the fixed Friday workload. | [E1](validation/chronos2-conversion-20261111/README.md) · [E2](validation/chronos2-swift-20261112/README.md) · [E4](validation/chronos2-demo-20261112/README.md) · [E5](validation/chronos2-api-20261112/README.md) |
| Task quality | Not measured | Numerical fidelity and fixed-scene operation do not establish held-out forecasting quality. | — |
| Repeated use / resource release | Passed (stated scope) | Release/reload, fixed and API production-service runs with zero leases, and normal Mac leave/reopen are checked. No memory measurement or iPhone acceptance. | [E2](validation/chronos2-swift-20261112/README.md) · [E4](validation/chronos2-demo-20261112/README.md) · [E5](validation/chronos2-api-20261112/README.md) |
| Stop / retry / recovery | Not measured | CPU fakes cover cancellation and stop joins; cancellation of the native model and normal app remain unmeasured. | — |

### Evidence and conditions

| Evidence | Date / record / route level | Device / OS / tools | Workload and recorded conditions |
|---|---|---|---|
| [E1](validation/chronos2-conversion-20261111/README.md) | 2026-11-11 / standard / conversion | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 Build version 27B52; Swift 6.4.2; macOS SDK 27.2; Python 3.11.9; chronos-forecasting 2.3.2; torch 2.9.0; coreai-torch 0.4.3; coreai-core 1.0.0b3 | 18 fixed synthetic series, raw length 1-8192, missing values and constant/periodic/trend inputs; all 21 quantiles x 16 forecast steps. One inference per case, no warmup. This measures numerical fidelity only. · variant=coreai-fp32-univariate; backend=Python Core AI GPU preference; actual delegate not profiled; route=Chronos-2 recipe: independent official CPU reference -> Core AI native verification -> bundle; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=e749b5653298b826770f57ac6a814967fad7162a; batch_size=1; concurrency=1; warmups=0; power=AC Power sampled before and after the run; thermal=Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; same after the run; temperature and GPU utilization not measured |
| [E2](validation/chronos2-swift-20261112/README.md) | 2026-11-12 / legacy / diagnostic | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; macOS/iOS SDK 27.2; Debug package | 18 fixed cases / 6048 raw values, then release/reload and the first 336 values again. Numerical validation, no warmup or performance benchmark. Power/thermal and a benchmark-plan hash were not recorded. · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; actual delegate not profiled; route=chronos-verify --verify; separate opt-in native test; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=25903277e60e99bfd85e217a96d69b18df82852c; batch_size=1; concurrency=1; warmups=0; unrecorded: power, thermal, plan_sha256 |
| [E3](validation/chronos2-models-20261112/README.md) | 2026-11-12 / legacy / diagnostic | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package tests; Release Mac/iOS applications | Pinned nine-file bundle import, five rejected replacements, lease gating and source-hash preservation; normal Mac import/preview and signed iOS build. No inference or performance measurement. · variant=coreai-fp32-univariate; backend=ManagedModelStore and normal Mac Models UI; no inference; route=Local bundle import -> inspect -> ready -> lease/release; explicit preview; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=d68828d601fea9d8b6756cc7fe2b2200749d7fa4; unrecorded: batch_size, concurrency, warmups, power, thermal, plan_sha256 |
| [E4](validation/chronos2-demo-20261112/README.md) | 2026-11-12 / legacy / native_app | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package and Release Mac application | 60 fixed scene/zone cases / 20160 raw values; native Mac GUI prediction and comparison, then close. CPU fakes cover stop/owner boundaries. No iPhone inference, user acceptance or performance measurement. Power/thermal and a benchmark-plan hash were not recorded. · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; actual delegate not profiled; route=Normal Mac Models -> explicit on-device prediction -> interval and clock changes -> leave/reopen; separate production-service native comparison; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=06e010133afd10d0d31532cfa804b97818944b30; batch_size=1; concurrency=1; warmups=0; unrecorded: power, thermal, plan_sha256 |
| [E5](validation/chronos2-api-20261112/README.md) | 2026-11-12 / legacy / native_app | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package and Release Mac application | Live API: 20,160 input values and 6,720 raw native outputs; zero leases after close. In the normal Mac app, the 17:00 forecast displays 120 against an observed 130; the 17:30 refetch produces 128. Nine annual metadata snapshots and CPU failure cases were checked. No controlled performance measurement or iPhone inference. · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; actual delegate not profiled; route=Normal Mac Models -> official API history -> explicit native forecast -> separate actuals; production Models/Store/Service integration test; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=05f8363b5e3de8f9531ebb2fb332b71bb9b795fb; batch_size=1; concurrency=1; warmups=0; unrecorded: power, thermal, plan_sha256 |

### Setup, limitations and progress

- [Conversion recipe](../Tools/Conversion/recipes/chronos2/README.md)

- [Conversion evidence](validation/chronos2-conversion-20261111/README.md)

- [Swift runtime evidence](validation/chronos2-swift-20261112/README.md)

- [Models import evidence](validation/chronos2-models-20261112/README.md)

- [Fixed NYC demo evidence](validation/chronos2-demo-20261112/README.md)

- [Official API demo evidence](validation/chronos2-api-20261112/README.md)

- [Implementation plan #548](https://github.com/oka-yuji/coreai-samples/issues/548)
<!-- model-report:end -->

## Verified NYC API dates

The date source supplements the three bundled scenes. Select **Verified NYC API dates**, check verified dates, choose a New York date and half-hour, and fetch the 21-day history. A prepared Models resource and explicit **On-device prediction** are required for API predictions. Fetch actual pickups separately after a successful prediction. Entering Models or switching the source starts neither a request nor inference.

The shipped evidence covers these 20 Manhattan zones, rather than every year listed in the official catalog:

| Boundary | New York time | UTC |
|---|---|---|
| Verified data, start inclusive | 2025-01-01 00:00 EST | 2025-01-01T05:00:00Z |
| Verified data, end exclusive | 2025-02-01 00:00 EST | 2025-02-01T05:00:00Z |
| Earliest forecast origin | 2025-01-22 00:00 EST | 2025-01-22T05:00:00Z |
| Latest forecast origin, inclusive | 2025-02-01 00:00 EST | 2025-02-01T05:00:00Z |

Forecast origins need 1,008 preceding half-hours. The data interval and selectable forecast origins therefore differ. The pinned dataset is `irx2-vgm4`, revision `2026-10-01T16:57:47.000Z`. A listed year with another revision, or a slot outside approved evidence, is **unverified** with a null count. An absent catalog year is **unpublished**. Neither becomes zero, and no aggregate request is made for those spans. A 21-day history containing either is refused with the corresponding reason. A 16-slot actuals window can have a verified prefix and an unverified suffix at January's end; only the prefix can advance the forecast. Unknown revisions never gain trust from dates alone.

The API publishes annual datasets; monthly TLC Parquet files are a separate publication. Annual metadata is not a completion watermark. Official domain, owner, provenance, annual name, publication state, TLC agency, column types and exact catalog revision still establish dataset identity. Metadata is captured and validated before and after all aggregate requests. Both captures must agree with the pinned catalog revision. This detects changes exposed by the provider's revision contract, not a transactional snapshot guarantee. Reconstructed counts must also exactly match the independent reference for every requested verified slot. An empty response for a positive reference count is a validation error; an absent group for a verified zero is observed zero. Errors retain the previous valid display and prediction mode.

### Independent January reconciliation

The runtime owns [the exact reference bytes](../CoreAISamplesKit/Sources/ChronosDemoRuntime/Resources/NYCVerified/january-2025.json), outside the unchanged fixed-scene `NYCData` resources. Its identity is SHA-256 `2da47735c51b5987dec9f6bee72c97acd39abe4870991e413dca30a1912c6530` (88,976 bytes). The resource records dataset ID/revision, half-open interval, 1,800-second spacing, `America/New_York`, 20 unique zone IDs, and both original source hashes. Loading checks the pinned bytes, identity, boundaries, shape, finite integer counts and zone set. The pure Feature receives a value-only verification policy; it performs no bundle IO. Adding support requires another independently verified resource and its explicitly approved policy entry.

| Original source | SHA-256 |
|---|---|
| [January 2025 Parquet](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet) | `9af277e4c0d3f9deb30644da822981e1e7df6af58313170fd3aa8a474485488a` |
| [February 2025 Parquet](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-02.parquet) | `037cba555a73663f3a51a2c27816e40e3feb364769942bdf122b9da31e377bd3` |

The supervisor's independent count compared all 20 zones × 1,488 January half-hours: 29,760 exact matches across five API captures at an unchanged revision. The initial January-only comparison **failed at nine cells**. February's file contains 30 January pickups, ten in the selected zones; including those ten explains all nine differences. The failed reference identity was `68d9a72188d1fb7bd8a3e6b07990c4163b404a67b233427eedfb15ea4dfdde55`. These are January counts, not whole-year numerical verification.

| Zone | January 31 local start | API | January-only count |
|---|---|---:|---:|
| 186 | 23:00 | 54 | 53 |
| 234 | 23:30 | 163 | 162 |
| 246 | 23:30 | 36 | 35 |
| 48 | 22:30 | 152 | 151 |
| 48 | 23:00 | 113 | 111 |
| 50 | 23:30 | 25 | 24 |
| 68 | 22:00 | 143 | 142 |
| 68 | 23:30 | 91 | 90 |
| 90 | 23:30 | 88 | 87 |

### Missingness, provenance and reuse

Intervals advance by 1,800 seconds in UTC. New York controls omit spring's nonexistent times; invalid provider rows in that local hour are rejected, never reassigned. The supervisor's query for 2025-03-09 02:00–03:00 in the 20 zones returned zero groups. This does not establish that other years contain no such rows. Spring is outside the shipped verified range; synthetic verification profiles retain the DST and invalid-row CPU tests. Both occurrences of an ambiguous autumn timestamp remain missing inside an explicitly verified profile. Observed zero, unverified, unpublished, ambiguous and not fetched are separate states.

`NYCAPIClient` owns coverage generations, transport and disposable cache. Requests use the two fixed HTTPS hosts, reject redirects and credentials, stop receiving above 8 MiB and accept fewer than 50,000 aggregate rows. Queries cover only verified intersections, split at New York years and seven UTC days. Only an explicit transient error receives one retry. Each window retains original HTTP bytes and their SHA-256, query URL/bounds, status, acquisition time, catalog and pre/post metadata evidence. Reference hashes identify independent evidence and never replace HTTP body hashes.

Cache keys and persisted windows include verification policy/reference identities and preprocessing `nyc-utc30m-verified-v3`. Old broad-year entries cannot bypass this policy. Reads check file type and size, raw hashes, metadata, current approved policy and reconstructed counts against the pinned reference. After a successful save, cache files are limited to 32 windows and 64 MiB. Each window has at most 32 MiB of raw evidence and a 48 MiB encoded file. One exclusive lock covers recovery of abandoned UUID staging files, writing and pruning; an active save can temporarily add one encoded file before pruning.

An API forecast retains all native 21×16 values per zone plus its immutable input snapshot. Its SHA-256 covers deterministic sorted-key JSON including UTC start/origin, nullable counts, revisions, preprocessing, verification identities and acquisition evidence. Snapshot validation rechecks the same policy and raw evidence. Only elapsed verified actuals can enter later model input, with the existing ambiguity mask preserved. Fetching future actuals never changes an earlier snapshot. The cache is not a saved forecast journal.

### Validation

CPU boundary, corruption, unknown-revision, reference-positive/empty-response, zero, pre/post metadata, cancellation and cache tests use:

```sh
swift test --package-path CoreAISamplesKit --filter 'ChronosVerifiedRangeTests|ChronosAPI|ChronosFeature'
COREAI_CHRONOS2_API_FIXTURES=/path/to/fixtures swift test --package-path CoreAISamplesKit --filter ChronosAPIReplayTests
```

The fixtures directory contains `api-baseline`, `api-fixed-history` and `api-fixed-actuals`. Replay checks 20,160 history values and 80 actuals against the production verification resource, retaining their original HTTP hashes. The actuals fixture covers two hours, without inferring additional observed hours. The separate nine-year metadata replay uses `COREAI_CHRONOS2_API_ANNUAL_METADATA=/path/to/capture` with `ChronosAPIReplayTests.annualMetadata`; its directory contains `api-baseline` and `api-annual-metadata`. Nine valid metadata records do not establish nine verified years.

Under the shared heavy-job lock, the parent runs live API/native acceptance:

```sh
COREAI_CHRONOS2_API_BUNDLE=/path/to/chronos2-bundle COREAI_CHRONOS2_API_REFERENCE=/path/to/fixed-scenes-reference.json COREAI_CHRONOS2_API_REPORT=/new/private/report.json swift test --package-path CoreAISamplesKit --filter ChronosAPINativeTests.liveHistoryAndNative
```

It retains Friday 2025-01-24 17:00 New York, all 20,160 fixed input values, all 6,720 native values, fixed `atol = rtol = 1e-4`, the source model identity and the zero-lease check. It writes new private `report.history.json` before inference and `report.json` after success. Default tests skip native execution. The parent also owns the renewed 29,760-count live reconciliation, actual GUI checks, signed builds and review. CPU replay does not establish live API, native, Mac/iPhone UI or user acceptance. The generated validation tables above retain their original evidence and are updated separately by the parent.
