# Chronos-2

[Home](../README.md) · [日本語](chronos2.ja.md)

A public FP32 Core AI conversion of Amazon Chronos-2 for one time series and a 16-step quantile forecast. Models offers the [downloadable bundle](https://huggingface.co/okayuji/chronos-2-coreai-fp32) for the NYC taxi demo. Fresh download and production-service native forecasts pass on Mac; normal download UI, physical iPhone and user acceptance remain open. [Release verification](validation/chronos2-download-20261113/README.md) records the scope and retained failures.

## Conversion contract

Use [the maintainer recipe](../Tools/Conversion/recipes/chronos2/README.md) with the pinned official snapshot. Inputs contain 1-8192 observations; missing values and left padding are NaN. The native input is Float32[1,T], T a multiple of 16. The output is Float32[1,21,16], with no state. This profile does not expose covariates, cross-learning, quantization or longer forecast horizons.

Weights are distributed separately on [Hugging Face](https://huggingface.co/okayuji/chronos-2-coreai-fp32). The recipe creates a local bundle and inventory. A fresh conversion can have a different asset/manifest hash; update the trusted catalog only after verifying that exact bundle.

## Conversion verification

[The recorded conditions and failures](validation/chronos2-conversion-20261111/README.md) distinguish conversion fidelity from forecasting quality. All 6,048 native output values pass `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`. This does not establish taxi-demand accuracy, performance or iPhone application support.

## Swift runtime

`ChronosCore` validates raw contexts and all 21 quantiles. `ChronosRuntime` owns one request at a time and waits for load/inference completion before releasing its model. The [Mac Swift verification](validation/chronos2-swift-20261112/README.md) passes the same 18 cases and a release/reload repeat. The fixed NYC route also passes production-service comparison and normal Mac UI checks; physical iPhone inference remains unverified.

```sh
swift run --package-path CoreAISamplesKit chronos-verify --help
```

Explicit model commands require the shared lock in `AGENTS.md`. JSON input is `{"context":[1,2,null,4]}`; `null` is missing. Results preserve the full raw `[1,21,16]` tensor and identity. Output files must be new and have an existing parent directory.

## Models download, import and preview

Open **Models**, select **Chronos-2**, and use the **Download model** control (a down-arrow icon in Models lists and on iPhone). The application anonymously downloads and checks the nine pinned files, totaling 478,864,956 bytes (about 479 MB). Allow additional space for managed and runtime copies. **Import model** remains available for a complete, known Chronos-2 FP32 folder; existing valid imports stay ready and do not require another download.

When the ready checkmark appears, open the NYC taxi demo, choose **On-device prediction** and select **Predict on this device**. Opening Models, downloading, importing, inspecting or preparing the bundle does not start forecasts automatically. The interactive **Precomputed preview** works without a model. A native failure never switches modes automatically. Mac and iOS use this acquisition UI; the signed iOS build has been checked, while physical iPhone download and inference remain unverified.

The application selects `bundle/` from HF commit `d3ead488046498faa8b3b5103b1923e52a313d0e` (`v1.0.0`). Public release `v1.0.1` is `f422aa3a7c1173566c99fbfceb97e874b0121dab`; it changes only the model card, and its nine bundle files are byte-identical. The release and application pins are both retained.

The trusted artifact is `chronos2-fp32`, contract `chronos2-fp32-v1`, from `amazon/chronos-2` revision `29ec3766d36d6f73f0696f85560a422f50e8498c`. Its `chronos2.json` SHA-256 is `786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572`. Models requires the exact original manifest bytes and all nine pinned file sizes and hashes. That byte identity fixes the metadata contract, including the state-free Float32 `[1,T]` to `[1,21,16]` ABI, without loading the model or creating a runtime snapshot during import. Even reformatting the manifest requires a new trusted identity.

The 40-character HF commit selects downloaded bytes; the 64-character manifest digest identifies the trusted resource and its existing storage/leases. Both contracts are checked. Adding a download locator leaves the `localManifest` identity and imported registration unchanged, and HTTP requests never use the manifest digest as an HF revision.

The existing Models store copies and checks the folder before promotion. Altered, missing, extra or symbolic-link entries are rejected; a rejected replacement preserves the previous registration and input folder. Active leases prevent replacement and deletion until the last lease is released. The source folder is retained when the managed copy is deleted.

CPU checks cover catalog/preview behavior, history windows, stop/owner boundaries and synthetic storage. The opt-in real-bundle import check is `COREAI_CHRONOS2_IMPORT_BUNDLE=/path/to/chronos2-bundle swift test --package-path CoreAISamplesKit --filter ChronosImportTests.knownBundle`, under the shared heavy-job lock. It checks import, full inspection, leases, rejected replacements and source hashes without inference. Its default skip is not an import pass; native import, app builds, iPhone inference and normal UI acceptance need separate evidence.

The [public download check](validation/chronos2-download-20261113/README.md) uses a new managed store and no migration candidates. Its real transfer validates all nine files, then production Models/Store/sessions/service compare sixty fixed scene/zone cases and all 20,160 raw values, with unchanged source hashes and zero leases after close. It reuses the existing fixed reference and measures conversion fidelity, not held-out forecasting quality or performance. To reproduce it under the shared lock, supply a new `COREAI_CHRONOS2_DOWNLOAD_OUTPUT`, the original bundle as `COREAI_CHRONOS2_DOWNLOAD_SOURCE` for hash checks, and `COREAI_CHRONOS2_DOWNLOAD_REFERENCE`, then run `swift test --package-path CoreAISamplesKit --filter ChronosDownloadNativeTests.freshModelsDownload`. The test skips by default; normal GUI, physical iPhone and user acceptance are separate.

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
| macos | ✅ | Normal Mac checks cover NYC, product sales and CPU-only Seoul Bike. Fresh public download/native comparison passed separately; normal download UI and user acceptance remain open. | [E4](validation/chronos2-demo-20261112/README.md) · [E5](validation/chronos2-api-20261112/README.md) · [E6](validation/chronos2-journal-20261112/README.md) · [E7](validation/chronos2-sales-20261113/README.md) · [E8](validation/chronos2-bike-20261113/README.md) · [E10](validation/chronos2-download-20261113/README.md) · [E11](validation/chronos2-download-20261113/README.md) |
| ios | △ | NYC, sales and Seoul Bike adapters plus download/import controls are connected; signed builds pass. Physical iPhone download, inference and user acceptance remain open. | [E4](validation/chronos2-demo-20261112/README.md) · [E5](validation/chronos2-api-20261112/README.md) · [E6](validation/chronos2-journal-20261112/README.md) · [E7](validation/chronos2-sales-20261113/README.md) · [E8](validation/chronos2-bike-20261113/README.md) · [E10](validation/chronos2-download-20261113/README.md) |

### Performance and quality

| Metric | Result | Statistic / samples | Scope and evidence |
|---|---|---|---|
| Bundle / payload size | 478,864,956 B | single; n=1 | [E1](validation/chronos2-conversion-20261111/README.md) · Nine-file local bundle, including reference vectors and metadata; not runtime memory. |
| Prepared cache size | Not measured | — | Not measured in this conversion-fidelity run; app behavior, task quality and performance require separate evidence. |
| First load, including specialization | Not measured | — | Not measured in this conversion-fidelity run; app behavior, task quality and performance require separate evidence. |
| Load from prepared cache | Not measured | — | Not measured in this conversion-fidelity run; app behavior, task quality and performance require separate evidence. |
| Request / pipeline completion time | Not measured | — | No controlled timing benchmark. Functional sales request intervals, including new-session validation/load, are recorded separately in the sales evidence. |
| Peak RSS | Not measured | — | Not measured in this conversion-fidelity run; app behavior, task quality and performance require separate evidence. |
| Peak physical footprint | 212,411,176 B | single; n=1 | [E7](validation/chronos2-sales-20261113/README.md) · Kernel lifetime peak of one Debug test process including import and both sales passes; not per-request/model-only/GPU memory or an iPhone budget. |
| Peak physical footprint | 56,198,560 B | single; n=1 | [E8](validation/chronos2-bike-20261113/README.md) · Whole Debug process kernel lifetime peak, including import and both passes; distinct from RSS, model-only memory and iPhone budget. |
| Resident memory during inference | Not measured | — | Not measured in this conversion-fidelity run; app behavior, task quality and performance require separate evidence. |
| Memory after release minus baseline | Not measured | — | Not measured in this conversion-fidelity run; app behavior, task quality and performance require separate evidence. |
| Task success fraction | Not measured | — | Not measured in this conversion-fidelity run; app behavior, task quality and performance require separate evidence. |
| Reference error (definition in evidence) | 0.00152587891 error | max; n=6048 | [E1](validation/chronos2-conversion-20261111/README.md) · Maximum absolute error over all scalars; each value passed atol 1e-4 + rtol 1e-4. n counts scalar comparisons, not independent runs. |
| Reference error (definition in evidence) | 8.39233398e-05 error | max; n=10752 | [E7](validation/chronos2-sales-20261113/README.md) · Sales: 16 fixed windows repeated after reload; all 21 x 16 raw values satisfy fixed atol/rtol 1e-4. Repeated scalar comparisons, not independent quality samples. |
| Reference error (definition in evidence) | 0.00170898438 error | max; n=3360 | [E8](validation/chronos2-bike-20261113/README.md) · CPU-only bike: five frozen windows twice; all raw values meet fixed atol/rtol1e-4. Repeats are not independent quality samples. |
| Reference error (definition in evidence) | 0.000228881836 error | max; n=20160 | [E10](validation/chronos2-download-20261113/README.md) · Freshly downloaded bundle, same 60 fixed NYC cases. Maximum absolute error across raw values; every value passes atol 1e-4 + rtol 1e-4. This is neither held-out task quality nor an independent repeat count. |
| Stop acknowledgement time | Not measured | — | Not measured in this conversion-fidelity run; app behavior, task quality and performance require separate evidence. |
| Forecast MAE | 4.44884534 count | aggregate; n=212 | [E7](validation/chronos2-sales-20261113/README.md) · Fixed sales first pass only: 212/224 common valid pairs; clipped unrounded median, seasonal naive in evidence. No held-out quality claim. |
| Forecast MAE | 449.954254 count | aggregate; n=80 | [E8](validation/chronos2-bike-20261113/README.md) · First-pass80 hourly pairs only; naive comparison and unfavorable per-origin results are retained. No held-out quality claim. |
| Forecast WAPE | 0.901678022 ratio | aggregate; n=212 | [E7](validation/chronos2-sales-20261113/README.md) · Same fixed sales pairs and common denominator; seasonal-naive comparison and zero/missing policy in evidence. |
| Forecast WAPE | 1.69434409 ratio | aggregate; n=80 | [E8](validation/chronos2-bike-20261113/README.md) · Same80 pairs and denominator; per-hour groups08:00-23:00 have five dates each. |

### Verification coverage

| Check | State | Scope / remaining work | Evidence |
|---|---|---|---|
| Conversion / reference fidelity | Passed (stated scope) | 18 synthetic cases, 60 fixed NYC scene/zone cases and 20 live API series pass frozen raw-tensor comparisons. The API inputs overlap the fixed Friday workload. Product sales adds 16 fixed windows, repeated after release/reopen, against official FP32. Fresh public download repeats the same 60 NYC cases; it adds no held-out series. Seoul Bike uses CPU-only for five frozen windows twice; earlier GPU failures remain recorded. | [E1](validation/chronos2-conversion-20261111/README.md) · [E2](validation/chronos2-swift-20261112/README.md) · [E4](validation/chronos2-demo-20261112/README.md) · [E5](validation/chronos2-api-20261112/README.md) · [E7](validation/chronos2-sales-20261113/README.md) · [E8](validation/chronos2-bike-20261113/README.md) · [E10](validation/chronos2-download-20261113/README.md) · [E11](validation/chronos2-download-20261113/README.md) |
| Task quality | Not measured | Fixed sales and bike diagnostics are recorded, including unfavorable results; no held-out general forecasting quality is established. | — |
| Repeated use / resource release | Passed (stated scope) | Release/reload, production-service zero leases and normal Mac leave/reopen are checked for NYC and sales. Sales records point memory and whole-process lifetime peak; iPhone acceptance remains open. Fresh public acquisition/native prediction also preserves source hashes and returns all leases. Seoul Bike also passes release/reopen and zero-lease checks. | [E2](validation/chronos2-swift-20261112/README.md) · [E4](validation/chronos2-demo-20261112/README.md) · [E5](validation/chronos2-api-20261112/README.md) · [E7](validation/chronos2-sales-20261113/README.md) · [E8](validation/chronos2-bike-20261113/README.md) · [E10](validation/chronos2-download-20261113/README.md) · [E11](validation/chronos2-download-20261113/README.md) |
| Stop / retry / recovery | Not measured | CPU fakes cover cancellation and stop joins; cancellation of the native model and normal app remain unmeasured. | — |

### Evidence and conditions

| Evidence | Date / record / route level | Device / OS / tools | Workload and recorded conditions |
|---|---|---|---|
| [E1](validation/chronos2-conversion-20261111/README.md) | 2026-11-11 / standard / conversion | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 Build version 27B52; Swift 6.4.2; macOS SDK 27.2; Python 3.11.9; chronos-forecasting 2.3.2; torch 2.9.0; coreai-torch 0.4.3; coreai-core 1.0.0b3 | 18 fixed synthetic series, raw length 1-8192, missing values and constant/periodic/trend inputs; all 21 quantiles x 16 forecast steps. One inference per case, no warmup. This measures numerical fidelity only. · variant=coreai-fp32-univariate; backend=Python Core AI GPU preference; actual delegate not profiled; route=Chronos-2 recipe: independent official CPU reference -> Core AI native verification -> bundle; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=e749b5653298b826770f57ac6a814967fad7162a; batch_size=1; concurrency=1; warmups=0; power=AC Power sampled before and after the run; thermal=Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; same after the run; temperature and GPU utilization not measured |
| [E2](validation/chronos2-swift-20261112/README.md) | 2026-11-12 / legacy / diagnostic | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; macOS/iOS SDK 27.2; Debug package | 18 fixed cases / 6048 raw values, then release/reload and the first 336 values again. Numerical validation, no warmup or performance benchmark. Power/thermal and a benchmark-plan hash were not recorded. · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; actual delegate not profiled; route=chronos-verify --verify; separate opt-in native test; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=25903277e60e99bfd85e217a96d69b18df82852c; batch_size=1; concurrency=1; warmups=0; unrecorded: power, thermal, plan_sha256 |
| [E3](validation/chronos2-models-20261112/README.md) | 2026-11-12 / legacy / diagnostic | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package tests; Release Mac/iOS applications | Pinned nine-file bundle import, five rejected replacements, lease gating and source-hash preservation; normal Mac import/preview and signed iOS build. No inference or performance measurement. · variant=coreai-fp32-univariate; backend=ManagedModelStore and normal Mac Models UI; no inference; route=Local bundle import -> inspect -> ready -> lease/release; explicit preview; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=d68828d601fea9d8b6756cc7fe2b2200749d7fa4; unrecorded: batch_size, concurrency, warmups, power, thermal, plan_sha256 |
| [E4](validation/chronos2-demo-20261112/README.md) | 2026-11-12 / legacy / native_app | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package and Release Mac application | 60 fixed scene/zone cases / 20160 raw values; native Mac GUI prediction and comparison, then close. CPU fakes cover stop/owner boundaries. No iPhone inference, user acceptance or performance measurement. Power/thermal and a benchmark-plan hash were not recorded. · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; actual delegate not profiled; route=Normal Mac Models -> explicit on-device prediction -> interval and clock changes -> leave/reopen; separate production-service native comparison; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=06e010133afd10d0d31532cfa804b97818944b30; batch_size=1; concurrency=1; warmups=0; unrecorded: power, thermal, plan_sha256 |
| [E5](validation/chronos2-api-20261112/README.md) | 2026-11-12 / legacy / native_app | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package and Release Mac application | The production client validates all 29,760 January counts. The native path compares 20,160 Friday inputs and 6,720 raw outputs, then closes with zero leases. Normal Mac checks cover explicit acquisition/prediction/actuals and the verified month-end stop; chart accessibility fixes have separate source attribution. No controlled performance measurement or iPhone inference. · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; actual delegate not profiled; route=Production Models/Store/Service with a pinned January API reference; separate normal Mac GUI; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=02aa27facf98b6d7d503f72a06e8ac0b123dec01; batch_size=1; concurrency=1; warmups=0; unrecorded: power, thermal, plan_sha256 |
| [E6](validation/chronos2-journal-20261112/README.md) | 2026-11-12 / legacy / native_app | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package and Release Mac application | One fixed Friday origin: 20,160 inputs, 6,720 raw outputs and 80 actual pairs; independent saved-file arithmetic and hash checks, zero leases. Normal Mac save/restart/check and locale correction have separate source attribution. No held-out quality or controlled performance measurement; power/thermal and benchmark-plan hash unrecorded. · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; actual delegate not profiled; route=Production Models/Store/service/session -> live verified API -> native prediction -> immutable journal -> separate actual check -> reload -> stop; separate normal Mac GUI; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=48adfc21c80ff7ceb1828f5b7326fde1c760d0be; batch_size=1; concurrency=1; warmups=0; unrecorded: power, thermal, plan_sha256 |
| [E7](validation/chronos2-sales-20261113/README.md) | 2026-11-13 / legacy / native_app | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package and Release Mac/iOS builds | Four SKUs x four retrospective origins, 365-point histories, two passes / 10752 raw comparisons; 212/224 first-pass paired actuals. Normal Mac chart fix at 1954168. Diagnostic timing and whole-process memory only; no controlled benchmark or held-out quality. · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; actual delegate not profiled; route=Production Models/Store/Service/Sessions with original sales CSV; independent raw/metric audit; separate normal Mac sidebar; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=55df3378a40a98f8e34c6fafa8e8a33bb1001e71; batch_size=1; concurrency=1; warmups=0; power=AC Power before and after; thermal=pmset reported no thermal/performance warnings before and after; temperature and GPU load unmeasured; unrecorded: plan_sha256 |
| [E8](validation/chronos2-bike-20261113/README.md) | 2026-11-13 / legacy / native_app | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package and Release Mac/iOS builds | Five frozen Seoul origins,336 preceding counts, all21x16 raw values, two passes/3360 comparisons and zero leases. First-pass80-hour diagnostics only; earlier GPU failure retained. No iPhone or controlled benchmark. · variant=coreai-fp32-univariate; backend=Swift Core AI cpuOnly; EFR=false; per-operator placement not profiled; route=Production Models/Store/Service/Sessions; independent source/raw/metric audit; separate normal Mac GUI at7251127; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=0008c7d7a74693172a93eb2a1b0de8908e91fb3e; batch_size=1; concurrency=1; warmups=0; power=AC Power before and after; thermal=pmset reported no thermal/performance warnings before and after; temperature and GPU load unmeasured; unrecorded: plan_sha256 |
| [E9](validation/chronos2-bike-20261113/README.md) | 2026-11-13 / legacy / diagnostic | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package and Release Mac/iOS builds | May17 tail scalar exceeds frozen atol/rtol1e-4 on each pass. EFR=true also failed a different scalar. CPU-only was subsequently adopted without changing assets or thresholds. · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; EFR=false; actual delegate not profiled; route=Same frozen five Seoul windows; full native report retained two scalar failures across two passes; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=1b47558cbc1943780815aaa391cf780f3d8d3ac1; batch_size=1; concurrency=1; warmups=0; power=AC Power before and after; thermal=pmset reported no thermal/performance warnings before and after; temperature and GPU load unmeasured; unrecorded: plan_sha256 |
| [E10](validation/chronos2-download-20261113/README.md) | 2026-11-13 / legacy / diagnostic | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package tests; Release Mac/iOS build checks | Nine anonymous GETs from d3ead488, 478864956 bytes; 60 fixed scene/zone cases and 20160 raw values compared, then source hashes and zero leases checked. The same fixed-scene reference is reused, not a new held-out set. Host identity captured after the run; power/thermal and benchmark-plan hash unrecorded. GUI, iPhone and user acceptance are separate. · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; actual delegate not profiled; route=Fresh public Models download -> full validation -> production Store/sessions/service -> fixed native forecasts -> close; drawing-only adapter; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=9c7da3d0de00ef86f8ca1235385a254bd82f7982; batch_size=1; concurrency=1; warmups=0; unrecorded: power, thermal, plan_sha256 |
| [E11](validation/chronos2-download-20261113/README.md) | 2026-11-13 / legacy / diagnostic | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package tests; Release Mac/iOS build checks | Second fresh download after sales integration and the opt-in catalog repair. Same 60 scene/zone cases and 20,160 raw values; no additional held-out data. Full case set, Float32-reconstructed raw output, original hashes and zero leases pass. Test 50.129 s and wrapper 63.754719333 s include validation work; not performance measurements. · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; actual delegate not profiled; route=Fresh production Models download -> corrected production-linked catalog -> Store/Sessions/Service -> same 60 fixed NYC cases -> close; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=ec02f94bde596787ef83c837499732b9c76f3c2a; batch_size=1; concurrency=1; warmups=0; unrecorded: power, thermal, plan_sha256 |

### Setup, limitations and progress

- [Conversion recipe](../Tools/Conversion/recipes/chronos2/README.md)

- [Conversion evidence](validation/chronos2-conversion-20261111/README.md)

- [Swift runtime evidence](validation/chronos2-swift-20261112/README.md)

- [Models import evidence](validation/chronos2-models-20261112/README.md)

- [Fixed NYC demo evidence](validation/chronos2-demo-20261112/README.md)

- [Official API demo evidence](validation/chronos2-api-20261112/README.md)

- [Implementation plan #548](https://github.com/oka-yuji/coreai-samples/issues/548)

- [Saved forecast evidence](validation/chronos2-journal-20261112/README.md)

- [TukuruSF execution analysis](validation/chronos2-tsf-20261112/README.md)

- [Product sales demo](chronos2-sales.md)

- [Product sales verification](validation/chronos2-sales-20261113/README.md)

- [Seoul Bike demo](chronos2-bike.md)

- [Seoul Bike verification](validation/chronos2-bike-20261113/README.md)

- [Public download evidence](validation/chronos2-download-20261113/README.md)
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

## Saved forecasts and actual checks

Successful on-device predictions are saved automatically. Precomputed previews do not create records. Wait for **Forecast saved on this device**, then open **Forecast history** to select a record. Records survive app restart. **Check published actuals** is a separate network action for the saved origin and the first two forecast hours across all twenty zones; it never reruns the model. A check cannot change the original forecast or earlier checks.

The detail shows the New York origin, source, local save time, model version, comparison status and pair count. MAE and WAPE use the nonnegative, unrounded median and only comparable pairs. WAPE is absent when the actual total is zero. Unknown revisions and out-of-range data remain independently unverified, distinct from unpublished data, an unchecked record, observed zero and an ambiguous repeated time. A failed request is an error, not missing or zero data. The shipped API verification scope remains January 2025.

A save failure leaves the prediction visible with an explicit retry. Retrying preserves the same record UUID, dates and bytes. Closing, changing selection or backgrounding invalidates unfinished work; completed records stay stored. A record that fails validation is shown as unreadable. The app does not automatically prune saved records or migrate the separate prototype app's files. Save times come from the device clock and are not independent timestamp certification.

Records use the app's Application Support directory, under `coreai-samples/ChronosJournal-v1`. Each record retains all twenty 1,008-value inputs and 21×16 raw outputs, the actual verified model identity, source bytes/hashes, preprocessing and acquisition evidence. Only presentation and metrics clip negative medians. Journal envelope version 2 preserves Date reference-clock values; the established API canonical-byte format remains separate. Reads validate the complete record, not just its outer hash. Limits are 192 MiB per payload and 256 MiB + 4 KiB per envelope. Directory entries, files and lock initialization retain no-follow and exclusive-publication checks.

The explicit integration test below uses production Models, Store, service, sessions and journal storage, with a drawing-only test adapter. It compares 20,160 API inputs and all 6,720 raw native outputs to the pinned references, saves the result, appends eighty actual values, reloads, verifies the original file is unchanged and checks zero leases after stop. Use the shared heavy-job lock. The output directory must not already exist. Default runs skip this model/network check.

```sh
COREAI_CHRONOS2_JOURNAL_BUNDLE=/path/to/chronos2-bundle \
COREAI_CHRONOS2_JOURNAL_REFERENCE=/path/to/fixed-scenes-reference.json \
COREAI_CHRONOS2_JOURNAL_OUTPUT=/new/output \
swift test --package-path CoreAISamplesKit --filter ChronosAPINativeTests.liveJournal
```

[Journal verification and retained failures](validation/chronos2-journal-20261112/README.md) and [TukuruSF execution analysis](validation/chronos2-tsf-20261112/README.md) distinguish automated checks, normal Mac operation, integration and pending user/iPhone acceptance.

## Product sales

The [product-sales demo](chronos2-sales.md) adds four original-source SKUs, 7/14-day forecasts, missing-day handling and seasonal-naive comparison. Data acquisition is explicit and respects the original noncommercial terms. [Native and normal Mac verification](validation/chronos2-sales-20261113/README.md) is separate from physical iPhone and user acceptance.
