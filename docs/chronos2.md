# Chronos-2

[Home](../README.md) · [日本語](chronos2.ja.md)

A reproducible FP32 Core AI conversion of Amazon Chronos-2 for one time series and a 16-step quantile forecast. The Mac Python Core AI path passes 18 fixed synthetic cases. Swift runtime, Models and the NYC sample are tracked in [#548](https://github.com/oka-yuji/coreai-samples/issues/548).

## Conversion contract

Use [the maintainer recipe](../Tools/Conversion/recipes/chronos2/README.md) with the pinned official snapshot. Inputs contain 1-8192 observations; missing values and left padding are NaN. The native input is Float32[1,T], T a multiple of 16. The output is Float32[1,21,16], with no state. This profile does not expose covariates, cross-learning, quantization or longer forecast horizons.

No weights are hosted by this repository. The recipe creates a local bundle and inventory for later Models registration. A fresh conversion can have a different asset/manifest hash; update the trusted catalog only after verifying that exact bundle.

## Conversion verification

[The recorded conditions and failures](validation/chronos2-conversion-20261111/README.md) distinguish conversion fidelity from forecasting quality. All 6,048 native output values pass `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`. This does not establish taxi-demand accuracy, performance or iPhone application support.

## Swift runtime

`ChronosCore` validates raw contexts and all 21 quantiles. `ChronosRuntime` owns one request at a time and waits for load/inference completion before releasing its model. The [Mac Swift verification](validation/chronos2-swift-20261112/README.md) passes the same 18 cases and a release/reload repeat. The iOS library builds; iPhone inference and the normal demo remain unverified.

```sh
swift run --package-path CoreAISamplesKit chronos-verify --help
```

Explicit model commands require the shared lock in `AGENTS.md`. JSON input is `{"context":[1,2,null,4]}`; `null` is missing. Results preserve the full raw `[1,21,16]` tensor and identity. Output files must be new and have an existing parent directory.

<!-- model-report:start -->
<a id="model-report"></a>

## Evaluation summary

[Measurement protocol](benchmarking.md) · [Source record](model-reports/records/chronos2.json)

Results below retain their original dates and workloads. Different rows may use different runs; they are not a controlled comparison. Legacy records explicitly retain unrecorded conditions. Missing values are not zero or a pass.

### Demo support

| Platform | State | Verified scope / remaining work | Evidence |
|---|---|---|---|
| macos | — | The native sample is not connected yet; conversion evidence does not establish app support. | — |
| ios | — | The native sample is not connected yet; conversion evidence does not establish app support. | — |

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
| Conversion / reference fidelity | Passed (stated scope) | Official FP32, Python Core AI and Swift Core AI pass the same 18-case gate. | [E1](validation/chronos2-conversion-20261111/README.md) · [E2](validation/chronos2-swift-20261112/README.md) |
| Task quality | Not measured | Held-out forecasting quality and normal app behavior are not established by numerical fidelity. | — |
| Repeated use / resource release | Passed (stated scope) | Mac Swift: 18 sequential requests and one release/reload repeat. CPU fakes test ownership and delayed completion. No memory measurement or normal app/iPhone acceptance. | [E2](validation/chronos2-swift-20261112/README.md) |
| Stop / retry / recovery | Not measured | CPU fakes cover cancellation and stop joins; cancellation of the native model and normal app remain unmeasured. | — |

### Evidence and conditions

| Evidence | Date / record / route level | Device / OS / tools | Workload and recorded conditions |
|---|---|---|---|
| [E1](validation/chronos2-conversion-20261111/README.md) | 2026-11-11 / standard / conversion | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 Build version 27B52; Swift 6.4.2; macOS SDK 27.2; Python 3.11.9; chronos-forecasting 2.3.2; torch 2.9.0; coreai-torch 0.4.3; coreai-core 1.0.0b3 | 18 fixed synthetic series, raw length 1-8192, missing values and constant/periodic/trend inputs; all 21 quantiles x 16 forecast steps. One inference per case, no warmup. This measures numerical fidelity only. · variant=coreai-fp32-univariate; backend=Python Core AI GPU preference; actual delegate not profiled; route=Chronos-2 recipe: independent official CPU reference -> Core AI native verification -> bundle; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=e749b5653298b826770f57ac6a814967fad7162a; batch_size=1; concurrency=1; warmups=0; power=AC Power sampled before and after the run; thermal=Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; same after the run; temperature and GPU utilization not measured |
| [E2](validation/chronos2-swift-20261112/README.md) | 2026-11-12 / legacy / diagnostic | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; macOS/iOS SDK 27.2; Debug package | 18 fixed cases / 6048 raw values, then release/reload and the first 336 values again. Numerical validation, no warmup or performance benchmark. Power/thermal and a benchmark-plan hash were not recorded. · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; actual delegate not profiled; route=chronos-verify --verify; separate opt-in native test; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=ae1714e82a3d97012165befcc3a95222a93077cd; batch_size=1; concurrency=1; warmups=0; unrecorded: power, thermal, plan_sha256 |

### Setup, limitations and progress

- [Conversion recipe](../Tools/Conversion/recipes/chronos2/README.md)

- [Conversion evidence](validation/chronos2-conversion-20261111/README.md)

- [Swift runtime evidence](validation/chronos2-swift-20261112/README.md)

- [Implementation plan #548](https://github.com/oka-yuji/coreai-samples/issues/548)
<!-- model-report:end -->
