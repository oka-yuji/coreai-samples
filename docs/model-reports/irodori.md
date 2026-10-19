# Irodori · FP32

[Home](../../README.md) · [日本語](irodori.ja.md)

Speech generation and playback

<!-- model-report:start -->
<a id="model-report"></a>

## Evaluation summary

[Measurement protocol](../benchmarking.md) · [Source record](records/irodori.json)

Results below retain their original dates and workloads. Different rows may use different runs; they are not a controlled comparison. Legacy records explicitly retain unrecorded conditions. Missing values are not zero or a pass.

### Demo support

| Platform | State | Verified scope / remaining work | Evidence |
|---|---|---|---|
| macos | △ | Runtime/UI implemented; actual app GPU generation and listening acceptance pending. | [E1](../irodori/demo.md) |
| ios | △ | Runtime/UI implemented; actual app GPU generation and listening acceptance pending. | [E2](../irodori/iphone.md) |

### Performance and quality

| Metric | Result | Statistic / samples | Scope and evidence |
|---|---|---|---|
| Bundle / payload size | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Prepared cache size | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| First load, including specialization | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Load from prepared cache | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Request / pipeline completion time | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Peak RSS | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Peak physical footprint | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Resident memory during inference | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Memory after release minus baseline | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Task success fraction | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Reference error (definition in evidence) | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Stop acknowledgement time | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Time to first playable audio | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Real-time factor (generation / audio duration) | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Character error rate | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Human quality score (rubric required) | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |

### Verification coverage

| Check | State | Scope / remaining work | Evidence |
|---|---|---|---|
| Conversion / reference fidelity | Not measured | No completed check for this scope under the common protocol. | — |
| Task quality | Not measured | No completed check for this scope under the common protocol. | — |
| Repeated use / resource release | Not measured | No completed check for this scope under the common protocol. | — |
| Stop / retry / recovery | Not measured | No completed check for this scope under the common protocol. | — |

### Evidence and conditions

| Evidence | Date / record / route level | Device / OS / tools | Workload and recorded conditions |
|---|---|---|---|
| [E1](../irodori/demo.md) | 2026-10-19 / legacy / plan | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | Native implementation and CPU/build checks; actual app generation/listening pending. Status reviewed on 2026-10-19; this is not a model-run date. · variant=FP32; backend=Core AI; GPU preferred; route=plan; batch_size=1; concurrency=1; unrecorded: source_revision, artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |
| [E2](../irodori/iphone.md) | 2026-10-19 / legacy / plan | iPhone 17 Pro / iOS 27.0.1 (24A446) / Xcode 27.0 (27A266a) | Native implementation and CPU/build checks; actual app generation/listening pending. Status reviewed on 2026-10-19; this is not a model-run date. · variant=FP32; backend=Core AI; GPU preferred; route=plan; batch_size=1; concurrency=1; unrecorded: source_revision, artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |

### Setup, limitations and progress

- [Mac](../irodori/demo.md)

- [iPhone](../irodori/iphone.md)

- [#93](https://github.com/oka-yuji/coreai-samples/issues/93)
<!-- model-report:end -->
