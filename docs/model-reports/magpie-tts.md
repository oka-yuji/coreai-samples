# Magpie TTS Multilingual v2607

[Home](../../README.md) · [日本語](magpie-tts.ja.md)

Speech generation

<!-- model-report:start -->
<a id="model-report"></a>

## Evaluation summary

[Measurement protocol](../benchmarking.md) · [Source record](records/magpie-tts.json)

Results below retain their original dates and workloads. Different rows may use different runs; they are not a controlled comparison. Legacy records explicitly retain unrecorded conditions. Missing values are not zero or a pass.

### Demo support

| Platform | State | Verified scope / remaining work | Evidence |
|---|---|---|---|
| macos | — | Native demo is not provided for this platform. | — |
| ios | — | Native demo is not provided for this platform. | — |

### Performance and quality

| Metric | Result | Statistic / samples | Scope and evidence |
|---|---|---|---|
| Bundle / payload size | 1,194,186,270 B | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/dfc037a004c38dd552fe72ef99b47d6dd7a904f7/docs/validation/magpie-conversion-20261017.md) · Sum of four Core AI assets only; source embeddings, tokenizers and Python host are additional. |
| Prepared cache size | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| First load, including specialization | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Load from prepared cache | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Request / pipeline completion time | 11.5601125 s | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/dfc037a004c38dd552fe72ef99b47d6dd7a904f7/docs/validation/magpie-conversion-20261017.md) · greeting: generation call; exclude loading and saving as documented. |
| Request / pipeline completion time | 9.7082895 s | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/dfc037a004c38dd552fe72ef99b47d6dd7a904f7/docs/validation/magpie-conversion-20261017.md) · instruction: generation call; exclude loading and saving as documented. |
| Request / pipeline completion time | 9.04915046 s | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/dfc037a004c38dd552fe72ef99b47d6dd7a904f7/docs/validation/magpie-conversion-20261017.md) · reading: generation call; exclude loading and saving as documented. |
| Peak RSS | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Peak physical footprint | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Resident memory during inference | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Memory after release minus baseline | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Task success fraction | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Reference error (definition in evidence) | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Stop acknowledgement time | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Time to first playable audio | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Real-time factor (generation / audio duration) | 3.50600353 ratio | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/dfc037a004c38dd552fe72ef99b47d6dd7a904f7/docs/validation/magpie-conversion-20261017.md) · greeting: derived 11.5601 s / 3.29723 audio s; excludes loading/saving. |
| Real-time factor (generation / audio duration) | 2.51868156 ratio | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/dfc037a004c38dd552fe72ef99b47d6dd7a904f7/docs/validation/magpie-conversion-20261017.md) · instruction: derived 9.70829 s / 3.85451 audio s; excludes loading/saving. |
| Real-time factor (generation / audio duration) | 2.66927664 ratio | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/dfc037a004c38dd552fe72ef99b47d6dd7a904f7/docs/validation/magpie-conversion-20261017.md) · reading: derived 9.04915 s / 3.39011 audio s; excludes loading/saving. |
| Character error rate | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Human quality score (rubric required) | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |

### Verification coverage

| Check | State | Scope / remaining work | Evidence |
|---|---|---|---|
| Conversion / reference fidelity | Passed (stated scope) | Pinned fixed-input FP32 comparisons passed; listening and publication are separate and pending. | [E1](https://github.com/oka-yuji/coreai-samples/blob/dfc037a004c38dd552fe72ef99b47d6dd7a904f7/docs/validation/magpie-conversion-20261017.md) |
| Task quality | Not measured | No completed check for this scope under the common protocol. | — |
| Repeated use / resource release | Not measured | No completed check for this scope under the common protocol. | — |
| Stop / retry / recovery | Not measured | No completed check for this scope under the common protocol. | — |

### Evidence and conditions

| Evidence | Date / record / route level | Device / OS / tools | Workload and recorded conditions |
|---|---|---|---|
| [E1](https://github.com/oka-yuji/coreai-samples/blob/dfc037a004c38dd552fe72ef99b47d6dd7a904f7/docs/validation/magpie-conversion-20261017.md) | 2026-10-17 / legacy / conversion | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | Three Japanese fixtures in an unmerged conversion PR. Python host; source/model loads and timing boundaries follow the pinned report. No native demo. · variant=FP32 research; backend=Core AI; GPU preferred; route=Python Core AI conversion host; source_revision=446bff55e1fe3d3aa29c771574e7d2b0fa322990; batch_size=1; concurrency=1; unrecorded: artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |

### Setup, limitations and progress

- [#165](https://github.com/oka-yuji/coreai-samples/issues/165)

- [PR #166](https://github.com/oka-yuji/coreai-samples/pull/166)

- [Pinned conversion results (unmerged PR)](https://github.com/oka-yuji/coreai-samples/blob/dfc037a004c38dd552fe72ef99b47d6dd7a904f7/docs/validation/magpie-conversion-20261017.md)
<!-- model-report:end -->
