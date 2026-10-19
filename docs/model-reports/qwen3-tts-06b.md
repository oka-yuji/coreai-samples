# Qwen3-TTS CustomVoice · 0.6B

[Home](../../README.md) · [日本語](qwen3-tts-06b.ja.md)

Speech generation

<!-- model-report:start -->
<a id="model-report"></a>

## Evaluation summary

[Measurement protocol](../benchmarking.md) · [Source record](records/qwen3-tts-06b.json)

Results below retain their original dates and workloads. Different rows may use different runs; they are not a controlled comparison. Legacy records explicitly retain unrecorded conditions. Missing values are not zero or a pass.

### Demo support

| Platform | State | Verified scope / remaining work | Evidence |
|---|---|---|---|
| macos | — | Native demo is not provided for this platform. | — |
| ios | — | Native demo is not provided for this platform. | — |

### Performance and quality

| Metric | Result | Statistic / samples | Scope and evidence |
|---|---|---|---|
| Bundle / payload size | 2,696,953,939 B | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/999105687d50d94964e4eb518361ae4b6adcac1e/docs/validation/qwen3-tts-conversion-20261017.md) · Core AI assets only; separate source/tokenizer/host inputs required. |
| Prepared cache size | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| First load, including specialization | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Load from prepared cache | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Request / pipeline completion time | 4.91792558 s | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/999105687d50d94964e4eb518361ae4b6adcac1e/docs/validation/qwen3-tts-conversion-20261017.md) · greeting: generation call; exclude loading and saving as documented. |
| Request / pipeline completion time | 11.7872787 s | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/999105687d50d94964e4eb518361ae4b6adcac1e/docs/validation/qwen3-tts-conversion-20261017.md) · instruction: generation call; exclude loading and saving as documented. |
| Request / pipeline completion time | 6.82656071 s | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/999105687d50d94964e4eb518361ae4b6adcac1e/docs/validation/qwen3-tts-conversion-20261017.md) · reading: generation call; exclude loading and saving as documented. |
| Peak RSS | 8,649,900,032 B | max; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/999105687d50d94964e4eb518361ae4b6adcac1e/docs/validation/qwen3-tts-conversion-20261017.md) · One diagnostic process, lifetime peak across three utterances, with original PyTorch model also resident. Not a native-app RAM requirement. |
| Peak physical footprint | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Resident memory during inference | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Memory after release minus baseline | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Task success fraction | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Reference error (definition in evidence) | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Stop acknowledgement time | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Time to first playable audio | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Real-time factor (generation / audio duration) | 1.33639282 ratio | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/999105687d50d94964e4eb518361ae4b6adcac1e/docs/validation/qwen3-tts-conversion-20261017.md) · greeting: derived 4.91793 s / 3.68 audio s; excludes loading/saving. |
| Real-time factor (generation / audio duration) | 1.32739625 ratio | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/999105687d50d94964e4eb518361ae4b6adcac1e/docs/validation/qwen3-tts-conversion-20261017.md) · instruction: derived 11.7873 s / 8.88 audio s; excludes loading/saving. |
| Real-time factor (generation / audio duration) | 1.39888539 ratio | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/999105687d50d94964e4eb518361ae4b6adcac1e/docs/validation/qwen3-tts-conversion-20261017.md) · reading: derived 6.82656 s / 4.88 audio s; excludes loading/saving. |
| Character error rate | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Human quality score (rubric required) | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |

### Verification coverage

| Check | State | Scope / remaining work | Evidence |
|---|---|---|---|
| Conversion / reference fidelity | Passed (stated scope) | Pinned fixed-input FP32 comparisons passed; listening and publication are separate and pending. | [E1](https://github.com/oka-yuji/coreai-samples/blob/999105687d50d94964e4eb518361ae4b6adcac1e/docs/validation/qwen3-tts-conversion-20261017.md) |
| Task quality | Not measured | No completed check for this scope under the common protocol. | — |
| Repeated use / resource release | Not measured | No completed check for this scope under the common protocol. | — |
| Stop / retry / recovery | Not measured | No completed check for this scope under the common protocol. | — |

### Evidence and conditions

| Evidence | Date / record / route level | Device / OS / tools | Workload and recorded conditions |
|---|---|---|---|
| [E1](https://github.com/oka-yuji/coreai-samples/blob/999105687d50d94964e4eb518361ae4b6adcac1e/docs/validation/qwen3-tts-conversion-20261017.md) | 2026-10-17 / legacy / conversion | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | Three Japanese fixtures in the recorded conversion snapshot. Python host; source/model loads and timing boundaries follow the pinned report. No native demo. · variant=FP32 research; backend=Core AI; GPU preferred; route=Python Core AI conversion host; source_revision=85e237c12c027371202489a0ec509ded67b5e4b5; batch_size=1; concurrency=1; unrecorded: artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |

### Setup, limitations and progress

- [#167](https://github.com/oka-yuji/coreai-samples/issues/167)

- [PR #168](https://github.com/oka-yuji/coreai-samples/pull/168)

- [Pinned conversion results](https://github.com/oka-yuji/coreai-samples/blob/999105687d50d94964e4eb518361ae4b6adcac1e/docs/validation/qwen3-tts-conversion-20261017.md)

- [Conversion recipe](../../Tools/Conversion/recipes/qwen3_tts/README.md)

- [Integration and remaining acceptance](../validation/qwen3-tts-conversion-20261017.md)
<!-- model-report:end -->
