# Qwen3-TTS CustomVoice · 1.7B

[Home](../../README.md) · [日本語](qwen3-tts-17b.ja.md)

Speech generation

<!-- model-report:start -->
<a id="model-report"></a>

## Evaluation summary

[Measurement protocol](../benchmarking.md) · [Source record](records/qwen3-tts-17b.json)

Results below retain their original dates and workloads. Different rows may use different runs; they are not a controlled comparison. Legacy records explicitly retain unrecorded conditions. Missing values are not zero or a pass.

### Demo support

| Platform | State | Verified scope / remaining work | Evidence |
|---|---|---|---|
| macos | — | Native demo is not provided for this platform. | — |
| ios | — | Native demo is not provided for this platform. | — |

### Performance and quality

| Metric | Result | Statistic / samples | Scope and evidence |
|---|---|---|---|
| Bundle / payload size | 6,602,093,201 B | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) · Core AI assets only; separate source/tokenizer/host inputs required. |
| Prepared cache size | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| First load, including specialization | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Load from prepared cache | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Request / pipeline completion time | 11.141144 s | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) · greeting: generation call; exclude loading and saving as documented. |
| Request / pipeline completion time | 10.1524363 s | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) · instruction: generation call; exclude loading and saving as documented. |
| Request / pipeline completion time | 10.088553 s | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) · reading: generation call; exclude loading and saving as documented. |
| Peak RSS | 20,477,050,880 B | max; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) · One diagnostic process, lifetime peak across three utterances, with original PyTorch model also resident. Not a native-app RAM requirement. |
| Peak physical footprint | 9,750,459,472 B | max; n=3 | [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) · Largest recorded kernel-lifetime peak at three sampled checkpoints; sampling does not prove the final process maximum. PyTorch co-resident. |
| Resident memory during inference | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Memory after release minus baseline | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Task success fraction | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Reference error (definition in evidence) | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Stop acknowledgement time | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Time to first playable audio | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Real-time factor (generation / audio duration) | 2.44323333 ratio | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) · greeting: derived 11.1411 s / 4.56 audio s; excludes loading/saving. |
| Real-time factor (generation / audio duration) | 2.64386363 ratio | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) · instruction: derived 10.1524 s / 3.84 audio s; excludes loading/saving. |
| Real-time factor (generation / audio duration) | 2.74145463 ratio | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) · reading: derived 10.0886 s / 3.68 audio s; excludes loading/saving. |
| Character error rate | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Human quality score (rubric required) | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |

### Verification coverage

| Check | State | Scope / remaining work | Evidence |
|---|---|---|---|
| Conversion / reference fidelity | Passed (stated scope) | Pinned fixed-input FP32 comparisons passed; listening and publication are separate and pending. | [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) |
| Task quality | Not measured | No completed check for this scope under the common protocol. | — |
| Repeated use / resource release | Not measured | No completed check for this scope under the common protocol. | — |
| Stop / retry / recovery | Not measured | No completed check for this scope under the common protocol. | — |

### Evidence and conditions

| Evidence | Date / record / route level | Device / OS / tools | Workload and recorded conditions |
|---|---|---|---|
| [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) | 2026-10-17 / legacy / conversion | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | Three Japanese fixtures in the recorded conversion snapshot. Python host; source/model loads and timing boundaries follow the pinned report. No native demo. · variant=FP32 research; backend=Core AI; GPU preferred; route=Python Core AI conversion host; source_revision=0c0e3051f131929182e2c023b9537f8b1c68adfe; batch_size=1; concurrency=1; unrecorded: artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |

### Setup, limitations and progress

- [#169](https://github.com/oka-yuji/coreai-samples/issues/169)

- [PR #170](https://github.com/oka-yuji/coreai-samples/pull/170)

- [Pinned conversion results](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md)

- [Conversion recipe](../../Tools/Conversion/recipes/qwen3_tts_17b/README.md)

- [Integration and remaining acceptance](../validation/qwen3-tts-17b-conversion-20261017.md)
<!-- model-report:end -->
