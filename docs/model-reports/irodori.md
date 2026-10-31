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
| macos | △ | Native generation is verified with synthetic references; GUI playback and subjective quality are not covered by these checks. | [E1](../irodori/demo.md) · [E4](../irodori/release-asset-validation.json) · [E5](../irodori/release-download-validation.json) · [E6](../irodori/release-download-validation.json) |
| ios | △ | An iPhone 17 Pro diagnostic verified reference-file generation, three WAV outputs and repeat equality. Microphone capture and normal-screen interaction are outside that diagnostic. | [E2](../irodori/iphone.md) · [E3](../irodori/reference-input-device.json) |

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
| [E1](../irodori/demo.md) | 2026-10-19 / legacy / plan | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | Native implementation and CPU/build checks without model execution. The date is the record review date. · variant=FP32; backend=Core AI; GPU preferred; route=plan; batch_size=1; concurrency=1; unrecorded: source_revision, artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |
| [E2](../irodori/iphone.md) | 2026-10-19 / legacy / plan | iPhone 17 Pro / iOS 27.0.1 (24A446) / Xcode 27.0 (27A266a) | Native implementation and CPU/build checks without model execution. The date is the record review date. · variant=FP32; backend=Core AI; GPU preferred; route=plan; batch_size=1; concurrency=1; unrecorded: source_revision, artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |
| [E3](../irodori/reference-input-device.json) | 2026-10-29 / legacy / diagnostic | iPhone 17 Pro / iOS 27.0.1 (24A446) / Xcode 27.0 (27A266a) | Synthetic 3.88-second reference; greeting, announcement, greeting; seed 1708; three WAVs and repeated greeting equality. This is not normal-screen or listening acceptance. · variant=Reference voice FP32 v2; backend=Core AI; GPU preferred; audio/speaker/duration/decoder EFR false; route=SpeechPhone explicit reference-file diagnostic; no playback or microphone capture; source_revision=ccc78f5d480b6e51b69b2d5042a14c4da04fea6e; code_revision=PR370 working tree based on a755889e87f027497d979d41e4469805b955faf6; batch_size=1; concurrency=1; warmups=0; power=USB connected, charging; thermal=nominal (0); unrecorded: artifact_revision, plan_sha256 |
| [E4](../irodori/release-asset-validation.json) | 2026-10-31 / standard / diagnostic | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a); Swift 6.4; SwiftPM Debug | Reference and no-reference WAVs match fixed original-asset baselines byte-for-byte; repeated reference and 0.5/30-second reference boundaries pass. No microphone capture or GUI playback. · variant=Reference voice FP32 v2 release; backend=Core AI; GPU preferred; route=Native integration test; no latency benchmark; specialization cache not reset; source_revision=ccc78f5d480b6e51b69b2d5042a14c4da04fea6e; artifact_revision=964e4c6f3d5f235fff5bac346d545a928112274412241d348ab3c65ff5f449a5; code_revision=c2e6c67be22847e848fd22668e63b93ea9aa62eb; batch_size=1; concurrency=1; warmups=0; power=AC attached; thermal=nominal (0) before and after |
| [E5](../irodori/release-download-validation.json) | 2026-11-01 / standard / diagnostic | M5 Max 128 GiB / macOS 27.2 (26B123) / Xcode 27.0 (27A266a); Swift 6.4; SwiftPM Debug | All 22 downloaded files pass size/SHA checks and become ready. Reference/no-reference generation, repeat equality and 0.5/30-second boundaries pass. The additional byte-equality gate against the earlier M4/macOS 27.0 WAVs fails; the same-host control is recorded separately. · variant=Reference voice FP32 v2 public release; backend=Core AI; GPU preferred; route=Anonymous Models acquisition and nativeSmoke, with the historical M4/macOS 27.0 byte-equality check; source_revision=ccc78f5d480b6e51b69b2d5042a14c4da04fea6e; artifact_revision=f5db5096365c7d1388a5002cfb3280f66cb19375; code_revision=ed013ff786fb9aa79c61c8addb8883ec321965b2; batch_size=1; concurrency=1; warmups=0; power=AC attached; thermal=nominal (0) before and after released-asset generation |
| [E6](../irodori/release-download-validation.json) | 2026-11-01 / legacy / diagnostic | M5 Max 128 GiB / macOS 27.2 (26B123) / Xcode 27.0 (27A266a); Swift 6.4; SwiftPM Debug | Both pre-strip WAVs match the anonymous public-release outputs byte-for-byte on this host. Repeated-reference equality and 0.5/30-second reference boundaries pass. This does not make the historical cross-environment comparison pass. Thermal state was not sampled separately for this control. · variant=Pre-strip reference FP32 v2 control against public release; backend=Core AI; GPU preferred; route=Pre-strip assets vs public release on the same M5/macOS 27.2 host; identical generation code and dependencies; source_revision=ccc78f5d480b6e51b69b2d5042a14c4da04fea6e; artifact_revision=8dca0128ec50d8b186a2f9acb6e6d658f65e9e21aae0ac766bad650e5ede4ea9; code_revision=1ae9634e29729afa8595c6d9caaad6f63850f862; batch_size=1; concurrency=1; warmups=0; power=AC attached; unrecorded: thermal |

### Setup, limitations and progress

- [Mac](../irodori/demo.md)

- [iPhone](../irodori/iphone.md)

- [#93](https://github.com/oka-yuji/coreai-samples/issues/93)

- [Models validation](../irodori/managed-validation.md)

- [#329](https://github.com/oka-yuji/coreai-samples/issues/329)

- [Reference voice runtime](../irodori/reference-runtime.md)

- [Release asset verification](../irodori/release-asset-validation.json)

- [Public download verification](../irodori/managed-validation.md#public-download-verification-2026-11-01)
<!-- model-report:end -->
