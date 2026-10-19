# Qwen Image 2.1 Turbo · macOS FP32

[Home](../../README.md) · [日本語](qwen-image21.ja.md)

Image generation and editing

<!-- model-report:start -->
<a id="model-report"></a>

## Evaluation summary

[Measurement protocol](../benchmarking.md) · [Source record](records/qwen-image21.json)

Results below retain their original dates and workloads. Different rows may use different runs; they are not a controlled comparison. Legacy records explicitly retain unrecorded conditions. Missing values are not zero or a pass.

### Demo support

| Platform | State | Verified scope / remaining work | Evidence |
|---|---|---|---|
| macos | — | Native demo is not provided for this platform. | — |
| ios | — | Native demo is not provided for this platform. | — |

### Performance and quality

| Metric | Result | Statistic / samples | Scope and evidence |
|---|---|---|---|
| Bundle / payload size | 62,370,841,163 B | single; n=1 | [E1](../research/qwen-image21-20261018/README.md) · Logical bundle files excluding bundle.json; disk payload, not RAM or prepared-cache size. |
| Prepared cache size | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| First load, including specialization | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Load from prepared cache | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Request / pipeline completion time | 163.63 s | single; n=1 | [E1](../research/qwen-image21-20261018/README.md) · Generate case, including 123.42 s of component loads; not warm interactive latency. |
| Request / pipeline completion time | 207.24 s | single; n=1 | [E1](../research/qwen-image21-20261018/README.md) · Edit case, including 148.77 s of loads; same diagnostic boundary. |
| Peak RSS | 110,662,795,264 B | single; n=1 | [E1](../research/qwen-image21-20261018/README.md) · Whole verification-process lifetime, including reference/diagnostic work. Not standalone app RAM. |
| Peak physical footprint | 7,751,065,968 B | single; n=1 | [E1](../research/qwen-image21-20261018/README.md) · Same verification process; different accounting from RSS, do not add them. |
| Resident memory during inference | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Memory after release minus baseline | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Task success fraction | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Reference error (definition in evidence) | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Stop acknowledgement time | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Image/text CLIP score | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Human quality score (rubric required) | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |

### Verification coverage

| Check | State | Scope / remaining work | Evidence |
|---|---|---|---|
| Conversion / reference fidelity | Passed (stated scope) | Two fixed FP32 generation/edit outputs and packaged replay; broader quality and mobile use unverified. | [E1](../research/qwen-image21-20261018/README.md) |
| Task quality | Not measured | No completed check for this scope under the common protocol. | — |
| Repeated use / resource release | Not measured | No completed check for this scope under the common protocol. | — |
| Stop / retry / recovery | Not measured | No completed check for this scope under the common protocol. | — |

### Evidence and conditions

| Evidence | Date / record / route level | Device / OS / tools | Workload and recorded conditions |
|---|---|---|---|
| [E1](../research/qwen-image21-20261018/README.md) | 2026-10-18 / legacy / conversion | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | 512×512 RGBA; batch1; CFG1; 8 sigma steps; generation and one-image edit. Python diagnostic host, staged loads and tensor capture. · variant=FP32; cache off; backend=Core AI; GPU preferred; route=Python conversion / diagnostic host; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; batch_size=1; concurrency=1; unrecorded: artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |

### Setup, limitations and progress

- [Conversion](../../Tools/Conversion/recipes/qwen_image21/README.md)

- [Mobile plan #222](https://github.com/oka-yuji/coreai-samples/issues/222)
<!-- model-report:end -->
