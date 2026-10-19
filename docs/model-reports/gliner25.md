# GLiNER2.5-multi · FP32

[Home](../../README.md) · [日本語](gliner25.ja.md)

Product extraction, inspection memos

<!-- model-report:start -->
<a id="model-report"></a>

## Evaluation summary

[Measurement protocol](../benchmarking.md) · [Source record](records/gliner25.json)

Results below retain their original dates and workloads. Different rows may use different runs; they are not a controlled comparison. Legacy records explicitly retain unrecorded conditions. Missing values are not zero or a pass.

### Demo support

| Platform | State | Verified scope / remaining work | Evidence |
|---|---|---|---|
| macos | △ | Product demo implemented; the linked record defers real-model integration. | [E1](../specs/product-comparison-demo.md#remaining-acceptance-and-next-work) |
| ios | △ | The current Models route downloads and validates the published bundle (self-test on iPhone 17 Pro, no model loaded); demo execution awaits device acceptance. | [E2](../validation/ios-models-integration-20261018.md#device-checks) · [E3](../validation/ios-models-integration-20261018.md#acquisition) |

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
| Precision | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Recall | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| F1 | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |

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
| [E1](../specs/product-comparison-demo.md#remaining-acceptance-and-next-work) | 2026-10-14 / legacy / plan | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | Product comparison real-model integration deferred; source-only and fictional preview results excluded. · variant=FP32; backend=Core AI; GPU preferred; route=plan; batch_size=1; concurrency=1; unrecorded: source_revision, artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |
| [E2](../validation/ios-models-integration-20261018.md#device-checks) | 2026-10-18 / legacy / plan | iPhone 17 Pro / iOS 27.0.1 (24A446) / Xcode 27.0 (27A266a) | See fixed cases in the evidence. · variant=FP32; backend=Core AI; GPU preferred; route=plan; batch_size=1; concurrency=1; unrecorded: source_revision, artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |
| [E3](../validation/ios-models-integration-20261018.md#acquisition) | 2026-10-19 / legacy / diagnostic | iPhone 17 Pro / iOS 27.0.1 (24A446) / Xcode 27.0 (27A266a) | Anonymous HTTPS download of the 8 listed files (1,247,733,386 B), then promotion and inspection through the library store, in 38.1 s; no model loaded. · variant=FP32; route=Models self-test (--acquire all); source_revision=2ca71aafb3446d9014e1c55c7ff51c9bc7209c47; artifact_revision=043e738f3a91222ec013b2d80e240e34abb46bdb; unrecorded: backend, code_revision, batch_size, concurrency, warmups, power, thermal, plan_sha256 |

### Setup, limitations and progress

- [Product demo](../specs/product-comparison-demo.md)

- [#210](https://github.com/oka-yuji/coreai-samples/issues/210)
<!-- model-report:end -->
