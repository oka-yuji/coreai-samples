# Clef-Flash 9B for Core AI

[Home](../README.md) · [日本語](clef-flash.ja.md) · [Download on Hugging Face](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide)

Give a document and a checklist of allowed answers. This model returns a probability for every answer and can reuse the document while you edit the checklist. The app uses it in **Decide** and **Recipes**.

**For:** Swift developers exploring local document classification and editable decision schemas. **Not for:** free-form chat, direct photo input, or calibrated production decisions.

Requirements: Apple silicon, macOS 27, Xcode 27 with Metal Toolchain, GPU execution. Tested on M4 Max 128 GB. A minimum-memory configuration has not been established. Recipes photo reading additionally needs Apple Intelligence with vision available on the Mac.

Before building the app, complete the [common setup](../README.md#get-started), including Rust/Cargo.


<!-- model-report:start -->
<a id="model-report"></a>

## Evaluation summary

[Measurement protocol](benchmarking.md) · [Source record](model-reports/records/clef-flash.json)

Results below retain their original dates and workloads. Different rows may use different runs; they are not a controlled comparison. Legacy records explicitly retain unrecorded conditions. Missing values are not zero or a pass.

### Demo support

| Platform | State | Verified scope / remaining work | Evidence |
|---|---|---|---|
| macos | ✅ | Decide native app JSON path verified; Recipes photo behavior has separate requirements. | [E1](clef-flash.md#recorded-result) |
| ios | — | Native demo is not provided for this platform. | — |

### Performance and quality

| Metric | Result | Statistic / samples | Scope and evidence |
|---|---|---|---|
| Bundle / payload size | 7,278,548,313 B | single; n=1 | [E3](clef-flash.md#input-and-files) · 15 model payload files; documentation/examples are additional. |
| Prepared cache size | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| First load, including specialization | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Load from prepared cache | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Request / pipeline completion time | 4.27701879 s | single; n=1 | [E1](clef-flash.md#recorded-result) · Fixed workload in the linked evidence. |
| Peak RSS | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Peak physical footprint | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Resident memory during inference | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Memory after release minus baseline | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Task success fraction | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Reference error (definition in evidence) | 0.01669393 error | max; n=40 | [E2](clef-flash.md#verified-scope-and-limits) · Maximum probability difference from same-uint4 PyTorch FP32; 40/40 top options agree. |
| Stop acknowledgement time | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Reused-document request time | 0.775898292 s | single; n=1 | [E1](clef-flash.md#recorded-result) · Fixed workload in the linked evidence. |
| Reused-document request time | 0.846601958 s | p95; n=20 | [E2](clef-flash.md#verified-scope-and-limits) · Different workload: fixed 8K document / 4 questions. Not a same-request speedup ratio. |

### Verification coverage

| Check | State | Scope / remaining work | Evidence |
|---|---|---|---|
| Conversion / reference fidelity | Passed (stated scope) | Recorded fixed-document same-quantized-weight comparisons; calibration/general accuracy not established. | [E2](clef-flash.md#verified-scope-and-limits) |
| Task quality | Not measured | No completed check for this scope under the common protocol. | — |
| Repeated use / resource release | Not measured | No completed check for this scope under the common protocol. | — |
| Stop / retry / recovery | Not measured | No completed check for this scope under the common protocol. | — |

### Evidence and conditions

| Evidence | Date / record / route level | Device / OS / tools | Workload and recorded conditions |
|---|---|---|---|
| [E1](clef-flash.md#recorded-result) | 2026-10-08 / legacy / native_app | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | One Japanese maintenance request, 5 questions, 2,645 input tokens; changed-question reuse is a separate request. · variant=uint4 backbone / FP16 head; backend=Core AI; GPU preferred; route=macOS app --decide-json; source_revision=17f0b0ad64efb65d273590632833508766b2aae6; artifact_revision=58e654c86485b29a47fef820226977742ff6868a; batch_size=1; concurrency=1; unrecorded: code_revision, warmups, power, thermal, plan_sha256 |
| [E2](clef-flash.md#verified-scope-and-limits) | 2026-10-08 / legacy / diagnostic | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | 10 fixed documents / 40 options; separate 8K four-question document, 20 reuse calls. · variant=uint4 backbone / FP16 head; backend=Core AI; GPU preferred; route=diagnostic; batch_size=1; concurrency=1; unrecorded: source_revision, artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |
| [E3](clef-flash.md#input-and-files) | 2026-10-09 / legacy / conversion | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | See fixed cases in the evidence. · variant=uint4 backbone / FP16 head; backend=Core AI; GPU preferred; route=conversion; batch_size=1; concurrency=1; unrecorded: source_revision, artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |

### Setup, limitations and progress

- [Recipes and Decide](clef-flash.md)
<!-- model-report:end -->

## Quick start

Can you make a recipe with what is in your fridge? **Recipes** reads a fridge photo, lets you check the proposed ingredients, and compares your inventory with a recipe URL. It uses Apple's on-device model for the photo, **Clef-Flash 9B through Core AI** for recipe requirements and equipment, and Swift code for quantities. **Decide** uses the same Clef bundle to score a document against editable questions. Both expose the model's uncalibrated probabilities.

**Requires access to the separately distributed Clef bundle.** The `hf download` line below works only with a Hugging Face account that has access to that repository. Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), sign in with `hf auth login` using that account, then get the app and decision bundle. Select the folder containing `decision.json` when opening the bundle:

```sh
git clone https://github.com/oka-yuji/coreai-samples.git
cd coreai-samples
hf download okayuji/clef-flash-9b-coreai-decide --local-dir ./models/clef-flash-9b-decide-v1
```

Open `CoreAISamples.xcodeproj`, choose the **CoreAISamples** scheme and **My Mac**, set the Run build configuration to **Release**, and run. The **EN / JP** control at the top right switches every Mac screen.

In the **Clef-Flash 9B** sidebar entry for recipe checks, choose **Open Clef-Flash bundle** and select `models/clef-flash-9b-decide-v1`. Try **Examples → Three-egg omelette**, review the suggested egg as `1 pcs`, confirm it, then **Check recipe**. The recipe needs three eggs, so the confirmed inventory is two short. Follow the [Recipes walkthrough](recipes-demo.md) for all five bundled photos and recipe URLs. Photos and inventory stay on the Mac; fetching a recipe URL needs an internet connection.

In the **Clef-Flash 9B** sidebar entry for document checklists, open the same bundle, choose **Examples → Maintenance agreement**, then **Run checklist**. Edit a question and run again to reuse the document. The [Decide guide](decide.md) covers review, JSON export and the Swift API.

The built app also runs the recorded request without opening a window. Set `DECIDE_APP` to its executable:

```sh
COREAI_DECISION_BUNDLE="$PWD/models/clef-flash-9b-decide-v1" "$DECIDE_APP" --decide-json models/clef-flash-9b-decide-v1/examples/maintenance-request.json response.json
```

This is the same JSON entry point used in the [recorded verification](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide/blob/58e654c86485b29a47fef820226977742ff6868a/VERIFICATION.md).

## Recorded result

The [request](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide/blob/58e654c86485b29a47fef820226977742ff6868a/examples/maintenance-request.json) is a fictional Japanese maintenance agreement with five questions. For “契約期間の満了時に、合意をし直さず自動的に更新する定めがあるか。” (whether renewal is automatic), the recorded answer object is:

```json
{"type":"noul","noul":0.0107}
```

`noul` is the probability of `true`. The [full response](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide/blob/58e654c86485b29a47fef820226977742ff6868a/examples/maintenance-response.json) contains every option's diagnostic probability and logit. This five-question complete read used 2,645 input tokens and took 4.277018792 seconds. The associated changed-question reuse run took 0.775898292 seconds and reused 2,067 document tokens.

These measurements were recorded on 2026-10-08, M4 Max 128 GB, macOS 27.0 (26A428), Xcode 27 / Swift 6.4, Release, one model operation at a time. Opening the selected bundle is outside these timings; runtime preparation/reloads inside a request are included.

## Verified scope and limits

Integration conditions: the same hardware/software above, serialized GPU execution, 2026-10-07–08. Sources are the copied raw JSON records in the [verification summary](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide/blob/58e654c86485b29a47fef820226977742ff6868a/VERIFICATION.md).

| Check | Result |
|---|---|
| Ten fixed documents against the same uint4 weights evaluated in PyTorch fp32 | Maximum probability difference 0.016693930; all 40 top options matched |
| Fixed four-question 8K document, 20 reuse calls | p95 0.846601958 seconds |
| One/twelve-question alternation on the same 8K document | About 1.5–2.7 seconds per call, including runtime reloads |
| Complete request | At most 9,216 rendered input tokens, including questions/options |
| Question frame | 1–16 questions, 1–8 options per question |

The sub-second result applies to the fixed four-question case. The at-or-below-512-token band was not tested in that integration comparison. Fidelity against the same quantized weights is separate from general task accuracy and probability calibration.

## Model and application roles

The bundle contains a uint4 text backbone, a separate fp16 joint schema head, original BF16 lexical vectors and the tokenizer. Gated-delta state updates inside the backbone use fp32. The head scores the supplied choices; the runtime applies a softmax for each question.

In Recipes, Apple's on-device `SystemLanguageModel.default` reads the photo. The user checks its suggestions. Clef reads recipe text to classify ingredient requirements and equipment; Swift code matches confirmed inventory and compares quantities. Downloading the bundle and fetching recipe URLs require internet access. The photo and inventory are not sent to the recipe site.

## Input and files

Use `DecisionModel` through the [Swift API](decide.md#swift-api). Requests contain `state` and ordered `questions` of type `choice`, `noul`, or `score`. The renderer follows `joint_schema_model.py` at upstream revision `17f0b0ad64efb65d273590632833508766b2aae6`. It does not use the ordinary Qwen chat template.

The model payload has 15 files totaling 7,278,548,313 bytes, checked during packaging on 2026-10-09. The main components are `backbone/*.aimodel`, `backbone/lexical.safetensors`, `backbone/tokenizer/`, `head/*.aimodel`, `decision.json`, `SHA256SUMS`, `LICENSE` and `NOTICE.md`. Documentation and recorded examples are additional files. Model version is the SHA-256 of `decision.json`: `67f69d5bd401104fd594ba74d94622882341c138be6149132d68f9857def852d`.

## Operational notes

Keep the complete folder structure. The loader verifies the manifest and `SHA256SUMS` before specialization. `SHA-256 mismatch: <file>.` indicates a changed or incomplete file. The first load prepares models for the GPU; leave space for both the download and Core AI's prepared cache. Another executable or OS build can need its own prepared copy.

The Decide screen reserves 1,024 tokens for questions when it keeps a reusable document context. Oversized requests are rejected without truncation. A new document or complete read invalidates the old context; `This document context is no longer current.` means a new context is needed. Use the companion runtime's reload handling when changing question shapes.

## Limitations

- Probabilities are uncalibrated scores over the supplied choices. Include unknown/not-stated choices and review the original document.
- The bundle has no vision encoder, audio input or free-form generation. It has not been validated on iOS or smaller-memory Macs.
- The fixed-document checks do not establish accuracy on arbitrary documents or real-fridge photos. Quantization can change outputs relative to the original model.
- Recipes uses live Kikkoman recipe pages and a separate on-device photo model. Network failure and unavailable Apple Intelligence are reported as errors.
- This is a developer demo. The recorded twenty-call workloads do not establish indefinite-session stability or production readiness.

## License and attribution

Model assets: Apache License 2.0, inherited from [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash/tree/17f0b0ad64efb65d273590632833508766b2aae6), post-trained from Qwen/Qwen3.5-9B. The [original license](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide/blob/main/LICENSE) and Alibaba Cloud copyright notice are preserved. The [conversion notice](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide/blob/main/NOTICE.md) lists changes and tool versions. This repository's code is [MIT](../LICENSE).

Converted by Yuji Oka. This independent conversion is not a Cloudflare or Apple release. Their names and trademarks identify the upstream model and platform; no endorsement is implied.
