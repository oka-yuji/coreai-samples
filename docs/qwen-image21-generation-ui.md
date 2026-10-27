# Qwen Image 2.1 Turbo generation UI

Implementation for [#220](https://github.com/oka-yuji/coreai-samples/issues/220), using practical v2, the [native runtime](qwen-image21-native-runtime.md), and the common Models session and residency contracts. The input, confirmation, progress, stop, result and PNG flows have CPU and synthetic UI evidence. The owner accepted the current candidate's appearance through an overall gallery review on October 28, as scoped below. Actual iPhone generation, device accessibility and acceptance of the device workflow remain pending. No production artifact identity or device memory budget is invented to enable the screen.

## Hosting and ownership

`ImageGenerationAdapter` uses the existing `image-generation` demo ID and is registered only when that demo is in the trusted catalog. The normal Models host supplies scrolling. `ImageGenerationContent` therefore lays out sections at their natural height; a standalone host must supply its own scroll view.

Every binding uses one immutable `ModelDemoSession`. The adapter checks both the gate's session and its permission before creating or returning a service. A separate `ManagedImageGeneration` belongs to each demo selection. Rebinding retains prompt, seed and confirmed reference, closes the previous Store and keeps its runner until exact-session release acknowledges completion. A release from an old session cannot stop a newer binding.

Feature/Store owns effects and tasks. The runner delivers progress only within the awaited request, with nonescaping callbacks throughout the native service and pipeline. It checks current permission and request identity before delivery. Stop joins the service and media actor before acknowledging completion. A cleanup failure keeps the stop action available and blocks another generation until retry succeeds.

## Inputs and generation

The screen never starts on appearance, preparation completion or language change. Explicit Generate or Edit validates a nonempty prompt, a seed from zero through `Int64.max`, and the existing input size limit. A blank seed is chosen once, displayed after starting, and retained for retry. Changing language does not change the prompt.

A reference starts with the system file picker. Decoding produces an original thumbnail and the exact centered square that the user must confirm. Dismissing that confirmation preserves any previously confirmed reference. Reference and generation callbacks carry owner/request tickets, so a delayed result from a previous selection cannot replace current state.

Preparing, reference encoding, text encoding, decoding and stopping use indeterminate progress. Only DiT's eight steps use a step indicator, with a caption that image finishing follows. Step progress is not an elapsed-time estimate. Cancellation preserves input and previously completed output. Failed or unfinished output cannot be saved as a completed image.

## Reference image boundary

The ingress version is `imageio-rgb8-max1024-confirmed-center512-v1`. It accepts one local image frame, positive dimensions no larger than 32,768 on either side and at most 268,435,456 pixels. The file is accessed within its security scope and is never modified. The source's color model and bit depth are checked from the image properties before either decode path: only 8-bit RGB sources are accepted, so a large CMYK or 16-bit image is refused before a thumbnail is made. A PNG whose IHDR color type is 3 (indexed) is refused from its header, because ImageIO reports indexed PNGs as 8-bit RGB.

For images whose longest side is at most 1,024 pixels, direct ImageIO decoding feeds the existing eight-bit RGB reader and EXIF orientation transform. This preserves the encoded RGB bytes, including low-alpha straight RGBA, and does not apply an ICC display transform. The canonical small RGB, tagged RGB and oriented fixtures match native preprocessing exactly.

Larger sources use ImageIO's orientation-aware thumbnail decode, bounded to a longest side of 1,024, before the existing centered 512-square preprocessing. This limits decoded memory but is a distinct ingress operation: its resampling is not claimed to reproduce Pillow's full-resolution resize. Non-eight-bit or non-RGB decoded images are rejected. Both original thumbnail and resulting crop are shown before acceptance.

## Results, saving and sharing

Only a completed result with matching request/report IDs and 512-square RGBA pixels becomes a generated asset. The result display uses a checkerboard behind transparency. Details include the exact seed, profile, raw manifest identity, pinned source revision and elapsed time; the license link targets that pinned revision.

PNG export preserves straight RGBA bytes and embeds JSON metadata for model, source revision, profile, manifest, seed, dimensions, eight steps, CFG and noise source. An editing result additionally includes the reference ingress version and SHA-256 of the confirmed RGBA bytes. The seed is a decimal string so large values remain exact. The prompt and source file path are not embedded. ImageIO may serialize the PNG description into XMP; readers must not assume a Pillow `Description` key.

Save uses the system `Transferable` exporter. Dismissing the panel is distinct from completing or cancelling the write: the export ticket remains active until its completion/cancellation callback, preventing duplicate saves. Sharing uses the same PNG data and the system share sheet. Saved output has no ownership relationship to model installation files, and cancellation, model release and model deletion do not remove it.

## Availability and validation

Production availability requires an actual verified local artifact and a calibrated `ImageMemoryBudget` tied to its manifest, hardware model and OS build. The device calibration table is currently empty. Unsupported bindings show that generation is unavailable and do not construct a native service. The iOS app has its own increased-memory-limit entitlement; actual available memory is still checked before every native phase.

Run the CPU interaction tests with:

```sh
swift test --package-path CoreAISamplesKit --filter 'ImageGenerationFeatureTests|ImageGenerationMediaTests|ImageGenerationRunnerTests|ManagedImageGenerationTests'
```

These tests cover no automatic generation, invalid input, deterministic retry input, stop/retry and stale callbacks, exact-session ownership, delayed delivery after cancellation, export completion tickets, a save retry that clears an earlier save failure, reference decoding refused without current permission, CMYK and 16-bit source refusal, original-file preservation and PNG pixel/metadata identity. A 2,048×1,024 oriented fixture exercises bounded thumbnail decoding before crop confirmation. The private synthetic UI preview also exercises the real View, Store, runner, media processing, picker and exporter without running a learned model.

QMU-01 and QMU-02 have CPU evidence. QMU-03 through QMU-06 still require the trusted candidate, measured device budget, normal Models flow on the target iPhone, accessibility/orientation checks and the user's acceptance. Synthetic UI success and signed builds do not replace those checks.

### Integration checkpoint, October 25, 2026

Commit `845b6ce` integrates main through the merged TeleOCR runtime in PR #325. The lint conflict retains both ImageGenerationFeature and OCRCore in the pure-logic rules. Strict lint, format, the full Swift package suite, macOS Release and the signed iOS Release build pass. The test output reports 918 tests across 36 runners, with 38 tests and four suites explicitly skipped for unavailable integration prerequisites. The signed iOS product retains file sharing, opening documents in place, multiple scenes and the increased-memory-limit entitlement. All 57 localization keys have Japanese translations.

Independent source review and a separate rebuttal check found no remaining confirmed defects in the UI branch. The production catalog and device calibration remain empty because the W4/F32 candidate failed its frozen numerical criteria and the selective-weight alternative has only partial component evidence. QMU-03 remains unverified; completing numerical and quality evaluation, binding a measured device budget and exercising the normal Models flow are still required before claiming working iPhone generation.

### Converter fidelity follow-up, October 26, 2026

The complete retained-eight-block source cohort passed 60 inputs and two repeats, but its Core AI fixed comparison failed 32 of 1,230 checks in two editing fixtures. The failure remains in the [attention diagnostic record](research/qwen-image21-attention-20261026.json). All 480 target-region comparisons passed; that did not override the full-joint failures.

The converter now uses explicit Float32 attention arithmetic in 256-query chunks for Boolean masks, including safe handling of fully masked rows and broadcast masks. FP16, BF16 and additive floating masks retain the previous SDPA route. The weights, output ABI and numerical thresholds are unchanged. Fresh formal assets pass all 1,230 fixed comparisons, and the new source cohort's 60 final latent/RGBA hashes match the preceding reference.

The free rollout still fails for `g08-20261021`: final latent nRMSE is 0.01758714 (limit 0.005), and final RGBA RMSE is 0.02754416 (limit 0.01). Of 46 completed cases, the other 45 pass. The process ended with SIGKILL before the remaining 14 completed; the termination cause is not established. A separate diagnostic replacing only text output with saved source output also fails on the same seed. Fixed-input agreement therefore does not establish end-to-end fidelity. The production catalog and measured device budget require the completed numerical, human-quality and physical-device evidence.

The remaining 14 cases subsequently passed. An audit combining both executions with identical input/code/component bindings recomputed all 1,230 fixed comparisons and all 60 final results: 59 rollouts pass and the same single failure remains. For two saved off-trajectory inputs, Core AI agrees with the original model within the fixed-input gates; most of the step-3 output change comes from the original model's response to its perturbed input. This does not establish quantization as the cause of that sensitivity.

The `w4-dit-g16-blocks16-23-f32` candidate changes only the DiT's quantized group size from 64 to 16 and has a separate profile identity. It retains the same 56 F16 Linear weights, other components, workload and frozen gates. Its own matched source and converted outputs must be evaluated before adoption. Long diagnostic batches will use separate processes per case because a four-rollout autorelease-pool control did not resolve the observed Python process memory growth; this is not a fix for same-process retention or evidence about Swift/iPhone memory.

The [completed G16 evaluation](research/qwen-image21-g16-20261026.json) retains a failed overall numerical result: all 60 free rollouts and 480 target comparisons pass, but 28 full-joint maximum-error comparisons fail at e10/e11. Separate physical iPhone 17 Pro execution of the current DiT passes two fixed inputs across two load/call/release cycles, including all eight full-valid/target raw comparisons. Its lifetime peak footprint is 1,571,167,568 bytes. This is component evidence and does not calibrate the full app or enable the production catalog.

An experimental custom Metal RoPE operation matches the original MPS complex arithmetic bit-for-bit on four captured query/key tensors. Formal full-model numerical acceptance, image-quality review and the normal Models workflow remain outstanding.

The FMA-only private DiT, retaining global safe256 attention, passes 98 expanded fixed inputs and six targeted free rollouts. The mobile Float32 conversion path now uses this rotary ordering. Fresh formal assets, complete numerical verification and their own physical-device measurements remain required; the preceding G16 failure and earlier component measurements remain separate evidence.

Fresh formal FMA assets now have a complete matched B cohort of 60 inputs and two repeats. All 60 final latent, RGBA and PNG outputs match the preceding G16 B byte-for-byte. A ten-call host preflight passes all 18 component comparisons, including all four Vision outputs and the previously worst DiT editing input at step 3. The complete converted-model suite subsequently passed all 1,230 fixed comparisons and all 60 final latent/RGBA comparisons. Source and asset inventories, all saved numerical outputs and the public acceptance audit passed. The overall profile remains unverified pending image-quality and native-device acceptance.

The first new-asset iPhone attempt ended in an OS `vm-pageshortage` termination before a final diagnostic report or raw output was produced. The executed binary matches the jetsam record. This attempt also changed the private harness from the convenience initializer to explicit cache lookup and specialization, so the cause cannot yet be assigned to FMA alone. Jetsam page counts and the earlier process footprint are different measurements. The [diagnostic record](research/qwen-image21-g16-20261026.json) preserves the failure; device budgets, normal Models operation and human acceptance remain unverified.

The local quality-review gallery now supplies Japanese instructions and case-specific inspection points, preserving the actual generation prompts, image identities, scoring axes and saved ratings. Its rating-import boundary accepts lossless JSON number spelling changes such as `1.0` to `1`, with strict rejection of changed values and identities. This usability correction does not complete the human-quality or physical-device acceptance gates.

### Owner visual review, October 28, 2026

At the owner's request, the current local research candidate's appearance is accepted using an overall qualitative gallery review, recorded as **QMI-OWNER-VISUAL-01 / qmi-owner-visual-v1** in the [bound acceptance record](research/qwen-image21-owner-quality-20261028.json). The owner found the precision-adjusted images closest to the original outputs and the Core AI images acceptable overall, without noticing an unacceptable output during the overview. The owner need not complete all 592 ratings for this visual decision. The initial QMV-01 label was corrected because issue #221 already uses that identifier for another criterion; the method and evidence are unchanged.

This is an explicit amendment to the visual-quality adoption method in QMP-02, made after viewing the outputs. The frozen numerical requirement remains unchanged. The old per-axis benchmark is still incomplete; no success percentage, zero-failure count or at-most-two-new-failures result is inferred. The existing ratings and numerical reports are retained. The bound profile and gallery identify the reviewed outputs; acceptance does not automatically transfer to a different candidate.

The fourth gallery entry, `g02-20261021`, is text-to-image with no reference image input. Its prompt requests one blue ceramic teapot on a white desk in soft natural light, with no stripe requirement. A0 is a comparison output, not an input or required target. Missing stripes therefore do not alone violate the prompt, and do not alone prove that one output follows the prompt better. Actual image-editing cases have their own reference input and preservation requirements, which the overview does not individually certify.

QMU-05 and QMP-08 now have visual-quality evidence, while their device-operation portions remain open. This documentation does not change the benchmark's `quality_verified` / `verified` flags or enable an artifact. Candidate evidence must preserve the explicit review method when integrated. The next engineering work is the native-device memory diagnosis and complete generation, editing, cancellation, retry, PNG saving and sharing through Models.

### Code integration decision, October 28, 2026

After the owner received the visual acceptance and remaining device-work status, they requested merging the current implementation into main. This changes the integration timing recorded earlier; it does not complete the outstanding functional acceptance. Issue #220 remains open for that work, and the PR references it without closing it.

The integration preserves the existing Irodori adapter, catalog and signed iOS target while adding the conditional image-generation adapter. It also retains main's OCR page-reading and static iOS conversion routes. The verified image artifact list and device calibrations are still empty, so this merge does not make Qwen image generation available through the ordinary iPhone Models catalog. The numerical and owner visual evidence remains valid for the bound candidate; native memory diagnosis and the real device workflow remain pending.
