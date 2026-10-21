# Qwen Image 2.1 Turbo generation UI

Implementation for [#220](https://github.com/oka-yuji/coreai-samples/issues/220), using practical v2, the [native runtime](qwen-image21-native-runtime.md), and the common Models session and residency contracts. The input, confirmation, progress, stop, result and PNG flows have CPU and synthetic UI evidence. Actual iPhone generation, device accessibility and human acceptance remain pending. No production artifact identity or device memory budget is invented to enable the screen.

## Hosting and ownership

`ImageGenerationAdapter` uses the existing `image-generation` demo ID and is registered only when that demo is in the trusted catalog. The normal Models host supplies scrolling. `ImageGenerationContent` therefore lays out sections at their natural height; a standalone host must supply its own scroll view.

Every binding uses one immutable `ModelDemoSession`. The adapter checks both the gate's session and its permission before creating or returning a service. A separate `ManagedImageGeneration` belongs to each demo selection. Rebinding retains prompt, seed and confirmed reference, closes the previous Store and keeps its runner until exact-session release acknowledges completion. A release from an old session cannot stop a newer binding.

Feature/Store owns effects and tasks. The runner delivers progress only within the awaited request, with nonescaping callbacks throughout the native service and pipeline. It checks current permission and request identity before delivery. Stop joins the service and media actor before acknowledging completion. A cleanup failure keeps the stop action available and blocks another generation until retry succeeds.

## Inputs and generation

The screen never starts on appearance, preparation completion or language change. Explicit Generate or Edit validates a nonempty prompt, a seed from zero through `Int64.max`, and the existing input size limit. A blank seed is chosen once, displayed after starting, and retained for retry. Changing language does not change the prompt.

A reference starts with the system file picker. Decoding produces an original thumbnail and the exact centered square that the user must confirm. Dismissing that confirmation preserves any previously confirmed reference. Reference and generation callbacks carry owner/request tickets, so a delayed result from a previous selection cannot replace current state.

Preparing, reference encoding, text encoding, decoding and stopping use indeterminate progress. Only DiT's eight steps use a step indicator, with a caption that image finishing follows. Step progress is not an elapsed-time estimate. Cancellation preserves input and previously completed output. Failed or unfinished output cannot be saved as a completed image.

## Reference image boundary

The ingress version is `imageio-rgb8-max1024-confirmed-center512-v1`. It accepts one local image frame, positive dimensions no larger than 32,768 on either side and at most 268,435,456 pixels. The file is accessed within its security scope and is never modified.

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

These tests cover no automatic generation, invalid input, deterministic retry input, stop/retry and stale callbacks, exact-session ownership, delayed delivery after cancellation, export completion tickets, original-file preservation and PNG pixel/metadata identity. A 2,048×1,024 oriented fixture exercises bounded thumbnail decoding before crop confirmation. The private synthetic UI preview also exercises the real View, Store, runner, media processing, picker and exporter without running a learned model.

QMU-01 and QMU-02 have CPU evidence. QMU-03 through QMU-06 still require the trusted candidate, measured device budget, normal Models flow on the target iPhone, accessibility/orientation checks and the user's acceptance. Synthetic UI success and signed builds do not replace those checks.
