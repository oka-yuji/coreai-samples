---
library_name: coreai
pipeline_tag: text-to-speech
language:
- ja
license: other
license_name: upstream-mit-and-apache-2.0
license_link: LICENSE
base_model:
- Aratako/Irodori-TTS-v4.1-Small-MF
- Aratako/Semantic-DACVAE-Japanese-32dim
tags:
- coreai
- apple
- on-device
- speech
- voice-cloning
---

# Irodori Small MF for Core AI

Generate Japanese speech on an Apple device, with an optional reference voice. This FP32 bundle connects Irodori to the [Core AI sample app](https://github.com/oka-yuji/coreai-samples).

For developers testing speech generation and reference-voice input on macOS 27 or iOS 27. It is an experimental conversion, with unresolved waveform-fidelity differences and listening acceptance still pending. It does not provide an iOS 26 or simulator runtime.

**Publication draft:** the bundle is prepared locally. The proposed repository is `okayuji/irodori-v4.1-small-mf-coreai-fp32`. External publication and its download command are awaiting the owner's decision.

## Quick start

Use the prepared model folder with the public `SpeechRuntime` library in `CoreAISamplesKit`. The following API calls are exercised by the native integration tests; the path placeholders must point to your own files:

```swift
import Foundation
import SpeechRuntime

let runtime = IrodoriRuntime()
_ = try await runtime.open(URL(filePath: "/path/to/Irodori-Reference"))
let audio = try await runtime.generate(text: "こんにちは。これは音声合成のテストです。", seed: 1708)
try audio.wav.write(to: URL(filePath: "/new/path/speech.wav"), options: .withoutOverwriting)
await runtime.release()
```

Reference input uses `SpeechReferenceAudio` from `SpeechCore` and `generate(text:seed:reference:)`. The app exposes recording and file selection under Models → Irodori. See [the iPhone guide](https://github.com/oka-yuji/coreai-samples/blob/main/docs/irodori/iphone.md) for preparation and operation.

## Recorded output

Input text: `こんにちは。これは音声合成のテストです。`

The native Mac check used a synthetic reference of approximately 3.86 seconds and produced a 4-second mono WAV. A reference → no-reference → same-reference sequence returned byte-identical WAVs for the repeated reference. The generated sample files are retained with the local verification record; no personal microphone recording is included in this distribution draft.

One primed-cache run reported 1.276 seconds for that reference generation on M4 Max / macOS 27.0 (26A428), Xcode 27, FP32, one request, seed 1708. This is the runtime's generation interval, excluding folder hashing, text normalization/tokenization and audio-file decoding/resampling. It is a sanity check, not a device performance promise. The [runtime verification record](https://github.com/oka-yuji/coreai-samples/blob/main/docs/irodori/reference-runtime.md) describes the scope and earlier failures.

## Input and files

- Japanese text, up to 255 body tokens; four MeanFlow steps.
- Optional reference: 48 kHz mono Float32, 0.5–30 seconds. App file input converts supported audio to this format.
- Output: 48 kHz mono Float32 WAV. Empty caption, no tail trimming, no added watermark.
- Model payload: 20 files, 3,836,181,569 bytes (3.84 GB). This is storage size, not working memory. License and documentation files are additional.

The six components are the text encoder, audio encoder, speaker encoder, conditioned duration predictor, conditioned DiT, and codec decoder. `irodori.json` identifies `irodori-coreai-v2`; `tokenizer.json` and every component file are pinned by the sample app's inventory. The legacy v1 text-only bundle remains separate.

## Verification

The added four components passed individual PyTorch/Core AI FP32 checks on the Mac. The preliminary end-to-end waveform check passed the existing exploratory NRMSE 0.001 threshold with a 0.5-second reference (0.00072797) and missed it with an approximately 8.26-second reference (0.00171340, cosine 0.99999853, matching 114 frames). The miss remains unresolved; it is not a listening score.

The native runtime disables frequent-reshape specialization for the audio encoder, speaker encoder and conditioned duration. DiT keeps its existing setting; the iOS codec decoder also disables frequent-reshape specialization to avoid a dynamic-allocation assertion. This configuration passed repeated-reference equality and the 0.5/30-second input checks. All-EFR execution previously returned different durations after switching voice mode. Neither result establishes iPhone quality or production readiness.

## Limitations and troubleshooting

- The reference diagnostic completed three generations on iPhone 17 Pro / iOS 27.0.1. Live microphone operation and personal listening acceptance remain separate; see the app's validation record.
- Original-PyTorch waveform parity is not established for the native reference path. Speaker similarity and naturalness have not been scored.
- First specialization can take longer and needs additional storage and memory. Working-memory and long-run behavior on iPhone are not established by the model-file size.
- The original upstream supports other conditioning modes and longer references; this bundle does not implement captions, arbitrary-length reference audio, or streaming output.
- `This model does not support a reference voice. Choose the reference-voice model.` means a legacy bundle was selected. Do not rename a v1 folder or edit its manifest to make it look like v2.
- A hash mismatch requires a complete, matching download or prepared folder. The app does not accept the original PyTorch checkpoint directly.

## License and attribution

The Irodori and Japanese codec model cards identify their fine-tuned weights as MIT. The pinned ModernBERT-Ja text backbone is MIT. The original DACVAE code and base-model metadata use Apache-2.0. The corresponding notices and license texts are retained in `LICENSE` and `NOTICE`; those upstream terms continue to apply to their components.

Original model: [Aratako/Irodori-TTS-v4.1-Small-MF](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF), revision `ccc78f5d480b6e51b69b2d5042a14c4da04fea6e`.
Codec: [Aratako/Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim), revision `47376ee24834d7a05a48ebabfe3cde29b3c5e214`.
Text backbone: [sbintuitions/modernbert-ja-310m](https://huggingface.co/sbintuitions/modernbert-ja-310m), revision `77675fc96a7e445e982e2ba90246b816efc74ec6`.

Follow the original model's ethical restrictions: obtain permission for reference voices and do not impersonate people or produce deceptive synthetic speech. This conversion is maintained independently of the original authors. Core AI is Apple's framework.
