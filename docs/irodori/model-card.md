---
library_name: coreai
pipeline_tag: text-to-speech
language:
- ja
license: other
license_name: upstream-mit-and-apache-2.0
license_link: https://huggingface.co/okayuji/irodori-v4.1-small-mf-coreai-fp32/blob/main/LICENSE
base_model:
- Aratako/Irodori-TTS-v4.1-Small-MF
- Aratako/Semantic-DACVAE-Japanese-32dim
- sbintuitions/modernbert-ja-310m
tags:
- coreai
- apple
- on-device
- speech
- voice-cloning
---

# Irodori Small MF for Core AI

Generate Japanese speech on an Apple device, with an optional reference voice. This FP32 conversion works with the [Core AI sample app and Swift runtime](https://github.com/oka-yuji/coreai-samples).

For developers exploring local speech generation on Apple silicon with macOS 27 or a physical iPhone with iOS 27. It is an experimental conversion with known waveform-fidelity differences. It does not support iOS 26, the simulator, or streaming speech. Building the sample requires Xcode 27 and Rust/Cargo.

## Quick start

In the iPhone sample app, open **Models → Irodori → Speech demo**, select **Reference voice (FP32)** and download the model. You can generate a voice from text alone, or supply your own recording or an audio file you have permission to use. See the [iPhone guide](https://github.com/oka-yuji/coreai-samples/blob/main/docs/irodori/iphone.md).

For a Swift project using `SpeechRuntime`, download the assets with the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```sh
hf download okayuji/irodori-v4.1-small-mf-coreai-fp32 --local-dir ./Irodori
```

Build the sample repository's native input library with `Tools/Irodori/build-inputs.sh`, then use the runtime API:

```swift
import Foundation
import SpeechRuntime

let runtime = IrodoriRuntime()
_ = try await runtime.open(URL(filePath: "./Irodori"))
let audio = try await runtime.generate(text: "こんにちは。これは音声合成のテストです。", seed: 1708)
try audio.wav.write(to: URL(filePath: "./speech.wav"), options: .withoutOverwriting)
await runtime.release()
```

The folder is verified and copied into runtime-managed storage before use. Keep space for model files and specialization caches. Use a new output filename if `speech.wav` already exists. Reference input uses `SpeechReferenceAudio` from `SpeechCore` and `generate(text:seed:reference:)`; the app handles recording and audio-file conversion.

## Recorded output

Input text: `こんにちは。これは音声合成のテストです。`

<audio controls src="https://huggingface.co/okayuji/irodori-v4.1-small-mf-coreai-fp32/resolve/main/samples/text-only.wav"></audio>

The included example is **5.48 seconds**, generated without a reference recording using seed `1708`. It is a synthetic voice, not a microphone recording. This example's generation interval was not recorded separately.

A separate synthetic-reference check used approximately **3.86 seconds** of reference audio and produced **4.00 seconds** of output. Reference → no-reference → same-reference generation returned identical repeated-reference WAVs. The release assets also produced byte-identical reference and no-reference WAVs to the earlier assets. These correctness checks ran on M4 Max, macOS 27.0 (26A428), Xcode 27.0 (27A266a), FP32, one request at a time. See the [release verification record](https://github.com/oka-yuji/coreai-samples/blob/main/docs/irodori/release-asset-validation.json).

An earlier run with the original v2 assets reported **1.276 seconds** for its reference generation on the same M4 Max/macOS/Xcode configuration, one request, seed `1708`. This is the runtime's generation interval, excluding folder verification, text normalization/tokenization and audio-file decoding/resampling. The historical log did not record cache state. This is a single recorded observation, not a device performance promise or a speed comparison with this release. The [verification record](https://github.com/oka-yuji/coreai-samples/blob/main/docs/irodori/release-asset-validation.json) includes this historical observation; the [runtime guide](https://github.com/oka-yuji/coreai-samples/blob/main/docs/irodori/reference-runtime.md) describes earlier failures.

## Input and files

- Japanese text, up to **255 body tokens**; four MeanFlow steps.
- Optional reference: **48 kHz mono Float32**, **0.5–30 seconds**, finite and non-silent. The app converts supported audio files to this format and rejects an overlong recording instead of silently truncating it.
- Output: **48 kHz mono Float32 WAV**. Empty caption, no tail trimming, no added watermark.
- Model payload: **20 files, 3,835,379,527 bytes (3.84 GB)**. Documentation, licenses and the audio example are additional. File size is not the working-memory requirement.

The six components are the text encoder, reference-audio encoder, speaker encoder, conditioned duration predictor, conditioned DiT, and codec decoder. `irodori.json` identifies `irodori-coreai-v2` and the asset inventory digest. The sample app pins the tokenizer and all component files. The legacy v1 text-only inventory remains separate.

## Verification

The four added reference components passed individual PyTorch/Core AI FP32 checks on the Mac. The preliminary Python end-to-end check passed the exploratory NRMSE **0.001** threshold with a **0.5-second** reference (**0.00072797**) and missed it with an approximately **8.26-second** reference (**0.00171340**, cosine **0.99999853**, matching **114 frames**). That miss remains unresolved; it is not a listening score.

The native runtime disables frequent-reshape specialization for the audio encoder, speaker encoder and conditioned duration. DiT keeps its existing policy. The iOS codec decoder also disables it to avoid a dynamic-allocation assertion. Repeated-reference equality and the **0.5/30-second** input boundaries passed on the Mac. All-EFR execution previously returned different durations after switching voice modes; that failure is retained in the runtime record.

Source debug locations were removed with the Core AI authoring API. Generic MLIR including all constants and resources, with only locations omitted, matched before stripping, after stripping and after reading the saved assets in separate processes. This preserves the existing model computations; it does not resolve the original-PyTorch fidelity difference.

The earlier reference diagnostic completed three generations and WAV saving on **iPhone 17 Pro / iOS 27.0.1**. That diagnostic did not capture a microphone or assess listening quality. Its [device report](https://github.com/oka-yuji/coreai-samples/blob/main/docs/irodori/reference-input-device.json) identifies the assets and conditions used.

## Limitations and troubleshooting

- Original-PyTorch waveform parity is not established for the native reference path. Speaker similarity and naturalness have not been scored.
- First specialization can take longer and needs additional storage and memory. The model-file size does not establish iPhone working-memory or endurance requirements.
- Captions, arbitrary-length references, multiple reference recordings and streaming output are not implemented by this bundle.
- `This model does not support a reference voice. Choose the reference-voice model.` means a legacy bundle was selected. Do not rename a v1 folder or edit its manifest to make it look like v2.
- A hash mismatch requires the matching release files and runtime version. The original PyTorch checkpoint cannot be opened directly by this runtime.
- A generated voice can coincidentally resemble a real person. This does not grant permission to present synthetic speech as that person's actual speech.

## License and attribution

Irodori, the Japanese codec fine-tune, ModernBERT-Ja and its Sarashina2 tokenizer ancestry use MIT terms. The original DACVAE code and base weights retain Apache-2.0 terms. The corresponding license texts, copyright notices, source revisions and conversion changes are in [LICENSE](https://huggingface.co/okayuji/irodori-v4.1-small-mf-coreai-fp32/blob/main/LICENSE) and [NOTICE](https://huggingface.co/okayuji/irodori-v4.1-small-mf-coreai-fp32/blob/main/NOTICE); the upstream terms continue to apply to their components.

- Speech model: [Aratako/Irodori-TTS-v4.1-Small-MF](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF), revision `ccc78f5d480b6e51b69b2d5042a14c4da04fea6e`.
- Codec: [Aratako/Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim), revision `47376ee24834d7a05a48ebabfe3cde29b3c5e214`.
- Text backbone: [sbintuitions/modernbert-ja-310m](https://huggingface.co/sbintuitions/modernbert-ja-310m), revision `77675fc96a7e445e982e2ba90246b816efc74ec6`.

Follow the original model's ethical restrictions: use your own voice or obtain the voice owner's explicit consent for the intended synthesis use. Do not clone or impersonate someone without that consent, or create synthetic speech intended to mislead people or spread misinformation. Having a recording or finding it online does not itself establish permission. A user's declaration that permission was obtained is not proof of another person's consent.

This conversion is maintained independently of the original model authors. Core AI is Apple's framework.
