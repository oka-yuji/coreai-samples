# Reference-voice runtime

The v2 runtime adds reference conditioning while retaining the v1 text-only bundle and `generate(text:seed:)` API. `generate(text:seed:reference:)` accepts a validated `SpeechReferenceAudio`. Legacy implementations reject a non-nil reference instead of dropping it.

Reference PCM is mono Float32 at 48 kHz, 0.5–30 seconds, finite and non-silent. Peaks above one are scaled down. The input is right-padded by reflection to the 1,920-sample codec hop. After audio encoding, incomplete four-frame speaker patches are discarded. The speaker encoder emits the masked mean token followed by the patch tokens. Its state and mask reach both duration prediction and every MeanFlow step. Output remains 48 kHz mono Float32 WAV, four steps, empty caption, no tail trimming or watermark.

The v2 folder pins 20 files: six `.aimodel` components, tokenizer and `irodori.json`. The release manifest digest is `964e4c6f3d5f235fff5bac346d545a928112274412241d348ab3c65ff5f449a5`; model payload total is 3,835,379,527 bytes. The manifest includes the asset inventory digest. The release assets omit source debug locations while retaining the same weights and operations. The legacy v1 inventory remains separate. Added component contracts are:

| Component | Input | Output |
|---|---|---|
| Reference audio | Float32 `[1,1,1920×R]`, R=13…750 | Float32 `[1,R,32]` |
| Reference speaker | Float32 `[1,4×P,32]`, Bool `[1,4×P]`, P=3…187 | Float32 `[1,P+1,768]` |
| Conditioned duration | Existing text state/mask and speaker state/mask, S=2…188 | Float32 log frame count `[1]` |
| Conditioned DiT | Existing latent/time/text/caption inputs and speaker state/mask, S=2…188 | Float32 `[1,F,32]`, F=13…750 |

Use the pinned exports and original prepared folder without modifying either source:

```sh
python3 Tools/Irodori/prepare-bundle.py \
  --artifacts /path/to/legacy/artifacts \
  --tokenizer /path/to/legacy/tokenizer.json \
  --reference-artifacts /path/to/reference-exports \
  --output /new/path/Irodori-Reference
```

The command verifies input and output hashes, materializes symbolic links, supports a source on another volume, and refuses an existing destination. Arbitrary PyTorch checkpoints cannot be imported into this runtime.

`IrodoriRuntime` keeps its busy gate across suspension and joins release. It drops the old pipeline when the reference frame count changes, including changes within one speaker patch, in addition to the existing generation-count and generated-frame policy. Audio encoder, speaker encoder and conditioned duration disable frequent-reshape specialization; DiT retains its existing policy. On iOS, the decoder also disables frequent-reshape specialization after the device check exposed a dynamic-allocation assertion.

## Validation and limits

The four added components passed individual PyTorch/Core AI FP32 checks on macOS 27.0 (26A428), M4 Max, Xcode 27. The preliminary Python end-to-end run passed the old exploratory NRMSE 0.001 threshold with a 0.5-second reference (`0.00072797`) and missed it with an approximately 8.26-second reference (`0.00171340`, cosine `0.99999853`, matching 114 frames). That miss remains a miss. It is not a listening or speaker-similarity assessment.

The initial Swift native all-EFR configuration also failed repeat equality: reference → no-reference → same reference returned 69 and 89 frames. Changing the three reference-front components to EFR false produced 100 frames and byte-identical repeated WAVs for the same roughly 3.86-second reference. The native check then passed both 0.5- and 30-second reference boundaries and generation without a reference. These are different fixtures from the original-PyTorch comparison; the native smoke is not a substitute for that fidelity check. Reference audio came from the existing synthetic speech fixture, not a user's microphone recording.

CPU tests cover input rejection, reflection and scaling, reference forwarding, old-model rejection, pipeline replacement, stop/release, session gating and v2 inventory. The existing Mac, signed iOS and SpeechPhone products are regression build targets. Opt-in native commands are listed in `CLAUDE.md`. Host checks do not cover microphone capture, normal-screen interaction or subjective listening quality on a physical iPhone.

## iPhone decoder correction (October 29, 2026)

Stage tracing on iPhone 17 Pro / iOS 27.0.1 isolated `Failed to resolve dynamic dimensions for memref.alloc` to the decoder call with latent `[1,100,32]`; the text, audio/speaker encoders, duration and four DiT calls completed first. Setting only the iOS decoder to EFR false produced the PCM and completed three reference-conditioned generations, saved WAVs and released resources. The [device report](reference-input-device.json) retains the environment, memory observations, output sizes and repeated-WAV equality. It does not certify live microphone input, listening quality, maximum-length device behavior or full PyTorch fidelity.

## Release asset verification

The release assets were saved with the Core AI debug-information stripping API. Generic MLIR including all constants and resources, with only debug locations omitted, matched before stripping, after stripping, and after loading the saved assets in separate processes. The native reference and unconditioned WAVs also matched byte-for-byte. Repeated-reference equality and the 0.5/30-second reference boundaries passed. See [the verification record](release-asset-validation.json). These checks preserve the previous runtime behavior and do not change the earlier waveform-fidelity findings.
