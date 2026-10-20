# Native Qwen Image preprocessing

[#217](https://github.com/oka-yuji/coreai-samples/issues/217) adds CPU preprocessing for the pinned Qwen Image 2.1 Turbo checkpoint. `ImageGenerationCore` contains immutable tensor contracts and calculations; `ImageGenerationRuntime` reads images and tokenizer files. Neither target loads a learned network. Core AI execution and application controls remain separate integration work.

The reference is Qwen revision `fc13ba1986fcb905854dc8de2f52eee395f49b5b`, Diffusers `b800f828f3951c4d206e154411bcbfa693cd322e`, Transformers 5.18.0, PyTorch 2.9.0 and Pillow 13.0.0.

## Contracts

| Boundary | Native behavior |
| --- | --- |
| Prompt | Exact source system/user/assistant template; 256 image slots for one reference; reject more than 512 processed tokens without truncation |
| Input size | A separate 262,144-byte guard precedes tokenization; valid long BPE tokens are covered by the 50,000-space fixture |
| Positions | Source Qwen3-VL interleaved M-RoPE, causal text mask, template removal and padding |
| Vision conditioning | Scatter merged features and three DeepStack taps only into image-token positions |
| DiT | Four latent tokens per image slot, block-causal mask, centered spatial RoPE, last target block extraction; J2304 or generation-only J1536 |
| Scheduler | Saved eight sigmas and terminal zero; delta/product rounding in model-output dtype, Float32 addition and per-step cast-back |
| Model timestep | Float32 sigma × 1000, cast to latent dtype, then divide by 1000 in that dtype |
| VAE | Channel/token transposition, checked mean/std, Float32 normalization and inverse, finite RGBA clipping and ties-to-even byte reconstruction |
| Profile/bundle | Version-two source/shape/dtype contracts and logical inventory; raw-byte SHA, filesystem checks and native descriptors belong to the importer/runtime |

`ImageGenerationPrecision` provides explicit Float16 and BFloat16 rounding. BFloat16 conversion preserves nearest-ties-to-even without relying on an unavailable Swift scalar type. Casting only once at the end of denoising is not equivalent to the source.

Physical tensor boundaries are serialized row-major. A future Core AI caller must still honor its actual NDArray descriptor strides. The profile's compute dtype and physical I/O dtype are independent.

## Images

The file reader accepts single-frame, 8-bit RGB images supported by ImageIO. It checks size before decoding, applies EXIF orientations 1–8, and preserves encoded RGB channel values instead of applying an ICC display transform. Unsupported representations fail explicitly. Tested fixtures are original PNGs, including tagged sRGB, unassociated alpha and orientation metadata; broader JPEG/HEIF decoder equivalence is not claimed.

`squareReference()` takes the centered square using integer crop coordinates and resizes it to 512 × 512. The result is the canonical reference that should be previewed and passed to both reference/native evaluation. The caller retains the source image separately.

RGBA resizing follows Pillow's premultiply/filter/unpremultiply behavior, including its choice of horizontal or vertical pass order. Same-size images retain their original hidden RGB values. These details affect byte equality at transparent edges. [Pillow resize](https://github.com/python-pillow/Pillow/blob/13.0.0/src/PIL/Image.py#L2208-L2217), [resampling implementation](https://github.com/python-pillow/Pillow/blob/13.0.0/src/libImaging/Resample.c), [alpha conversion](https://github.com/python-pillow/Pillow/blob/13.0.0/src/libImaging/Convert.c#L406-L436).

VLM input composites onto white, uses the original merged patch order and duplicated temporal patch, and normalizes as `(pixel - 127.5) / 127.5`. VAE input retains straight RGBA and computes `(pixel / 255) * 2 - 1`. The different Float32 rounding orders are intentional. [Pinned Transformers preprocessing](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/image_processing_backends.py).

## Reproducibility

The native RNG is separately versioned as `splitmix64-boxmuller-f64-packed-v1`. Its seed does not claim equality with PyTorch. Cross-runtime comparisons use saved noise and explicit input fingerprints.

The golden suite includes 30 evaluation prompts, exact 512-token generation/edit boundaries, their 513-token failures, and a valid long whitespace input: 35 cases. It also contains source rotary values, image hashes and scheduler trajectories. Saved-noise replay uses actual FP32 source predictions, selecting the source pipeline's last 1,024 target tokens. Replaying those predictions at another host dtype tests arithmetic, not half-precision network quality.

Regeneration validates the pinned source profile and all 60 fixture runs, including their file inventory and actual input hashes, before writing output. Swift tests also require the frozen source revision, fixture manifest hash and complete 35-case identity set, so a shortened suite cannot silently reduce coverage.

```sh
swift test --package-path CoreAISamplesKit --filter ImageGeneration
QMI_TOKENIZER=/path/to/pinned/snapshot/processor \
  swift test --package-path CoreAISamplesKit --filter ImageGeneration
```

The first command runs CPU calculations using checked-in goldens. The second also uses the real local tokenizer; it performs no network access or learned-model inference. Golden comparisons themselves run without Python.

Fixture regeneration uses the locked conversion environment and new output directories:

```sh
"$QI_PYTHON" -m Tools.Conversion.adapters.qwen_image21.native_fixture \
  --snapshot "$QI_SNAPSHOT" --fixtures "$QI_FIXTURES" \
  --reference "$QI_FP32_REFERENCE" --output "$QI_NATIVE_GOLDENS"
"$QI_PYTHON" -m Tools.Conversion.adapters.qwen_image21.native_image_fixture \
  --snapshot "$QI_SNAPSHOT" --output "$QI_NATIVE_GOLDENS"
```

The native library also builds directly for iOS, from `CoreAISamplesKit`:

```sh
xcodebuild -scheme ImageGenerationRuntime -configuration Release \
  -destination 'generic/platform=iOS' -derivedDataPath /path/to/build \
  CODE_SIGNING_ALLOWED=NO build
```

This is library build evidence. It does not establish device inference, full-model memory fit or end-to-end mobile image quality.

## Acceptance evidence — 2026-10-21

QMI practical v2 / #217, macOS 27.0 (26A428), M4 Max, Xcode 27 and Swift 6.4. The production implementation is `636a881`; the final review correction strengthens fixture validation and tests without changing that implementation.

| Condition | Evidence |
| --- | --- |
| QMN-01 | `layoutsMatchOriginal`, `tokenizerMatchesOriginal`, `conditioningMatchesOriginal`: all 35 pinned prompt cases, source positions/masks and feature scatter |
| QMN-02 | `ImageGenerationImageTests`: source PNG channels/orientation, both resize pass orders, white-composited VLM and straight RGBA VAE tensors; independent optimized Swift/Pillow comparison: 180 random and edge-size cases, exact bytes |
| QMN-03 | `schedulerMatchesOriginal`, `savedNoiseReplayMatchesOriginal`, RGBA reconstruction: all eight steps at Float32/Float16/BFloat16; independent `-O` replay: 24 synthetic steps and 24 saved-prediction hashes |
| QMN-04 | Invalid token/shape/dtype/profile/nonfinite tests; 513-token failures and the separate byte limit; native fixture generation rejects changed sources and incomplete suites before writing |
| QMN-05 | CPU Swift suite: 15 tests passed with the real local tokenizer; full package runners reported 746 tests passed with learned-model opt-ins skipped; macOS Release app and iOS Release library builds succeeded |

Strict SwiftLint reported zero violations in 409 files. The format gate passed for 13 Swift files. The locked conversion environment passed 164 tests, with 13 unrelated optional Magpie/Qwen TTS tests skipped. After the fixture guard correction, regeneration produced 18 files byte-identical to the committed goldens; the 15 Swift tests passed again.

After integrating #215 at `120143f`, the conversion suite passed 165 tests with the same 13 optional skips. That base update changes failed-reference reporting only; the Swift implementation and golden resources are unchanged.

Two independent reviews covered numeric/layout/tokenizer and profile/image/I/O boundaries. One confirmed defect was fixed: a two-input subset could regenerate only seven golden cases and silently reduce coverage. The new regression covers subset, reference-profile and snapshot changes. Review also checked optimized Swift arithmetic and raw image byte order. JPEG/HEIF equivalence and iPhone decoding remain unverified.

Local evidence is under `.local/validation/217/`: `all-goldens-final.log`, `full-tests.log`, `macos-build.log`, `native-ios-build-final.log`, `review-fix-swift.log`, `python-final.log`, `native-goldens-r5.log`, `swiftlint-review-fix.log` and `format-review-fix.log`. No learned weights are included in the checked-in fixture resources.

The approved single-PR scope is retained: Core and Runtime are separate targets, while the token, pixel and scheduler contracts share one original-data verification boundary. Handwritten implementation is 959 changed lines (955 additions / 4 deletions), within the planned 650–1,000 range. Test code and fixture generators are separate. Merge after #215; #218 owns network integration, and #221 owns final mobile acceptance.

The principal lesson is to preserve intermediate rounding, including the resize axis order and half-precision Euler products. Final Float32 comparisons alone do not establish equivalent computation. The complete-case guard now prevents future fixture regeneration from silently weakening that evidence.
