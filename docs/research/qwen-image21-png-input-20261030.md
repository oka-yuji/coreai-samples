# Transparent PNG input boundary

The native e10 trial under `e250f91` completed generation but failed the frozen numerical gate. Its fully transparent 224,246 source pixels changed from white RGB to black before the VAE. Alpha and the white-composited Vision input were unchanged. The original result remains in [the device record](qwen-image21-native-complex-device-20261029.json).

The same encoded fixture and four existing goldens passed all 100 Mac ImageIO option combinations. On iPhone 17 Pro / iOS 27.0.1 (24A446), cache true/false, immediate true/false and the CoreGraphics PNG provider all returned `premultipliedLast`. Both the URL reader and the ordinary small-image preview failed the independent pixel hash. Cache flags alone were not a correction.

Standard RGB8/RGBA8 PNG decoding now uses the pinned, unchanged LodePNG sources through a bounded C bridge. ImageIO remains responsible for metadata/orientation and the existing other-format/large-thumbnail paths. CgBI RGBA compatibility remains covered by its existing test. The diagnostic uses the same Data whose hash it validated, and Data slices with nonzero indices are accepted safely.

The CPU verification includes the independent e10 PNG and RGBA hashes, existing image goldens, malformed input, dimensions, RGB+tRNS and Adam7 cases. An independent C audit exercised 45 cases and 705 allocation-failure sites under AddressSanitizer/UBSan, with no live allocation left behind. The decompressed-output limit is not an exact ceiling on all temporary allocations.

`CoreAISamples-ImageTests` hosts the shared Swift Testing image suites in the ordinary iOS app. It creates no separate demo app. SwiftPM tool-hosted tests were refused on the physical device. The hosted bundle uses the Bundle resource lookup API and signs its nested resource-only bundles before its own signature, without applying `--deep` to production targets.

The iPhone CPU image checks pass. The corrected main-app native trials also completed with the same bundle, prompts, seeds and supplied noise. Their [bound device record](qwen-image21-png-input-device-20261030.json) records the implementation, binary, request, source and final-output hashes.

| Fixed input | Runtime with trace/journal | Final latent nRMSE | Final RGBA absolute RMSE | Process lifetime footprint peak |
|---|---:|---:|---:|---:|
| e10 / seed 20261018, edit | 282.6949 s | 0.0000024700 | 0.0000031351 | 3,648,277,736 B |
| g02 / seed 20261021, generate | 290.5902 s | 0.0000053921 | 0.0000070080 | 3,685,289,144 B |

Both pass the frozen final latent / RGBA gates (0.005 / 0.01). All 1,048,576 e10 VAE-input Float32 values are bit-identical to the independently decoded source. The saved RGBA and PNG also match the traced decoder output. This resolves the input discrepancy; the prior failed edit is retained separately.

These are two explicit diagnostic trials with unchanged estimated admission budgets. Runtime includes trace/journal work, and process-lifetime peaks are not phase peaks. The provisional 60/90-second latency target, measured production budgets, continuous use, ordinary Models operation and owner acceptance remain unfulfilled. The corrected binary UUID is `CBC3AF1C-8C83-36AC-9B27-933BB29BAB44`; its recorded source-file hashes match implementation `c55e9aa`.

The final CPU regression run reports 998 tests in 166 suites, with 43 explicitly skipped tests. An earlier complete run stalled in an unrelated VideoSearch test helper; its log is retained, the isolated 43 VideoSearch tests passed, and the subsequent full run passed. Strict lint, formatting, macOS Release and signed iOS Release also pass.
