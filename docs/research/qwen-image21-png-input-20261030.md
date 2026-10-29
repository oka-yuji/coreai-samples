# Transparent PNG input boundary

The native e10 trial under `e250f91` completed generation but failed the frozen numerical gate. Its fully transparent 224,246 source pixels changed from white RGB to black before the VAE. Alpha and the white-composited Vision input were unchanged. The original result remains in [the device record](qwen-image21-native-complex-device-20261029.json).

The same encoded fixture and four existing goldens passed all 100 Mac ImageIO option combinations. On iPhone 17 Pro / iOS 27.0.1 (24A446), cache true/false, immediate true/false and the CoreGraphics PNG provider all returned `premultipliedLast`. Both the URL reader and the ordinary small-image preview failed the independent pixel hash. Cache flags alone were not a correction.

Standard RGB8/RGBA8 PNG decoding now uses the pinned, unchanged LodePNG sources through a bounded C bridge. ImageIO remains responsible for metadata/orientation and the existing other-format/large-thumbnail paths. CgBI RGBA compatibility remains covered by its existing test. The diagnostic uses the same Data whose hash it validated, and Data slices with nonzero indices are accepted safely.

The CPU verification includes the independent e10 PNG and RGBA hashes, existing image goldens, malformed input, dimensions, RGB+tRNS and Adam7 cases. An independent C audit exercised 45 cases and 705 allocation-failure sites under AddressSanitizer/UBSan, with no live allocation left behind. The decompressed-output limit is not an exact ceiling on all temporary allocations.

`CoreAISamples-ImageTests` hosts the shared Swift Testing image suites in the ordinary iOS app. It creates no separate demo app. SwiftPM tool-hosted tests were refused on the physical device. The hosted bundle uses the Bundle resource lookup API and signs its nested resource-only bundles before its own signature, without applying `--deep` to production targets.

The iPhone CPU image checks pass. The corrected full native editing and generation trials, device budgets and ordinary Models acceptance remain separate checks; their completion must not be inferred from these CPU tests.
