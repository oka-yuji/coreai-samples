# Object finder native pixel-format comparison

2026-10-17. Issue #159, PR #160, base `540d98c`. The comparison adopts supported native 8-bit 420v/420f input and retains BGRA for unrecognized or unavailable native pixel subtypes. This is not a blanket HDR-mode exclusion: video HDR state and active color space are separate device properties. The output contract is explicitly RGBA8/sRGB with immediate rendering; this does not prove the precise release time of every original buffer.

After committing input/preset configuration, capture reads activeFormat and the output's available formats. It selects the native subtype only if both recognized and supported, otherwise BGRA, or fails through existing cleanup. Prior output settings are reset with the SDK's empty native-format dictionary; final output and 30 fps are configured before capture starts. The output format list order is not used as preference.

| Condition | Evidence |
|---|---|
| V1–V2 | `nativeAndFallback`, `unsupported`; independent CPU probe checked 32 native/support combinations, three SDK constant roundtrips and 13 unrecognized pixel types. |
| V3 | Synthetic 720×1280 CVPixelBuffers: BT.601/709 matrices × video/full range × four seeds, with independently varying Cb/Cr. All 16 cases matched exactly across 2,764,800 RGB channels each. The native direct RGBA8/sRGB/eager path was compared with Core Image's BGRA conversion followed by the previous reader path. Native buffer size 1,479,680 bytes; BGRA 3,686,400 bytes (59.861% less in this fixture). |
| V4 | 478 CPU tests passed / 32 conditional skips (510 discovered), including 34 ObjectFinder tests. Strict format passed for 3 changed Swift files, SwiftLint found 0 violations in 316 files, and Mac Release / signed iOS Release / signature verification passed. |
| V5 | Live iPhone appearance, ISP conversion, rotation and Vision/model behavior remain a user device check. |

The eight initial BT.709 fixtures were expanded after review because their Cb/Cr values were correlated. The zero-difference criterion was not relaxed. These are synthetic Core Video/Core Image comparisons, not a capture-ISP equivalence test, all-gamut guarantee or an iPhone memory/FPS measurement. No sensor, Vision request or Core AI model was started by the probe. The raw input-buffer capacity is not the whole-app memory footprint.

CIContext cache-on/off was also compared with three alternating pairs of separate processes, 8 warmups and 80 frames each. RGB hashes matched, median conversion times were approximately 1.9 ms and own-process peak RSS approximately 166 MB; the predeclared 10% memory-reduction gate was not met. Cache policy remains unchanged. The reference program was rerun after expanding fixtures to retain a source/binary-identifiable comparison.

Environment: M4 Max, macOS 27.0, Swift 6.4 / Swift 6 / -O, SDK27. An existing demo process remained open, so absolute GPU timings may be affected. Local evidence: ignored `.local/object-finder-next/image-comparison-plan.json`, `ImageComparison.swift`, `format-independent-*.json`, `cache-final-*.json`, final metadata and build/test logs. Independent review checked SDK contracts, selection, lifecycle and the limits of the comparison; no unresolved source defect remains.

The behavior reference is updated in `docs/object-finder.md`. The provisional 3 SP and one-unit scope are unchanged. The timer change was integrated separately in PR #158, and unverified model replacements are not part of this change.
