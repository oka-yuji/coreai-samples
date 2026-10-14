# EmbeddingGemma 2 native demo validation

2026-10-15. Runtime issue #112 and demo issue #113. The implementation is complete for native text/image input, local image search, and signed Mac/iOS builds. Device execution and the user's manual acceptance remain open.

## Measured results

The original model revision is `914f7f89142e33e77833254d9c9b90c3cef7303b`. The assets use safe FP16 normalization, S128 text, and a 70-soft-token vision budget. The host is M4 Max 128 GB, macOS 27.0 (26A428), Swift 6.4 in Swift 6 mode, and swift-transformers 1.3.4.

| Check | Result |
|---|---|
| Native text tokenizer | 12/12 exact ID sequences, including prefixes and BOS/EOS |
| Native image token counts | 8/8 exact |
| Pixel preprocessing | Maximum difference 1/255 for five recipe images; zero for three gradients |
| Native final vectors | 20/20 pass; minimum cosine 0.999483553; maximum absolute difference 0.004896015 |
| Unit norm | Maximum deviation 0.000413965 |
| Application path | Five images indexed; three queries match the original FP32 top result; original images readable; queued cancellation and retry succeed |
| Builds | Mac Release and signed iOS Release succeed |
| View rendering | Japanese gallery inspected at 1040×760 and 390×844 points |
| iPhone execution | Not performed: iPhone 17 Pro/iOS 27.0.1 was locked during installation, CoreDevice error 10003 |

These are twenty fixed numerical cases and three small application scenarios, not broad retrieval-quality or sustained performance measurements. The warmed application run indexed five images in 0.767 seconds; the initial run took 11.637 seconds. The timing includes preprocessing and is not a controlled benchmark.

The vector gate was fixed before the native implementation: finite values, cosine ≥ 0.999, maximum absolute difference ≤ 0.01, and norm deviation ≤ 0.001. Raw vectors were independently recalculated using `math.fsum`. All gates remain unchanged.

## Evidence

- [Verification conditions, independent arithmetic, and source hashes](embeddinggemma2-native-20261015/verification.json)
- [Raw native vectors](embeddinggemma2-native-20261015/vectors.json)
- [Token and pixel comparison](embeddinggemma2-native-20261015/preprocessing.json)
- [Application queries and rankings](embeddinggemma2-native-20261015/searches.json)
- [Application summary](embeddinggemma2-native-20261015/summary.json)
- [Wide Japanese gallery](embeddinggemma2-native-20261015/gallery-ja.png)
- [Compact Japanese gallery](embeddinggemma2-native-20261015/gallery-compact-ja.png)

The gallery images are snapshots of the actual SwiftUI view hosted by AppKit, using the successfully computed application state. Computer-use automation failed to initialize, so mouse/touch operations and the system file picker were not exercised. The app's integration harness sends the same Store actions used by its controls. Original-image decoding was verified separately from presentation of the detail sheet.

## Acceptance mapping

The original acceptance wording remains in the linked Issues; IDs are preserved here.

| ID | Evidence / state |
|---|---|
| R1 | `bundleBoundaries`, `bundleEnumeration`, `checksum`; real asset signature checked before inference |
| R2 | `tokenization` with `EG2_TOKENIZER`; preprocessing report; overlong input rejection |
| R3 | `invalidInputs`, `geometry`, `orientation`, `patches`, `colorModes`; pixel comparison |
| R4 | 20 native vectors and independent arithmetic above |
| R5 | `failureRecovery`, `cancellation`; actor gate holds ownership across suspension |
| R6 | Full CPU suite, Mac and iOS Release builds, format and lint checks |
| U1 | `guards`, `snapshot`, `indexAndRetry`; real application searches |
| U2 | `stop`, `closeAndReload`, `storeCancellation`; queued stop/retry in the app |
| U3 | Failure/retry reducer tests and runtime failure recovery |
| U4 | Mac/iOS Release, CPU suite, Swift formatting, lint, and Japanese catalog checks |
| U5 | App Store/runtime path and rendered gallery verified; interactive file-picker/detail-sheet operation remains unverified |
| U6 | User acceptance pending |

The real Store/runner cancellation test uses a suspended service to represent an in-flight operation. The application scenario cancels before dispatch, reported as `queuedStopAndRetry`. Neither is represented as a measurement of cancellation inside the OS GPU call.

SwiftLint's checked-in baseline uses relative paths; this tool version emits absolute `/tmp` paths and fails to match the existing 35 violations. A temporary baseline expanded the same 41 existing entries to absolute paths. No rule, source text, or allowed violation was added; the resulting full lint run passed. `check_format.py --base origin/main` passed for 22 changed Swift files. The pre-existing baseline behavior is not repaired in this feature.

## Review and findings

An independent review inspected runtime, preprocessing, validation boundaries, cancellation, UI state transitions, file scope, resources, and the native application checks. Confirmed findings were corrected and rechecked:

- Gray/CMYK images need RGB conversion; replacing the color-space tag only works for RGB component counts.
- URL enumeration can return `/tmp` while the root uses `/private/tmp`; relative paths cannot be made by deleting a fixed character count.
- Three Japanese UI keys were missing.
- Validation now rejects an existing output directory, uses a unique remote run ID, and exits a cancelled wait instead of spinning.
- The job wrapper checks the whole process group before releasing its lock. A CPU reproduction that left a SIGTERM-ignoring child alive was corrected and independently rechecked; the child is killed after the grace period.

The original FP32 fixtures, raw vectors, source hashes, and all thresholds were checked independently. No unresolved confirmed code finding remains. Physical-device execution, interactive UI behavior, and real GPU cancellation remain outside that conclusion.

The article idea remains recorded separately: explain the actual conversion, the plain-FP16 failure, the normalization changes, and the verification process after the demo is reviewed. No article or model release was published by this work.
