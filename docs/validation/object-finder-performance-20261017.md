# Object finder image-work optimization

2026-10-17 JST. Issue #155, plan v3, PR #156, based on main `d9648b9`. The user approved the three prioritized improvements and requested main integration after feature-level verification. Whole-device performance measurement is outside this acceptance scope.

The change preserves the 30 fps native preview, model, interpolation, proposal count, similarity threshold and stop/restart behavior. It replaces per-pixel ArraySlice appends with a preallocated RGB copy, creates analysis images only when requested, and excludes ineligible candidates before Vision tracking. It does not select a different camera pixel format or retain the model across target changes.

## Acceptance evidence

| Condition | Evidence |
|---|---|
| P1 | `pixelsAndAlpha` checks exact RGB at four image sizes and rejects alpha 0, 128 and 254. Existing `imageInput` and `colorModes` cover in-memory/file equality, row/crop orientation and grayscale/CMYK inputs. |
| P2 | `imageDemand` suspends preparation and inference independently and verifies zero requests during preparation and no additional request while inference remains suspended. |
| P3 | `demandAndRestart`, `timeBoundaries` verify latest input, one delivery per request, spacing, non-finite timestamps, timestamp restart and discarded pending demand. The capture actor uses this same buffer at its only CGImage creation entry points. |
| P4 | `trackingThreshold` uses a non-default threshold and verifies the exact candidates received by tracking. `trackingEligibility` retains IDs, scores, order and overlapping eligible regions. `runnerAndStore` checks the Feature-to-Service threshold connection. |
| P5 | Existing `stopAndStale`, `cancelQueryRestart`, `rotateDuringInference`, `livePreviewFollowsFeature`, interruption and cancellation tests pass. `requestFailure` checks the new failure boundary and cleanup. |
| P6 | **474 tests passed, 32 conditionally skipped, 506 discovered**. All 30 ObjectFinder tests passed. Strict format passed for 14 changed Swift files; SwiftLint found 0 violations in 313 files. Mac Release and signed iOS Release builds succeeded, and strict signature verification passed. The Release CPU comparison below preserves RGB values and reduces extraction time. |
| P7 | User device confirmation remains unchecked. Neither sustained iPhone FPS nor a whole-app speedup is claimed. |

Environment: M4 Max, macOS 27.0 (26A428), Apple Swift 6.4, Swift 6 package language mode, Xcode 27.0 and SDK 27. The configured package isolation features are unchanged. No Core AI model or physical-camera workload was started for these checks.

## Function-level performance

The actual changed `EmbeddingImageReader` was compared with the `d9648b9` implementation. An independent Swift 6 / `-O` / AddressSanitizer probe also compared 92 color-format, padding, dimension and alpha cases: zero mismatches, 44 expected rejections and no sanitizer diagnostics. Both use the same CGContext draw and alpha validation. The input is an immutable synthetic opaque image; each size has three warmups and 21 measured pairs with alternating order, no excluded samples. Values are median / nearest-rank p95 milliseconds, compiled with `-O -whole-module-optimization -swift-version 6`.

| Input | Before | After |
|---|---:|---:|
| 320 × 320 | 0.746 / 1.134 | 0.063 / 0.086 |
| 640 × 480 | 2.240 / 2.353 | 0.178 / 0.193 |
| 720 × 1280 | 6.754 / 6.998 | 0.638 / 0.678 |

RGB values matched for every measured pair. This measures only CPU extraction on this Mac. Candidate crops can be smaller than a full camera frame, and model execution remains separate. The existing Mac demo process was not stopped; this is not an isolated whole-machine baseline. Raw values, source hashes, probe and command logs are retained in the ignored `.local/object-finder-perf` directory.

## Review and reconciliation

Independent review covers capture demand/lifetime/rotation, RGB memory and color contracts, threshold propagation, tracking eligibility and their tests. The independent coverage audit clarified that the 0.15-second gate concerns source-frame timestamps, not a strict wall-time delivery interval. No unresolved source finding remains in the reviewed ranges. Source review and CPU fakes do not establish physical-camera quality or end-to-end speed.

An independent seeded CPU comparison also matched old and new selection for 50,000 candidate/threshold/limit cases and checked 10,000 eligibility-order cases. These checks supplement the repository tests; they are not device measurements.

The shared `ObjectCandidateSelection.matching` predicate is used by both pre-tracking filtering and final selection to keep finite-value/range checks consistent. It deliberately does not perform overlap suppression before tracking. Capture remains on its existing serial executor and its analysis buffer retains only the latest camera input; no new unchecked Sendable exception is introduced.

`docs/object-finder.md` now reflects demand-based image creation and pre-tracking eligibility. The original 5 SP provisional estimate and feature scope are retained. Remaining article-driven ideas—native YUV, Core Image cache policy, model reuse and alternative detectors—remain comparison candidates rather than part of this implementation. Device-wide measurements are not used as a reason to hold the user-requested integration.
