# Object finder validation

2026-10-17 JST. Plan v1; issues #147, #149 and #150. This record is not final acceptance: static-photo inference has not run, and the user will perform iPhone touch acceptance after integration.

## Recorded checks

The environment is Apple Swift 6.4, Swift 6 language mode for new package targets, Xcode 27.0 and the macOS/iPhoneOS 27 SDKs. The new modules enable `NonisolatedNonsendingByDefault` and `InferIsolatedConformances`. The full CPU suite passed 383 tests, including 17 ObjectFinder tests. Strict format passed for 20 changed Swift files; SwiftLint reported no violations in 303 files. The configured Japanese localization is complete for all 680 catalog keys.

| Condition | Evidence and current state |
|---|---|
| F1–F2 | `invalidRegions`, `filtering`: invalid/non-finite regions and scores, preserved coordinates, overlap suppression, limits and empty matches |
| F3 | `imageInput`: in-memory/file RGB equality, known colored rows and top-row crop; existing EmbeddingRuntime tests remain part of the full suite |
| F4 / C6 | Pending: real eyeglass photos, negative photos, manually fixed boxes, raw similarities and selected boxes |
| C1–C3 | `startGuards`, `stopAndStale`, `notificationAndErrors`: valid inputs, stopped/reopened search, stale results, stable notification and distinct errors |
| C4 | `runnerAndStore`, `cancellationGate`, `deniedPermission`, `stopDuringPreparation`, `rotateDuringInference`, `interruptionDuringInference`: controlled delayed completion, permission rejection, stop acknowledgement, stale output and error propagation using the real service with fake devices |
| C8 automated portion | `flatOrientation`, `sharedOrientationDuringPreparation`, `rotationDuringInference`: downward-facing device, differently ordered start/orientation commands, old sessions and old rotation revisions |
| F5 / C5 | Swift tests, changed-Swift strict format, SwiftLint and Mac/iOS Release builds are checked separately from model inference; generated camera purpose text and en/ja strings were inspected |
| C7–C8 device acceptance | Pending for the user: actual camera, localized boxes, haptic, denial/settings, background/close, restart, rotation including a locked interface and 180-degree turns |

The attached device inventory contained no connected iPhone. A signed `CoreAISamples-iOS.app` was built; it has not been installed by this work. Existing device models and data were not changed.

## Self-review

Independent source review covered region preprocessing and coordinate contracts; actor/queue isolation and capture lifetime; state/cancellation; UI, resources, localization and target wiring. An independent follow-up checked the fixes against the original counterexamples. Model execution and user touch acceptance were not inferred from source review.

Corrections made during review:

- Flat or unknown device poses now preserve the last valid orientation.
- Orientation is interpreted once in the Feature and passed with session/revision information. A service that only polls after inference can otherwise miss an intermediate rotation and reject every later frame while the device remains flat.
- Rotation clears displayed candidates and invalidates old analysis, including rotation away and back to the same angle. Capture is stopped and restarted so the old pipeline cannot label buffered images with the new angle.
- Rotation and loss of candidates cancel queued haptics.
- Old capture callbacks check a generation. All terminal capture failures set failure state before finishing the stream; the consumer checks that state before and after inference because finishing an async stream does not discard its buffered element.

No unresolved confirmed source defect remained in the reviewed ranges. This does not complete F4/C6. Unmeasured items include localization quality for real objects, analysis frequency, sustained resource use, physical-camera interruption and stopping during an actual GPU call. VoiceOver and Dynamic Type require device acceptance.

## Reproduction data

Static checks are prepared using two public eyeglass photos and the existing five recipe images as negative cases for the eyeglasses query. Source images, raw private paths and build logs stay in the ignored local workspace. Approximate manual object rectangles were fixed before inference. Calibration and held-out photos are separate; held-out failures must not be removed or used to lower the threshold afterward.

- Calibration positive: [Eyeglass on the table, Nihar Gogoi](https://commons.wikimedia.org/wiki/File:Eyeglass_on_the_table_(India,_2016).jpg).
- Held-out positive: [Purple tortoiseshell glasses](https://commons.wikimedia.org/wiki/File:Stylish_purple_tortoiseshell_glasses_resting_on_a_wooden_table_surrounded_by_blurred_objects_in_a_cozy_setting.jpg).

The shared-model rule requires the already-running Mac demo to finish before the validation job starts. The user was asked whether that app may be closed. No model-validation success or merge is claimed while this gate is pending.
