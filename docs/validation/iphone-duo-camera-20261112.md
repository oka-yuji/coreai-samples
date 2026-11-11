# iPhone Duo camera rotation

Issue [#534](https://github.com/oka-yuji/coreai-samples/issues/534) replaces device-pose inference in the camera-search path with `AVCaptureDevice.RotationCoordinator`. This record separates CPU/build evidence from the unmeasured camera behavior.

## Contract and ownership

The MainActor observer retains the selected camera and coordinator, owns KVO registrations and holds the preview layer weakly. It starts only when the preview is mounted in a window and an explicit search has selected its camera. Initial values are read before observation. KVO callbacks synchronously assert MainActor isolation, matching the SDK's main-queue delivery contract and [Apple's rotation guidance](https://developer.apple.com/documentation/avfoundation/supporting-device-rotation-in-your-camera-app).

Preview angles update only the preview connection. Capture angles reach the Feature as values tagged with search session, camera ID and observation generation. The Feature accepts a newer source only in the current active search; invalid, duplicate and stale capture notifications issue no rotation effect. A capture-angle change advances its revision, clears old results and uses the existing stop/restart analysis loop. Preview-only changes do not enter that loop. Before the first observation, capture uses the Feature's retained angle (90 degrees for a new state). A differing first capture observation uses the same restart path; hardware startup latency is unmeasured. Reconnecting the same camera reapplies preview rotation without replacing the observer. Replacing its camera, session or mounted layer ends old observations before creating new ones.

Resolving the selected camera on MainActor happens immediately after capture starts. A missing device reports camera unavailability through the existing cleanup path; it does not leave capture running with a fallback angle. Unmounting ends observation while retaining the selected camera until the search ends.

The capture actor still owns session configuration, start, stop and the video output. Its existing back-camera order, pixel-format selection, 30 fps settings and tracking logic are unchanged. The actor returns only camera ID, capture generation and its existing frame stream to MainActor. Stop invalidates observation before awaiting camera teardown; normal completion/cancellation/failure does the same, once per search. No new `@unchecked Sendable`, `nonisolated(unsafe)` or unstructured Task was introduced.

Finite values wrap into `[0, 360)`, round to the nearest integer degree with ties away from zero, then wrap again. Only 0, 90, 180 and 270 are accepted. For example, -90 becomes 270, 450 becomes 90, 89.5 becomes 90, 359.5 becomes 0, while 45 and 90.5 are rejected. This is a fixed policy for the SDK connection's supported quarter turns, not arbitrary-angle image reprojection. The actual connection also checks support before assignment.

Cached boxes are filtered against the current preview angle at drawing time, not just when a capture callback arrives. A capture-angle change immediately rejects the current capture generation, so a delayed layout or queued old frame cannot restore boxes before capture reconnects, including a turn away and back to the same angle. A new capture generation clears old candidates and rejects earlier callbacks even if a stale layout still names the previous generation. Changing observation source at the same capture angle clears the previous visible result without advancing the angle revision. Only the current parent layer can lay out or detach the preview, and the UI callback captures its Store weakly.

## CPU coverage

| Condition | Evidence |
|---|---|
| CAM-01: old capture results are rejected after rotation | `ObjectCameraRotationTests.captureChange`, existing `rotationDuringInference` and `livePreviewFollowsFeature`. |
| CAM-02: preview-only changes do not restart capture or analysis | `previewChange` checks no Feature commands, unchanged capture state and mismatched cached-box suppression. Existing service tests cover capture restart after a versioned orientation command. |
| CAM-03: invalid/duplicate normalized angles are ignored | `ignoredAngles`, `normalizedAngles`, `invalidAngles`. |
| CAM-04: old camera/observation/search notifications are rejected | `staleSources`, `staleSession`, `sameAngleReconnect`, `angleChangeBeforeLayout`; existing managed stop/retry tests remain in the suite. |
| Rotation-device failure | `rotationDeviceFailure` injects camera unavailability after capture startup and verifies camera/observer teardown before reporting failure, without starting model preparation. |
| Stop ordering | `LiveObjectFinderServiceTests.stopDuringPreparation` observes rotation teardown before model cleanup completes and verifies one teardown per search. |

Eight temporary mutations were detected by assertion failures: removing cached-frame angle filtering, accepting an old observation, accepting a repeated angle, accepting an older source binding, losing the capture-generation floor delaying observation teardown until cleanup, retaining the old generation after an angle change and omitting the initial angle seed. All mutated sources were restored before the final checks.

The targeted command is `swift test --package-path CoreAISamplesKit --force-resolved-versions --filter ObjectFinderTests`. It uses fake capture/model devices; no camera or learned model is opened. Final repository gates are recorded with the PR.

## Build environment

The default and explicitly selected developer directory both identify Xcode 27.2 (`27B52`), Swift 6.4.2 and iPhoneOS 27.2. The update replaced the previous default installation; two different Xcode releases were not tested. The deployment target remains iOS 27.0. The required macOS and unsigned iOS Release builds, strict lint, formatting, localization, privacy and dependency-pin checks remain the PR gates.

## Physical acceptance and angle readings

No physical camera measurement or real-model execution is claimed. The available Duo used for Models verification is a simulator. CAM-06 through CAM-09 remain unchecked for the owner on main.

| Device/display | Orientation | Preview angle | Capture angle | Preview/box alignment |
|---|---|---|---|---|
| Duo outer | Portrait | Not measured | Not measured | Pending |
| Duo outer | Landscape | Not measured | Not measured | Pending |
| Duo inner | Landscape | Not measured | Not measured | Pending |
| Duo inner | Portrait | Not measured | Not measured | Pending |
| Ordinary iPhone | Portrait | Not measured | Not measured | Pending |

The current equal-angle rule intentionally hides boxes when capture and preview differ. If a persistent difference is observed on hardware, retain both readings and plan a separate coordinate-transform change. Ordinary iPhone landscape, front-camera behavior, preview placement and model quality are outside this change.
