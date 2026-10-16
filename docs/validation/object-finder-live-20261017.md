# Object finder live-preview validation

2026-10-17 JST. Issue #153, plan v2, PR #154, based on main `99af79d`. The user reported that detection worked well on their iPhone and requested smoother camera motion and a visible free-text field. That feedback establishes basic use of the previous build; it does not measure the new preview or complete the earlier static-photo checks.

## Acceptance evidence

| Condition | Evidence and state |
|---|---|
| L1 | `previewStartsBeforeModel`: camera startup and its latest-frame stream remain available while model preparation is suspended. `stopDuringPreparation` verifies cleanup and its acknowledgement. |
| L2 | `applyTypedQuery`, `startGuards`: edits preserve the submitted target, applying waits for stop acknowledgement, and restart uses the new draft. |
| L3 | `cancelQueryRestart`, `stopAndStale`, `runnerAndStore`: stop, background and dismissal cancel pending restarts; old sessions cannot publish or restart a newer search. |
| L4 | `trackingGuards`, `previewCoordinates`, `livePreviewFollowsFeature`: invalid/low-confidence/expired boxes, rotation and older capture generations are rejected. `interruptionDuringTracking` rejects a result when the camera fails during tracking. These are CPU checks, not physical Vision tracking measurements. |
| L5 | Full package tests: **467 passed and 32 conditionally skipped**, **499 discovered**, including **24 passing ObjectFinder tests**. The skipped tests require model bundles or optional fixtures and are not counted as successful checks. Strict format: **14 changed Swift files passed**. SwiftLint: **0 violations in 310 files**. Mac Release and signed iOS Release builds succeeded. All **696 keys** have the configured Japanese translation; the English source and seven new Japanese values were checked. Generated camera-purpose text and file-sharing keys are present, and strict code-signature verification passed. |
| L6 | **User confirmation pending:** preview smoothness during inference, arbitrary Japanese typing, applying the edited target and canceling a pending change. Capture requests 30 fps, but actual sustained FPS has not been measured. |
| L7 | **User confirmation pending:** boxes follow objects and clear on tracking loss, stopping, target changes and rotation, including rotating away and back while inference is running. |

Environment: Apple Swift 6.4, Swift 6 package language mode, Xcode 27.0, macOS/iPhoneOS 27 SDKs. No model or physical-camera execution was performed for these checks. The available connected iPhone ran iOS 26.6.2, below the app's iOS 27 minimum. The intended iPhone was unavailable; this work built the app without installing it or modifying device data.

Commands (raw logs are in the ignored `.local/object-finder-live` directory):

```sh
./Tools/Irodori/build-inputs.sh
env -u COREAI_TEST_BUNDLES -u COREAI_DECISION_BUNDLE swift test --package-path CoreAISamplesKit
python3 Tools/Format/check_format.py --base origin/main
swiftlint lint --config .swiftlint.yml --baseline .swiftlint-baseline.json --strict --no-cache
xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples -configuration Release -destination 'platform=macOS' -derivedDataPath .local/object-finder/MacDerivedData build
xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples-iOS -configuration Release -destination 'generic/platform=iOS' -derivedDataPath .local/object-finder/DerivedData build
codesign --verify --deep --strict .local/object-finder/DerivedData/Build/Products/Release-iphoneos/CoreAISamples-iOS.app
```

The Mac build retains an unrelated unused-result warning in `ModelMenu.swift`; both app targets report skipped App Intents metadata extraction. Neither build has an error.

## Self-review

Independent reviews covered capture/session ownership and lifecycle; tracking, coordinate and generation contracts; and UI, state transitions and localization. The parent reviewed the final corrections and the cross-layer connections. Independent follow-up reproduced the pure-Swift generation gate and re-read changed ranges.

Two confirmed P2 defects were corrected:

- A pending target change originally disabled Stop. `canStop` now permits cancellation while waiting for shutdown, and the UI displays **Cancel change**. The reducer test verifies that the eventual stop acknowledgement cannot restart it.
- Native boxes originally bypassed the Feature's rotation invalidation. `ObjectPreviewGate` now requires the expected generation supplied by the Feature; clearing the frame hides boxes synchronously. Returning to the previous angle cannot revive an old generation. Expiry also runs from a display-link tick when camera callbacks stop.

A separate orientation-delay candidate was addressed by applying the latest UI angle on the MainActor independently of model inference. Reconnecting capture reapplies that current angle instead of a stale start argument. The raw preview layer is private; the view can attach and lay it out without access to the capture session. Follow-up found no unresolved confirmed or plausible source defect in the reviewed scope.

The SDK declarations were checked for frame-duration resets and automatic frame-rate configuration, the imported Vision tracker-capacity error API, and tracking revision behavior. The preview/capture boundary was compared with [Apple's AVCam sample](https://developer.apple.com/documentation/avfoundation/avcam-building-a-camera-app) and [device-rotation guidance](https://developer.apple.com/documentation/avfoundation/supporting-device-rotation-in-your-camera-app). Revision 1 is explicitly selected for the `.fast` tracker because revision 2 ignores that setting. One new `concurrency_escape` baseline entry is limited to `ObjectFinderPreview`; its ownership conditions are recorded in `CLAUDE.md`.

## Reconciliation and remaining work

The behavior reference in `docs/object-finder.md` now describes native preview, draft/apply/cancel behavior and tracking expiry instead of synchronized analyzed snapshots. Issue #153 retains the original L1–L7 conditions and provisional 8 SP estimate. The single-unit plan remains appropriate because capture generations, renderer invalidation and the Feature's stop/restart acknowledgement must agree at the integration point. The review added the explicit generation gate and independent expiry check within that contract.

Source implementation and automated verification are complete. Physical preview/tracker behavior, accessibility and sustained thermal load remain unmeasured. L6/L7 stay unchecked for the user's next device test; parent #150 also retains the unperformed F4/C6 static-photo checks and detailed lifecycle acceptance. No additional feature scope is introduced. Main integration and physical acceptance are recorded separately from these source/build results.
