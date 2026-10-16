# Object finder expiry scheduling

2026-10-17. Issue #157, PR #158, base main `0d868bc`. This is a feature-level reduction of unnecessary maintenance work, not a claim about whole-device energy use.

The previous UIView installed a CADisplayLink requesting 30 fps until dismantling, even with no displayed candidates. The runtime now owns at most one nonrepeating Timer. Same-deadline position changes preserve its identity. Clearing, invalid generation, expiry and current-parent detach invalidate it; a weak proxy prevents a retain cycle and callback identity rejects replaced timers. Unattached layers never schedule a timer. Preview video still comes directly from AVCaptureVideoPreviewLayer at the existing capture setting.

| Condition | Evidence |
|---|---|
| E1 | `ObjectPreviewExpiryTests.deadlines`: only visible finite unexpired candidates in the expected generation expose a deadline. |
| E2 | `ObjectPreviewExpiryTests.invalidation` and existing rotation/generation tests: clear and older callbacks cannot revive the old deadline or boxes. |
| E3 | Independent actual-source lifecycle probe: 34 assertions, 8 Timer instances, zero failures. Foundation Timer/RunLoop and CALayer were real; UIKit/AVCapture were shimmed and uptime was controlled. Empty, low-score, detached and wrong-generation states have no timer; same deadline keeps identity; changed deadline invalidates; old callbacks do not clear new state; early fire reschedules; expiry clears; correct-parent detach invalidates; preview deallocates through a weak proxy. No camera/model was opened. |
| E4 | 476 CPU tests passed, 32 conditionally skipped (508 discovered); ObjectFinder 32 passed. Strict format: 4 changed Swift files. SwiftLint: 0 violations in 314 files. Mac Release and signed iOS Release succeeded; strict code-signature verification passed. |
| E5 | iPhone natural firing, drawing and reopening remain a user device check. Timer is not a real-time deadline guarantee. |

Independent source review and lifecycle verification found no unresolved confirmed or plausible defect. Creation, registration, invalidation and selector execution stay on MainActor, matching the SDK27 NSTimer thread and ownership contract. The existing Sendable exception's safety table was updated without adding an exception.

Environment: M4 Max, macOS 27.0, Swift 6.4 / Swift 6, Xcode and SDK 27. Existing required build/test commands are unchanged; logs are in ignored `.local/object-finder-next/expiry-*`. The reproducible independent probe is retained in the private guide evidence. No measured battery percentage or iPhone FPS improvement is inferred from removing polling.

`docs/object-finder.md` describes the new ownership. The provisional 3 SP and single-unit plan remain unchanged. Other camera/model comparison candidates are evaluated separately, so no unsupported model or color-format change is bundled into this lifecycle change.
