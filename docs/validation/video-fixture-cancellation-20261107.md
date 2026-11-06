# Video fixture cancellation

Issue #519, the verified follow-up from #429. Production video export is unchanged.

SyntheticVideo.write now checks cancellation before creating its folder/writer, before each frame, and before/after finishing. An unsuccessful exit cancels the locally owned writer and then removes only that call's temporary folder. There is no concurrent access to the non-Sendable writer from an onCancel handler. A successful output is kept for the caller's existing cleanup.

- VWC-01/02: The same parameterized regression failed on the old helper with four expectation failures: both pre-cancelled and second-frame-cancelled calls returned success and requested the remaining frames. After the fix both cases throw CancellationError, request respectively zero and two frames, and leave their private parent directory empty. No timing wait controls either ordering.
- VWC-03: VideoSearchTests reports 44 tests in five suites with no failures; the model-dependent case remains skipped. Existing frame colors/orientation/sampling, clip timing/audio and export cancellation retain their checks.
- VWC-04: The full package suite reports 1,280 tests across 43 runs without failures. Strict SwiftLint, changed-file swift-format, Rust input build, Mac Release build, privacy and package-pin checks pass.

The original multi-hour suspension remains unconfirmed. These checks do not show that cancellation interrupts an AVFoundation call that never returns; the external deadline and capture plan in the #429 investigation remain relevant. The fix is one S unit within the existing test helper, with no product state or dependency changes and no SP estimate.
