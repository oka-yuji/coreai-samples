# Video test suspension investigation

Issue #429. Source behavior at `fcedb7f`, macOS 27.2 (26B123), Swift 6.4, Xcode 27.0 (27A266a). No trained model is loaded by these experiments.

## Result and limits

The previously reported multi-hour suspension was not reproduced. Five simultaneous processes, repeated for ten batches, completed all 50 VideoSearchTests runs without a timeout or nonzero exit. Each reported 43 tests in five suites; the model-dependent integration case remained skipped. Process durations ranged from 0.778 to 1.551 seconds, with median 1.5365 seconds. The earlier single `swift test --skip-build --filter VideoSearchTests` run also passed.

These runs use SwiftPM's testing helper directly, avoiding a shared package-build lock. They exercise the existing video test binary and a temporary instrumented copy that records each writer append/finish and native export/observation boundary. The instrumented source is not part of the submitted change. A sampler was configured at 75 seconds, followed by termination of only that owned process after 95 seconds; no run reached either threshold, so no stalled stack was captured. Raw logs and temporary probes remain in the ignored local validation directory.

A separate explicit GUI conversation check was running during part of the experiment; this is not a claim of an otherwise idle machine or an exact replay of the earlier five-worktree build load. Timing is included only to bound these attempts. The original 7-hour-38-minute observation remains valid and unexplained.

## Wait boundaries

| Boundary | Source location | Evidence |
|---|---|---|
| Test fixture waits for an input to accept each frame | `CoreAISamplesKit/Tests/VideoSearchTests/VideoFrameSourceTests.swift:146` | SDK27 PixelBufferReceiver.append is async and documented to suspend for input readiness. The fixture has no explicit cancellation checks. |
| Test fixture waits for the writer to finish | Same file, `:151` | An async finishWriting call follows all appends. The old fixture has no failure cleanup for its folder. |
| Product waits for native export and its progress observer | `CoreAISamplesKit/Sources/VideoSearchRuntime/VideoClipExporter.swift:15-18` | The task-group scope joins its observer. SDK27 export(to:as:) has a cancellation handler; cancelling the parent requests native cancellation. |

An isolated SDK probe created the states sequence before export, completed the export, and only then iterated the states. It finished successfully in 0.145 seconds. The hypothesis that every late observer waits forever is therefore rejected for this exact case. This does not establish that every possible observer race is safe.

A second probe cancelled its own task before SyntheticVideo.colors. The old helper still returned a video, so the expectation of CancellationError failed in 0.133 seconds. This identifies a concrete cancellation defect in the test fixture, not the location or cause of the historical multi-hour suspension. Follow-up #519 fixes only this verified boundary.

## Recommended changes and remaining work

1. Apply #519: check cancellation before creating the writer, between frames and around finishing; clean up the current fixture on unsuccessful exit. Keep the non-Sendable writer in the same execution context. Do not call cancelWriting concurrently with appending from a cancellation handler.
2. Retain an external process deadline when diagnosing a future suspension. Swift Testing's time limit requests cooperative cancellation; it cannot force a non-cooperating OS wait to return. Capture the last active test, writer/export stage and a completed sample before terminating the owned process.
3. Do not adopt blanket serialization as proof of a fix. It is a comparison condition and may hide the relevant ordering. If a future failure is captured, compare the minimal writer test separately from native export and the progress observer.
4. Do not change product export behavior from the present evidence. If an observer remains active after native completion in a captured failure, test an explicit observer cancellation before changing that lifetime contract.

VHT-01 has a bounded non-reproduction record. VHT-02 remains unconfirmed for the original suspension; the separate fixture cancellation defect is located above. VHT-03 has the concrete #519 follow-up and the capture/restart conditions above. The original suspension is not marked fixed.

## Primary references

- [Swift Testing time limits](https://github.com/swiftlang/swift-testing/blob/main/Sources/Testing/Testing.docc/LimitingExecutionTime.md): timeout requests task cancellation.
- [SE-0304 cancellation and task-group lifetime](https://github.com/swiftlang/swift-evolution/blob/main/proposals/0304-structured-concurrency.md): cancellation is cooperative, and child tasks are joined.
- [PixelBufferReceiver.append](https://developer.apple.com/documentation/avfoundation/avassetwriterinput/pixelbufferreceiver/append(_:with:)): readiness suspension.
- [AVAssetExportSession.export](https://developer.apple.com/documentation/avfoundation/avassetexportsession/export(to:as:isolation:)): parent-task cancellation cancels export.
- SDK27 AVAssetWriter.h marks the writer non-Sendable and warns against concurrent cancelWriting and append operations. The SDK AVFoundation.swiftinterface exposes the export cancellation handler and the receiver's nonisolated(nonsending) append.
