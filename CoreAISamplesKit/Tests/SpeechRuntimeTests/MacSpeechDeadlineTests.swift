#if os(macOS)
    import Dispatch
    import Foundation
    import SpeechCore
    import Synchronization
    import Testing

    @testable import SpeechRuntime

    struct MacSpeechDeadlineTests {
        private final class Silent: MacSpeechRecordingDevice {
            let stopAction: @Sendable () -> Void
            init(_ stop: @escaping @Sendable () -> Void) { stopAction = stop }
            func prepare() -> Bool { true }
            func start(duration: TimeInterval) -> Bool { duration == 10 }
            func stop() { stopAction() }
        }
        private struct TimeState {
            var now = ContinuousClock.now
            var dates: [ContinuousClock.Instant] = []
            var fire: (@Sendable (ContinuousClock.Instant) -> Void)?
            var cancellations = 0
            var stops = 0
        }
        @Test(
            "VRC-03 A successfully started silent device fails at the fifteen-second elapsed deadline and releases its owned file",
            .timeLimit(.minutes(1)))
        func noTerminalEvent() async throws {
            let times = Mutex(TimeState())
            let deadline = MacSpeechRecordingDeadline(
                now: { times.withLock { $0.now } },
                schedule: { remaining, fire in
                    times.withLock {
                        $0.dates.append($0.now.advanced(by: remaining))
                        $0.fire = fire
                    }
                    return { times.withLock { $0.cancellations += 1 } }
                })
            let permissions = AsyncStream<@Sendable (Bool) -> Void>.makeStream()
            let began = AsyncStream<Void>.makeStream()
            let url = URL.temporaryDirectory.appending(path: "irodori-reference-" + UUID().uuidString + ".wav")
            defer {
                permissions.continuation.finish()
                began.continuation.finish()
                try? FileManager.default.removeItem(at: url)
            }
            let backend = MacSpeechRecordingBackend(
                requestPermission: { permissions.continuation.yield($0) },
                reserveFile: {
                    try Data("synthetic".utf8).write(to: url, options: .withoutOverwriting)
                    return url
                }, makeDevice: { _, _, _ in Silent { times.withLock { $0.stops += 1 } } })
            let recorder = MacSpeechReferenceRecorder(
                backend: backend,
                cleanup: MacSpeechRecordingCleanup(remove: { try FileManager.default.removeItem(at: $0) }),
                deadline: deadline)
            async let captured: URL = recorder.record(ticket: 91) { began.continuation.yield(()) }
            var p = permissions.stream.makeAsyncIterator()
            let approve = try #require(await p.next())
            #expect(times.withLock { $0.dates.isEmpty })
            let start = times.withLock { s in
                s.now = s.now + .seconds(60)
                return s.now
            }
            approve(true)
            var s = began.stream.makeAsyncIterator()
            try #require(await s.next() != nil)
            let scheduled = times.withLock { ($0.dates, $0.fire) }
            #expect(scheduled.0 == [start + .seconds(15)])
            if let callback = scheduled.1 {
                times.withLock { $0.now = start + .seconds(15) }
                callback(start + .seconds(15))
                callback(start + .seconds(15))
            } else {
                await recorder.finish(ticket: 91)
            }
            do {
                _ = try await captured
                Issue.record("The silent device returned a successful file instead of timing out.")
            } catch { #expect(error as? SpeechReferenceError == .recordingFailed) }
            #expect(times.withLock { $0.cancellations == 1 && $0.stops == 1 })
            #expect(!FileManager.default.fileExists(atPath: url.path))
        }
    }
#endif
