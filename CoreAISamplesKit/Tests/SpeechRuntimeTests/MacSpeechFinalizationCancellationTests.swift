#if os(macOS)
    import Foundation
    import SpeechCore
    import Synchronization
    import Testing

    @testable import SpeechRuntime

    struct MacSpeechFinalizationCancellationTests {
        private final class Device: MacSpeechRecordingDevice {
            let onStart: @Sendable () -> Void
            let onStop: @Sendable () -> Void
            init(start: @escaping @Sendable () -> Void, stop: @escaping @Sendable () -> Void) {
                onStart = start
                onStop = stop
            }
            func prepare() -> Bool { true }
            func start(duration: TimeInterval) -> Bool {
                onStart()
                return true
            }
            func stop() { onStop() }
        }
        @Test(
            "VRC-03 A cancellation during terminal device stop or file removal wins and transfers no successful file",
            arguments: ["success-stop", "failure-stop", "failure-remove"])
        func cancellationDuringFinalization(_ path: String) async throws {
            let url = URL.temporaryDirectory.appending(path: "irodori-reference-" + UUID().uuidString + ".wav")
            defer { try? FileManager.default.removeItem(at: url) }
            let stops = Mutex(0)
            let removals = Mutex(0)
            let cleanup = MacSpeechRecordingCleanup(remove: { file in
                removals.withLock { $0 += 1 }
                if path == "failure-remove" { withUnsafeCurrentTask { $0?.cancel() } }
                try FileManager.default.removeItem(at: file)
            })
            let backend = MacSpeechRecordingBackend(
                requestPermission: { $0(true) },
                reserveFile: {
                    try Data("test".utf8).write(to: url, options: .withoutOverwriting)
                    return url
                },
                makeDevice: { _, _, finished in
                    Device(
                        start: { finished(path == "success-stop") },
                        stop: {
                            stops.withLock { $0 += 1 }
                            if path.hasSuffix("-stop") { withUnsafeCurrentTask { $0?.cancel() } }
                        })
                })
            let recorder = MacSpeechReferenceRecorder(backend: backend, cleanup: cleanup)
            let cancelled = await withTaskGroup(of: Bool.self) { group in
                group.addTask {
                    do {
                        _ = try await recorder.record(ticket: 1) {}
                        return false
                    } catch is CancellationError { return Task.isCancelled } catch { return false }
                }
                return await group.next() ?? false
            }
            #expect(cancelled)
            #expect(stops.withLock { $0 } == 1 && removals.withLock { $0 } == 1)
            #expect(!FileManager.default.fileExists(atPath: url.path))
        }
    }
#endif
