#if os(macOS)
    import Foundation
    import SpeechCore
    import Testing

    @testable import SpeechRuntime

    struct MacSpeechRecordingTests {
        @Test("VRC-01 Stop completes while microphone permission is pending and a late reply cannot create a device")
        func stopPendingPermission() async throws {
            let (events, sink) = AsyncStream<@Sendable (Bool) -> Void>.makeStream()
            let backend = MacSpeechRecordingBackend(
                requestPermission: { reply in sink.yield(reply) },
                reserveFile: {
                    Issue.record("An expired permission callback attempted to reserve a recording file.")
                    throw SpeechReferenceError.recordingFailed
                },
                makeDevice: { _, _, _ in
                    Issue.record("An expired permission callback attempted to create a device.")
                    throw SpeechReferenceError.recordingFailed
                })
            let recorder = MacSpeechReferenceRecorder(backend: backend)
            let recording = Task { try await recorder.record(ticket: 1) {} }
            var iterator = events.makeAsyncIterator()
            let reply = try #require(await iterator.next())
            await recorder.stop()
            let result = await recording.result
            #expect(throws: CancellationError.self) { try result.get() }
            reply(true)
            sink.finish()
        }
    }
#endif
