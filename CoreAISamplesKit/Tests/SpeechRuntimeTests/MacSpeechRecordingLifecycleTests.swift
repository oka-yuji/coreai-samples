#if os(macOS)
    import AVFAudio
    import Foundation
    import SpeechCore
    import Testing

    @testable import SpeechRuntime

    @Suite(.timeLimit(.minutes(1)))
    struct MacSpeechRecordingLifecycleTests {
        static func permissionOnly(
            _ request: @escaping @Sendable (@escaping @Sendable (Bool) -> Void) -> Void
        ) -> MacSpeechRecordingBackend {
            MacSpeechRecordingBackend(
                requestPermission: request,
                reserveFile: {
                    Issue.record("A recording file was reserved without a current permission grant.")
                    throw SpeechReferenceError.recordingFailed
                },
                makeDevice: { _, _, _ in
                    Issue.record("A device was created without a current permission grant.")
                    throw SpeechReferenceError.recordingFailed
                })
        }

        static func isCaptureFormat(_ settings: [String: Any]) -> Bool {
            (settings[AVFormatIDKey] as? NSNumber)?.uint32Value == kAudioFormatLinearPCM
                && (settings[AVSampleRateKey] as? NSNumber)?.doubleValue == 16_000
                && (settings[AVNumberOfChannelsKey] as? NSNumber)?.intValue == 1
                && (settings[AVLinearPCMBitDepthKey] as? NSNumber)?.intValue == 32
                && (settings[AVLinearPCMIsFloatKey] as? NSNumber)?.boolValue == true
                && (settings[AVLinearPCMIsBigEndianKey] as? NSNumber)?.boolValue == false
        }

        @Test("VRC-01 Only an explicit uncancelled request asks for permission and an overlapping request is busy")
        func permissionOnlyForExplicitRequest() async throws {
            let (replies, sink) = AsyncStream<@Sendable (Bool) -> Void>.makeStream()
            let recorder = MacSpeechReferenceRecorder(backend: Self.permissionOnly { reply in sink.yield(reply) })
            let cancelled = await Task {
                withUnsafeCurrentTask { $0?.cancel() }
                return try await recorder.record(ticket: 0) {}
            }.result
            #expect(throws: CancellationError.self) { try cancelled.get() }
            let recording = Task { try await recorder.record(ticket: 1) {} }
            var iterator = replies.makeAsyncIterator()
            let reply = try #require(await iterator.next())
            await #expect(throws: SpeechError.busy) { try await recorder.record(ticket: 2) {} }
            await recorder.finish(ticket: 1)
            DispatchQueue.global().async { reply(false) }
            let result = await recording.result
            #expect(throws: SpeechReferenceError.permissionDenied) { try result.get() }
            sink.finish()
            let remaining = await iterator.next()
            #expect(remaining == nil)
        }

        @Test("VRC-01 Late or repeated permission answers for an ended request cannot reach the next request")
        func lateAnswersStayWithTheirRequest() async throws {
            let (replies, sink) = AsyncStream<@Sendable (Bool) -> Void>.makeStream()
            let recorder = MacSpeechReferenceRecorder(backend: Self.permissionOnly { reply in sink.yield(reply) })
            var iterator = replies.makeAsyncIterator()
            let first = Task { try await recorder.record(ticket: 1) {} }
            let expired = try #require(await iterator.next())
            first.cancel()
            let cancelled = await first.result
            #expect(throws: CancellationError.self) { try cancelled.get() }
            let second = Task { try await recorder.record(ticket: 2) {} }
            let current = try #require(await iterator.next())
            expired(true)
            expired(true)
            current(false)
            current(true)
            let result = await second.result
            #expect(throws: SpeechReferenceError.permissionDenied) { try result.get() }
            sink.finish()
        }

        @Test("VRC-03 Stopping a pending request returns after the request is released and later stops are harmless")
        func stopReleasesPendingRequest() async throws {
            let (replies, sink) = AsyncStream<@Sendable (Bool) -> Void>.makeStream()
            let (stops, stopSink) = AsyncStream<Void>.makeStream()
            let recorder = MacSpeechReferenceRecorder(
                backend: Self.permissionOnly { reply in sink.yield(reply) }, didRequestStop: { stopSink.yield() })
            var iterator = replies.makeAsyncIterator()
            let first = Task { try await recorder.record(ticket: 1) {} }
            let expired = try #require(await iterator.next())
            await recorder.stop()
            let stopped = await first.result
            #expect(throws: CancellationError.self) { try stopped.get() }
            await recorder.finish(ticket: 1)
            await recorder.stop()
            stopSink.finish()
            var requests = 0
            for await _ in stops { requests += 1 }
            #expect(requests == 1)
            let next = Task { try await recorder.record(ticket: 2) {} }
            let current = try #require(await iterator.next())
            expired(true)
            current(false)
            let result = await next.result
            #expect(throws: SpeechReferenceError.permissionDenied) { try result.get() }
            sink.finish()
        }

        @Test(
            "VRC-03 A failed reservation or device creation reports recordingFailed and leaves no reserved file",
            arguments: [false, true])
        func failedStartRemovesReservedFile(reserves: Bool) async throws {
            let (replies, sink) = AsyncStream<@Sendable (Bool) -> Void>.makeStream()
            let (files, fileSink) = AsyncStream<URL>.makeStream()
            let (formats, formatSink) = AsyncStream<Bool>.makeStream()
            let backend = MacSpeechRecordingBackend(
                requestPermission: { reply in sink.yield(reply) },
                reserveFile: {
                    guard reserves else { throw SpeechReferenceError.unreadable }
                    let url = URL.temporaryDirectory.appending(path: "irodori-reference-\(UUID().uuidString).wav")
                    try Data().write(to: url, options: .withoutOverwriting)
                    fileSink.yield(url)
                    return url
                },
                makeDevice: { url, settings, _ in
                    formatSink.yield(Self.isCaptureFormat(settings))
                    try Data("partial".utf8).write(to: url)
                    throw SpeechReferenceError.unreadable
                })
            let recorder = MacSpeechReferenceRecorder(backend: backend)
            let recording = Task {
                try await recorder.record(ticket: 1) { Issue.record("A failed start was reported as started.") }
            }
            var iterator = replies.makeAsyncIterator()
            let reply = try #require(await iterator.next())
            reply(true)
            let result = await recording.result
            #expect(throws: SpeechReferenceError.recordingFailed) { try result.get() }
            fileSink.finish()
            formatSink.finish()
            sink.finish()
            var reserved: [URL] = []
            for await url in files { reserved.append(url) }
            var checked: [Bool] = []
            for await format in formats { checked.append(format) }
            #expect(reserved.count == (reserves ? 1 : 0))
            #expect(checked == (reserves ? [true] : []))
            for url in reserved {
                #expect(!FileManager.default.fileExists(atPath: url.path))
                try? FileManager.default.removeItem(at: url)
            }
        }

        @Test("VRC-04 The Mac VoxCPM2 engine builds the Mac recorder while Mac Irodori keeps the inert recorder")
        func engineRecorders() async {
            #expect(SpeechEngine.voxcpm2.makeRecorder() is MacSpeechReferenceRecorder)
            let irodori = SpeechEngine.irodori.makeRecorder()
            #expect(irodori is SpeechReferenceRecorder)
            await #expect(throws: SpeechReferenceError.recordingFailed) { try await irodori.record(ticket: 1) {} }
        }
    }
#endif
