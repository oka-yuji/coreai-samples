#if os(macOS)
    import AVFAudio
    import Foundation
    import SpeechCore

    actor MacSpeechReferenceRecorder: SpeechReferenceRecording {
        private static let maximumSeconds: TimeInterval = 10
        private static var captureSettings: [String: Any] {
            [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: 16_000,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 32,
                AVLinearPCMIsFloatKey: true,
                AVLinearPCMIsBigEndianKey: false,
            ]
        }

        private let backend: MacSpeechRecordingBackend
        private let cleanup: MacSpeechRecordingCleanup
        private let didRequestStop: @Sendable () -> Void
        private var activeTicket: Int?
        private var isRecording = false
        private var device: (any MacSpeechRecordingDevice)?
        private var ownedURL: URL?
        private var continuation: AsyncStream<MacSpeechRecordingEvent>.Continuation?
        private var isStopRequested = false
        private var waiters: [CheckedContinuation<Void, Never>] = []

        init(
            backend: MacSpeechRecordingBackend = .live, cleanup: MacSpeechRecordingCleanup = .shared,
            didRequestStop: @escaping @Sendable () -> Void = {}
        ) {
            self.backend = backend
            self.cleanup = cleanup
            self.didRequestStop = didRequestStop
        }

        func record(ticket: Int, started: @Sendable () async -> Void) async throws -> URL {
            guard activeTicket == nil else { throw SpeechError.busy }
            try Task.checkCancellation()
            let resolved = cleanup.retryPending()
            try Task.checkCancellation()
            guard resolved else { throw SpeechReferenceError.recordingFailed }
            let (events, sink) = AsyncStream.makeStream(of: MacSpeechRecordingEvent.self)
            activeTicket = ticket
            continuation = sink
            let url: URL
            do {
                url = try await withTaskCancellationHandler {
                    try await capture(events, sink: sink, started: started)
                } onCancel: {
                    sink.yield(.cancelled)
                    sink.finish()
                }
            } catch {
                if settle(keeping: nil) { throw CancellationError() }
                throw error
            }
            if settle(keeping: url) { throw CancellationError() }
            return url
        }

        func finish(ticket: Int) {
            guard activeTicket == ticket, isRecording, !isStopRequested, !Task.isCancelled else { return }
            continuation?.yield(.finishRequested)
        }

        func stop() async {
            guard activeTicket != nil else {
                _ = cleanup.retryPending()
                return
            }
            isStopRequested = true
            continuation?.yield(.cancelled)
            continuation?.finish()
            didRequestStop()
            await withCheckedContinuation { waiters.append($0) }
        }

        private func capture(
            _ events: AsyncStream<MacSpeechRecordingEvent>,
            sink: AsyncStream<MacSpeechRecordingEvent>.Continuation,
            started: @Sendable () async -> Void
        ) async throws -> URL {
            try checkCancellation()
            backend.requestPermission { granted in sink.yield(.permissionAnswered(granted)) }
            var recording: URL?
            for await event in events {
                try checkCancellation()
                switch event {
                case .permissionAnswered(let granted):
                    guard recording == nil else { continue }
                    guard granted else { throw SpeechReferenceError.permissionDenied }
                    recording = try begin(sink)
                    await started()
                case .deviceFinished(let success):
                    guard success, let recording else { throw SpeechReferenceError.recordingFailed }
                    return recording
                case .finishRequested:
                    if let recording { return recording }
                case .cancelled:
                    throw CancellationError()
                }
            }
            throw CancellationError()
        }

        private func begin(_ sink: AsyncStream<MacSpeechRecordingEvent>.Continuation) throws -> URL {
            let url: URL
            do {
                url = try backend.reserveFile()
            } catch {
                throw SpeechReferenceError.recordingFailed
            }
            ownedURL = url
            let created: any MacSpeechRecordingDevice
            do {
                created = try backend.makeDevice(url, Self.captureSettings) { success in
                    sink.yield(.deviceFinished(success))
                }
            } catch {
                throw SpeechReferenceError.recordingFailed
            }
            device = created
            guard created.prepare(), created.start(duration: Self.maximumSeconds) else {
                throw SpeechReferenceError.recordingFailed
            }
            isRecording = true
            return url
        }

        private func checkCancellation() throws {
            try Task.checkCancellation()
            if isStopRequested { throw CancellationError() }
        }

        private func settle(keeping result: URL?) -> Bool {
            continuation?.finish()
            continuation = nil
            device?.stop()
            device = nil
            if let ownedURL, ownedURL != result { cleanup.discard(ownedURL) }
            let cancelled = isStopRequested || Task.isCancelled
            if cancelled, let result, result == ownedURL { cleanup.discard(result) }
            ownedURL = nil
            activeTicket = nil
            isRecording = false
            isStopRequested = false
            let pending = waiters
            waiters.removeAll()
            for waiter in pending { waiter.resume() }
            return cancelled
        }
    }
#endif
