import AVFAudio
import Foundation
import OSLog
import SpeechCore

protocol SpeechReferenceRecording: Actor {
    func record(ticket: Int, started: @Sendable () async -> Void) async throws -> URL
    func finish(ticket: Int) async
    func stop() async
}

#if os(iOS)
    private enum RecordingEvent: Sendable {
        case finished(Bool)
        case failed, finishRequested, cancelled, interrupted
    }

    private final class RecordingDelegate: NSObject, AVAudioRecorderDelegate {
        let continuation: AsyncStream<RecordingEvent>.Continuation
        init(_ continuation: AsyncStream<RecordingEvent>.Continuation) { self.continuation = continuation }
        func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
            continuation.yield(.finished(flag))
        }
        func audioRecorderEncodeErrorDidOccur(_ recorder: AVAudioRecorder, error: (any Error)?) {
            continuation.yield(.failed)
        }
    }

    actor SpeechReferenceRecorder: SpeechReferenceRecording {
        private let permission: @Sendable () async -> Bool
        private let session = AVAudioSession.sharedInstance()
        private var device: AVAudioRecorder?
        private var delegate: RecordingDelegate?
        private var continuation: AsyncStream<RecordingEvent>.Continuation?
        private var observers: [NSObjectProtocol] = []
        private var busy = false
        private var activeTicket: Int?
        private var cancelled = false
        private var activated = false
        private var waiters: [CheckedContinuation<Void, Never>] = []
        private let logger = Logger(subsystem: "com.coreai-samples", category: "ReferenceRecording")

        init(permission: @escaping @Sendable () async -> Bool = { await AVAudioApplication.requestRecordPermission() })
        {
            self.permission = permission
        }

        func record(ticket: Int, started: @Sendable () async -> Void) async throws -> URL {
            guard !busy else { throw SpeechError.busy }
            try Task.checkCancellation()
            busy = true
            activeTicket = ticket
            cancelled = false
            let (events, sink) = AsyncStream.makeStream(of: RecordingEvent.self)
            continuation = sink
            var recordingURL: URL?
            do {
                let result = try await withTaskCancellationHandler {
                    guard await permission() else { throw SpeechReferenceError.permissionDenied }
                    try checkCancellation()
                    let url = URL.temporaryDirectory.appending(path: "irodori-reference-\(UUID().uuidString).wav")
                    recordingURL = url
                    try session.setCategory(.record, mode: .default)
                    activated = await withCheckedContinuation { pending in
                        session.activate(options: []) { success, _ in pending.resume(returning: success) }
                    }
                    guard activated else { throw SpeechReferenceError.recordingFailed }
                    try checkCancellation()
                    let names = [
                        AVAudioSession.didBecomeInactiveNotification, AVAudioSession.mediaServicesWereResetNotification,
                    ]
                    observers = names.map { name in
                        NotificationCenter.default.addObserver(forName: name, object: nil, queue: nil) { _ in
                            sink.yield(.interrupted)
                        }
                    }
                    let created = try AVAudioRecorder(
                        url: url,
                        settings: [
                            AVFormatIDKey: kAudioFormatLinearPCM,
                            AVSampleRateKey: 48_000,
                            AVNumberOfChannelsKey: 1,
                            AVLinearPCMBitDepthKey: 32,
                            AVLinearPCMIsFloatKey: true,
                            AVLinearPCMIsBigEndianKey: false,
                        ])
                    let receiver = RecordingDelegate(sink)
                    delegate = receiver
                    device = created
                    created.delegate = receiver
                    guard created.prepareToRecord(), created.record(forDuration: 30) else {
                        throw SpeechReferenceError.recordingFailed
                    }
                    await started()
                    for await event in events {
                        try checkCancellation()
                        switch event {
                        case .finishRequested, .finished(true): return url
                        case .cancelled: throw CancellationError()
                        case .failed, .finished(false), .interrupted: throw SpeechReferenceError.recordingFailed
                        }
                    }
                    throw CancellationError()
                } onCancel: {
                    sink.yield(.cancelled)
                    sink.finish()
                }
                await cleanup()
                finishOperation()
                return result
            } catch {
                await cleanup()
                if let recordingURL { try? FileManager.default.removeItem(at: recordingURL) }
                finishOperation()
                throw error
            }
        }

        func finish(ticket: Int) {
            guard activeTicket == ticket, busy, !Task.isCancelled else { return }
            continuation?.yield(.finishRequested)
        }

        func stop() async {
            guard busy else { return }
            cancelled = true
            continuation?.yield(.cancelled)
            continuation?.finish()
            await withCheckedContinuation { waiters.append($0) }
        }

        private func checkCancellation() throws {
            try Task.checkCancellation()
            if cancelled { throw CancellationError() }
        }

        private func cleanup() async {
            continuation?.finish()
            device?.delegate = nil
            device?.stop()
            device = nil
            delegate = nil
            for observer in observers { NotificationCenter.default.removeObserver(observer) }
            observers.removeAll()
            if activated {
                let success = await withCheckedContinuation { pending in
                    session.deactivate(options: .notifyOthersOnDeactivation) { success, _ in
                        pending.resume(returning: success)
                    }
                }
                if !success { logger.warning("Reference audio session deactivation failed") }
                activated = false
            }
        }

        private func finishOperation() {
            continuation = nil
            activeTicket = nil
            busy = false
            let pending = waiters
            waiters.removeAll()
            for waiter in pending { waiter.resume() }
        }
    }
#else
    actor SpeechReferenceRecorder: SpeechReferenceRecording {
        func record(ticket: Int, started: @Sendable () async -> Void) throws -> URL {
            throw SpeechReferenceError.recordingFailed
        }
        func finish(ticket: Int) {}
        func stop() {}
    }
#endif
