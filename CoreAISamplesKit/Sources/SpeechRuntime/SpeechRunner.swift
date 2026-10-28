#if os(macOS) || os(iOS)
    import Foundation
    import SpeechCore
    import SpeechFeature

    @MainActor
    public protocol SpeechService {
        func open(path: String) async throws -> SpeechModel
        func generate(text: String, seed: UInt64) async throws -> SpeechAudio
        func generate(text: String, seed: UInt64, reference: SpeechReferenceAudio?) async throws -> SpeechAudio
        func play(_ data: Data) async throws
        func recordReference(ticket: Int, started: @MainActor () -> Void) async throws -> SpeechReferenceAudio
        func finishReferenceRecording(ticket: Int) async
        func importReference(_ url: URL) async throws -> SpeechReferenceAudio
        func playReference(_ reference: SpeechReferenceAudio) async throws
        func stop() async
    }

    extension SpeechService {
        public func recordReference(ticket: Int, started: @MainActor () -> Void) async throws -> SpeechReferenceAudio {
            throw SpeechReferenceError.recordingFailed
        }
        public func finishReferenceRecording(ticket: Int) async {}
        public func importReference(_ url: URL) async throws -> SpeechReferenceAudio {
            throw SpeechReferenceError.unreadable
        }
        public func playReference(_ reference: SpeechReferenceAudio) async throws {
            try await play(referenceWAV(reference))
        }
        public func generate(text: String, seed: UInt64, reference: SpeechReferenceAudio?) async throws -> SpeechAudio {
            guard reference == nil else { throw SpeechError.referenceNotSupported }
            return try await generate(text: text, seed: seed)
        }
    }

    @MainActor
    public final class SpeechRunner {
        private let service: any SpeechService
        public init(service: any SpeechService) { self.service = service }

        public func run(_ effect: SpeechEffect, send: (SpeechAction) -> Void) async {
            let session: Int
            switch effect {
            case .open(let value, _), .generate(let value, _, _), .play(let value, _), .stop(let value): session = value
            case .generateReference(let value, _, _, _), .recordReference(let value),
                .finishReferenceRecording(let value), .importReference(let value, _), .playReference(let value, _):
                session = value
            }
            do {
                switch effect {
                case .open(_, let path):
                    let model = try await service.open(path: path)
                    try Task.checkCancellation()
                    send(.opened(session, model))
                case .generate(_, let text, let seed):
                    let audio = try await service.generate(text: text, seed: seed)
                    try Task.checkCancellation()
                    send(.generated(session, audio))
                case .generateReference(_, let text, let seed, let reference):
                    let audio = try await service.generate(text: text, seed: seed, reference: reference)
                    try Task.checkCancellation()
                    send(.generated(session, audio))
                case .recordReference:
                    let reference = try await service.recordReference(ticket: session) {
                        if !Task.isCancelled { send(.referenceRecordingStarted(session)) }
                    }
                    try Task.checkCancellation()
                    send(.referencePrepared(session, reference))
                case .finishReferenceRecording:
                    await service.finishReferenceRecording(ticket: session)
                case .importReference(_, let url):
                    let reference = try await service.importReference(url)
                    try Task.checkCancellation()
                    send(.referencePrepared(session, reference))
                case .playReference(_, let reference):
                    try await service.playReference(reference)
                    try Task.checkCancellation()
                    send(.playbackFinished(session))
                case .play(_, let data):
                    try await service.play(data)
                    try Task.checkCancellation()
                    send(.playbackFinished(session))
                case .stop:
                    await service.stop()
                    try Task.checkCancellation()
                    send(.stopped(session))
                }
            } catch {
                guard !Task.isCancelled else { return }
                if error is CancellationError {
                    switch effect {
                    case .play, .playReference: send(.playbackInterrupted(session))
                    default: break
                    }
                } else if error as? SpeechReferenceError == .permissionDenied {
                    send(.microphoneDenied(session))
                } else {
                    send(.failed(session, error.localizedDescription))
                }
            }
        }
    }
#endif
