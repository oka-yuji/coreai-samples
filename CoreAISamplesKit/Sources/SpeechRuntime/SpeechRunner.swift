#if os(macOS) || os(iOS)
    import Foundation
    import SpeechCore
    import SpeechFeature

    @MainActor
    public protocol SpeechService {
        func open(path: String) async throws -> SpeechModel
        func generate(text: String, seed: UInt64) async throws -> SpeechAudio
        func play(_ data: Data) async throws
        func stop() async
    }

    @MainActor
    public final class SpeechRunner {
        private let service: any SpeechService
        public init(service: any SpeechService) { self.service = service }

        public func run(_ effect: SpeechEffect, send: (SpeechAction) -> Void) async {
            let session: Int
            switch effect {
            case .open(let value, _), .generate(let value, _, _), .play(let value, _), .stop(let value): session = value
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
                    if case .play = effect { send(.playbackInterrupted(session)) }
                } else {
                    send(.failed(session, error.localizedDescription))
                }
            }
        }
    }
#endif
