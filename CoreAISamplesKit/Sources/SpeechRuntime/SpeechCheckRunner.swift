#if os(macOS) || os(iOS)
    import Foundation
    import SpeechFeature

    @MainActor
    public protocol SpeechDeviceChecking {
        func run(_ request: SpeechCheckRequest, progress: @MainActor (String) -> Void) async throws -> String
        func stop() async
    }
    @MainActor
    public final class SpeechCheckRunner {
        private let service: any SpeechDeviceChecking
        public init(service: any SpeechDeviceChecking) { self.service = service }
        public func run(_ effect: SpeechCheckEffect, send: (SpeechCheckAction) -> Void) async {
            switch effect {
            case .run(let session, let request):
                do {
                    let path = try await service.run(request) { if !Task.isCancelled { send(.progress(session, $0)) } }
                    try Task.checkCancellation()
                    send(.completed(session, path))
                } catch {
                    if !Task.isCancelled {
                        let message =
                            error is CancellationError
                            ? "The device check was interrupted." : error.localizedDescription
                        send(.failed(session, message))
                    }
                }
            case .stop(let session):
                await service.stop()
                if !Task.isCancelled { send(.stopped(session)) }
            }
        }
    }
#endif
