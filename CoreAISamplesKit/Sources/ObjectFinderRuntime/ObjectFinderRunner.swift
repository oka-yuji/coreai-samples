import Foundation
import ObjectFinderCore
import ObjectFinderFeature

@MainActor
public protocol ObjectFinderService {
    func search(
        folder: URL, query: String, minimumSimilarity: Double, rotation: ObjectCameraRotation, started: () -> Void,
        receive: (ObjectFinderFrame) -> Void
    ) async throws
    func stop() async
    func notify()
    func setRotation(_ rotation: ObjectCameraRotation)
}

public struct ObjectFinderRunner {
    private let service: any ObjectFinderService

    public init(service: any ObjectFinderService) { self.service = service }

    @MainActor
    public func run(_ effect: ObjectFinderEffect, send: (ObjectFinderAction) -> Void) async {
        switch effect {
        case .search(let session, let folder, let query, let minimumSimilarity, let rotation):
            do {
                try await service.search(
                    folder: folder, query: query, minimumSimilarity: minimumSimilarity, rotation: rotation,
                    started: { send(.started(session)) },
                    receive: { send(.analyzed(session, $0)) })
                if !Task.isCancelled { send(.failed(session, .cameraInterrupted)) }
            } catch is CancellationError {
                return
            } catch {
                guard !Task.isCancelled else { return }
                send(.failed(session, error as? ObjectFinderFailure ?? .operation(String(describing: error))))
            }
        case .stop(let session):
            await service.stop()
            send(.stopped(session))
        case .notify:
            guard !Task.isCancelled else { return }
            service.notify()
        case .orient(let rotation):
            guard !Task.isCancelled else { return }
            service.setRotation(rotation)
        }
    }
}
