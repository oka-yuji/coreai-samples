import Foundation
import ObjectFinderCore
import ObjectFinderFeature

@MainActor
public protocol ObjectFinderService {
    func search(
        folder: URL, query: String, minimumSimilarity: Double, mode: ObjectSearchMode, rotation: ObjectCameraRotation,
        started: () -> Void, receive: (ObjectFinderFrame) -> Void
    ) async throws
    func stop(session: Int, retainingModel: Bool) async throws
    func notify()
    func setRotation(_ rotation: ObjectCameraRotation)
}

public struct ObjectFinderReleaseFailure: LocalizedError, Equatable, Sendable {
    public let failure: ObjectFinderFailure

    public var errorDescription: String? {
        let detail = if case .operation(let message) = failure { message } else { String(describing: failure) }
        return "The camera search could not release its camera and model: \(detail)"
    }
}

public struct ObjectFinderRunner {
    private let service: any ObjectFinderService
    private let permits: @MainActor () -> Bool

    public init(service: any ObjectFinderService, permits: @escaping @MainActor () -> Bool = { true }) {
        self.service = service
        self.permits = permits
    }

    @MainActor
    public func run(_ effect: ObjectFinderEffect, send: (ObjectFinderAction) -> Void) async {
        guard !Task.isCancelled else { return }
        switch effect {
        case .search(let session, let folder, let query, let minimumSimilarity, let mode, let rotation):
            guard permits() else { return send(.failed(session, .notPermitted)) }
            do {
                try await service.search(
                    folder: folder, query: query, minimumSimilarity: minimumSimilarity, mode: mode, rotation: rotation,
                    started: { send(.started(session)) },
                    receive: { send(.analyzed(session, $0)) })
                if !Task.isCancelled { send(.failed(session, .cameraInterrupted)) }
            } catch is CancellationError {
                return
            } catch {
                guard !Task.isCancelled else { return }
                send(.failed(session, error as? ObjectFinderFailure ?? .operation(String(describing: error))))
            }
        case .stop(let session, let retainingModel):
            do {
                try await service.stop(session: session, retainingModel: retainingModel)
                if !Task.isCancelled { send(.stopped(session)) }
            } catch {
                guard !Task.isCancelled else { return }
                send(.stopFailed(session, error as? ObjectFinderFailure ?? .operation(String(describing: error))))
            }
        case .notify:
            guard !Task.isCancelled, permits() else { return }
            service.notify()
        case .orient(let rotation):
            guard !Task.isCancelled else { return }
            service.setRotation(rotation)
        }
    }

    @MainActor
    public func release(_ session: Int, send: (ObjectFinderAction) -> Void) async throws {
        send(.stop)
        do {
            try await service.stop(session: session, retainingModel: false)
            send(.stopped(session))
        } catch {
            let failure = error as? ObjectFinderFailure ?? .operation(String(describing: error))
            send(.stopFailed(session, failure))
            throw ObjectFinderReleaseFailure(failure: failure)
        }
    }
}
