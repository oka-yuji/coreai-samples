import FeatureStore
import Foundation
import ImageGenerationFeature
import ModelLibraryFeature

@MainActor
public final class ManagedImageGeneration {
    public typealias Permission = @MainActor @Sendable () -> Bool
    public typealias Factory = @MainActor (ModelDemoSession, @escaping Permission) -> (any ImageGenerationService)?

    private struct Bound {
        let token: UUID
        let session: ModelDemoSession
        let store: Store<ImageGenerationFeature>
        let runner: ImageGenerationRunner
    }

    private let makeService: Factory
    private var bound: Bound?
    private var retired: [LibraryTicket: Bound] = [:]
    private var retained: ImageGenerationState?

    public init(makeService: @escaping Factory) { self.makeService = makeService }

    public var state: ImageGenerationState? { bound?.store.state ?? retained }

    public func store(for session: ModelDemoSession, isPermitted: @escaping Permission) -> Store<
        ImageGenerationFeature
    >? {
        guard isPermitted() else { return nil }
        if let bound, bound.session == session { return bound.store }
        let token = UUID()
        let permission: Permission = { [weak self] in
            guard let self, self.bound?.token == token else { return false }
            return isPermitted()
        }
        guard let service = makeService(session, permission) else { return nil }
        let runner = ImageGenerationRunner(service: service, isPermitted: permission)
        let store = Store<ImageGenerationFeature>(
            initialState: ImageGenerationState(owner: token, availability: .ready, retaining: state), run: runner.run)
        if let bound {
            bound.store.send(.ended)
            retired[bound.session.id] = bound
        }
        bound = Bound(token: token, session: session, store: store, runner: runner)
        retained = nil
        return store
    }

    public func release(_ session: ModelDemoSession) async throws {
        let target = bound.flatMap { $0.session == session ? $0 : nil } ?? retired[session.id]
        guard let target, target.session == session else { return }
        target.store.send(.ended)
        try await target.runner.release()
        if bound?.token == target.token {
            retained = target.store.state
            bound = nil
        }
        retired[session.id] = nil
    }
}
