import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTransfer

public final class ModelLibraryRunner {
    private enum Hold {
        case acquiring([CheckedContinuation<Void, Never>])
        case leased(ResourceLease)
        case revoked
    }

    private let resources: ModelLibraryResourceRunner
    private let registry: ModelDemoAdapterRegistry
    private let store: ManagedModelStore
    private var holds: [LibraryTicket: Hold] = [:]
    private var stopping: [LibraryTicket: [CheckedContinuation<String?, Never>]] = [:]
    private var releasedThrough: [UUID: Int] = [:]
    private lazy var provisions = Store<DemoProvision>(initialState: DemoProvision.State()) { [weak self] session, _ in
        await self?.prepare(session)
    }

    public init(resources: ModelLibraryResourceRunner, registry: ModelDemoAdapterRegistry, store: ManagedModelStore) {
        self.resources = resources
        self.registry = registry
        self.store = store
    }

    public func run(_ effect: ModelLibraryEffect, send: (ModelLibraryAction) -> Void) async {
        guard case .stopDemo(let session, let ticket) = effect else {
            await resources.run(effect, send: send)
            return
        }
        if let failure = await stop(session) {
            send(.demoStopFailed(session.id, ticket, failure))
        } else {
            send(.demoStopped(session.id, ticket))
        }
    }

    func gate(for session: ModelDemoSession, in library: Store<ModelLibraryFeature>) -> DemoExecutionGate {
        DemoExecutionGate(session: session) { [weak self, weak library] in
            guard let self, let library else { return false }
            return library.state.permits(session.id, selection: session.selection) && !isRevoked(session)
        }
    }

    func provide(_ session: ModelDemoSession) {
        provisions.send(session)
    }

    func prepare(_ session: ModelDemoSession) async {
        guard session.id.number > releasedThrough[session.id.owner, default: 0] else { return }
        switch holds[session.id] {
        case .acquiring: return await withCheckedContinuation { wait(for: session.id, $0) }
        case .leased, .revoked: return
        case nil: holds[session.id] = .acquiring([])
        }
        let lease = await store.lease(identity: session.identity, owner: session.id.owner)
        let installed = await store.generation(of: session.identity) != nil
        if !installed { await lease.release() }
        let waiting: [CheckedContinuation<Void, Never>] =
            if case .acquiring(let waiting) = holds[session.id] { waiting } else { [] }
        holds[session.id] = installed ? .leased(lease) : .revoked
        for continuation in waiting { continuation.resume() }
    }

    private func isRevoked(_ session: ModelDemoSession) -> Bool {
        if case .revoked = holds[session.id] { true } else { false }
    }

    private func wait(for id: LibraryTicket, _ continuation: CheckedContinuation<Void, Never>) {
        guard case .acquiring(let waiting) = holds[id] else { return continuation.resume() }
        holds[id] = .acquiring(waiting + [continuation])
    }

    private func stop(_ session: ModelDemoSession) async -> String? {
        if stopping[session.id] != nil {
            return await withCheckedContinuation { stopping[session.id]?.append($0) }
        }
        stopping[session.id] = []
        let failure = await shutDown(session)
        for continuation in stopping.removeValue(forKey: session.id) ?? [] { continuation.resume(returning: failure) }
        return failure
    }

    private func shutDown(_ session: ModelDemoSession) async -> String? {
        do {
            try await registry.adapter(for: session.selection.demoID)?.stop(session)
        } catch {
            return error.localizedDescription
        }
        if case .acquiring = holds[session.id] { await withCheckedContinuation { wait(for: session.id, $0) } }
        releasedThrough[session.id.owner] = max(releasedThrough[session.id.owner, default: 0], session.id.number)
        if case .leased(let lease) = holds.removeValue(forKey: session.id) { await lease.release() }
        return nil
    }
}

nonisolated enum DemoProvision: Feature {
    struct State: Equatable, Sendable {}

    static func reduce(_ state: inout State, _ action: ModelDemoSession) -> [Command<ModelDemoSession, LibraryTicket>] {
        [.run(action, id: action.id)]
    }
}
