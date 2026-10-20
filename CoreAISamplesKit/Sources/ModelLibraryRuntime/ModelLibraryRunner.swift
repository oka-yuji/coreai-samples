import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTransfer

public final class ModelLibraryRunner {
    private enum Lease {
        case acquiring([CheckedContinuation<Void, Never>])
        case held(ResourceLease)
        case revoked
    }

    private struct LibraryReference {
        weak var store: Store<ModelLibraryFeature>?
    }

    private let resources: ModelLibraryResourceRunner
    private let registry: ModelDemoAdapterRegistry
    private let store: ManagedModelStore
    private let retryDelays: [Duration]
    private var leases: [LibraryTicket: (session: ModelDemoSession, lease: Lease)] = [:]
    private var abandoned: [LibraryTicket: ModelDemoSession] = [:]
    private var stopping: [LibraryTicket: [CheckedContinuation<String?, Never>]] = [:]
    private var releasedThrough: [UUID: Int] = [:]
    private var libraries: [UUID: LibraryReference] = [:]
    private var resident: ModelDemoSession?
    private var bound: Set<LibraryTicket> = []
    private lazy var work = Store<DemoRuntimeWork>(initialState: DemoRuntimeWork.State()) { [weak self] job, _ in
        switch job {
        case .prepare(let session): await self?.prepare(session)
        case .abandon(let owner): await self?.release(owner: owner)
        case .retry: await self?.reattempt()
        }
    }

    public init(
        resources: ModelLibraryResourceRunner, registry: ModelDemoAdapterRegistry, store: ManagedModelStore,
        retryDelays: [Duration] = [.seconds(1), .seconds(2), .seconds(4)]
    ) {
        self.resources = resources
        self.registry = registry
        self.store = store
        self.retryDelays = retryDelays
    }

    public func run(_ effect: ModelLibraryEffect, send: (ModelLibraryAction) -> Void) async {
        if case .prepareDemo(let session) = effect {
            await prepare(session)
            return
        }
        guard case .stopDemo(let session, let ticket) = effect else {
            await resources.run(effect, send: send)
            return
        }
        retryAbandoned()
        if let failure = await stop(session) {
            send(.demoStopFailed(session.id, ticket, failure))
        } else {
            send(.demoStopped(session.id, ticket))
        }
    }

    public func abandon(owner: UUID) {
        work.send(.abandon(owner))
    }

    public func retryAbandoned() {
        guard !abandoned.isEmpty else { return }
        work.send(.retry)
    }

    func gate(for session: ModelDemoSession, in library: Store<ModelLibraryFeature>) -> DemoExecutionGate {
        if library.state.owner == session.id.owner, library.state.session == session {
            libraries[session.id.owner] = LibraryReference(store: library)
        }
        return DemoExecutionGate(session: session) { [weak self, weak library] in
            guard let self, let library else { return false }
            return resident == session && library.state.permits(session.id, selection: session.selection)
                && isLeased(session) && stopping[session.id] == nil
        }
    }

    func bind(_ session: ModelDemoSession, in library: Store<ModelLibraryFeature>) -> DemoExecutionGate? {
        let gate = gate(for: session, in: library)
        guard gate.isPermitted() else { return nil }
        bound.insert(session.id)
        return gate
    }

    func provide(_ session: ModelDemoSession) {
        work.send(.prepare(session))
    }

    func prepare(_ session: ModelDemoSession) async {
        retryAbandoned()
        guard !Task.isCancelled, session.id.number > releasedThrough[session.id.owner, default: 0],
            stopping[session.id] == nil, let state = libraries[session.id.owner]?.store?.state,
            state.session == session, state.permits(session.id, selection: session.selection)
        else { return }
        guard resident == nil || resident == session else {
            publish(.occupied, for: session)
            return
        }
        resident = session
        switch leases[session.id]?.lease {
        case .acquiring: return await withCheckedContinuation { wait(for: session.id, $0) }
        case .held:
            publish(.available, for: session)
            return
        case .revoked: return
        case nil: leases[session.id] = (session, .acquiring([]))
        }
        publish(.preparing, for: session)
        let lease = await store.lease(identity: session.identity, owner: session.id.owner)
        let installed = await store.generation(of: session.identity) != nil
        if !installed { await lease.release() }
        let waiting: [CheckedContinuation<Void, Never>] =
            if case .acquiring(let waiting)? = leases[session.id]?.lease { waiting } else { [] }
        leases[session.id] = (session, installed ? .held(lease) : .revoked)
        for continuation in waiting { continuation.resume() }
        if installed {
            if Task.isCancelled || libraries[session.id.owner]?.store == nil {
                abandoned[session.id] = session
                retryAbandoned()
            } else {
                publish(.available, for: session)
            }
            return
        }
        libraries[session.id.owner]?.store?.send(
            .resourceLost(artifactID: session.resource.artifactID, session: session.id))
    }

    func release(owner: UUID) async {
        libraries[owner] = nil
        releasedThrough[owner] = .max
        for session in leases.values.map(\.session) where session.id.owner == owner {
            abandoned[session.id] = session
        }
        for delay in [.zero] + retryDelays where !abandoned.isEmpty {
            if delay > .zero { try? await Task.sleep(for: delay) }
            await reattempt()
        }
    }

    func reattempt() async {
        for session in abandoned.values {
            if await stop(session) == nil { abandoned[session.id] = nil }
        }
    }

    private func isLeased(_ session: ModelDemoSession) -> Bool {
        if case .held? = leases[session.id]?.lease { true } else { false }
    }

    private func publish(_ preparation: DemoPreparation, for session: ModelDemoSession) {
        libraries[session.id.owner]?.store?.send(.demoPreparationChanged(session.id, preparation))
    }

    private func wait(for id: LibraryTicket, _ continuation: CheckedContinuation<Void, Never>) {
        guard let entry = leases[id], case .acquiring(let waiting) = entry.lease else { return continuation.resume() }
        leases[id] = (entry.session, .acquiring(waiting + [continuation]))
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
        guard resident == session else {
            if libraries[session.id.owner]?.store?.state.session == session {
                releasedThrough[session.id.owner] = max(
                    releasedThrough[session.id.owner, default: 0], session.id.number)
            }
            return nil
        }
        do {
            if bound.contains(session.id) {
                try await registry.adapter(for: session.selection.demoID)?.stop(session)
            }
        } catch {
            return error.localizedDescription
        }
        if case .acquiring? = leases[session.id]?.lease { await withCheckedContinuation { wait(for: session.id, $0) } }
        releasedThrough[session.id.owner] = max(releasedThrough[session.id.owner, default: 0], session.id.number)
        if case .held(let lease)? = leases.removeValue(forKey: session.id)?.lease { await lease.release() }
        bound.remove(session.id)
        if resident == session { resident = nil }
        return nil
    }
}

nonisolated enum DemoRuntimeWork: Feature {
    enum Job: Equatable, Sendable {
        case prepare(ModelDemoSession)
        case abandon(UUID)
        case retry
    }

    struct State: Equatable, Sendable {}

    static func reduce(_ state: inout State, _ action: Job) -> [Command<Job, Never>] {
        [.run(action, id: nil)]
    }
}
