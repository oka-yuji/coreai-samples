import ChronosFeature
import FeatureStore
import Foundation
import ModelLibraryFeature

@MainActor
public final class ChronosDemoSessions {
    private final class Permission { var active = true }
    private struct Entry {
        let session: ModelDemoSession
        let store: Store<ChronosFeature>
        let runner: ChronosRunner
        let permission: Permission
    }

    public typealias Factory =
        @MainActor (ValidatedModelResource, @escaping @MainActor () -> Bool) -> any ChronosService
    private let makeService: Factory
    private let observe: @MainActor (LibraryTicket, ChronosFeature.Action) -> Void
    private var entries: [LibraryTicket: Entry] = [:]
    private var latest: [DemoSelection: LibraryTicket] = [:]

    public init(
        makeService: @escaping Factory = { LiveChronosService(resource: $0, permits: $1) },
        observe: @escaping @MainActor (LibraryTicket, ChronosFeature.Action) -> Void = { _, _ in }
    ) {
        self.makeService = makeService
        self.observe = observe
    }

    public func bind(_ session: ModelDemoSession, permits: @escaping @MainActor () -> Bool) -> Store<ChronosFeature>? {
        guard session.selection == ChronosDemoCatalog.selection, permits() else { return nil }
        if let entry = entries[session.id] {
            return entry.session == session && entry.permission.active ? entry.store : nil
        }
        if let previous = latest[session.selection], entries[previous] != nil { return nil }
        let permission = Permission()
        let allowed: @MainActor () -> Bool = { [weak self] in
            permission.active && self?.latest[session.selection] == session.id && permits()
        }
        let runner = ChronosRunner(service: makeService(session.resource, allowed), permits: allowed)
        let observe = observe
        let store = Store<ChronosFeature>(initialState: ChronosFeature.State()) { effect, send in
            await runner.run(effect) { action in
                send(action)
                observe(session.id, action)
            }
        }
        entries[session.id] = Entry(session: session, store: store, runner: runner, permission: permission)
        latest[session.selection] = session.id
        store.send(.appear)
        return store
    }

    public func stop(_ session: ModelDemoSession) async {
        guard let entry = entries[session.id], entry.session == session else { return }
        entry.permission.active = false
        entry.store.send(.ended)
        await entry.runner.release()
        guard entries[session.id]?.store === entry.store else { return }
        if latest[session.selection] == session.id { latest[session.selection] = nil }
        entries[session.id] = nil
    }

    public static func preview(service: (any ChronosService)? = nil) -> Store<ChronosFeature> {
        let runner = ChronosRunner(service: service ?? LiveChronosService())
        return Store<ChronosFeature>(initialState: ChronosFeature.State()) { effect, send in
            await runner.run(effect, send: send)
        }
    }
}
