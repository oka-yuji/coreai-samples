import FeatureStore
import ForecastFeature
import Foundation
import ModelLibraryFeature
import ModelLibraryRuntime

@MainActor
public final class ForecastDemoSessions {
    private final class Permission { var active = true }

    private struct Entry {
        let session: ModelDemoSession
        let store: Store<ForecastFeature>
        let runner: ForecastRunner
        let permission: Permission
    }

    public typealias Factory =
        @MainActor (ValidatedModelResource, @escaping @MainActor () -> Bool) -> any ForecastService
    private let makeService: Factory
    private let observe: @MainActor (LibraryTicket, ForecastFeature.Action) -> Void
    private var entries: [LibraryTicket: Entry] = [:]
    private var latest: LibraryTicket?

    public init(
        makeService: Factory? = nil,
        observe: @escaping @MainActor (LibraryTicket, ForecastFeature.Action) -> Void = { _, _ in }
    ) {
        self.makeService = makeService ?? { LiveForecastService(resource: $0, permits: $1) }
        self.observe = observe
    }

    public func bind(_ session: ModelDemoSession, gate: DemoExecutionGate) -> Store<ForecastFeature>? {
        guard gate.session == session else { return nil }
        return bind(session, permits: gate.isPermitted)
    }

    public func bind(
        _ session: ModelDemoSession, permits: @escaping @MainActor () -> Bool
    ) -> Store<ForecastFeature>? {
        guard session.selection == ChronosDemoCatalog.salesSelection, permits() else { return nil }
        if let entry = entries[session.id] {
            return entry.session == session && entry.permission.active ? entry.store : nil
        }
        guard latest == nil else { return nil }
        let permission = Permission()
        let allowed: @MainActor () -> Bool = { [weak self] in
            permission.active && self?.latest == session.id && permits()
        }
        let runner = ForecastRunner(service: makeService(session.resource, allowed), permits: allowed)
        let observe = observe
        let store = Store<ForecastFeature>(initialState: ForecastFeature.State(liveAvailable: true)) { effect, send in
            await runner.run(effect) { action in
                send(action)
                observe(session.id, action)
            }
        }
        entries[session.id] = Entry(session: session, store: store, runner: runner, permission: permission)
        latest = session.id
        store.send(.appear)
        return store
    }

    public func stop(_ session: ModelDemoSession) async {
        guard let entry = entries[session.id], entry.session == session else { return }
        entry.permission.active = false
        entry.store.send(.ended)
        await entry.runner.release()
        guard entries[session.id]?.store === entry.store else { return }
        entries[session.id] = nil
        if latest == session.id { latest = nil }
    }

    public static func preview(service: (any ForecastService)? = nil) -> Store<ForecastFeature> {
        let runner = ForecastRunner(service: service ?? LiveForecastService())
        return Store<ForecastFeature>(initialState: ForecastFeature.State()) { effect, send in
            await runner.run(effect, send: send)
        }
    }
}
