import FeatureStore
import Foundation
import ModelLibraryFeature
import ReceiptFeature

@MainActor
public final class ReceiptDemoSessions {
    private final class Permission { var active = true }
    private struct Entry {
        let session: ModelDemoSession
        let store: Store<ReceiptFeature>
        let runner: ReceiptRunner
        let permission: Permission
    }

    private let makeService: @MainActor () -> any ReceiptService
    private let observe: @MainActor (LibraryTicket, ReceiptAction) -> Void
    private var entries: [LibraryTicket: Entry] = [:]
    private var latest: [DemoSelection: LibraryTicket] = [:]

    public convenience init(makeService: @escaping @MainActor () -> any ReceiptService = { LiveReceiptService() }) {
        self.init(makeService: makeService, observe: { _, _ in })
    }

    init(
        makeService: @escaping @MainActor () -> any ReceiptService,
        observe: @escaping @MainActor (LibraryTicket, ReceiptAction) -> Void
    ) {
        self.makeService = makeService
        self.observe = observe
    }

    public func bind(_ session: ModelDemoSession, permits: @escaping @MainActor () -> Bool) -> Store<ReceiptFeature>? {
        guard permits() else { return nil }
        if let entry = entries[session.id] { return entry.session == session ? entry.store : nil }
        if let prior = latest[session.selection], entries[prior] != nil { return nil }
        let permission = Permission()
        let runner = ReceiptRunner(service: makeService()) { [weak self] in
            permission.active && self?.latest[session.selection] == session.id && permits()
        }
        let observe = observe
        let store = Store<ReceiptFeature>(
            initialState: ReceiptState(bundle: session.resource.directory, session: session.id.number)
        ) { effect, send in
            await runner.run(effect) { action in
                send(action)
                observe(session.id, action)
            }
        }
        entries[session.id] = Entry(session: session, store: store, runner: runner, permission: permission)
        latest[session.selection] = session.id
        store.send(.appeared)
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
}
