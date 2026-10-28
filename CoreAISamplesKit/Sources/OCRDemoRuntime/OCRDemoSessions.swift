import FeatureStore
import Foundation
import ModelLibraryFeature
import OCRFeature

@MainActor
public final class OCRDemoSessions {
    private final class Permission {
        var active = true
    }

    private struct Entry {
        let session: ModelDemoSession
        let store: Store<OCRFeature>
        let runner: OCRRunner
        let permission: Permission
    }

    private let makeService: @MainActor () -> any OCRService
    private let observe: @MainActor (LibraryTicket, OCRAction) -> Void
    private var entries: [LibraryTicket: Entry] = [:]
    private var latest: [DemoSelection: LibraryTicket] = [:]
    private var retained: [DemoSelection: OCRInput] = [:]

    public convenience init(makeService: @escaping @MainActor () -> any OCRService = { LiveOCRService() }) {
        self.init(makeService: makeService, observe: { _, _ in })
    }

    init(
        makeService: @escaping @MainActor () -> any OCRService,
        observe: @escaping @MainActor (LibraryTicket, OCRAction) -> Void
    ) {
        self.makeService = makeService
        self.observe = observe
    }

    public func bind(_ session: ModelDemoSession, permits: @escaping @MainActor () -> Bool) -> Store<OCRFeature>? {
        guard permits() else { return nil }
        if let entry = entries[session.id] { return entry.session == session ? entry.store : nil }
        if let prior = latest[session.selection], entries[prior] != nil { return nil }
        let permission = Permission()
        let runner = OCRRunner(service: makeService()) { [weak self] in
            permission.active && self?.latest[session.selection] == session.id && permits()
        }
        let observe = observe
        let store = Store<OCRFeature>(
            initialState: OCRState(bundle: session.resource.directory, input: retained[session.selection])
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
        if latest[session.selection] == session.id {
            retained[session.selection] = entry.store.state.input
            latest[session.selection] = nil
        }
        entries[session.id] = nil
    }
}
