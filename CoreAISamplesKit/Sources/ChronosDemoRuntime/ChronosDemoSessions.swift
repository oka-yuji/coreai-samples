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
        let journal: Store<ChronosJournalFeature>
        let permission: Permission
    }

    public typealias Factory =
        @MainActor (ValidatedModelResource, @escaping @MainActor () -> Bool) -> any ChronosService
    private let makeService: Factory
    private let journalAssembly: ChronosJournalAssembly
    private let observe: @MainActor (LibraryTicket, ChronosFeature.Action) -> Void
    private let observeJournal: @MainActor (LibraryTicket, ChronosJournalFeature.Action) -> Void
    private var entries: [LibraryTicket: Entry] = [:]
    private var latest: [DemoSelection: LibraryTicket] = [:]

    public init(
        makeService: Factory? = nil, journal: ChronosJournalAssembly = .shared,
        observe: @escaping @MainActor (LibraryTicket, ChronosFeature.Action) -> Void = { _, _ in },
        observeJournal: @escaping @MainActor (LibraryTicket, ChronosJournalFeature.Action) -> Void = { _, _ in }
    ) {
        journalAssembly = journal
        self.makeService = makeService ?? { LiveChronosService(resource: $0, api: journal.api, permits: $1) }
        self.observe = observe
        self.observeJournal = observeJournal
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
        let observeJournal = observeJournal
        let journal = journalAssembly.makeStore(permits: allowed, observe: { observeJournal(session.id, $0) })
        let runner = ChronosRunner(service: makeService(session.resource, allowed), permits: allowed, journal: journal)
        let observe = observe
        let store = Store<ChronosFeature>(initialState: ChronosFeature.State()) { effect, send in
            await runner.run(effect) { action in
                send(action)
                observe(session.id, action)
            }
        }
        entries[session.id] = Entry(
            session: session, store: store, runner: runner, journal: journal, permission: permission)
        latest[session.selection] = session.id
        store.send(.appear)
        return store
    }

    public func stop(_ session: ModelDemoSession) async {
        guard let entry = entries[session.id], entry.session == session else { return }
        entry.permission.active = false
        entry.journal.send(.ended)
        entry.store.send(.ended)
        await entry.runner.release()
        guard entries[session.id]?.store === entry.store else { return }
        if latest[session.selection] == session.id { latest[session.selection] = nil }
        entries[session.id] = nil
    }

    public func journal(for session: ModelDemoSession) -> Store<ChronosJournalFeature>? {
        guard let entry = entries[session.id], entry.session == session, entry.permission.active else { return nil }
        return entry.journal
    }

    public static func preview(
        service: (any ChronosService)? = nil, journal: Store<ChronosJournalFeature>? = nil
    ) -> Store<ChronosFeature> {
        let runner = ChronosRunner(service: service ?? LiveChronosService(), journal: journal)
        return Store<ChronosFeature>(initialState: ChronosFeature.State()) { effect, send in
            await runner.run(effect, send: send)
        }
    }
}
