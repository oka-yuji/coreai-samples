import FeatureStore
import Foundation
import ModelLibraryFeature
import SpeechCore
import SpeechFeature

@MainActor
public final class SpeechDemoSessions {
    private struct Entry {
        let session: ModelDemoSession
        let store: Store<SpeechFeature>
        let service: ManagedSpeechService
    }

    private let makeService: @MainActor () -> any SpeechService
    private var entries: [LibraryTicket: Entry] = [:]
    private var latest: [DemoSelection: LibraryTicket] = [:]
    private var retained: [DemoSelection: SpeechState] = [:]

    public init(makeService: @escaping @MainActor () -> any SpeechService = { LiveSpeechService(defaults: nil) }) {
        self.makeService = makeService
    }

    public func state(for selection: DemoSelection) -> SpeechState? {
        latest[selection].flatMap { entries[$0]?.store.state } ?? retained[selection]
    }

    public func bind(
        _ session: ModelDemoSession, examples: [SpeechExample], permits: @escaping @MainActor () -> Bool
    ) -> Store<SpeechFeature> {
        if let entry = entries[session.id], entry.session == session { return entry.store }
        let prior = state(for: session.selection)
        let service = ManagedSpeechService(service: makeService(), directory: session.resource.directory) {
            [weak self] in
            self?.latest[session.selection] == session.id && permits()
        }
        let model = SpeechModel(name: "Irodori v4.1 Small MF (FP32)", directory: session.resource.directory)
        let store = Store<SpeechFeature>(
            initialState: SpeechState(managedModel: model, examples: examples, retaining: prior),
            run: SpeechRunner(service: service).run)
        entries[session.id] = Entry(session: session, store: store, service: service)
        latest[session.selection] = session.id
        store.send(.appeared)
        return store
    }

    public func stop(_ session: ModelDemoSession) async {
        guard let entry = entries[session.id], entry.session == session else { return }
        entry.service.deactivate()
        entry.store.send(.ended)
        await entry.service.stop()
        if latest[session.selection] == session.id {
            var snapshot = SpeechState(examples: entry.store.state.examples)
            snapshot.text = entry.store.state.text
            snapshot.seed = entry.store.state.seed
            retained[session.selection] = snapshot
            latest[session.selection] = nil
        }
        entries[session.id] = nil
    }
}
