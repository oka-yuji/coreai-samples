import ChronosDemoRuntime
import ChronosFeature
import FeatureStore
import ModelLibraryFeature
import ModelLibraryRuntime
import SwiftUI

public final class ChronosAdapter: ModelDemoAdapter {
    public let demoID = ChronosDemoCatalog.selection.demoID
    private let sessions = ChronosDemoSessions()

    public init() {}

    public func preview(for selection: DemoSelection) -> AnyView { AnyView(ChronosPreview()) }

    public func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
        guard gate.session == session, gate.isPermitted(), let store = sessions.bind(session, gate: gate)
        else {
            return AnyView(Text("Close and reopen the forecast to start this session.", bundle: .module))
        }
        return AnyView(ChronosScreen(store: store, journal: sessions.journal(for: session)))
    }

    public func stop(_ session: ModelDemoSession) async throws { await sessions.stop(session) }
}

public struct ChronosPreview: View {
    @State private var store: Store<ChronosFeature>
    @State private var journal: Store<ChronosJournalFeature>
    @Environment(\.scenePhase) private var phase
    private let scrollsContent: Bool

    public init(scrollsContent: Bool = true) {
        self.scrollsContent = scrollsContent
        let journal = ChronosJournalAssembly.shared.makeStore()
        _journal = State(initialValue: journal)
        _store = State(initialValue: ChronosDemoSessions.preview(journal: journal))
    }

    public var body: some View {
        ChronosScreen(store: store, scrollsContent: scrollsContent, journal: journal)
            .onAppear { store.send(.appear) }
            .onDisappear { store.send(.suspend) }
            .onChange(of: phase) { _, phase in store.send(phase == .active ? .appear : .suspend) }
    }
}
