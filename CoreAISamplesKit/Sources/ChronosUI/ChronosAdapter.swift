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
        guard gate.session == session, gate.isPermitted(), let store = sessions.bind(session, permits: gate.isPermitted)
        else {
            return AnyView(Text("Close and reopen the forecast to start this session.", bundle: .module))
        }
        return AnyView(ChronosScreen(store: store))
    }

    public func stop(_ session: ModelDemoSession) async throws { await sessions.stop(session) }
}

public struct ChronosPreview: View {
    @State private var store = ChronosDemoSessions.preview()
    @Environment(\.scenePhase) private var phase
    private let scrollsContent: Bool

    public init(scrollsContent: Bool = true) { self.scrollsContent = scrollsContent }

    public var body: some View {
        ChronosScreen(store: store, scrollsContent: scrollsContent)
            .onAppear { store.send(.appear) }
            .onDisappear { store.send(.suspend) }
            .onChange(of: phase) { _, phase in store.send(phase == .active ? .appear : .suspend) }
    }
}
