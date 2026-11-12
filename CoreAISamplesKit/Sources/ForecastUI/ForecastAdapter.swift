import FeatureStore
import ForecastFeature
import ForecastRuntime
import ModelLibraryFeature
import ModelLibraryRuntime
import SwiftUI

public final class ForecastAdapter: ModelDemoAdapter {
    public let demoID = ChronosDemoCatalog.salesSelection.demoID
    private let sessions = ForecastDemoSessions()

    public init() {}

    public func preview(for selection: DemoSelection) -> AnyView { AnyView(ForecastPreview()) }

    public func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
        guard let store = sessions.bind(session, gate: gate) else {
            return AnyView(Text("Close and reopen the forecast to start this session.", bundle: .module))
        }
        return AnyView(ForecastScreen(store: store))
    }

    public func stop(_ session: ModelDemoSession) async throws { await sessions.stop(session) }
}

public struct ForecastPreview: View {
    @State private var store = ForecastDemoSessions.preview()
    @Environment(\.scenePhase) private var phase
    private let scrollsContent: Bool

    public init(scrollsContent: Bool = true) { self.scrollsContent = scrollsContent }

    public var body: some View {
        ForecastScreen(store: store, scrollsContent: scrollsContent)
            .onAppear { store.send(.appear) }
            .onDisappear { store.send(.suspend) }
            .onChange(of: phase) { _, phase in store.send(phase == .active ? .appear : .suspend) }
    }
}
