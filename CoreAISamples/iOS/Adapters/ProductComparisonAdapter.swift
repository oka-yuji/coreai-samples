#if os(iOS)
    import FeatureStore
    import ModelLibraryFeature
    import ModelLibraryRuntime
    import ProductFeature
    import ProductRuntime
    import SwiftUI

    @MainActor
    final class ProductComparisonAdapter: ModelDemoAdapter {
        let demoID = "comparison"
        private let sessions = ProductDemoSessions()

        func preview(for selection: DemoSelection) -> AnyView {
            guard let store = sessions.store(for: selection) else { return AnyView(ProductComparisonPlaceholder()) }
            return AnyView(ProductComparisonContent(store: store))
        }

        func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
            guard gate.session == session, gate.isPermitted() else { return preview(for: session.selection) }
            let store = sessions.bind(session, permits: gate.isPermitted)
            return AnyView(ProductComparisonContent(store: store).onAppear { store.send(.appeared) })
        }

        func stop(_ session: ModelDemoSession) async throws {
            try await sessions.stop(session)
        }
    }
#endif
