#if os(iOS)
    import FeatureStore
    import ModelLibraryFeature
    import ModelLibraryRuntime
    import ProductFeature
    import ProductRuntime
    import SwiftUI

    @MainActor
    final class ProductComparisonAdapter: ModelDemoAdapter {
        private struct Entry {
            let identity: ModelArtifactIdentity
            let store: Store<ProductFeature>
            let runner: ProductRunner
            var gate: DemoExecutionGate
        }

        let demoID = "comparison"
        private var entries: [DemoSelection: Entry] = [:]

        func preview(for selection: DemoSelection) -> AnyView {
            guard let store = entries[selection]?.store else { return AnyView(ProductComparisonPlaceholder()) }
            return AnyView(ProductComparisonContent(store: store))
        }

        func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
            let selection = session.selection
            let identity = session.identity
            let entry: Entry
            if let current = entries[selection], current.identity == identity {
                entry = Entry(identity: identity, store: current.store, runner: current.runner, gate: gate)
            } else {
                let managed = ProductAssembly.makeManagedStore(
                    resource: session.resource, keeping: entries[selection]?.store.state
                ) { [weak self] in
                    guard let entry = self?.entries[selection], entry.identity == identity else { return false }
                    return entry.gate.isPermitted()
                }
                entry = Entry(identity: identity, store: managed.store, runner: managed.runner, gate: gate)
            }
            entries[selection] = entry
            let store = entry.store
            return AnyView(ProductComparisonContent(store: store).onAppear { store.send(.appeared) })
        }

        func stop(_ session: ModelDemoSession) async throws {
            guard let entry = entries[session.selection], entry.identity == session.identity else { return }
            entry.store.send(.closed)
            try await entry.runner.release()
        }
    }
#endif
