#if os(iOS)
    import FeatureStore
    import ImageSearchFeature
    import ImageSearchRuntime
    import ModelLibraryFeature
    import ModelLibraryRuntime
    import SwiftUI

    @MainActor
    final class ImageSearchAdapter: ModelDemoAdapter {
        let demoID = "image-search"
        private var searches: [DemoSelection: ManagedImageSearch] = [:]

        func preview(for selection: DemoSelection) -> AnyView {
            var state = searches[selection]?.state ?? ImageSearchState()
            state.execution = .managed
            state.selection = nil
            return AnyView(ImageSearchContent(state: state, send: { _ in }, isPermitted: { false }))
        }

        func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
            let search = searches[session.selection] ?? ManagedImageSearch { LiveImageSearchService() }
            searches[session.selection] = search
            let identity = session.identity
            let store = search.store(
                folder: session.resource.directory,
                identity: "\(identity.modelID)/\(identity.artifactID)@\(identity.revision)",
                isPermitted: gate.isPermitted)
            return AnyView(ImageSearchStoreContent(store: store, isPermitted: gate.isPermitted))
        }

        func stop(_ session: ModelDemoSession) async throws {
            try await searches[session.selection]?.release()
        }
    }

    private struct ImageSearchStoreContent: View {
        let store: Store<ImageSearchFeature>
        let isPermitted: @MainActor @Sendable () -> Bool

        var body: some View {
            ImageSearchContent(state: store.state, send: store.send, isPermitted: isPermitted)
        }
    }
#endif
