#if os(iOS)
    import FeatureStore
    import ModelLibraryFeature
    import ModelLibraryRuntime
    import SwiftUI
    import VideoSearchFeature
    import VideoSearchRuntime

    @MainActor
    final class VideoSearchAdapter: ModelDemoAdapter {
        let demoID = "video-search"
        private var searches: [DemoSelection: ManagedVideoSearch] = [:]

        func preview(for selection: DemoSelection) -> AnyView {
            var state = searches[selection]?.state ?? VideoSearchState()
            state.execution = .managed
            return AnyView(VideoSearchContent(state: state, send: { _ in }, isPermitted: { false }))
        }

        func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
            let search = searches[session.selection] ?? ManagedVideoSearch { LiveVideoSearchService() }
            searches[session.selection] = search
            let identity = session.identity
            let store = search.store(
                folder: session.resource.directory,
                identity: "\(identity.modelID)/\(identity.artifactID)@\(identity.revision)",
                isPermitted: gate.isPermitted)
            return AnyView(VideoSearchStoreContent(store: store, isPermitted: gate.isPermitted))
        }

        func stop(_ session: ModelDemoSession) async throws {
            try await searches[session.selection]?.release()
        }
    }

    private struct VideoSearchStoreContent: View {
        let store: Store<VideoSearchFeature>
        let isPermitted: @MainActor @Sendable () -> Bool

        var body: some View {
            VideoSearchContent(state: store.state, send: store.send, isPermitted: isPermitted)
        }
    }
#endif
