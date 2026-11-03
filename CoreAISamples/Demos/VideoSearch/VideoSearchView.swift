#if os(macOS)
    import FeatureStore
    import SwiftUI
    import VideoSearchFeature

    struct VideoSearchView: View {
        let store: Store<VideoSearchFeature>
        @Environment(MacModelLibrary.self) private var library: MacModelLibrary?
        @State private var attempted: URL?

        private var state: VideoSearchState { store.state }

        var body: some View {
            VStack(spacing: 0) {
                modelControls.padding().frame(maxWidth: .infinity, alignment: .leading)
                Divider()
                ScrollView {
                    VideoSearchContent(
                        state: state, send: store.send, isPermitted: { true }, canUseModel: canUseModel
                    )
                    .frame(maxWidth: 1_200)
                    .frame(maxWidth: .infinity)
                    .padding()
                }
            }
            .demoNavigationTitle("EmbeddingGemma 2")
            .onAppear { library?.use(.embeddingGemma) }
            .onDisappear {
                store.send(.disappeared)
                library?.stopUsing(.embeddingGemma)
                attempted = nil
            }
            .onChange(of: library?.resource(inUse: .embeddingGemma)?.directory) { _, new in
                if new == nil { attempted = nil }
            }
            .onChange(of: modelToLoad, initial: true) { _, url in
                guard let url else { return }
                attempted = url
                store.send(.load(url))
            }
        }

        private var modelControls: some View {
            HStack(spacing: 10) {
                if let library, library.installed(.embeddingGemma) == nil {
                    Text("Download \(DemoModel.embeddingGemma.title) from its heading in the sidebar.")
                        .font(.callout).foregroundStyle(.secondary)
                } else {
                    if state.ready { Label("Model ready", systemImage: "checkmark.circle") }
                    Text(verbatim: state.modelFolder?.lastPathComponent ?? "EmbeddingGemma 2")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        }

        private var canUseModel: Bool {
            guard let library else { return true }
            return library.resource(inUse: .embeddingGemma)?.directory == state.modelFolder
        }

        private var modelToLoad: URL? {
            guard let directory = library?.resource(inUse: .embeddingGemma)?.directory, !state.isBusy,
                state.modelFolder != directory || !state.ready, attempted != directory
            else { return nil }
            return directory
        }
    }
#endif
