import FeatureStore
import ImageSearchFeature
import SwiftUI
import UniformTypeIdentifiers

struct ImageSearchView: View {
    let store: Store<ImageSearchFeature>
    #if os(iOS)
        @State private var choosingModel = false
        @State private var modelRequest = 0
    #else
        @Environment(MacModelLibrary.self) private var library: MacModelLibrary?
        @State private var attempted: URL?
    #endif

    private var state: ImageSearchState { store.state }

    var body: some View {
        VStack(spacing: 0) {
            modelControls.padding().frame(maxWidth: .infinity, alignment: .leading)
            Divider()
            ScrollView {
                ImageSearchContent(state: state, send: store.send, isPermitted: { true }, canUseModel: canUseModel)
                    .frame(maxWidth: 1_200)
                    .frame(maxWidth: .infinity)
                    .padding()
            }
        }
        .demoNavigationTitle("EmbeddingGemma 2")
        #if os(iOS)
            .fileImporter(isPresented: $choosingModel, allowedContentTypes: [.folder]) { result in
                switch result {
                case .success(let url): store.send(.load(url))
                case .failure(let error): store.send(.selectionFailed(modelRequest, error.localizedDescription))
                }
            }
            .onDisappear { store.send(.disappeared) }
        #else
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
        #endif
    }

    private var modelControls: some View {
        HStack(spacing: 10) {
            #if os(iOS)
                Menu {
                    ForEach(ImageSearchAssembly.preparedModels, id: \.self) { folder in
                        Button {
                            store.send(.load(folder))
                        } label: {
                            Text(verbatim: folder.lastPathComponent)
                        }
                    }
                    Button("Open model folder…") {
                        modelRequest = state.session
                        choosingModel = true
                    }
                    if let folder = state.modelFolder {
                        Button("Reload selected model") { store.send(.load(folder)) }
                    }
                } label: {
                    Label(
                        LocalizedStringKey(state.ready ? "Model ready" : "Open model"),
                        systemImage: state.ready ? "checkmark.circle" : "folder"
                    )
                }
                .disabled(state.isBusy)
                .accessibilityIdentifier("image-search-model")
                folderName
            #else
                if let library, library.installed(.embeddingGemma) == nil {
                    Text("Download \(DemoModel.embeddingGemma.title) from its heading in the sidebar.")
                        .font(.callout).foregroundStyle(.secondary)
                } else {
                    if state.ready { Label("Model ready", systemImage: "checkmark.circle") }
                    folderName
                }
            #endif
        }
    }

    private var folderName: some View {
        Text(verbatim: modelName).font(.caption).foregroundStyle(.secondary)
    }

    private var modelName: String {
        #if os(iOS)
            return state.modelFolder?.lastPathComponent ?? "EmbeddingGemma 2"
        #else
            return library?.artifactTitle(.embeddingGemma) ?? state.modelFolder?.lastPathComponent ?? "EmbeddingGemma 2"
        #endif
    }

    private var canUseModel: Bool {
        #if os(iOS)
            return true
        #else
            guard let library else { return true }
            return library.resource(inUse: .embeddingGemma)?.directory == state.modelFolder
        #endif
    }

    #if os(macOS)
        private var modelToLoad: URL? {
            guard let directory = library?.resource(inUse: .embeddingGemma)?.directory, !state.isBusy,
                state.modelFolder != directory || !state.ready, attempted != directory
            else { return nil }
            return directory
        }
    #endif
}
