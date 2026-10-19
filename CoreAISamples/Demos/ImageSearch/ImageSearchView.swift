import FeatureStore
import ImageSearchFeature
import SwiftUI
import UniformTypeIdentifiers

struct ImageSearchView: View {
    let store: Store<ImageSearchFeature>
    @State private var choosingModel = false
    @State private var modelRequest = 0

    private var state: ImageSearchState { store.state }

    var body: some View {
        VStack(spacing: 0) {
            modelControls.padding().frame(maxWidth: .infinity, alignment: .leading)
            Divider()
            ScrollView {
                ImageSearchContent(state: state, send: store.send, isPermitted: { true })
                    .frame(maxWidth: 1_200)
                    .frame(maxWidth: .infinity)
                    .padding()
            }
        }
        .demoNavigationTitle("EmbeddingGemma 2")
        .fileImporter(isPresented: $choosingModel, allowedContentTypes: [.folder]) { result in
            switch result {
            case .success(let url): store.send(.load(url))
            case .failure(let error): store.send(.selectionFailed(modelRequest, error.localizedDescription))
            }
        }
        .onDisappear { store.send(.disappeared) }
    }

    private var modelControls: some View {
        HStack(spacing: 10) {
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
            Text(verbatim: state.modelFolder?.lastPathComponent ?? "EmbeddingGemma 2")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}
