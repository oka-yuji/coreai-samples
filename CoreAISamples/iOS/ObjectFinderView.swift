#if os(iOS)
    import FeatureStore
    import Foundation
    import ObjectFinderCore
    import ObjectFinderFeature
    import ObjectFinderRuntime
    import SwiftUI
    import UniformTypeIdentifiers

    @MainActor
    enum ObjectFinderAssembly {
        struct Session {
            let store: Store<ObjectFinderFeature>
            let preview: ObjectFinderPreview
        }

        static func makeSession() -> Session {
            let prepared = URL.documentsDirectory.appending(path: ImageSearchAssembly.preparedFolderName)
            let folder =
                LaunchArguments.value("--embedding-bundle").map { URL(filePath: $0) }
                ?? (FileManager.default.fileExists(atPath: prepared.appending(path: "embedding-manifest.json").path)
                    ? prepared : nil)
            let preview = ObjectFinderPreview()
            let store = Store<ObjectFinderFeature>(
                initialState: ObjectFinderState(folder: folder),
                run: ObjectFinderRunner(service: LiveObjectFinderService(preview: preview)).run)
            return Session(store: store, preview: preview)
        }

        static var examples: [String] {
            guard let url = Bundle.main.url(forResource: "object-finder-examples", withExtension: "json"),
                let data = try? Data(contentsOf: url), let values = try? JSONDecoder().decode([String].self, from: data)
            else { return [] }
            return values
        }
    }

    struct ObjectFinderView: View {
        let store: Store<ObjectFinderFeature>
        let preview: ObjectFinderPreview
        @State private var choosesModel = false

        var body: some View {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Text("Name an object, then point your camera around the room.")
                        .foregroundStyle(.secondary)
                    ObjectFinderContent(store: store, preview: preview, chooseFolder: { choosesModel = true }) {
                        ForEach(ImageSearchAssembly.preparedModels, id: \.self) { folder in
                            Button {
                                store.send(.folderSelected(folder))
                            } label: {
                                Label {
                                    Text(verbatim: folder.lastPathComponent)
                                } icon: {
                                    Image(systemName: store.state.folder == folder ? "checkmark.circle" : "circle")
                                }
                            }
                        }
                        Button {
                            choosesModel = true
                        } label: {
                            Label(
                                store.state.folder == nil ? "Open model folder…" : "Change model folder…",
                                systemImage: "folder")
                        }
                        if let folder = store.state.folder {
                            Text(verbatim: folder.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("EmbeddingGemma 2")
            .fileImporter(isPresented: $choosesModel, allowedContentTypes: [.folder]) { result in
                switch result {
                case .success(let url): store.send(.folderSelected(url))
                case .failure(let error): store.send(.inputFailed(error.localizedDescription))
                }
            }
        }
    }
#endif
