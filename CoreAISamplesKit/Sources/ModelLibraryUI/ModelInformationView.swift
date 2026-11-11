import FeatureStore
import ModelLibraryFeature
import SwiftUI

public struct ModelInformationView: View {
    private let store: Store<ModelLibraryFeature>
    private let model: LibraryModel
    private let inUse: Bool
    @Environment(\.locale) private var locale

    public init(store: Store<ModelLibraryFeature>, model: LibraryModel, inUse: Bool = false) {
        self.store = store
        self.model = model
        self.inUse = inUse
    }

    private var artifact: ModelArtifact? { store.state.selectedArtifact(for: model.id) }

    public var body: some View {
        let artifacts = store.state.artifacts(for: model.id)
        Form {
            LabeledContent {
                Text(verbatim: ModelClassificationPresentation.classification(for: model, locale: locale))
            } label: {
                libraryText("Category")
            }
            if artifacts.count > 1 || (artifact == nil && !artifacts.isEmpty) {
                Picker(
                    selection: Binding(
                        get: { store.state.selectedArtifacts[model.id] ?? "" },
                        set: { store.send(.selectArtifact(modelID: model.id, artifactID: $0)) })
                ) {
                    ForEach(artifacts, id: \.id) { libraryText($0.title).tag($0.id) }
                } label: {
                    libraryText("Variant")
                }
                .disabled(store.state.pendingDestination != nil || inUse)
            } else if let artifact, artifact.sourceKind != nil {
                LabeledContent {
                    Text(verbatim: artifact.title)
                } label: {
                    libraryText("Variant")
                }
            }
            if let artifact,
                let bytes = store.state.installations[artifact.id]?.resource?.sizeBytes ?? artifact.sizeBytes
            {
                LabeledContent {
                    Text(verbatim: bytes.formatted(.byteCount(style: .file)))
                } label: {
                    libraryText("Model size")
                }
            }
            if let download = artifact?.download {
                LabeledContent {
                    Text(verbatim: download.repository)
                } label: {
                    libraryText("Repository")
                }
                LabeledContent {
                    Text(verbatim: download.revision).textSelection(.enabled)
                } label: {
                    libraryText("Revision")
                }
            } else if artifact?.localBundle != nil {
                LabeledContent {
                    libraryText("Local bundle")
                } label: {
                    libraryText("Source")
                }
                if artifact?.contract == "qwen-image21-mobile-v2" {
                    libraryText("For research and evaluation only. Commercial use requires a separate license.")
                        .font(.callout)
                    if let upstream = artifact?.upstream,
                        let license = URL(
                            string:
                                "https://huggingface.co/\(upstream.repository)/blob/\(upstream.revision)/LICENSE"
                        )
                    {
                        Link(destination: license) { libraryText("Qwen Research License") }
                    }
                }
            }
        }
    }
}
