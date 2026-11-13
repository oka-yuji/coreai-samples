import FeatureStore
import Foundation
import ModelLibraryFeature
import SwiftUI
import UniformTypeIdentifiers

public struct ModelAcquisitionControls: View {
    let store: Store<ModelLibraryFeature>
    let model: LibraryModel
    let requestDeletion: (ModelArtifact) -> Void
    var showsActionTitle = false
    var deletionDisabled = false
    @State private var choosingBundle = false
    @State private var importArtifactID: String?

    public init(
        store: Store<ModelLibraryFeature>, model: LibraryModel, requestDeletion: @escaping (ModelArtifact) -> Void,
        showsActionTitle: Bool = false, deletionDisabled: Bool = false
    ) {
        self.store = store
        self.model = model
        self.requestDeletion = requestDeletion
        self.showsActionTitle = showsActionTitle
        self.deletionDisabled = deletionDisabled
    }

    public var body: some View {
        Group {
            if let artifact = store.state.selectedArtifact(for: model.id), artifact.sourceKind != nil {
                acquisition(artifact, installation: store.state.installations[artifact.id] ?? ModelInstallation())
            } else {
                Image(systemName: "info.circle")
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 44, minHeight: 44)
                    .accessibilityLabel(libraryText("Description only"))
            }
        }
        .fileImporter(isPresented: $choosingBundle, allowedContentTypes: [.folder]) { result in
            guard let artifactID = importArtifactID else { return }
            importArtifactID = nil
            switch result {
            case .success(let folder): store.send(.importRequested(artifactID, folder))
            case .failure(let error):
                if (error as? CocoaError)?.code != .userCancelled {
                    store.send(.importSelectionFailed(artifactID, error.localizedDescription))
                }
            }
        }
    }

    private func acquisition(_ artifact: ModelArtifact, installation: ModelInstallation) -> some View {
        let policy = ModelAcquisitionPolicy(artifact: artifact)
        return HStack(spacing: 4) {
            switch installation.phase {
            case .missing, .failed:
                if let method = policy.preferredMethod {
                    acquisitionButton(
                        artifact, method: method,
                        retrying: installation.phase == .failed && !(method == .download && policy.allowsImport))
                    if method == .download, policy.allowsImport {
                        acquisitionButton(artifact, method: .importBundle)
                    }
                }
            case .ready:
                Image(systemName: "checkmark").foregroundStyle(.secondary)
                    .accessibilityLabel(libraryText(policy.readyTitle))
                if policy.allowsImport {
                    Button {
                        acquire(artifact, method: .importBundle)
                    } label: {
                        Image(systemName: "folder.badge.plus")
                            .frame(minWidth: 44, minHeight: 44)
                            .contentShape(Rectangle())
                    }
                    .accessibilityLabel(libraryText("Import again"))
                    .disabled(store.state.session?.selection.modelID == model.id || deletionDisabled)
                }
                Button {
                    requestDeletion(artifact)
                } label: {
                    Image(systemName: "trash")
                        .frame(minWidth: 44, minHeight: 44)
                        .contentShape(Rectangle())
                }
                .foregroundStyle(.secondary)
                .accessibilityLabel(libraryText("Delete model"))
                .accessibilityValue(Text(verbatim: model.name))
                .disabled(store.state.session?.selection.modelID == model.id || deletionDisabled)
            case .downloading(let progress), .importing(let progress):
                if let method = ModelAcquisitionPolicy.transferMethod(for: installation.phase) {
                    transferProgress(artifact, method: method, progress: progress)
                }
            case .inspecting, .validating:
                ProgressView().accessibilityLabel(libraryText("Checking model"))
                Button {
                    store.send(.cancelRequested(artifact.id))
                } label: {
                    Image(systemName: "xmark")
                        .frame(minWidth: 44, minHeight: 44)
                        .contentShape(Rectangle())
                }
                .accessibilityLabel(libraryText("Cancel"))
            case .stopping:
                ProgressView().accessibilityLabel(libraryText("Stopping"))
            case .deleting:
                ProgressView().accessibilityLabel(libraryText("Deleting model"))
            }
        }
        .buttonStyle(.plain)
    }

    private func acquisitionButton(
        _ artifact: ModelArtifact, method: ModelAcquisitionPolicy.Method, retrying: Bool = false
    ) -> some View {
        Button {
            acquire(artifact, method: method)
        } label: {
            if showsActionTitle {
                libraryText(method.actionTitle(retrying: retrying)).padding(.leading, 12)
            }
            Image(systemName: method.systemImage(retrying: retrying))
                .frame(minWidth: 44, minHeight: 44)
                .contentShape(Rectangle())
        }
        .accessibilityLabel(libraryText(method.actionTitle(retrying: retrying)))
        .accessibilityValue(Text(verbatim: model.name))
        .disabled(method == .importBundle && (store.state.session?.selection.modelID == model.id || deletionDisabled))
    }

    private func transferProgress(
        _ artifact: ModelArtifact, method: ModelAcquisitionPolicy.Method, progress: ModelTransferProgress?
    ) -> some View {
        Group {
            VStack(spacing: 3) {
                if let fraction = progress?.fraction {
                    ProgressView(value: fraction)
                        .accessibilityLabel(
                            libraryText(method.progressTitle))
                } else {
                    ProgressView().accessibilityLabel(
                        libraryText(method.progressTitle))
                }
            }
            .frame(width: 30)
            Button {
                store.send(.cancelRequested(artifact.id))
            } label: {
                Image(systemName: "xmark")
                    .frame(minWidth: 44, minHeight: 44)
                    .contentShape(Rectangle())
            }
            .accessibilityLabel(libraryText(method.cancellationTitle))
        }
    }

    private func acquire(_ artifact: ModelArtifact, method: ModelAcquisitionPolicy.Method) {
        if method == .importBundle {
            importArtifactID = artifact.id
            choosingBundle = true
        } else {
            store.send(.downloadRequested(artifact.id))
        }
    }
}

public struct LibraryFailureView: View {
    let failure: ModelLibraryFailure

    public init(failure: ModelLibraryFailure) {
        self.failure = failure
    }

    public var body: some View {
        Label {
            switch failure {
            case .unavailable: libraryText("This demo is not available on this device.")
            case .downloadNotConfigured: libraryText("A download source has not been configured for this model.")
            case .invalidDownload: libraryText("The model download configuration is invalid.")
            case .invalidResource: libraryText("The model files do not match the required bundle.")
            case .modelInUse: libraryText("Close this model's demo before changing its files.")
            case .operation(let message): Text(verbatim: message)
            }
        } icon: {
            Image(systemName: "exclamationmark.triangle")
        }
        .font(.callout)
        .foregroundStyle(.secondary)
    }
}

public struct ModelDeletionRequest {
    let model: LibraryModel
    let artifact: ModelArtifact

    public init(model: LibraryModel, artifact: ModelArtifact) {
        self.model = model
        self.artifact = artifact
    }
}

private struct ModelDeletionAlert: ViewModifier {
    let store: Store<ModelLibraryFeature>
    @Binding var deletion: ModelDeletionRequest?
    @Environment(\.locale) private var locale

    func body(content: Content) -> some View {
        content.alert(
            libraryText("Delete this model?"),
            isPresented: Binding(get: { deletion != nil }, set: { if !$0 { deletion = nil } }),
            presenting: deletion
        ) { request in
            Button(role: .cancel) {
                deletion = nil
            } label: {
                libraryText("Cancel")
            }
            Button(role: .destructive) {
                store.send(.deleteRequested(request.artifact.id))
                deletion = nil
            } label: {
                libraryText("Delete")
            }
        } message: { request in
            Text(verbatim: message(request.model, request.artifact))
        }
    }

    private func message(_ model: LibraryModel, _ artifact: ModelArtifact) -> String {
        let explanation = String(
            localized: "Only model files are removed. Your inputs and saved results remain.", bundle: .module,
            locale: locale)
        let name = store.state.artifacts(for: model.id).count > 1 ? model.name + " · " + artifact.title : model.name
        let size = store.state.installations[artifact.id]?.resource?.sizeBytes
            .formatted(.byteCount(style: .file))
        return [name, explanation, size].compactMap { $0 }.joined(separator: "\n\n")
    }
}

extension View {
    public func modelDeletionAlert(
        store: Store<ModelLibraryFeature>, deletion: Binding<ModelDeletionRequest?>
    ) -> some View {
        modifier(ModelDeletionAlert(store: store, deletion: deletion))
    }
}

func libraryText(_ key: String) -> Text {
    Text(LocalizedStringKey(key), bundle: .module)
}
