#if os(iOS)
    import FeatureStore
    import ModelLibraryFeature
    import SwiftUI

    struct ModelAcquisitionControls: View {
        let store: Store<ModelLibraryFeature>
        let model: LibraryModel
        let requestDeletion: (ModelArtifact) -> Void

        var body: some View {
            if let artifact = store.state.selectedArtifact(for: model.id), artifact.download != nil {
                acquisition(artifact, installation: store.state.installations[artifact.id] ?? ModelInstallation())
            } else {
                Image(systemName: "info.circle")
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 44, minHeight: 44)
                    .accessibilityLabel(libraryText("Description only"))
            }
        }

        private func acquisition(_ artifact: ModelArtifact, installation: ModelInstallation) -> some View {
            HStack(spacing: 4) {
                switch installation.phase {
                case .missing, .failed:
                    Button {
                        store.send(.downloadRequested(artifact.id))
                    } label: {
                        Image(systemName: installation.phase == .failed ? "arrow.clockwise" : "arrow.down.circle")
                            .frame(minWidth: 44, minHeight: 44)
                            .contentShape(Rectangle())
                    }
                    .accessibilityLabel(
                        libraryText(installation.phase == .failed ? "Retry download" : "Download model")
                    )
                    .accessibilityValue(Text(verbatim: model.name))
                case .ready:
                    Image(systemName: "checkmark").foregroundStyle(.secondary)
                        .accessibilityLabel(libraryText("Downloaded"))
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
                    .disabled(store.state.session?.selection.modelID == model.id)
                case .downloading(let progress):
                    VStack(spacing: 3) {
                        if let fraction = progress?.fraction {
                            ProgressView(value: fraction)
                                .accessibilityLabel(libraryText("Downloading"))
                        } else {
                            ProgressView().accessibilityLabel(libraryText("Downloading"))
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
                    .accessibilityLabel(libraryText("Cancel download"))
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
    }

    struct LibraryFailureView: View {
        let failure: ModelLibraryFailure

        var body: some View {
            Label {
                switch failure {
                case .unavailable: libraryText("This demo is not available on this device.")
                case .downloadNotConfigured: libraryText("A download source has not been configured for this model.")
                case .invalidDownload: libraryText("The model download configuration is invalid.")
                case .invalidResource: libraryText("The model files do not match the required bundle.")
                case .modelInUse: libraryText("Close this model's demo before deleting its files.")
                case .operation(let message): Text(verbatim: message)
                }
            } icon: {
                Image(systemName: "exclamationmark.triangle")
            }
            .font(.callout)
            .foregroundStyle(.secondary)
        }
    }

    struct ModelDeletionRequest {
        let model: LibraryModel
        let artifact: ModelArtifact
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
        func modelDeletionAlert(
            store: Store<ModelLibraryFeature>, deletion: Binding<ModelDeletionRequest?>
        ) -> some View {
            modifier(ModelDeletionAlert(store: store, deletion: deletion))
        }
    }

    func libraryText(_ key: String) -> Text {
        Text(LocalizedStringKey(key), bundle: .module)
    }
#endif
