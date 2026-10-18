#if os(iOS)
    import FeatureStore
    import ModelLibraryFeature
    import SwiftUI

    struct ModelAcquisitionControls: View {
        let store: Store<ModelLibraryFeature>
        let model: LibraryModel
        let requestDeletion: () -> Void

        private var installation: ModelInstallation { store.state.installations[model.id] ?? ModelInstallation() }

        var body: some View {
            if model.artifact.download == nil {
                Image(systemName: "info.circle")
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 44, minHeight: 44)
                    .accessibilityLabel(libraryText("Description only"))
            } else {
                acquisition
            }
        }

        private var acquisition: some View {
            HStack(spacing: 4) {
                switch installation.phase {
                case .missing, .failed:
                    Button {
                        store.send(.downloadRequested(model.id))
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
                    Button(action: requestDeletion) {
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
                        store.send(.cancelRequested(model.id))
                    } label: {
                        Image(systemName: "xmark")
                            .frame(minWidth: 44, minHeight: 44)
                            .contentShape(Rectangle())
                    }
                    .accessibilityLabel(libraryText("Cancel download"))
                case .inspecting, .validating:
                    ProgressView().accessibilityLabel(libraryText("Checking model"))
                    Button {
                        store.send(.cancelRequested(model.id))
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

    func libraryText(_ key: String) -> Text {
        Text(LocalizedStringKey(key), bundle: .module)
    }
#endif
