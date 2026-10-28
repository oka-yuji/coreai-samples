#if os(macOS)
    import FeatureStore
    import Foundation
    import ModelLibraryFeature
    import SwiftUI
    import UniformTypeIdentifiers

    public struct OCRModelControls: View {
        private let store: Store<ModelLibraryFeature>
        private let open: (DemoSelection) -> Void
        @State private var importing = false
        @State private var deleting = false

        public init(store: Store<ModelLibraryFeature>, open: @escaping (DemoSelection) -> Void) {
            self.store = store
            self.open = open
        }

        public var body: some View {
            VStack(alignment: .leading, spacing: 10) {
                Text(verbatim: "TeleOCR").font(.headline)
                Text("Read text and tables from a document image", bundle: .module).foregroundStyle(.secondary)
                if let artifact = store.state.selectedArtifact(for: "teleocr") {
                    let installation = store.state.installations[artifact.id] ?? ModelInstallation()
                    Text(verbatim: artifact.title).font(.caption)
                    Text("Import the evaluated FP16 model folder.", bundle: .module).font(.callout)
                    controls(artifact, installation)
                    if let failure = installation.failure { failureText(failure) }
                }
                if let failure = store.state.failure { failureText(failure) }
            }
            .padding(.vertical, 4)
            .fileImporter(isPresented: $importing, allowedContentTypes: [.folder]) { result in
                guard let artifact = store.state.selectedArtifact(for: "teleocr") else { return }
                switch result {
                case .success(let folder): store.send(.importRequested(artifact.id, folder))
                case .failure(let error):
                    if (error as? CocoaError)?.code != .userCancelled {
                        store.send(.importSelectionFailed(artifact.id, error.localizedDescription))
                    }
                }
            }
            .confirmationDialog(
                Text("Remove the imported model?", bundle: .module), isPresented: $deleting, titleVisibility: .visible
            ) {
                Button(role: .destructive) {
                    if let artifact = store.state.selectedArtifact(for: "teleocr") {
                        store.send(.deleteRequested(artifact.id))
                    }
                } label: {
                    Text("Remove Model", bundle: .module)
                }
                Button(role: .cancel) {
                } label: {
                    Text("Cancel", bundle: .module)
                }
            } message: {
                Text("This removes the app’s imported copy of TeleOCR.", bundle: .module)
            }
        }

        @ViewBuilder private func controls(_ artifact: ModelArtifact, _ installation: ModelInstallation) -> some View {
            switch installation.phase {
            case .missing, .failed:
                Button {
                    importing = true
                } label: {
                    Text("Import Model…", bundle: .module)
                }
            case .ready:
                HStack {
                    Label {
                        Text("Imported", bundle: .module)
                    } icon: {
                        Image(systemName: "checkmark.circle")
                    }
                    if let bytes = artifact.sizeBytes {
                        Text(bytes.formatted(.byteCount(style: .file))).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button {
                        open(TeleOCRDemoCatalog.selection)
                    } label: {
                        Text("Open OCR", bundle: .module)
                    }
                    .buttonStyle(.borderedProminent).disabled(store.state.pendingDestination != nil)
                    Button {
                        importing = true
                    } label: {
                        Text("Import Again…", bundle: .module)
                    }
                    .disabled(store.state.session != nil)
                    Button(role: .destructive) {
                        deleting = true
                    } label: {
                        Text("Remove Model", bundle: .module)
                    }
                    .disabled(store.state.session != nil)
                }
            case .importing(let progress), .downloading(let progress):
                HStack {
                    ProgressView(value: progress?.fraction) { Text("Importing…", bundle: .module) }
                    Button {
                        store.send(.cancelRequested(artifact.id))
                    } label: {
                        Text("Cancel", bundle: .module)
                    }
                }
            case .inspecting, .validating:
                ProgressView { Text("Checking model…", bundle: .module) }
            case .stopping, .deleting:
                ProgressView { Text("Finishing model operation…", bundle: .module) }
            }
        }

        private func failureText(_ failure: ModelLibraryFailure) -> some View {
            let message: String
            switch failure {
            case .operation(let detail): message = detail
            case .modelInUse: message = "Close the OCR demo before replacing or removing its model."
            default: message = "The model is not available. Check its folder and try importing again."
            }
            return Text(LocalizedStringKey(message), bundle: .module).font(.callout).foregroundStyle(.red)
                .textSelection(.enabled)
        }
    }
#endif
