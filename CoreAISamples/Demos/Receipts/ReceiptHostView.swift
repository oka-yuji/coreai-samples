#if os(macOS)
    import FeatureStore
    import ModelLibraryFeature
    import ModelLibraryUI
    import SwiftUI

    struct ReceiptHostView: View {
        let library: MacModelLibrary
        @State private var deletion: ModelDeletionRequest?

        var body: some View {
            Group {
                if let store = library.store, let rendering = library.rendering {
                    if store.state.pendingDestination != nil {
                        VStack(spacing: 12) {
                            ProgressView("Finishing the previous demo…")
                            if store.state.failure != nil {
                                Text("The demo could not finish stopping. Try again.")
                                Button("Try again") { store.send(.retryDemoStop) }
                            }
                        }
                    } else if let session = store.state.session, session.selection == ReceiptDemoCatalog.selection {
                        rendering.content(for: session)
                    } else {
                        welcome(store).modelDeletionAlert(store: store, deletion: $deletion)
                    }
                } else {
                    ContentUnavailableView {
                        Label("Receipt Desk is unavailable", systemImage: "exclamationmark.triangle")
                    } description: {
                        Text(LocalizedStringKey(library.problem ?? "The receipt catalog could not be loaded."))
                    }
                }
            }
            .demoNavigationTitle("JEV-27B")
            .onAppear { library.refresh() }
        }

        private func welcome(_ store: Store<ModelLibraryFeature>) -> some View {
            VStack(alignment: .leading, spacing: 22) {
                Label("Receipt Desk", systemImage: "receipt").font(.largeTitle.bold())
                Text("Sort receipts into a reviewed expense ledger").font(.title2)
                Text(
                    "Read receipt images and PDFs, review JEV’s category suggestions, then save a ledger with organized copies."
                )
                .foregroundStyle(.secondary).frame(maxWidth: 540, alignment: .leading)
                HStack(spacing: 14) {
                    Label("Local OCR", systemImage: "doc.viewfinder")
                    Image(systemName: "arrow.right").foregroundStyle(.tertiary)
                    Label("JEV on Core AI", systemImage: "cpu")
                    Image(systemName: "arrow.right").foregroundStyle(.tertiary)
                    Label("Reviewed ledger", systemImage: "folder")
                }.font(.callout)
                Divider().frame(maxWidth: 540)
                if let model = store.state.catalog.model(ReceiptDemoCatalog.selection.modelID),
                    let artifact = store.state.selectedArtifact(for: model.id)
                {
                    ModelAcquisitionControls(
                        store: store, model: model,
                        requestDeletion: { deletion = ModelDeletionRequest(model: model, artifact: $0) },
                        showsActionTitle: true)
                    if let size = artifact.sizeBytes {
                        Text("Model download: \(size.formatted(.byteCount(style: .file)))").font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    if store.state.installations[artifact.id]?.phase == .ready {
                        Button("Open Receipt Desk") { store.send(.openDemo(ReceiptDemoCatalog.selection)) }
                            .buttonStyle(.borderedProminent).controlSize(.large)
                    }
                    if let failure = store.state.installations[artifact.id]?.failure {
                        LibraryFailureView(failure: failure)
                    }
                }
                Text("After downloading JEV, receipt processing stays on this Mac.").font(.callout).foregroundStyle(
                    .secondary)
                Text(
                    "Includes four synthetic sample receipts. Supports JPY receipts; you confirm the fields and categories."
                )
                .font(.caption).foregroundStyle(.secondary).frame(maxWidth: 540, alignment: .leading)
                Spacer()
            }.padding(32).frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
    }
#endif
