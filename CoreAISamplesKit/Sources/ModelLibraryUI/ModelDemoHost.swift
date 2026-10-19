#if os(iOS)
    import FeatureStore
    import ModelLibraryFeature
    import SwiftUI

    struct ModelDemoHost<Renderer: ModelDemoRendering>: View {
        let store: Store<ModelLibraryFeature>
        let renderer: Renderer
        let selection: DemoSelection
        @State private var explanationExpanded = false
        @State private var showingModel = false
        @State private var deletion: ModelDeletionRequest?

        private var model: LibraryModel? { store.state.catalog.model(selection.modelID) }
        private var demo: DemoDescriptor? { store.state.catalog.demo(selection.demoID) }
        private var artifact: ModelArtifact? { store.state.selectedArtifact(for: selection.modelID) }

        var body: some View {
            Group {
                if let model, let demo {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 20) {
                            Text(verbatim: model.classification).font(.footnote).foregroundStyle(.secondary)
                            Text(LocalizedStringKey(demo.title)).font(.title3.bold())
                            Text(LocalizedStringKey(demo.summary)).foregroundStyle(.secondary)
                            DisclosureGroup(isExpanded: $explanationExpanded) {
                                VStack(alignment: .leading, spacing: 15) {
                                    ForEach(Array(demo.explanation.enumerated()), id: \.offset) { _, section in
                                        VStack(alignment: .leading, spacing: 5) {
                                            Text(LocalizedStringKey(section.title)).font(.headline)
                                            Text(LocalizedStringKey(section.text)).font(.body)
                                        }
                                    }
                                }
                                .padding(.vertical, 12)
                            } label: {
                                libraryText("About this demo")
                            }
                            contents(model: model)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                    }
                    .navigationTitle(Text(verbatim: model.name))
                    .navigationSubtitle(
                        store.state.variantTitle(for: model.id).map { Text("Variant: \($0)", bundle: .module) }
                            ?? Text(verbatim: "")
                    )
                    .toolbar {
                        ToolbarItem(placement: .topBarTrailing) {
                            Button {
                                showingModel = true
                            } label: {
                                Image(systemName: "info.circle")
                            }
                            .frame(minWidth: 44, minHeight: 44)
                            .accessibilityLabel(libraryText("Model information"))
                        }
                    }
                    .sheet(isPresented: $showingModel) { modelInformation(model) }
                    .modelDeletionAlert(store: store, deletion: $deletion)
                } else {
                    ContentUnavailableView {
                        libraryText("Demo unavailable")
                    } description: {
                        libraryText("This demo is not available on this device.")
                    }
                }
            }
        }

        @ViewBuilder
        private func contents(model: LibraryModel) -> some View {
            switch store.state.access(to: selection) {
            case .unavailable:
                LibraryFailureView(failure: .unavailable)
            case .stopping:
                if let failure = store.state.failure {
                    LibraryFailureView(failure: failure)
                    Button {
                        store.send(.retryDemoStop)
                    } label: {
                        libraryText("Retry stopping")
                    }
                    .buttonStyle(.borderedProminent)
                } else {
                    ProgressView { libraryText("Stopping the previous demo…") }
                }
            case .preview:
                VStack(alignment: .leading, spacing: 12) {
                    if artifact?.download == nil {
                        libraryText("Description only").font(.headline)
                        libraryText(
                            "Demo descriptions are available. Downloads and execution are not available in this build."
                        )
                        .font(.callout)
                    } else {
                        libraryText("You can read this demo now.").font(.headline)
                        libraryText(
                            store.state.isConnected(selection)
                                ? "Download the model to enable its controls."
                                : "Demo execution is not connected in this build."
                        )
                        .font(.callout)
                        ModelAcquisitionControls(store: store, model: model) {
                            deletion = ModelDeletionRequest(model: model, artifact: $0)
                        }
                    }
                    if let artifact, let failure = store.state.installations[artifact.id]?.failure {
                        LibraryFailureView(failure: failure)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 16))
                renderer.preview(for: selection).disabled(true).allowsHitTesting(false)
            case .ready(let session):
                if store.state.permits(session.id, selection: selection) {
                    renderer.content(for: session)
                } else {
                    LibraryFailureView(failure: .unavailable)
                }
            }
        }

        private func modelInformation(_ model: LibraryModel) -> some View {
            let artifacts = store.state.artifacts(for: model.id)
            return NavigationStack {
                Form {
                    LabeledContent {
                        Text(verbatim: model.classification)
                    } label: {
                        libraryText("Category")
                    }
                    if artifacts.count > 1 || (artifact == nil && !artifacts.isEmpty) {
                        Picker(
                            selection: Binding(
                                get: { store.state.selectedArtifacts[model.id] ?? "" },
                                set: { store.send(.selectArtifact(modelID: model.id, artifactID: $0)) })
                        ) {
                            ForEach(artifacts, id: \.id) { Text(verbatim: $0.title).tag($0.id) }
                        } label: {
                            libraryText("Variant")
                        }
                        .disabled(store.state.pendingDestination != nil)
                    } else if let artifact, artifact.download != nil {
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
                    }
                }
                .navigationTitle(Text(verbatim: model.name))
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button {
                            showingModel = false
                        } label: {
                            libraryText("Done")
                        }
                    }
                }
            }
        }
    }
#endif
