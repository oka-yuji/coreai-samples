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
        @Environment(\.locale) private var locale

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
                            if model.id == "irodori", store.state.artifacts(for: model.id).count > 1 {
                                Picker(
                                    selection: Binding(
                                        get: {
                                            store.state.selectedArtifact(for: model.id)?.id ?? model.defaultArtifactID
                                        },
                                        set: { store.send(.selectArtifact(modelID: model.id, artifactID: $0)) }
                                    )
                                ) {
                                    ForEach(store.state.artifacts(for: model.id), id: \.id) { item in
                                        libraryText(
                                            item.contract == "irodori-coreai-v1" ? "Text only (legacy)" : item.title
                                        )
                                        .tag(item.id)
                                    }
                                } label: {
                                    libraryText("Model")
                                }
                                .disabled(store.state.pendingDestination != nil)
                            }
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
                    .navigationBarBackButtonHidden()
                    .background(ModelNavigationGuard())
                    .accessibilityAction(.escape, returnToLibrary)
                    .navigationSubtitle(
                        store.state.variantTitle(for: model.id).map {
                            Text(
                                "Variant: \(String(localized: String.LocalizationValue($0), bundle: .module, locale: locale))",
                                bundle: .module)
                        }
                            ?? Text(verbatim: "")
                    )
                    .toolbar {
                        ToolbarItem(placement: .topBarLeading) {
                            Button(action: returnToLibrary) {
                                Label {
                                    libraryText("Back to Models")
                                } icon: {
                                    Image(systemName: "chevron.backward")
                                }
                            }
                            .disabled(store.state.pendingDestination != nil)
                        }
                        ToolbarItem(placement: .topBarTrailing) {
                            Button {
                                showingModel = true
                            } label: {
                                Label {
                                    libraryText("Model information")
                                } icon: {
                                    Image(systemName: "info.circle")
                                }
                            }
                        }
                    }
                    .sheet(isPresented: $showingModel) {
                        NavigationStack {
                            ModelInformationView(store: store, model: model)
                                .navigationTitle(Text(verbatim: model.name))
                                .toolbar {
                                    ToolbarItem(placement: .confirmationAction) {
                                        Button(role: .confirm) {
                                            showingModel = false
                                        } label: {
                                            Label {
                                                libraryText("Done")
                                            } icon: {
                                                Image(systemName: "checkmark")
                                            }
                                        }
                                    }
                                }
                        }
                    }
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

        private func returnToLibrary() {
            if store.state.selection == selection,
                ModelLibraryNavigation.resolve(.showSidebar, state: store.state).destination == .library
            {
                store.send(.closeDemo)
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
                    if artifact?.sourceKind == nil {
                        libraryText("Description only").font(.headline)
                        libraryText(
                            "Demo descriptions are available. Downloads and execution are not available in this build."
                        )
                        .font(.callout)
                    } else {
                        if selection.modelID == "irodori" {
                            libraryText("1. Prepare the model").font(.headline)
                            libraryText(
                                artifact?.download != nil
                                    ? "Download the Reference voice model to enable recording and audio files."
                                    : artifact?.contract == "irodori-coreai-v2"
                                        ? "Import the Reference voice model folder from Files to enable recording and audio files."
                                        : "Import the prepared text-only model folder from Files. Choose Reference voice to use recordings or audio files."
                            )
                            .font(.callout)
                            if let bytes = artifact?.sizeBytes {
                                Text("Model files: \(bytes.formatted(.byteCount(style: .file)))", bundle: .module)
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            if artifact?.localBundle != nil {
                                libraryText("Importing keeps the original folder and creates a separate app copy.")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            if let guide = URL(
                                string: "https://github.com/oka-yuji/coreai-samples/blob/main/docs/irodori/iphone.md")
                            {
                                Link(destination: guide) { libraryText("Model preparation guide") }
                            }
                        } else {
                            libraryText("You can read this demo now.").font(.headline)
                            libraryText(
                                store.state.isConnected(selection)
                                    ? artifact?.localBundle == nil
                                        ? "Download the model to enable its controls."
                                        : "Import the model to enable its controls."
                                    : "Demo execution is not connected in this build."
                            )
                            .font(.callout)
                        }
                        ModelAcquisitionControls(
                            store: store, model: model,
                            requestDeletion: { deletion = ModelDeletionRequest(model: model, artifact: $0) },
                            showsActionTitle: selection.modelID == "irodori")
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
                    if selection.modelID == "irodori" {
                        Label {
                            libraryText("1. Model ready")
                        } icon: {
                            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                        }
                        .font(.callout)
                    }
                    renderer.content(for: session)
                } else {
                    LibraryFailureView(failure: .unavailable)
                }
            }
        }
    }
#endif
