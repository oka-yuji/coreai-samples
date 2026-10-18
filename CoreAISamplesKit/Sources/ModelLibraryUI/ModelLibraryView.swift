#if os(iOS)
    import FeatureStore
    import ModelLibraryFeature
    import SwiftUI

    @MainActor
    public protocol ModelDemoRendering {
        func content(for session: ModelDemoSession) -> AnyView
        func preview(for selection: DemoSelection) -> AnyView
    }

    public struct ModelLibraryView<Renderer: ModelDemoRendering>: View {
        private let store: Store<ModelLibraryFeature>
        private let renderer: Renderer
        private let leadingToolbar: AnyView?
        @State private var deletion: ModelDeletionRequest?
        @State private var showsAbout = false
        @ScaledMetric(relativeTo: .body) private var tileWidth = 88

        public init(store: Store<ModelLibraryFeature>, renderer: Renderer, leadingToolbar: AnyView? = nil) {
            self.store = store
            self.renderer = renderer
            self.leadingToolbar = leadingToolbar
        }

        private var navigation: Binding<[DemoSelection]> {
            Binding(
                get: { store.state.selection.map { [$0] } ?? [] },
                set: { path in
                    if let selection = path.last { store.send(.openDemo(selection)) } else { store.send(.closeDemo) }
                })
        }

        private var query: Binding<String> {
            Binding(get: { store.state.query }, set: { store.send(.queryChanged($0)) })
        }

        public var body: some View {
            NavigationStack(path: navigation) {
                models
                    .navigationTitle(Text(verbatim: "Models"))
                    .searchable(text: query, prompt: libraryText("Search models and demos"))
                    .toolbar {
                        if let leadingToolbar {
                            ToolbarItem(placement: .topBarLeading) { leadingToolbar }
                                .sharedBackgroundVisibility(.hidden)
                        }
                        DefaultToolbarItem(kind: .search, placement: .bottomBar)
                        ToolbarSpacer(.fixed, placement: .bottomBar)
                        ToolbarItem(placement: .bottomBar) {
                            Button {
                                showsAbout = true
                            } label: {
                                Image(systemName: "info")
                            }
                            .frame(minWidth: 44, minHeight: 44)
                            .accessibilityLabel(libraryText("About this app"))
                        }
                    }
                    .navigationDestination(for: DemoSelection.self) { selection in
                        ModelDemoHost(store: store, renderer: renderer, selection: selection)
                    }
            }
            .onAppear { store.send(.refresh) }
            .sheet(isPresented: $showsAbout) {
                NavigationStack {
                    VStack(alignment: .leading, spacing: 18) {
                        libraryText("Explore converted models through practical demos.").font(.body)
                        libraryText("You can read each demo before downloading its model.").foregroundStyle(.secondary)
                    }
                    .padding()
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                    .navigationTitle(libraryText("About this app"))
                    .toolbar {
                        ToolbarItem(placement: .confirmationAction) {
                            Button {
                                showsAbout = false
                            } label: {
                                libraryText("Done")
                            }
                        }
                    }
                }
                .presentationDetents([.medium, .large])
            }
            .modelDeletionAlert(store: store, deletion: $deletion)
        }

        private var models: some View {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 26) {
                    VStack(alignment: .leading, spacing: 6) {
                        libraryText("Choose a demo for a model.").font(.body)
                        libraryText(
                            store.state.catalog.models.flatMap(\.artifacts).allSatisfy { $0.download == nil }
                                ? "Demo descriptions are available. Downloads and execution are not available in this build."
                                : "You can read each demo before downloading its model."
                        )
                        .font(.footnote).foregroundStyle(.secondary)
                    }
                    if store.state.sections.isEmpty {
                        if store.state.query.isEmpty {
                            ContentUnavailableView {
                                libraryText("No models available")
                            } description: {
                                libraryText("No published models support this device yet.")
                            }
                        } else {
                            ContentUnavailableView.search(text: store.state.query)
                        }
                    }
                    ForEach(store.state.sections) { section in
                        modelSection(section)
                    }
                    if let failure = store.state.failure { LibraryFailureView(failure: failure) }
                }
                .padding()
            }
            .disabled(store.state.pendingDestination != nil)
        }

        private func modelSection(_ section: ModelSection) -> some View {
            VStack(alignment: .leading, spacing: 18) {
                HStack(alignment: .firstTextBaseline) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(verbatim: section.model.name).font(.headline)
                        Text(verbatim: classification(section.model)).font(.footnote).foregroundStyle(.secondary)
                    }
                    Spacer(minLength: 8)
                    ModelAcquisitionControls(store: store, model: section.model) {
                        deletion = ModelDeletionRequest(model: section.model, artifact: $0)
                    }
                }
                LazyVGrid(columns: [GridItem(.adaptive(minimum: tileWidth), spacing: 16)], spacing: 20) {
                    ForEach(section.demos) { demo in
                        Button {
                            store.send(.openDemo(DemoSelection(modelID: section.id, demoID: demo.id)))
                        } label: {
                            ModelDemoTile(demo: demo)
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel(Text(LocalizedStringKey(demo.title)))
                        .accessibilityValue(Text(verbatim: section.model.name))
                    }
                }
                if let artifact = store.state.selectedArtifact(for: section.id),
                    let failure = store.state.installations[artifact.id]?.failure
                {
                    LibraryFailureView(failure: failure)
                }
                Divider()
            }
            .accessibilityElement(children: .contain)
        }

        private func classification(_ model: LibraryModel) -> String {
            let size = store.state.selectedArtifact(for: model.id).flatMap {
                store.state.installations[$0.id]?.resource?.sizeBytes ?? $0.sizeBytes
            }
            return [model.classification, size.map { $0.formatted(.byteCount(style: .file)) }]
                .compactMap { $0 }.joined(separator: " · ")
        }
    }

    private struct ModelDemoTile: View {
        let demo: DemoDescriptor
        @ScaledMetric(relativeTo: .title) private var iconSize = 64

        var body: some View {
            VStack(spacing: 8) {
                Image(systemName: demo.systemImage)
                    .font(.title2)
                    .frame(width: iconSize, height: iconSize)
                    .foregroundStyle(.tint)
                    .background(Color.accentColor.opacity(0.12), in: RoundedRectangle(cornerRadius: 16))
                    .accessibilityHidden(true)
                Text(LocalizedStringKey(demo.title)).font(.footnote).multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, minHeight: 44)
            .contentShape(Rectangle())
        }
    }
#endif
