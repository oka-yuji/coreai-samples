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
        @State private var preferredColumn: ModelLibraryNavigation.Column
        @State private var columnVisibility = NavigationSplitViewVisibility.all
        @Environment(\.locale) private var locale
        @Environment(\.horizontalSizeClass) private var horizontalSizeClass
        @ScaledMetric(relativeTo: .body) private var tileWidth = 88

        public init(store: Store<ModelLibraryFeature>, renderer: Renderer, leadingToolbar: AnyView? = nil) {
            self.store = store
            self.renderer = renderer
            self.leadingToolbar = leadingToolbar
            _preferredColumn = State(
                initialValue: ModelLibraryNavigation.resolve(.synchronize, state: store.state).column)
        }

        private var compactColumn: Binding<NavigationSplitViewColumn> {
            Binding(
                get: { preferredColumn == .sidebar ? .sidebar : .detail },
                set: { column in
                    if column == .sidebar, horizontalSizeClass == .compact {
                        navigate(.showSidebar)
                    } else {
                        preferredColumn = column == .sidebar ? .sidebar : .detail
                    }
                })
        }

        private var query: Binding<String> {
            Binding(get: { store.state.query }, set: { store.send(.queryChanged($0)) })
        }

        public var body: some View {
            NavigationSplitView(columnVisibility: $columnVisibility, preferredCompactColumn: compactColumn) {
                models
                    .navigationSplitViewColumnWidth(min: 240, ideal: 280)
                    .navigationTitle(Text(verbatim: "Models"))
                    .searchable(text: query, prompt: libraryText("Search models and demos"))
                    .toolbar {
                        if let leadingToolbar {
                            ToolbarItem(placement: .topBarLeading) { leadingToolbar }
                        }
                        DefaultToolbarItem(kind: .search, placement: .bottomBar)
                        ToolbarSpacer(.fixed, placement: .bottomBar)
                        ToolbarItem(placement: .bottomBar) {
                            Button {
                                showsAbout = true
                            } label: {
                                Label {
                                    libraryText("About this app")
                                } icon: {
                                    Image(systemName: "info")
                                }
                            }
                        }
                    }
            } detail: {
                if let selection = store.state.selection {
                    ModelDemoHost(store: store, renderer: renderer, selection: selection)
                        .id(selection)
                        .navigationSplitViewColumnWidth(min: 240, ideal: 320)
                } else {
                    ContentUnavailableView {
                        Label {
                            libraryText("Choose a demo for a model.")
                        } icon: {
                            Image(systemName: "square.grid.2x2")
                        }
                    } description: {
                        libraryText("You can read each demo before preparing its model.")
                    }
                    .navigationSplitViewColumnWidth(min: 240, ideal: 320)
                }
            }
            .navigationSplitViewStyle(.balanced)
            .onAppear { store.send(.refresh) }
            .onChange(of: store.state.selection) { _, _ in navigate(.synchronize) }
            .onChange(of: store.state.pendingDestination) { _, _ in navigate(.synchronize) }
            .onChange(of: horizontalSizeClass) { _, sizeClass in
                if sizeClass == .regular { columnVisibility = .all }
                navigate(.synchronize)
            }
            .sheet(isPresented: $showsAbout) {
                NavigationStack {
                    VStack(alignment: .leading, spacing: 18) {
                        libraryText("Explore converted models through practical demos.").font(.body)
                        libraryText("You can read each demo before preparing its model.").foregroundStyle(.secondary)
                    }
                    .padding()
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                    .navigationTitle(libraryText("About this app"))
                    .toolbar {
                        ToolbarItem(placement: .confirmationAction) {
                            Button(role: .confirm) {
                                showsAbout = false
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
                .presentationDetents([.medium, .large])
            }
            .modelDeletionAlert(store: store, deletion: $deletion)
        }

        private func navigate(_ intent: ModelLibraryNavigation.Intent) {
            let update = ModelLibraryNavigation.resolve(intent, state: store.state)
            preferredColumn = update.column
            if let destination = update.destination {
                switch destination {
                case .library: store.send(.closeDemo)
                case .demo(let selection): store.send(.openDemo(selection))
                }
                preferredColumn = ModelLibraryNavigation.resolve(.synchronize, state: store.state).column
            }
        }

        private var models: some View {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 26) {
                    VStack(alignment: .leading, spacing: 6) {
                        libraryText("Choose a demo for a model.").font(.body)
                        libraryText(
                            store.state.catalog.models.flatMap(\.artifacts).allSatisfy { $0.sourceKind == nil }
                                ? "Demo descriptions are available. Downloads and execution are not available in this build."
                                : "You can read each demo before preparing its model."
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
                        Text(
                            verbatim: store.state.rowDetail(
                                for: section.model,
                                classification: ModelClassificationPresentation.classification(
                                    for: section.model, locale: locale), locale: locale)
                        )
                        .font(.footnote).foregroundStyle(.secondary)
                    }
                    Spacer(minLength: 8)
                    ModelAcquisitionControls(store: store, model: section.model) {
                        deletion = ModelDeletionRequest(model: section.model, artifact: $0)
                    }
                }
                LazyVGrid(columns: [GridItem(.adaptive(minimum: tileWidth), spacing: 16)], spacing: 20) {
                    ForEach(section.demos) { demo in
                        Button {
                            navigate(.select(DemoSelection(modelID: section.id, demoID: demo.id)))
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
