#if os(macOS)
    import ChronosUI
    import CoreAI
    import FeatureStore
    import ForecastUI
    import Foundation
    import ModelLibraryFeature
    import ModelLibraryRuntime
    import ModelLibraryStorage
    import ModelLibraryTransfer
    import ModelLibraryUI
    import Observation
    import SwiftUI

    typealias MacModelChoice = Store<MacModelSelection>

    @MainActor
    @Observable
    final class MacModelLibrary {
        let store: Store<ModelLibraryFeature>?
        let rendering: ModelLibraryRendering?
        let problem: String?
        private(set) var downloadedSize: Int64 = 0
        private(set) var availableSpace: Int64 = 0

        init() {
            do {
                let catalog = try MacDemoCatalog.make()
                let assembly = MacModelLibraryServices.assembly
                try catalog.validateConnections(
                    adapterIDs: Array(assembly.registry.connectedDemos),
                    unconnectedDemoIDs: MacModelLibraryServices.unconnectedDemoIDs)
                let library = assembly.makeLibrary(catalog: catalog, platform: .macOS, capabilities: [:], owner: UUID())
                store = library.store
                rendering = library.rendering
                problem = nil
            } catch {
                store = nil
                rendering = nil
                problem = error.localizedDescription
            }
        }

        static func clearPrepared(_ directory: URL, artifact: ModelArtifact) throws {
            let assets = Set(
                artifact.files.compactMap { file -> String? in
                    let parts = file.path.split(separator: "/")
                    return parts.firstIndex { $0.hasSuffix(".aimodel") }.map { parts[...$0].joined(separator: "/") }
                })
            for asset in assets.sorted() {
                try AIModelCache.default.deleteEntries(
                    for: directory.appending(path: asset, directoryHint: .isDirectory))
            }
            ModelStorage.prepared.remove(ModelStorage.key(directory))
        }

        func refresh() {
            if store?.state.session == nil { store?.send(.refresh) }
            measureStorage()
        }

        func measureStorage() {
            downloadedSize =
                ModelStorage.size(of: MacModelLibraryServices.root) + ModelStorage.size(of: ModelStorage.modelsRoot)
            availableSpace = ModelStorage.availableSpace()
        }

        func use(_ model: DemoModel) { store?.send(.useRequested(model.libraryID)) }

        func stopUsing(_ model: DemoModel) { store?.send(.useEnded(model.libraryID)) }

        func isInUse(_ model: DemoModel) -> Bool {
            store?.state.usages[model.libraryID]?.isActive == true
                || MacModelLibraryServices.runtimeUses.state.counts[model.libraryID, default: 0] > 0
        }

        func resource(inUse model: DemoModel) -> ValidatedModelResource? {
            guard let resource = store?.state.usages[model.libraryID]?.resource, resource == installed(model) else {
                return nil
            }
            return resource
        }

        func installed(_ model: DemoModel) -> ValidatedModelResource? {
            guard let state = store?.state, let artifact = state.selectedArtifact(for: model.libraryID) else {
                return nil
            }
            return state.installations[artifact.id]?.resource
        }

        func artifactTitle(_ model: DemoModel) -> String? {
            store?.state.selectedArtifact(for: model.libraryID)?.title
        }

        func leave() { store?.send(.closeDemo) }

        func enter(_ phase: ScenePhase) {
            if phase == .background { leave() }
            if phase == .active { MacModelLibraryServices.assembly.runner.retryAbandoned() }
        }

        func close() {
            guard let store else { return }
            leave()
            for (artifact, installation) in store.state.installations where installation.phase.isBusy {
                store.send(.cancelRequested(artifact))
            }
            MacModelLibraryServices.assembly.runner.abandon(owner: store.state.owner)
        }
    }

    struct ChronosPreviewView: View {
        let library: MacModelLibrary
        var selection = ChronosDemoCatalog.selection
        @State private var deletion: ModelDeletionRequest?
        @Environment(\.locale) private var locale

        var body: some View {
            Group {
                if let store = library.store, let rendering = library.rendering {
                    if store.state.pendingDestination != nil {
                        VStack {
                            ProgressView("Finishing the previous demo…")
                            if store.state.failure != nil {
                                Button("Try again") { store.send(.retryDemoStop) }
                            }
                        }
                    } else if let session = store.state.session, session.selection == selection {
                        rendering.content(for: session)
                    } else {
                        ScrollView {
                            VStack(spacing: 0) {
                                if let model = store.state.catalog.model(selection.modelID) {
                                    if let demo = store.state.catalog.demo(selection.demoID) {
                                        VStack(alignment: .leading, spacing: 8) {
                                            Text(
                                                verbatim: ModelClassificationPresentation.classification(
                                                    for: model, locale: locale)
                                            ).font(.footnote).foregroundStyle(.secondary)
                                            Text(LocalizedStringKey(demo.summary)).foregroundStyle(.secondary)
                                            DisclosureGroup("About this demo") {
                                                VStack(alignment: .leading, spacing: 12) {
                                                    ForEach(Array(demo.explanation.enumerated()), id: \.offset) {
                                                        _, section in
                                                        VStack(alignment: .leading, spacing: 5) {
                                                            Text(LocalizedStringKey(section.title)).font(.headline)
                                                            Text(LocalizedStringKey(section.text))
                                                        }
                                                    }
                                                }.padding(.vertical, 8)
                                            }
                                        }
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .fixedSize(horizontal: false, vertical: true)
                                        .padding(.horizontal)
                                    }
                                    HStack {
                                        ModelAcquisitionControls(
                                            store: store, model: model,
                                            requestDeletion: {
                                                deletion = ModelDeletionRequest(model: model, artifact: $0)
                                            },
                                            showsActionTitle: true)
                                        if let artifact = store.state.selectedArtifact(for: model.id),
                                            store.state.installations[artifact.id]?.phase == .ready
                                        {
                                            Button("Open on-device forecast") {
                                                store.send(.openDemo(selection))
                                            }
                                            .buttonStyle(.borderedProminent)
                                        }
                                    }.padding()
                                    if let artifact = store.state.selectedArtifact(for: model.id),
                                        let failure = store.state.installations[artifact.id]?.failure
                                    {
                                        LibraryFailureView(failure: failure)
                                    }
                                }
                                if selection == ChronosDemoCatalog.salesSelection {
                                    ForecastPreview(scrollsContent: false)
                                } else {
                                    ChronosPreview(scrollsContent: false)
                                }
                            }
                        }.modelDeletionAlert(store: store, deletion: $deletion)
                    }
                } else {
                    ContentUnavailableView {
                        Label("Chronos-2", systemImage: "exclamationmark.triangle")
                    } description: {
                        LibraryFailureView(failure: .unavailable)
                    }
                }
            }
            .demoNavigationTitle("Chronos-2")
            .onAppear { library.refresh() }
            .onDisappear { library.leave() }
        }
    }
#endif
