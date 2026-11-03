#if os(macOS)
    import CoreAI
    import FeatureStore
    import Foundation
    import ModelLibraryFeature
    import ModelLibraryRuntime
    import ModelLibraryStorage
    import ModelLibraryTransfer
    import Observation
    import SwiftUI

    @MainActor
    enum MacModelLibraryServices {
        static let root = ModelStorage.root.appending(path: "ModelLibrary")
        static let unconnectedDemoIDs: Set<String> = [
            "chat", "decide", "recipes", "image-search", "video-search", "comparison", "speech",
        ]
        static let assembly = ModelLibraryAssembly(
            adapters: [OCRMacAdapter()], store: ManagedModelStore(root: root), transfer: ArtifactTransfer())
    }

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
#endif
