#if os(macOS)
    import FeatureStore
    import Foundation
    import ModelLibraryFeature
    import ModelLibraryRuntime
    import ModelLibraryStorage
    import ModelLibraryTransfer
    import Observation
    import SwiftUI

    @MainActor
    private enum OCRMacServices {
        static let assembly = ModelLibraryAssembly(
            adapters: [OCRMacAdapter()],
            store: ManagedModelStore(root: ModelStorage.root.appending(path: "ModelLibrary")),
            transfer: ArtifactTransfer())
    }

    @MainActor
    @Observable
    final class OCRMacLibrary {
        let store: Store<ModelLibraryFeature>?
        let rendering: ModelLibraryRendering?
        let problem: String?

        init() {
            do {
                let catalog = try TeleOCRDemoCatalog.make(artifact: TeleOCRModelArtifact.make())
                let assembly = OCRMacServices.assembly
                try catalog.validateConnections(adapterIDs: Array(assembly.registry.connectedDemos))
                let environment = LibraryEnvironment(
                    platform: .macOS, connectedDemos: assembly.registry.connectedDemos, capabilities: [:])
                let store = Store<ModelLibraryFeature>(
                    initialState: ModelLibraryState(catalog: catalog, environment: environment, owner: UUID()),
                    run: assembly.runner.run)
                self.store = store
                rendering = ModelLibraryRendering(registry: assembly.registry, runner: assembly.runner, library: store)
                problem = nil
            } catch {
                store = nil
                rendering = nil
                problem = error.localizedDescription
            }
        }

        func refresh() {
            if store?.state.session == nil { store?.send(.refresh) }
        }

        func leave() { store?.send(.closeDemo) }

        func enter(_ phase: ScenePhase) {
            if phase == .background { leave() }
            if phase == .active { OCRMacServices.assembly.runner.retryAbandoned() }
        }

        func close() {
            guard let store else { return }
            leave()
            for (artifact, installation) in store.state.installations where installation.phase.isBusy {
                store.send(.cancelRequested(artifact))
            }
            OCRMacServices.assembly.runner.abandon(owner: store.state.owner)
        }
    }
#endif
