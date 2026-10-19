import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTransfer

public struct ModelLibraryAssembly {
    public struct Library {
        public let store: Store<ModelLibraryFeature>
        public let rendering: ModelLibraryRendering
    }

    public let registry: ModelDemoAdapterRegistry
    public let runner: ModelLibraryRunner

    public init(
        adapters: [any ModelDemoAdapter], store: ManagedModelStore, transfer: ArtifactTransfer,
        migrationCandidates: @escaping @Sendable () -> [URL] = { [] }
    ) {
        let registry = ModelDemoAdapterRegistry()
        adapters.forEach(registry.register)
        self.registry = registry
        runner = ModelLibraryRunner(
            resources: ModelLibraryResourceRunner(
                store: store, transfer: transfer, migrationCandidates: migrationCandidates),
            registry: registry, store: store)
    }

    public func makeLibrary(
        catalog: ModelCatalog, capabilities: [DemoCapability: CapabilityState], owner: UUID
    ) -> Library {
        let environment = LibraryEnvironment(
            platform: .iOS, connectedDemos: registry.connectedDemos, capabilities: capabilities)
        let library = Store<ModelLibraryFeature>(
            initialState: ModelLibraryState(catalog: catalog, environment: environment, owner: owner),
            run: runner.run)
        return Library(
            store: library, rendering: ModelLibraryRendering(registry: registry, runner: runner, library: library))
    }
}
