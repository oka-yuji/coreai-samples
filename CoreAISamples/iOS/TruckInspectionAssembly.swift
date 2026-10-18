#if os(iOS)
    import FeatureStore
    import Foundation
    import ModelLibraryFeature
    import ModelLibraryRuntime
    import TruckInspectionFeature
    import TruckInspectionRuntime

    @MainActor
    enum TruckInspectionAssembly {
        static func modelFolders(_ candidates: [URL]) -> [URL] {
            InspectionCatalogLoader.modelFolders(in: candidates)
        }

        static func makeStore(assetFolders: [URL]) throws -> Store<TruckInspectionFeature> {
            let catalog = try InspectionCatalogLoader.load()
            let model = LiveTruckInspectionModel(
                automaticPlacement: LaunchArguments.value("--compute-units") == "default")
            return Store(
                initialState: InspectionState(catalog: catalog, modelFolder: assetFolders.first),
                run: TruckInspectionRunner(model: model, speech: LiveInspectionSpeech()).run)
        }

        static func makeManagedStore(
            resource: ValidatedModelResource, gate: DemoExecutionGate, retaining previous: InspectionState?
        ) throws -> (store: Store<TruckInspectionFeature>, runner: TruckInspectionRunner) {
            let identity = resource.identity
            var state = InspectionState(
                catalog: try previous?.catalog ?? InspectionCatalogLoader.load(), modelFolder: resource.directory,
                execution: .managed, artifact: "\(identity.modelID)/\(identity.artifactID)@\(identity.revision)")
            if let previous { state.retain(previous) }
            let model = LiveTruckInspectionModel(
                automaticPlacement: LaunchArguments.value("--compute-units") == "default")
            let runner = TruckInspectionRunner(model: model, speech: LiveInspectionSpeech(), permits: gate.isPermitted)
            return (Store(initialState: state, run: runner.run), runner)
        }
    }
#endif
