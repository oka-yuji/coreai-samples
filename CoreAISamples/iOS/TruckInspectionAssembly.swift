#if os(iOS)
    import FeatureStore
    import Foundation
    import TruckInspectionFeature
    import TruckInspectionRuntime

    @MainActor
    enum TruckInspectionAssembly {
        static func makeStore(assetFolders: [URL]) throws -> Store<TruckInspectionFeature> {
            let catalog = try InspectionCatalogLoader.load()
            let model = LiveTruckInspectionModel(
                automaticPlacement: LaunchArguments.value("--compute-units") == "default")
            return Store(
                initialState: InspectionState(catalog: catalog, modelFolder: assetFolders.first),
                run: TruckInspectionRunner(model: model, speech: LiveInspectionSpeech()).run)
        }
    }
#endif
