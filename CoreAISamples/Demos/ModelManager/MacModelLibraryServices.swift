#if os(macOS)
    import FeatureStore
    import Foundation
    import ModelLibraryFeature
    import ModelLibraryRuntime
    import ModelLibraryStorage
    import ModelLibraryTransfer

    @MainActor
    enum MacModelLibraryServices {
        static let root: URL = {
            let directory = ModelStorage.root.appending(path: "ModelLibrary")
            try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            ModelStorage.excludeFromBackup(directory)
            return directory
        }()
        static let unconnectedDemoIDs: Set<String> = [
            "chat", "decide", "recipes", "image-search", "video-search", "comparison", "speech",
        ]
        static let runtimeUses = Store<ModelRuntimeUses>(initialState: ModelRuntimeUses.State()) { _, _ in }

        static func acquire(_ path: String, modelID: String) async throws -> @MainActor () async -> Void {
            let catalog = try MacDemoCatalog.make()
            guard let model = catalog.model(modelID), let artifact = model.artifacts.first else {
                throw ManagedModelStoreError.invalidDefinition
            }
            let lease = try await storage.acquire(
                model: model, artifact: artifact, directory: URL(filePath: path), owner: UUID())
            runtimeUses.send(.acquired(modelID))
            return {
                await lease.release()
                runtimeUses.send(.released(modelID))
            }
        }

        static let storage = ManagedModelStore(root: root, removeMigratedSources: true)
        static let assembly: ModelLibraryAssembly = {
            let legacy = ModelStorage.directory(for: "okayuji/Qwen3.8-27B-coreai-4bit")
            return ModelLibraryAssembly(
                adapters: [OCRMacAdapter()], store: storage, transfer: ArtifactTransfer(),
                migrationCandidates: { [legacy] })
        }()
    }

#endif
