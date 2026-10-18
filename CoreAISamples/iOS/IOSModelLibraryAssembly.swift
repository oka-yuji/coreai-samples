#if os(iOS)
    import AVFoundation
    import FeatureStore
    import Foundation
    import ModelLibraryFeature
    import ModelLibraryRuntime
    import ModelLibraryStorage
    import ModelLibraryTransfer
    import Speech

    @MainActor
    enum IOSModelLibraryAssembly {
        struct Library {
            let store: Store<ModelLibraryFeature>
            let rendering: ModelLibraryRendering
        }

        private static let store = ManagedModelStore(root: ModelStorage.root.appending(path: "ModelLibrary"))
        private static let resources = ModelLibraryResourceRunner(
            store: store, transfer: ArtifactTransfer(), migrationCandidates: legacyFolders)

        static func make(catalog: ModelCatalog) -> Library {
            let registry = ModelDemoAdapterRegistry()
            let runner = ModelLibraryRunner(resources: resources, registry: registry, store: store)
            let environment = LibraryEnvironment(
                platform: .iOS, connectedDemos: registry.connectedDemos, capabilities: capabilities())
            let library = Store<ModelLibraryFeature>(
                initialState: ModelLibraryState(catalog: catalog, environment: environment, owner: UUID()),
                run: runner.run)
            return Library(
                store: library, rendering: ModelLibraryRendering(registry: registry, runner: runner, library: library))
        }

        static func capabilities() -> [DemoCapability: CapabilityState] {
            let camera: CapabilityState =
                switch AVCaptureDevice.authorizationStatus(for: .video) {
                case .authorized: .available
                case .notDetermined: .undetermined
                case .denied, .restricted: .denied
                @unknown default: .unsupported
                }
            let microphone: CapabilityState =
                switch AVAudioApplication.shared.recordPermission {
                case .granted: .available
                case .undetermined: .undetermined
                case .denied: .denied
                @unknown default: .unsupported
                }
            let speech: CapabilityState =
                switch SFSpeechRecognizer.authorizationStatus() {
                case .authorized: .available
                case .notDetermined: .undetermined
                case .denied, .restricted: .denied
                @unknown default: .unsupported
                }
            return [.camera: camera, .microphone: microphone, .speechRecognition: speech, .network: .available]
        }

        nonisolated private static func legacyFolders() -> [URL] {
            let documents = URL.documentsDirectory
            let items =
                (try? FileManager.default.contentsOfDirectory(at: documents, includingPropertiesForKeys: nil)) ?? []
            let gliner = items.filter {
                FileManager.default.fileExists(atPath: $0.appending(path: "contract.json").path(percentEncoded: false))
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            return gliner + ["embeddinggemma2", "embeddinggemma2-int4-int8"].map { documents.appending(path: $0) }
        }
    }
#endif
