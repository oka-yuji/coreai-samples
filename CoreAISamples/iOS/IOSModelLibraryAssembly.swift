#if os(iOS)
    import AVFoundation
    import FeatureStore
    import Foundation
    import ModelLibraryFeature
    import ModelLibraryRuntime
    import ModelLibraryStorage
    import ModelLibraryTransfer
    import Speech
    import SwiftUI

    @MainActor
    enum IOSModelLibraryAssembly {
        struct Library {
            let store: Store<ModelLibraryFeature>
            let rendering: ModelLibraryRendering
        }

        private static let store = ManagedModelStore(root: ModelStorage.root.appending(path: "ModelLibrary"))
        private static let registry = ModelDemoAdapterRegistry()
        private static let runner = ModelLibraryRunner(
            resources: ModelLibraryResourceRunner(
                store: store, transfer: ArtifactTransfer(), migrationCandidates: legacyFolders),
            registry: registry, store: store)

        static func make(catalog: ModelCatalog) -> Library {
            let environment = LibraryEnvironment(
                platform: .iOS, connectedDemos: registry.connectedDemos, capabilities: capabilities())
            let library = Store<ModelLibraryFeature>(
                initialState: ModelLibraryState(catalog: catalog, environment: environment, owner: UUID()),
                run: runner.run)
            return Library(
                store: library, rendering: ModelLibraryRendering(registry: registry, runner: runner, library: library))
        }

        static func enter(_ phase: ScenePhase, library: Store<ModelLibraryFeature>) {
            if phase == .active { runner.retryAbandoned() }
            for action in ModelLibraryRendering.actions(entering: phase, capabilities: capabilities()) {
                library.send(action)
            }
        }

        static func close(_ library: Store<ModelLibraryFeature>) {
            library.send(.suspendDemo)
            runner.abandon(owner: library.state.owner)
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
