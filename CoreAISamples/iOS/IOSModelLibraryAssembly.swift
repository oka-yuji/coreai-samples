#if os(iOS)
    import AVFoundation
    import FeatureStore
    import Foundation
    import ImageGenerationRuntime
    import ImageGenerationUI
    import ModelLibraryFeature
    import ModelLibraryRuntime
    import ModelLibraryStorage
    import ModelLibraryTransfer
    import Speech
    import SwiftUI

    @MainActor
    enum IOSModelLibraryAssembly {
        static let adapters: [any ModelDemoAdapter] = {
            var result: [any ModelDemoAdapter] = [
                ProductComparisonAdapter(), ExtractionAdapter(), InspectionAdapter(), ImageSearchAdapter(),
                VideoSearchAdapter(), ObjectFinderAdapter(), SpeechAdapter(),
            ]
            if (try? IOSDemoCatalog.make().demos.contains { $0.id == "image-generation" }) == true {
                result.append(
                    ImageGenerationAdapter { session, permission in
                        guard let budget = ImageGenerationDeviceProfiles.budget(for: session.resource) else {
                            return nil
                        }
                        return LiveImageGenerationService(session: session, budget: budget, isPermitted: permission)
                    })
            }
            return result
        }()
        private static let root: URL = {
            let root = ModelStorage.root.appending(path: "ModelLibrary")
            try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            ModelStorage.excludeFromBackup(root)
            return root
        }()
        private static let assembly = ModelLibraryAssembly(
            adapters: adapters, store: ManagedModelStore(root: root), transfer: ArtifactTransfer(),
            migrationCandidates: legacyFolders)

        static func catalog(searchTerms: [String: [String]] = [:]) throws -> ModelCatalog {
            try IOSDemoCatalog.make(searchTerms: searchTerms, speechArtifact: IrodoriModelArtifact.make())
        }

        static func make(catalog: ModelCatalog) -> ModelLibraryAssembly.Library {
            let previews = Set(["image-generation"]).intersection(catalog.demos.map(\.id))
                .subtracting(adapters.map(\.demoID))
            do {
                try catalog.validateConnections(adapterIDs: adapters.map(\.demoID), unconnectedDemoIDs: previews)
            } catch {
                preconditionFailure("Invalid model demo connections: \(error)")
            }
            return assembly.makeLibrary(catalog: catalog, capabilities: capabilities(), owner: UUID())
        }

        static func enter(_ phase: ScenePhase, library: Store<ModelLibraryFeature>) {
            if phase == .active { assembly.runner.retryAbandoned() }
            for action in ModelLibraryRendering.actions(entering: phase, capabilities: capabilities()) {
                library.send(action)
            }
        }

        static func close(_ library: Store<ModelLibraryFeature>) {
            library.send(.suspendDemo)
            assembly.runner.abandon(owner: library.state.owner)
        }

        static func runSelfTest(acquire target: String? = nil) async -> Int32 {
            guard let output = LaunchArguments.value("--model-library-selftest") else { return 2 }
            let folder = LaunchArguments.value("--import-local-bundle").map {
                URL(filePath: $0, relativeTo: .documentsDirectory)
            }
            let definitions = Result { try catalog() }
            let previews = (try? definitions.get().demos.map(\.id)) ?? []
            return await ModelLibrarySelfTest.run(
                output: URL(filePath: output, relativeTo: .documentsDirectory), root: root,
                registry: assembly.registry, capabilities: capabilities(), acquire: target, assembly: assembly,
                importBundle: folder,
                unconnectedDemoIDs: Set(["image-generation"]).intersection(previews).subtracting(
                    adapters.map(\.demoID)),
                catalog: { try definitions.get() })
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
