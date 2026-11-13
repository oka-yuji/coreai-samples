import Foundation
import Testing

@testable import ModelLibraryFeature

struct MobileImageLabArtifactTests {
    @Test("QD-02 The public inventory changes only the three declared metadata files")
    func inventory() throws {
        let artifact = MobileImageLabArtifact.published
        let legacy = QwenImage21Artifact.nativeComplex
        let url = try #require(
            Bundle.module.url(
                forResource: "qwen-native-complex-bundle", withExtension: "json", subdirectory: "Resources"))
        let original = try JSONDecoder().decode(Inventory.self, from: Data(contentsOf: url))
        let replacements = [
            ModelArtifactFile(
                path: "README.txt", bytes: 538,
                sha256: "26383ebfd2abdfbd07f9405a1e0553f1edf22758d753c8109890b837ece2a5f5"),
            ModelArtifactFile(
                path: "NOTICE.txt", bytes: 642,
                sha256: "7b42feee2cf48a78b2ad8051d269ac03f106b8fbb076f132770c80879b2586d1"),
            ModelArtifactFile(
                path: "bundle.json", bytes: 23_018,
                sha256: "9d61c180a8c72b67ba34e8ece6e842fec489dc98fc68b2b875fa07a1900dd3e5"),
        ]
        let changed = Set(replacements.map(\.path))
        #expect(artifact.files.count == 51)
        #expect(Set(artifact.files.map(\.path)) == Set(original.assets.keys).union(["bundle.json"]))
        #expect(artifact.sizeBytes == 13_783_495_057 && artifact.totalFileBytes == artifact.sizeBytes)
        #expect(artifact.files.filter { !legacy.files.contains($0) }.count == 3)
        for expected in replacements { #expect(artifact.files.first { $0.path == expected.path } == expected) }
        for file in artifact.files where !changed.contains(file.path) {
            let expected = try #require(original.assets[file.path])
            #expect(file.bytes == expected.bytes && file.sha256 == expected.sha256)
        }
        #expect(artifact.sourceKind == .localManifest && artifact.isSourceValid)
        #expect(artifact.sourceRevision == replacements.last?.sha256)
        #expect(artifact.localBundle?.profileID == QwenImage21Artifact.profileID)
        #expect(artifact.contract == legacy.contract && artifact.upstream == legacy.upstream)
        #expect(
            artifact.download
                == ModelDownload(
                    repository: "okayuji/mobile-image-lab-coreai", revision: "fdd2b8cb17f48976207039f1f780360ba60275dc",
                    root: "bundle"))
        #expect(ModelAcquisitionPolicy(artifact: artifact).preferredMethod == .download)
        #expect(ModelAcquisitionPolicy(artifact: artifact).allowsImport)
        #expect(ModelAcquisitionPolicy(artifact: legacy).preferredMethod == .importBundle)
    }

    @Test("QD-04 Selecting the retained legacy installation never starts a download or migration")
    func legacySelection() throws {
        let catalog = try IOSDemoCatalog.make()
        let model = try #require(catalog.model("qwen-image21-turbo"))
        let legacy = QwenImage21Artifact.nativeComplex
        let resource = ValidatedModelResource(
            modelID: model.id, artifactID: legacy.id, contract: legacy.contract,
            revision: QwenImage21Artifact.manifestSHA256, upstreamRevision: legacy.upstream?.revision,
            directory: URL(filePath: "/synthetic/legacy"), sizeBytes: 13_783_495_099, sourceKind: .localManifest)
        var state = ModelLibraryState(catalog: catalog, environment: LibraryEnvironment(platform: .iOS), owner: UUID())
        state.installations[legacy.id]?.phase = .ready
        state.installations[legacy.id]?.resource = resource
        #expect(state.selectedArtifact(for: model.id) == MobileImageLabArtifact.published)
        #expect(ModelLibraryFeature.reduce(&state, .selectArtifact(modelID: model.id, artifactID: legacy.id)).isEmpty)
        #expect(state.selectedArtifact(for: model.id) == legacy && state.session == nil)
        #expect(state.installations[legacy.id]?.resource == resource)
        state.query = "Qwen"
        #expect(state.sections.map(\.id).contains(model.id))
    }

    @Test("QD-06 Image connection diagnostics resolve only the two explicit complete identities")
    func connectionTargets() throws {
        for artifact in [MobileImageLabArtifact.published, QwenImage21Artifact.nativeComplex] {
            let expected = ModelArtifactIdentity(
                modelID: "qwen-image21-turbo", artifactID: artifact.id, contract: "qwen-image21-mobile-v2",
                revision: try #require(artifact.sourceRevision), sourceKind: .localManifest)
            #expect(IOSDemoCatalog.imageGenerationIdentity(for: artifact.id) == expected)
        }
        for target: String? in [nil, "", "all", "qwen-image21-turbo", "unknown", MobileImageLabArtifact.manifestSHA256]
        {
            #expect(IOSDemoCatalog.imageGenerationIdentity(for: target) == nil)
        }
    }

    @Test("QD-04 Synthetic public completion cannot bypass cancellation or delete a live image session")
    func cancellationAndDeletion() throws {
        let catalog = try IOSDemoCatalog.make()
        let model = try #require(catalog.model("qwen-image21-turbo"))
        let artifact = MobileImageLabArtifact.published
        let resource = ValidatedModelResource(
            modelID: model.id, artifactID: artifact.id, contract: artifact.contract,
            revision: MobileImageLabArtifact.manifestSHA256, upstreamRevision: artifact.upstream?.revision,
            directory: URL(filePath: "/synthetic/public-image"), sizeBytes: 13_783_495_057, sourceKind: .localManifest)
        var state = ModelLibraryState(
            catalog: catalog, environment: LibraryEnvironment(platform: .iOS, connectedDemos: ["image-generation"]),
            owner: UUID())
        let download = ModelLibraryFeature.reduce(&state, .downloadRequested(artifact.id))
        let request = try #require(state.installations[artifact.id]?.requestID)
        #expect(download == [.run(.download(model, artifact, request), id: .resource(artifact.id))])
        let cancellation = ModelLibraryFeature.reduce(&state, .cancelRequested(artifact.id))
        #expect(cancellation.first == .cancel(.resource(artifact.id)))
        #expect(state.installations[artifact.id]?.phase == .stopping)
        let stopping = state
        #expect(ModelLibraryFeature.reduce(&state, .downloaded(artifact.id, request, resource.directory)).isEmpty)
        #expect(ModelLibraryFeature.reduce(&state, .validated(artifact.id, request, resource)).isEmpty)
        #expect(state == stopping)
        #expect(ModelLibraryFeature.reduce(&state, .transferStopped(artifact.id, request)).isEmpty)
        #expect(state.installations[artifact.id]?.phase == .missing)
        _ = ModelLibraryFeature.reduce(&state, .downloadRequested(artifact.id))
        let retry = try #require(state.installations[artifact.id]?.requestID)
        #expect(retry != request)
        _ = ModelLibraryFeature.reduce(&state, .downloaded(artifact.id, retry, resource.directory))
        #expect(ModelLibraryFeature.reduce(&state, .validated(artifact.id, retry, resource)).isEmpty)
        #expect(state.installations[artifact.id]?.resource == resource)
        let ready = state
        #expect(ModelLibraryFeature.reduce(&state, .deleteRequested(artifact.id)).count == 1)
        #expect(state.installations[artifact.id]?.phase == .deleting)
        state = ready
        _ = ModelLibraryFeature.reduce(&state, .openDemo(DemoSelection(modelID: model.id, demoID: "image-generation")))
        let session = try #require(state.session)
        #expect(session.resource == resource)
        #expect(ModelLibraryFeature.reduce(&state, .deleteRequested(artifact.id)).isEmpty)
        #expect(state.failure == .modelInUse && state.installations[artifact.id]?.resource == resource)
        #expect(state.installations[QwenImage21Artifact.nativeComplex.id] == ModelInstallation())
    }

    private struct Inventory: Decodable {
        struct File: Decodable {
            let bytes: Int64
            let sha256: String
        }
        let assets: [String: File]
    }
}
