import Foundation
import ModelLibraryFeature
import Testing

@testable import ModelLibraryStorage

struct IrodoriDownloadTests {
    @Test("IRDL-01 The reference model downloads from its pinned release while legacy import remains available")
    func catalogDownload() async throws {
        let legacy = try IrodoriModelArtifact.make()
        let artifact = try IrodoriModelArtifact.makeReferenceDownload()
        let catalog = try IOSDemoCatalog.make(speechArtifact: legacy, referenceSpeechArtifact: artifact)
        let model = try #require(catalog.model("irodori"))
        var state = ModelLibraryState(catalog: catalog, environment: LibraryEnvironment(platform: .iOS), owner: UUID())
        #expect(state.selectedArtifact(for: model.id) == artifact)
        #expect(legacy.sourceKind == .localManifest && legacy.files.count == 14)
        #expect(artifact.sourceKind == .huggingFace && artifact.localBundle == nil)
        #expect(artifact.sourceRevision == "f5db5096365c7d1388a5002cfb3280f66cb19375")
        #expect(artifact.files.count == 22 && artifact.totalFileBytes == 3_835_396_408)
        #expect(artifact.files.contains { $0.path == "LICENSE" } && artifact.files.contains { $0.path == "NOTICE" })
        let effects = ModelLibraryFeature.reduce(&state, .downloadRequested(artifact.id))
        let request = try #require(state.installations[artifact.id]?.requestID)
        #expect(effects == [.run(.download(model, artifact, request), id: .resource(artifact.id))])
        #expect(state.installations[artifact.id]?.phase == .downloading(nil) && state.session == nil)
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        #expect(try await store.inspect(model: model, artifact: artifact).isMissing)
        #expect(
            try await store.stagingDirectory(for: model, artifact: artifact).lastPathComponent
                == artifact.sourceRevision)
        #expect(try JSONDecoder().decode(ModelArtifact.self, from: JSONEncoder().encode(artifact)) == artifact)
    }

    @Test(
        "IRDL-02 Altered speech distributions never receive a staging directory or an executable resource",
        arguments: ["repository", "revision", "root", "hash", "missing", "extra", "contract", "upstream"])
    func alteredDistribution(_ field: String) async throws {
        let known = try IrodoriModelArtifact.makeReferenceDownload()
        let source = try #require(known.download)
        var files = known.files
        if field == "hash" {
            files[0] = ModelArtifactFile(
                path: files[0].path, bytes: files[0].bytes, sha256: String(repeating: "a", count: 64))
        } else if field == "missing" {
            files.removeFirst()
        } else if field == "extra" {
            files.append(ModelArtifactFile(path: "extra.json", bytes: 2, sha256: String(repeating: "a", count: 64)))
        }
        let artifact = ModelArtifact(
            id: known.id, title: known.title,
            contract: field == "contract" ? IrodoriModelArtifact.contract : known.contract,
            platforms: known.platforms, sizeBytes: files.reduce(0) { $0 + $1.bytes },
            download: ModelDownload(
                repository: field == "repository" ? "other/model" : source.repository,
                revision: field == "revision" ? String(repeating: "a", count: 40) : source.revision,
                root: field == "root" ? "other" : source.root),
            upstream: field == "upstream" ? nil : known.upstream, files: files)
        #expect(artifact.isSourceValid)
        let model = LibraryModel(
            id: "irodori", name: "Irodori", category: "Speech", artifacts: [artifact],
            defaultArtifactID: artifact.id, platforms: [.iOS], demos: [])
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let inspected = try await store.inspect(model: model, artifact: artifact)
        #expect(inspected.resource == nil && inspected.reason?.contains("Irodori inventory") == true)
        await #expect(throws: ManagedModelStoreError.contractMismatch("Irodori inventory")) {
            try await store.stagingDirectory(for: model, artifact: artifact)
        }
    }
}
