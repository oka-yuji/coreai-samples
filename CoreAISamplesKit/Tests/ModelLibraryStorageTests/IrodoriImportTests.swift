import Foundation
import ModelLibraryFeature
import SpeechAssets
import Testing

@testable import ModelLibraryStorage

struct IrodoriImportTests {
    @Test("REF-CORE-03 Reference bundles have a separate identity and pin all twenty files")
    func referenceDefinition() async throws {
        let legacy = try IrodoriModelArtifact.make()
        let artifact = try IrodoriModelArtifact.make(profile: .referenceVoice)
        #expect(artifact.id != legacy.id && artifact.contract != legacy.contract)
        #expect(artifact.sourceRevision == IrodoriBundleInventory.referenceManifestSHA256)
        #expect(artifact.files.count == 20 && artifact.isSourceValid)
        #expect(artifact.files.contains { $0.path.contains("reference-audio-fp32") })
        #expect(artifact.files.contains { $0.path.contains("reference-speaker-fp32") })
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        #expect(try await store.inspect(model: model(artifact), artifact: artifact).isMissing)
        let staging = try await store.stagingDirectory(for: model(artifact), artifact: artifact)
        #expect(staging.lastPathComponent == artifact.sourceRevision)
    }

    @Test("IR-ASSET-01 The Irodori identity pins the manifest and all fourteen files")
    func definition() async throws {
        let artifact = try IrodoriModelArtifact.make()
        #expect(artifact.sourceKind == .localManifest)
        #expect(artifact.sourceRevision == IrodoriBundleInventory.manifestSHA256)
        #expect(artifact.localBundle?.manifestPath == "irodori.json")
        #expect(artifact.isSourceValid && artifact.download == nil)
        #expect(artifact.files.count == 14)
        #expect(artifact.totalFileBytes == 3_375_061_725)
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let model = model(artifact)
        #expect(try await store.inspect(model: model, artifact: artifact).isMissing)
        let staging = try await store.stagingDirectory(for: model, artifact: artifact)
        #expect(staging.lastPathComponent == artifact.sourceRevision)
        let restored = try JSONDecoder().decode(ModelArtifact.self, from: JSONEncoder().encode(artifact))
        #expect(restored == artifact)
    }

    @Test("IR-ASSET-02 Legacy local manifests retain their source identity and default filename")
    func legacyManifest() throws {
        let bytes = Data(
            #"{"manifestSHA256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","profileID":"image"}"#
                .utf8)
        let source = try JSONDecoder().decode(ModelLocalBundle.self, from: bytes)
        #expect(source.manifestPath == "bundle.json")
        #expect(source == ModelLocalBundle(manifestSHA256: String(repeating: "a", count: 64), profileID: "image"))
        #expect(!ModelLocalBundle(manifestSHA256: source.manifestSHA256, profileID: "x", manifestPath: "../x").isValid)
    }

    @Test("IR-ASSET-01 A manifest with unsupported speech settings is rejected", arguments: ["fp16", "other"])
    func manifest(_ precision: String) throws {
        let raw = Data(
            "{\"format\":\"irodori-coreai-v1\",\"modelID\":\"Aratako/Irodori-TTS-v4.1-Small-MF\",\"precision\":\"\(precision)\",\"sampleRate\":48000}"
                .utf8)
        #expect(throws: (any Error).self) { try IrodoriBundleInventory.validateManifest(raw) }
        let valid = Data(
            "{\"format\":\"irodori-coreai-v1\",\"modelID\":\"Aratako/Irodori-TTS-v4.1-Small-MF\",\"precision\":\"fp32\",\"sampleRate\":48000}"
                .utf8)
        try IrodoriBundleInventory.validateManifest(valid)
    }

    @Test("IOS-IR-02 Replacing the trusted inventory with attacker supplied hashes is rejected")
    func alteredDefinition() async throws {
        let known = try IrodoriModelArtifact.make()
        let altered = ModelArtifact(
            id: known.id, title: known.title, contract: known.contract, platforms: known.platforms,
            sizeBytes: known.sizeBytes,
            files: known.files.map {
                $0.path == "tokenizer.json"
                    ? ModelArtifactFile(path: $0.path, bytes: $0.bytes, sha256: String(repeating: "a", count: 64)) : $0
            }, localBundle: known.localBundle)
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let inspected = try await store.inspect(model: model(altered), artifact: altered)
        #expect(inspected.resource == nil)
        #expect(inspected.reason?.contains("Irodori inventory") == true)
    }

    @Test("IOS-IR-02 Incomplete Irodori folders never become ready")
    func missing() async throws {
        let artifact = try IrodoriModelArtifact.make()
        let fixture = StoreFixture()
        defer { fixture.remove() }
        try fixture.write(["irodori.json": Data("{}".utf8)], to: fixture.documents)
        let store = fixture.store()
        let missing = try #require(artifact.files.first?.path)
        await #expect(throws: ManagedModelStoreError.missingFile(missing)) {
            try await store.copyForImport(from: fixture.documents, model: model(artifact), artifact: artifact) { _ in }
        }
        #expect(try await store.inspect(model: model(artifact), artifact: artifact).isMissing)
    }

    @Test("IR-ASSET-01 Import rejects unexpected files and symbolic links before copying", arguments: [false, true])
    func unsafeSource(link: Bool) async throws {
        let artifact = try IrodoriModelArtifact.make()
        let fixture = StoreFixture()
        defer { fixture.remove() }
        try fixture.write(["irodori.json": Data("{}".utf8)], to: fixture.documents)
        let extra = fixture.documents.appending(path: ".DS_Store")
        if link {
            try FileManager.default.createSymbolicLink(
                at: extra, withDestinationURL: fixture.documents.appending(path: "irodori.json"))
        } else {
            try Data().write(to: extra)
        }
        let store = fixture.store()
        let expected: ManagedModelStoreError = link ? .symbolicLink : .unexpectedFile(".DS_Store")
        await #expect(throws: expected) {
            try await store.copyForImport(from: fixture.documents, model: model(artifact), artifact: artifact) { _ in }
        }
        #expect(try await store.inspect(model: model(artifact), artifact: artifact).isMissing)
        #expect(try Data(contentsOf: fixture.documents.appending(path: "irodori.json")) == Data("{}".utf8))
    }

    @Test("IR-ASSET-01 Import rejects a changed manifest before copying the model payload")
    func changedManifest() async throws {
        let artifact = try IrodoriModelArtifact.make()
        let fixture = StoreFixture()
        defer { fixture.remove() }
        for file in artifact.files {
            let url = fixture.documents.appending(path: file.path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data().write(to: url)
            let handle = try FileHandle(forWritingTo: url)
            try handle.truncate(atOffset: UInt64(file.bytes))
            try handle.close()
        }
        let store = fixture.store()
        await #expect(throws: ManagedModelStoreError.checksumMismatch("irodori.json")) {
            try await store.copyForImport(from: fixture.documents, model: model(artifact), artifact: artifact) { _ in }
        }
        #expect(try await store.inspect(model: model(artifact), artifact: artifact).isMissing)
    }

    @Test(
        "IOS-IR-02 The known Irodori folder imports intact and rejects changes before reuse",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_IRODORI_IMPORT_BUNDLE"] != nil))
    func knownBundle() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["COREAI_IRODORI_IMPORT_BUNDLE"])
        let source = URL(filePath: path, directoryHint: .isDirectory)
        let artifact = try IrodoriModelArtifact.make()
        let model = model(artifact)
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let staging = try await store.copyForImport(from: source, model: model, artifact: artifact) { _ in }
        let resource = try await store.promote(staging: staging, model: model, artifact: artifact)
        #expect(resource.sourceKind == .localManifest)
        #expect(try await store.inspect(model: model, artifact: artifact).resource == resource)
        let lease = await store.lease(identity: resource.identity, owner: UUID())
        await #expect(throws: ManagedModelStoreError.inUse) { try await store.delete(identity: resource.identity) }
        await lease.release()
        let first = try #require(artifact.files.first)
        let file = resource.directory.appending(path: first.path)
        let bytes = try Data(contentsOf: file)
        try Data(repeating: 0, count: bytes.count).write(to: file)
        #expect(try await store.inspect(model: model, artifact: artifact).resource == nil)
        try bytes.write(to: file)
        let extra = resource.directory.appending(path: "extra.json")
        try Data("{}".utf8).write(to: extra)
        #expect(try await store.inspect(model: model, artifact: artifact).resource == nil)
        try FileManager.default.removeItem(at: extra)
        try FileManager.default.removeItem(at: file)
        #expect(try await store.inspect(model: model, artifact: artifact).resource == nil)
        try FileManager.default.createSymbolicLink(at: file, withDestinationURL: source.appending(path: first.path))
        #expect(try await store.inspect(model: model, artifact: artifact).resource == nil)
        #expect(try Data(contentsOf: source.appending(path: first.path)) == bytes)
    }

    private func model(_ artifact: ModelArtifact) -> LibraryModel {
        LibraryModel(
            id: "irodori", name: "Irodori", category: "Speech", artifacts: [artifact], defaultArtifactID: artifact.id,
            platforms: [.iOS], demos: [])
    }
}
