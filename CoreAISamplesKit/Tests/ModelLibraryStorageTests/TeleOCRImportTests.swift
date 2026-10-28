import CryptoKit
import Foundation
import ModelLibraryFeature
import Testing

@testable import ModelLibraryStorage

struct TeleOCRImportTests {
    @Test("The TeleOCR identity pins the evaluated dynamic FP16 bundle and all nineteen files")
    func definition() async throws {
        let artifact = try TeleOCRModelArtifact.make()
        #expect(artifact.sourceKind == .localManifest && artifact.download == nil)
        #expect(artifact.localBundle?.manifestPath == "metadata.json")
        #expect(artifact.sourceRevision == TeleOCRModelArtifact.manifestSHA256)
        #expect(artifact.files.count == 19 && artifact.totalFileBytes == 2_847_208_089)
        #expect(artifact.platforms == [.macOS] && artifact.isSourceValid)
        #expect(artifact.upstream?.revision == TeleOCRModelArtifact.revision)
        #expect(try JSONDecoder().decode(ModelArtifact.self, from: JSONEncoder().encode(artifact)) == artifact)
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        #expect(try await store.inspect(model: model(artifact), artifact: artifact).isMissing)
        #expect(
            try await store.stagingDirectory(for: model(artifact), artifact: artifact).lastPathComponent
                == artifact.sourceRevision)
    }

    @Test(
        "The TeleOCR metadata rejects unsupported precision, platform, source, assets and file hashes",
        arguments: ["precision", "platform", "source", "revision", "assets", "files"])
    func metadata(_ field: String) throws {
        var value = try metadata()
        try TeleOCRModelArtifact.validateManifest(JSONSerialization.data(withJSONObject: value))
        switch field {
        case "precision": value["precision"] = "int4"
        case "platform": value["platform"] = "ios"
        case "source": value["source"] = ["model_id": "other/model", "revision": TeleOCRModelArtifact.revision]
        case "revision": value["source"] = ["model_id": TeleOCRModelArtifact.modelID, "revision": "other"]
        case "assets": value["assets"] = ["vision": "other.aimodel"]
        default: value["files"] = ["main.aimodel/main.mlirb": String(repeating: "a", count: 64)]
        }
        #expect(throws: ManagedModelStoreError.contractMismatch("TeleOCR metadata")) {
            try TeleOCRModelArtifact.validateManifest(JSONSerialization.data(withJSONObject: value))
        }
    }

    @Test(
        "A replacement TeleOCR inventory cannot redefine trusted file hashes or the contract",
        arguments: [false, true])
    func alteredDefinition(contract: Bool) async throws {
        let known = try TeleOCRModelArtifact.make()
        let artifact = ModelArtifact(
            id: known.id, title: known.title, contract: contract ? "teleocr-other" : known.contract,
            platforms: known.platforms, sizeBytes: known.sizeBytes, upstream: known.upstream,
            files: known.files.map {
                !contract && $0.path == "tokenizer/tokenizer.json"
                    ? ModelArtifactFile(path: $0.path, bytes: $0.bytes, sha256: String(repeating: "a", count: 64)) : $0
            }, localBundle: known.localBundle)
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let inspected = try await fixture.store().inspect(model: model(artifact), artifact: artifact)
        #expect(inspected.resource == nil && inspected.reason != nil)
    }

    @Test("Incomplete TeleOCR folders never become ready and leave their source unchanged")
    func missing() async throws {
        let artifact = try TeleOCRModelArtifact.make()
        let fixture = StoreFixture()
        defer { fixture.remove() }
        try fixture.write(["metadata.json": Data("{}".utf8)], to: fixture.documents)
        let store = fixture.store()
        let first = try #require(artifact.files.first?.path)
        await #expect(throws: ManagedModelStoreError.missingFile(first)) {
            try await store.copyForImport(from: fixture.documents, model: model(artifact), artifact: artifact) { _ in }
        }
        #expect(try await store.inspect(model: model(artifact), artifact: artifact).isMissing)
        #expect(try Data(contentsOf: fixture.documents.appending(path: "metadata.json")) == Data("{}".utf8))
    }

    @Test("TeleOCR imports reject unexpected files and symbolic links before copying", arguments: [false, true])
    func unsafeSource(link: Bool) async throws {
        let artifact = try TeleOCRModelArtifact.make()
        let fixture = StoreFixture()
        defer { fixture.remove() }
        try fixture.write(["metadata.json": Data("{}".utf8)], to: fixture.documents)
        let extra = fixture.documents.appending(path: ".DS_Store")
        if link {
            try FileManager.default.createSymbolicLink(
                at: extra, withDestinationURL: fixture.documents.appending(path: "metadata.json"))
        } else {
            try Data().write(to: extra)
        }
        let expected: ManagedModelStoreError = link ? .symbolicLink : .unexpectedFile(".DS_Store")
        await #expect(throws: expected) {
            try await fixture.store().copyForImport(
                from: fixture.documents, model: model(artifact), artifact: artifact
            ) { _ in }
        }
    }

    @Test("A changed TeleOCR manifest is rejected before copying the model payload")
    func changedManifest() async throws {
        let artifact = try TeleOCRModelArtifact.make()
        let fixture = StoreFixture()
        defer { fixture.remove() }
        try sparseSource(artifact, at: fixture.documents)
        let store = fixture.store()
        await #expect(throws: ManagedModelStoreError.checksumMismatch("metadata.json")) {
            try await store.copyForImport(from: fixture.documents, model: model(artifact), artifact: artifact) { _ in }
        }
        #expect(try await store.inspect(model: model(artifact), artifact: artifact).isMissing)
    }

    @Test(
        "The evaluated TeleOCR bundle imports intact, survives a rejected replacement and respects leases",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_TELEOCR_IMPORT_BUNDLE"] != nil))
    func knownBundle() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["COREAI_TELEOCR_IMPORT_BUNDLE"])
        let source = URL(filePath: path, directoryHint: .isDirectory)
        let artifact = try TeleOCRModelArtifact.make()
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
        try sparseSource(artifact, at: fixture.documents)
        await #expect(throws: ManagedModelStoreError.checksumMismatch("metadata.json")) {
            try await store.copyForImport(from: fixture.documents, model: model, artifact: artifact) { _ in }
        }
        #expect(try await store.inspect(model: model, artifact: artifact).resource == resource)
        #expect(try await store.delete(identity: resource.identity) == resource.sizeBytes)
        #expect(try await store.inspect(model: model, artifact: artifact).isMissing)
        for file in artifact.files {
            #expect(try fileHash(source.appending(path: file.path)) == file.sha256)
        }
    }

    private func model(_ artifact: ModelArtifact) -> LibraryModel {
        LibraryModel(
            id: "teleocr", name: "TeleOCR", category: "OCR", artifacts: [artifact], defaultArtifactID: artifact.id,
            platforms: [.macOS], demos: [])
    }

    private func metadata() throws -> [String: Any] {
        let artifact = try TeleOCRModelArtifact.make()
        return [
            "format": "coreai-samples-teleocr/1", "precision": "fp16",
            "source": ["model_id": TeleOCRModelArtifact.modelID, "revision": TeleOCRModelArtifact.revision],
            "assets": ["vision": "vision.aimodel", "embed": "embed.aimodel", "main": "main.aimodel"],
            "files": Dictionary(
                uniqueKeysWithValues: artifact.files.filter { $0.path != "metadata.json" }.map {
                    ($0.path, $0.sha256)
                }),
        ]
    }

    private func sparseSource(_ artifact: ModelArtifact, at source: URL) throws {
        for file in artifact.files {
            let url = source.appending(path: file.path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data().write(to: url)
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.truncate(atOffset: UInt64(file.bytes))
        }
    }

    private func fileHash(_ file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var hash = SHA256()
        while try autoreleasepool(invoking: {
            guard let data = try handle.read(upToCount: 1_048_576), !data.isEmpty else { return false }
            hash.update(data: data)
            return true
        }) {}
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
