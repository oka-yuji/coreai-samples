import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import Testing

struct MobileImageLabDownloadTests {
    @Test("QD-04 Public and legacy staging remain separate and incomplete public files cannot be promoted")
    func stagingAndIncompleteInventory() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ManagedModelStore(root: root)
        let model = try #require(IOSDemoCatalog.make().model("qwen-image21-turbo"))
        let artifact = MobileImageLabArtifact.published
        let publicStaging = try await store.stagingDirectory(for: model, artifact: artifact)
        let legacyStaging = try await store.stagingDirectory(for: model, artifact: QwenImage21Artifact.nativeComplex)
        #expect(publicStaging != legacyStaging)
        #expect(publicStaging.lastPathComponent == MobileImageLabArtifact.manifestSHA256)
        #expect(legacyStaging.lastPathComponent == QwenImage21Artifact.manifestSHA256)
        #expect(publicStaging.deletingLastPathComponent().lastPathComponent == artifact.id)
        await #expect(throws: ManagedModelStoreError.missingFile("LICENSE")) {
            try await store.promote(staging: publicStaging, model: model, artifact: artifact)
        }
        if case .missing = try await store.inspect(model: model, artifact: artifact, verification: .full) {
        } else {
            Issue.record("An incomplete public inventory must remain uninstalled")
        }
        #expect(artifact.files.count == 51 && artifact.totalFileBytes == 13_783_495_057)
        #expect(FileManager.default.fileExists(atPath: legacyStaging.path))
    }

    @Test(
        "QD-04 Managed storage rejects noncanonical image downloads even when the local manifest is trusted",
        arguments: ["id", "repository", "revision", "root", "inventory", "profile", "source", "legacy"])
    func rejectsChangedHybrid(field: String) async throws {
        let artifact = field == "legacy" ? QwenImage21Artifact.nativeComplex : MobileImageLabArtifact.published
        let download = try #require(MobileImageLabArtifact.published.download)
        var files = artifact.files
        if field == "inventory" {
            let first = try #require(files.first)
            files[0] = ModelArtifactFile(
                path: first.path, bytes: first.bytes, sha256: String(repeating: "f", count: 64))
        }
        let changed = ModelArtifact(
            id: field == "id" ? "unknown" : artifact.id, title: artifact.title, contract: artifact.contract,
            platforms: artifact.platforms, sizeBytes: artifact.sizeBytes,
            download: ModelDownload(
                repository: field == "repository" ? "example/foreign" : download.repository,
                revision: field == "revision" ? String(repeating: "f", count: 40) : download.revision,
                root: field == "root" ? "foreign" : download.root), upstream: artifact.upstream, files: files,
            localBundle: field == "source"
                ? nil
                : field == "profile"
                    ? ModelLocalBundle(manifestSHA256: MobileImageLabArtifact.manifestSHA256, profileID: "unknown")
                    : artifact.localBundle)
        let model = LibraryModel(
            id: "qwen-image21-turbo", name: "Mobile Image Lab", category: "Image Generation", artifacts: [changed],
            defaultArtifactID: changed.id, platforms: [.iOS], demos: [])
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ManagedModelStore(root: root)
        await #expect(throws: ManagedModelStoreError.self) {
            try await store.stagingDirectory(for: model, artifact: changed)
        }
    }
}
