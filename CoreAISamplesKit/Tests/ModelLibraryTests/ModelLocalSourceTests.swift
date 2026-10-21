import Foundation
import ModelLibraryFeature
import Testing

struct ModelLocalSourceTests {
    private let hash = String(repeating: "a", count: 64)

    @Test("Legacy downloaded artifacts decode without a local source and retain their encoded fields")
    func legacyArtifactCompatibility() throws {
        let original = ModelArtifact(
            id: "old", title: "Old", contract: "legacy", platforms: [.iOS],
            download: ModelDownload(repository: "owner/model", revision: String(repeating: "b", count: 40)),
            files: [ModelArtifactFile(path: "weights.bin", bytes: 3, sha256: hash)])
        let bytes = try JSONEncoder().encode(original)
        let object = try #require(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
        #expect(object["localBundle"] == nil)
        let restored = try JSONDecoder().decode(ModelArtifact.self, from: bytes)
        #expect(restored == original)
        #expect(restored.sourceKind == .huggingFace)
        #expect(restored.sourceRevision == original.download?.revision)
        #expect(restored.isSourceValid)
    }

    @Test("Local resources use the raw manifest hash and cannot masquerade as Hugging Face resources")
    func localSourceIdentity() throws {
        let artifact = localArtifact()
        let model = LibraryModel(
            id: "qwen-image", name: "Qwen Image", category: "Image generation", artifacts: [artifact],
            defaultArtifactID: artifact.id, platforms: [.iOS], demos: [])
        _ = try ModelCatalog(models: [model], demos: [])
        let directory = URL(filePath: "/fixture/model")
        let local = ValidatedModelResource(
            modelID: model.id, artifactID: artifact.id, contract: artifact.contract, revision: hash,
            upstreamRevision: nil, directory: directory, sizeBytes: 12, sourceKind: .localManifest)
        let downloaded = ValidatedModelResource(
            modelID: model.id, artifactID: artifact.id, contract: artifact.contract, revision: hash,
            upstreamRevision: nil, directory: directory, sizeBytes: 12)
        #expect(local.matches(model: model, artifact: artifact))
        #expect(!downloaded.matches(model: model, artifact: artifact))
        #expect(local.identity != downloaded.identity)
        #expect(artifact.download == nil)
        #expect(artifact.sourceRevision == hash)
    }

    @Test("Conflicting sources, unknown manifest hashes and overflowing file sizes are rejected")
    func invalidSourceDefinitions() {
        let definitions = [
            localArtifact(
                download: ModelDownload(repository: "owner/model", revision: String(repeating: "b", count: 40))),
            localArtifact(manifestHash: String(repeating: "b", count: 64)),
            localArtifact(manifestHash: String(repeating: "a", count: 40)),
            localArtifact(files: [
                ModelArtifactFile(path: "bundle.json", bytes: .max, sha256: hash),
                ModelArtifactFile(path: "weights.bin", bytes: 1, sha256: hash),
            ]),
        ]
        for artifact in definitions {
            let model = LibraryModel(
                id: "qwen-image", name: "Qwen Image", category: "Image generation", artifacts: [artifact],
                defaultArtifactID: artifact.id, platforms: [.iOS], demos: [])
            #expect(throws: ModelCatalogError.invalidArtifact(artifact.id)) {
                try ModelCatalog(models: [model], demos: [])
            }
        }
    }

    private func localArtifact(
        download: ModelDownload? = nil, manifestHash: String? = nil, files: [ModelArtifactFile]? = nil
    ) -> ModelArtifact {
        ModelArtifact(
            id: "qwen-mobile", title: "Mobile", contract: "qwen-image21-mobile-v2", platforms: [.iOS],
            download: download, files: files ?? [ModelArtifactFile(path: "bundle.json", bytes: 12, sha256: hash)],
            localBundle: ModelLocalBundle(manifestSHA256: manifestHash ?? hash, profileID: "qmi-mobile-w4-v1"))
    }
}
