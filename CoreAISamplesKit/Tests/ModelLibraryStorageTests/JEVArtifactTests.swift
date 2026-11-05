import Foundation
import ModelLibraryFeature
import Testing

@testable import ModelLibraryStorage

struct JEVArtifactTests {
    @Test("The Mac JEV artifact pins the public bundle root and every published file")
    func inventory() throws {
        let catalog = try MacDemoCatalog.make()
        let artifact = try #require(catalog.model("jev-27b")?.artifacts.first)
        #expect(try artifact == JEVModelArtifact.make())
        #expect(artifact.files.count == 13)
        #expect(artifact.sourceKind == .huggingFace && artifact.isSourceValid)
        #expect(artifact.download?.repository == "okayuji/jev-27b-coreai")
        #expect(artifact.download?.revision == "acc4cc835f64a75ab9ecee9d5d2e119a3bd29212")
        #expect(artifact.download?.root == "bundle")
        #expect(artifact.files.first { $0.path == "decision.json" }?.sha256 == JEVModelArtifact.manifestSHA256)
        #expect(catalog.allows(ReceiptDemoCatalog.selection, platform: .macOS, audience: .release))
        #expect(!catalog.allows(ReceiptDemoCatalog.selection, platform: .iOS, audience: .release))
    }

    @Test("The JEV family rejects changed inventories and decision manifests")
    func alteredDefinition() async throws {
        let artifact = try JEVModelArtifact.make()
        let changed = ModelArtifact(
            id: artifact.id, title: artifact.title, contract: artifact.contract, platforms: artifact.platforms,
            sizeBytes: artifact.sizeBytes, download: artifact.download, upstream: artifact.upstream,
            files: Array(artifact.files.dropLast()))
        let catalog = try ReceiptDemoCatalog.make(artifact: changed)
        let model = try #require(catalog.models.first)
        let fixture = StoreFixture()
        defer { fixture.remove() }
        await #expect(throws: ManagedModelStoreError.contractMismatch("JEV inventory")) {
            try await fixture.store().stagingDirectory(for: model, artifact: changed)
        }
        let url = try #require(
            Bundle.module.url(forResource: "jev-manifest", withExtension: "json", subdirectory: "Resources"))
        let data = try Data(contentsOf: url)
        try JEVModelArtifact.validateManifest(data)
        #expect(throws: ManagedModelStoreError.contractMismatch("JEV decision.json")) {
            try JEVModelArtifact.validateManifest(data + Data("\n".utf8))
        }
    }
}
