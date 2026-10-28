import ModelLibraryFeature
import Testing

struct TeleOCRCatalogTests {
    @Test("The OCR catalog exposes one Mac document demo backed by the supplied artifact")
    func catalog() throws {
        let hash = String(repeating: "a", count: 64)
        let artifact = ModelArtifact(
            id: "teleocr-fp16", title: "FP16", contract: "teleocr-dynamic-fp16-v1", platforms: [.macOS],
            files: [ModelArtifactFile(path: "metadata.json", bytes: 0, sha256: hash)],
            localBundle: ModelLocalBundle(manifestSHA256: hash, profileID: "fixture", manifestPath: "metadata.json"))
        let catalog = try TeleOCRDemoCatalog.make(artifact: artifact)
        #expect(catalog.models.count == 1 && catalog.demos.count == 1)
        #expect(catalog.models[0].platforms == [.macOS] && catalog.models[0].artifacts == [artifact])
        #expect(catalog.models[0].defaultArtifactID == artifact.id && catalog.demos[0].id == "ocr")
        try catalog.validateConnections(adapterIDs: ["ocr"])
        #expect(throws: ModelCatalogError.invalidDemo("ocr")) { try catalog.validateConnections(adapterIDs: []) }
    }
}
