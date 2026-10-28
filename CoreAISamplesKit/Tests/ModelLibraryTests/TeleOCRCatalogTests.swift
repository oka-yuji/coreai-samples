import ModelLibraryFeature
import Testing

struct TeleOCRCatalogTests {
    @Test(
        "OCR model failures expose useful messages instead of unlocalized internal names",
        arguments: [
            (
                "insufficientSpace",
                "There is not enough disk space to import this model. Free some space and try again."
            ),
            ("ledgerCorrupted", "The model library record is damaged. Model changes are unavailable."),
            ("transfer in progress", "Another window is transferring this model. Wait for it to finish."),
        ])
    func failureMessages(reason: String, message: String) {
        let notice = TeleOCRModelNotice(.operation(reason))
        #expect(notice.message == message && notice.detail == nil)
    }

    @Test("Unknown model failures preserve details under a common explanation")
    func unknownFailure() {
        let notice = TeleOCRModelNotice(.operation("fixture failure"))
        #expect(notice.message == "The model operation could not be completed." && notice.detail == "fixture failure")
        #expect(TeleOCRModelNotice(.invalidResource).detail == nil)
        #expect(TeleOCRModelNotice(.modelInUse).message == "Close the OCR demo before replacing or removing its model.")
    }

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
