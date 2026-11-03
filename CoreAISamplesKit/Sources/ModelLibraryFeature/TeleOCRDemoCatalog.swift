public enum TeleOCRDemoCatalog {
    public static let selection = DemoSelection(modelID: "teleocr", demoID: "ocr")

    public static func make(artifact: ModelArtifact) throws -> ModelCatalog {
        let demo = DemoDescriptor(
            id: selection.demoID, title: "Document reading", systemImage: "doc.viewfinder",
            summary: "Read text and tables from a document image", platforms: [.macOS])
        return try ModelCatalog(models: [model(artifact: artifact)], demos: [demo])
    }

    static func model(artifact: ModelArtifact) -> LibraryModel {
        LibraryModel(
            id: selection.modelID, name: "TeleOCR", category: "Documents", architecture: "1.4B",
            searchTerms: ["OCR", "document", "table"], artifacts: [artifact], defaultArtifactID: artifact.id,
            platforms: [.macOS], demos: [ModelDemoLink(demoID: selection.demoID, platforms: [.macOS])])
    }
}
