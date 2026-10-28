public enum TeleOCRDemoCatalog {
    public static let selection = DemoSelection(modelID: "teleocr", demoID: "ocr")

    public static func make(artifact: ModelArtifact) throws -> ModelCatalog {
        let demo = DemoDescriptor(
            id: selection.demoID, title: "Document reading", systemImage: "doc.viewfinder",
            summary: "Read text and tables from a document image", platforms: [.macOS])
        let model = LibraryModel(
            id: selection.modelID, name: "TeleOCR", category: "Documents", architecture: "1.4B",
            searchTerms: ["OCR", "document", "table"], artifacts: [artifact], defaultArtifactID: artifact.id,
            platforms: [.macOS], demos: [ModelDemoLink(demoID: demo.id, platforms: [.macOS])])
        return try ModelCatalog(models: [model], demos: [demo])
    }
}
