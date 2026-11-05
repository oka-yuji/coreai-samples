public enum ReceiptDemoCatalog {
    public static let selection = DemoSelection(modelID: "jev-27b", demoID: "receipt-desk")

    public static func make(artifact: ModelArtifact) throws -> ModelCatalog {
        try ModelCatalog(models: [model(artifact: artifact)], demos: [demo])
    }

    static var demo: DemoDescriptor {
        DemoDescriptor(
            id: selection.demoID, title: "Sort receipts into a reviewed expense ledger", systemImage: "receipt",
            summary: "Sort receipts into a reviewed expense ledger", platforms: [.macOS])
    }

    static func model(artifact: ModelArtifact) -> LibraryModel {
        LibraryModel(
            id: selection.modelID, name: "JEV-27B", category: "Receipt Classification", architecture: "27B",
            searchTerms: ["receipt", "expense", "JEV"], artifacts: [artifact], defaultArtifactID: artifact.id,
            platforms: [.macOS], demos: [ModelDemoLink(demoID: selection.demoID, platforms: [.macOS])])
    }
}
