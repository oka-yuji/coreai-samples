public enum IOSDemoCatalog {
    public static func make(searchTerms: [String: [String]] = [:]) throws -> ModelCatalog {
        let demos = [
            DemoDescriptor(
                id: "comparison", title: "Comparison demo", systemImage: "rectangle.split.3x1",
                summary: "Compare product specifications from source text.",
                explanation: [
                    DemoExplanation(
                        title: "How it works",
                        text:
                            "Add up to four product descriptions from URLs or pasted text, then compare extracted specifications and their source evidence."
                    )
                ],
                platforms: [.iOS], searchTerms: searchTerms["comparison"]),
            DemoDescriptor(
                id: "extraction", title: "Extraction demo", systemImage: "text.magnifyingglass",
                summary: "Extract structured fields and review their evidence.",
                explanation: [
                    DemoExplanation(
                        title: "How it works",
                        text:
                            "Enter a note and inspect the extracted text spans and scores for each field alongside the original text."
                    )
                ],
                platforms: [.iOS], searchTerms: searchTerms["extraction"]),
            DemoDescriptor(
                id: "inspection", title: "Inspection demo", systemImage: "checklist",
                summary: "Review truck inspection answers entered by text or voice.",
                explanation: [
                    DemoExplanation(
                        title: "How it works",
                        text:
                            "Enter inspection answers, review the interpreted fields, and correct them before sharing the inspection record."
                    )
                ],
                platforms: [.iOS], searchTerms: searchTerms["inspection"]),
            DemoDescriptor(
                id: "image-search", title: "Image search demo", systemImage: "photo.badge.magnifyingglass",
                summary: "Find selected images using a text query.",
                explanation: [
                    DemoExplanation(
                        title: "How it works",
                        text:
                            "Select images, prepare their search index, and compare the matching images for your query. Similarity scores help rank candidates."
                    )
                ],
                platforms: [.iOS], searchTerms: searchTerms["image-search"]),
            DemoDescriptor(
                id: "object-finder", title: "Camera search demo", systemImage: "viewfinder",
                summary: "Find camera regions that match your words.",
                explanation: [
                    DemoExplanation(
                        title: "How it works",
                        text:
                            "Enter what you want to find and inspect matching regions in the camera preview. Matches are similarity candidates and may need human review."
                    )
                ],
                platforms: [.iOS], searchTerms: searchTerms["object-finder"]),
        ]
        let models = [
            LibraryModel(
                id: "gliner2.5-multi", name: "GLiNER2.5-multi", category: "Information Extraction",
                searchTerms: searchTerms["gliner2.5-multi"] ?? [],
                artifacts: [unconfigured("gliner2.5-multi-unconfigured")],
                defaultArtifactID: "gliner2.5-multi-unconfigured", platforms: [.iOS],
                demos: ["comparison", "extraction", "inspection"].map {
                    ModelDemoLink(
                        demoID: $0, platforms: [.iOS], optionalCapabilities: $0 == "comparison" ? [.network] : [])
                }),
            LibraryModel(
                id: "embeddinggemma2", name: "EmbeddingGemma 2", category: "Multimodal Embedding",
                searchTerms: searchTerms["embeddinggemma2"] ?? [],
                artifacts: [unconfigured("embeddinggemma2-unconfigured")],
                defaultArtifactID: "embeddinggemma2-unconfigured", platforms: [.iOS],
                demos: ["image-search", "object-finder"].map {
                    ModelDemoLink(demoID: $0, platforms: [.iOS])
                }),
        ]
        return try ModelCatalog(models: models, demos: demos)
    }

    private static func unconfigured(_ id: String) -> ModelArtifact {
        ModelArtifact(id: id, title: "Unconfigured", contract: "unconfigured", platforms: [.iOS])
    }
}
