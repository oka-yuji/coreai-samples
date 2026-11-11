public enum MacDemoCatalog {
    public static func make(
        searchTerms: [String: [String]] = [:], speechArtifact: ModelArtifact, ocrArtifact: ModelArtifact,
        qwenArtifact: ModelArtifact, clefArtifact: ModelArtifact, jevArtifact: ModelArtifact,
        chronosArtifact: ModelArtifact
    ) throws -> ModelCatalog {
        let mac: Set<LibraryPlatform> = [.macOS]
        let demos = [
            ("chat", "Talk, search the web and run multi-step tasks on a local model", "bubble.left.and.bubble.right"),
            ("decide", "Check a document against an editable checklist", "checklist"),
            ("recipes", "Can you make this recipe with what you have?", "carrot"),
            ("image-search", "Find selected images with your own words", "photo.badge.magnifyingglass"),
            ("video-search", "Find moments in a video with your own words", "film.stack"),
            ("comparison", "Compare product specifications from official descriptions", "text.magnifyingglass"),
            ("speech", "Generate and play speech from text with Irodori", "waveform"),
            (TeleOCRDemoCatalog.selection.demoID, "Read text and tables from a document image", "doc.viewfinder"),
        ].map { id, summary, systemImage in
            DemoDescriptor(
                id: id, title: summary, systemImage: systemImage, summary: summary, platforms: mac,
                searchTerms: searchTerms[id])
        }
        func model(
            _ id: String, _ name: String, _ category: String, _ artifacts: [ModelArtifact], _ demoIDs: [String]
        ) -> LibraryModel {
            LibraryModel(
                id: id, name: name, category: category, searchTerms: searchTerms[id] ?? [], artifacts: artifacts,
                defaultArtifactID: artifacts[0].id, platforms: mac,
                demos: demoIDs.map { ModelDemoLink(demoID: $0, platforms: mac) })
        }
        let embedding = [
            PublishedModelArtifacts.embeddingFP16(platforms: mac), PublishedModelArtifacts.embeddingPTQ(platforms: mac),
        ]
        return try ModelCatalog(
            models: [
                model(
                    "qwen3.8-27b", "Qwen3.8 27B", "Language Model", [qwenArtifact],
                    ["chat"]),
                model(
                    "clef-flash-9b", "Clef-Flash 9B", "Document Classification",
                    [clefArtifact], ["decide", "recipes"]),
                model(
                    "embeddinggemma2", "EmbeddingGemma 2", "Multimodal Embedding", embedding,
                    ["image-search", "video-search"]),
                model(
                    "gliner2.5-multi", "GLiNER2.5-multi", "Information Extraction",
                    [PublishedModelArtifacts.glinerFP32(platforms: mac)], ["comparison"]),
                model("irodori", "Irodori", "Speech Synthesis", [speechArtifact], ["speech"]),
                TeleOCRDemoCatalog.model(artifact: ocrArtifact),
                ReceiptDemoCatalog.model(artifact: jevArtifact),
                ChronosDemoCatalog.model(artifact: chronosArtifact, platforms: mac, searchTerms: searchTerms),
            ],
            demos: demos + [ReceiptDemoCatalog.demo, ChronosDemoCatalog.demo(platforms: mac, searchTerms: searchTerms)])
    }

}
