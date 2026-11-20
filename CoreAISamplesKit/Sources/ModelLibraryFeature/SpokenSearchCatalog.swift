public enum SpokenSearchCatalog {
    public static let selection = DemoSelection(modelID: "embeddinggemma2", demoID: "spoken-search")
    public static let title = "Search spoken words"
    public static let symbol = "waveform.badge.magnifyingglass"

    static func demo(platform: LibraryPlatform, searchTerms: [String: [String]]) -> DemoDescriptor {
        DemoDescriptor(
            id: selection.demoID, title: title, systemImage: symbol,
            summary: "Find a word in a Japanese recording and play from each match.",
            explanation: [
                DemoExplanation(
                    title: "How it works",
                    text:
                        "Apple Speech transcribes the recording on this device. Word search works without downloading EmbeddingGemma 2."
                )
            ],
            platforms: [platform], searchTerms: searchTerms[selection.demoID], allowsPreviewInteraction: true,
            preparesModelOnOpen: false)
    }
}
