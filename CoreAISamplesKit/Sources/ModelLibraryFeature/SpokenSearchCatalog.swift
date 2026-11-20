public enum SpokenSearchCatalog {
    public static let selection = DemoSelection(modelID: "embeddinggemma2", demoID: "spoken-search")
    public static let title = "Search a recording"
    public static let symbol = "waveform.badge.magnifyingglass"

    static func demo(platform: LibraryPlatform, searchTerms: [String: [String]]) -> DemoDescriptor {
        DemoDescriptor(
            id: selection.demoID, title: title, systemImage: symbol,
            summary: "Find a spoken word or a related passage in a Japanese recording.",
            explanation: [
                DemoExplanation(
                    title: "How it works",
                    text:
                        "Apple Speech transcribes the recording on this device. Word finds the words you entered. Topic uses EmbeddingGemma 2 to find passages related to a question. Both modes play the original recording."
                )
            ],
            platforms: [platform], searchTerms: searchTerms[selection.demoID], allowsPreviewInteraction: true,
            preparesModelOnOpen: false)
    }
}
