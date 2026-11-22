extension LocalLanguageModel {
    public struct ReloadPolicy: Sendable, Equatable {
        public var readInterval: Int
        public var footprintLimitBytes: UInt64
        public var graphicsGrowthLimitBytes: UInt64?

        public init(
            readInterval: Int = 10, footprintLimitBytes: UInt64 = 18_000_000_000,
            graphicsGrowthLimitBytes: UInt64? = nil
        ) {
            self.readInterval = readInterval
            self.footprintLimitBytes = footprintLimitBytes
            self.graphicsGrowthLimitBytes = nil
        }

        static let verifiedGraphicsGrowthLimits: [(modelID: String, mainHash: String, bytes: UInt64)] = [
            ("Qwen/Qwen3.5-0.8B", "41b46bd34a69c7a25e017122323e72c338ff1fb1a78f17888e5e04bb0c77a2ba", 536_870_912)
        ]

        func graphicsGrowthLimit(modelID: String?, assetHash: String?) -> UInt64 { 0 }
    }
}

struct ReloadState {
    private(set) var completedReads = 0
    private var seenPromptIDs: Set<String> = []

    mutating func observe(_ conversation: ReloadConversation) -> Bool {
        let prompts = conversation.promptIDs
        let isNew =
            prompts.count == 1 && !conversation.hasHistory
            && prompts.first.map { !seenPromptIDs.contains($0) } == true
        seenPromptIDs.formUnion(prompts)
        return isNew
    }

    func shouldReload(
        policy: LocalLanguageModel.ReloadPolicy, isNewConversation: Bool, footprint: UInt64,
        graphicsGrowth: UInt64? = nil, graphicsLimit: UInt64 = 0
    ) -> Bool {
        (policy.readInterval > 0 && completedReads >= policy.readInterval && isNewConversation)
            || (policy.footprintLimitBytes > 0 && footprint >= policy.footprintLimitBytes)
    }

    mutating func completedRead() { completedReads += 1 }
    mutating func finishedRead() {}
    mutating func graphicsGrowth(sample: UInt64?) -> UInt64? { nil }
    mutating func didLoad() { completedReads = 0 }
}

enum ReloadError: Error, Equatable {
    case negativeInterval(Int)
}

struct ReloadConversation {
    let promptIDs: [String]
    let hasHistory: Bool
}
