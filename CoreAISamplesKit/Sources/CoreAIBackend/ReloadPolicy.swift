extension LocalLanguageModel {
    public struct ReloadPolicy: Sendable, Equatable {
        public var readInterval: Int
        public var footprintLimitBytes: UInt64

        public init(readInterval: Int = 10, footprintLimitBytes: UInt64 = 18_000_000_000) {
            self.readInterval = readInterval
            self.footprintLimitBytes = footprintLimitBytes
        }
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

    func shouldReload(policy: LocalLanguageModel.ReloadPolicy, isNewConversation: Bool, footprint: UInt64) -> Bool {
        (policy.readInterval > 0 && completedReads >= policy.readInterval && isNewConversation)
            || (policy.footprintLimitBytes > 0 && footprint >= policy.footprintLimitBytes)
    }

    mutating func completedRead() { completedReads += 1 }
    mutating func didLoad() { completedReads = 0 }
}

enum ReloadError: Error, Equatable {
    case negativeInterval(Int)
}

struct ReloadConversation {
    let promptIDs: [String]
    let hasHistory: Bool
}
