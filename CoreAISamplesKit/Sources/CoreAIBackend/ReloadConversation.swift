import FoundationModels

extension Runtime {
    static func reloadConversation(_ transcript: Transcript) -> ReloadConversation {
        let prompts = transcript.compactMap { entry -> String? in
            if case .prompt(let prompt) = entry { return prompt.id }
            return nil
        }
        let hasHistory = transcript.contains { entry in
            switch entry {
            case .response, .reasoning, .toolCalls, .toolOutput: true
            default: false
            }
        }
        return ReloadConversation(promptIDs: prompts, hasHistory: hasHistory)
    }
}
