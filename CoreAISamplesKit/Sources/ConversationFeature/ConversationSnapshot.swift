import Foundation

public struct ConversationSnapshot: Codable, Equatable, Sendable {
    public enum Status: String, Codable, Sendable {
        case completed, interrupted
    }

    public struct Metrics: Codable, Equatable, Sendable {
        public var steps: Int
        public var promptTokens: Int
        public var reusedTokens: Int
        public var generatedTokens: Int
    }

    public static let currentFormatVersion = 1

    public var formatVersion: Int
    public var id: String
    public var title: String
    public var createdAt: Date
    public var updatedAt: Date
    public var modelID: String
    public var instructionsVersion: String
    public var toolsVersion: String
    public var turns: [ConversationTurn]
    public var transcript: TranscriptRecord
    public var status: Status
    public var metrics: Metrics

    public init(_ state: ConversationState) {
        let steps = state.turns.flatMap(\.steps)
        formatVersion = Self.currentFormatVersion
        id = state.id
        title = state.title
        createdAt = state.createdAt
        updatedAt = state.updatedAt
        modelID = state.modelID
        instructionsVersion = state.instructionsVersion
        toolsVersion = state.toolsVersion
        turns = state.turns
        transcript = state.transcript
        status = state.isRunning ? .interrupted : .completed
        metrics = Metrics(
            steps: steps.count, promptTokens: steps.map(\.promptTokens).reduce(0, +),
            reusedTokens: steps.map(\.reusedTokens).reduce(0, +),
            generatedTokens: steps.map(\.generatedTokens).reduce(0, +))
    }

    public func markdown() -> String {
        func fence(_ text: String) -> String {
            let longestRun = text.split(whereSeparator: { $0 != "`" }).map(\.count).max() ?? 0
            let delimiter = String(repeating: "`", count: max(4, longestRun + 1))
            return "\(delimiter)\n\(text)\n\(delimiter)"
        }
        var lines = [
            "# Conversation",
            "",
            "- title: \(title)",
            "- id: \(id)",
            "- model: \(modelID.isEmpty ? "(none)" : modelID)",
            "- created: \(createdAt.formatted(.iso8601))",
            "- updated: \(updatedAt.formatted(.iso8601))",
            "- status: \(status.rawValue)",
            "",
        ]
        for entry in ConversationEntry.render(transcript, turns, live: false) {
            switch entry.kind {
            case .user:
                lines += ["## You", "", entry.text, ""]
            case .thinking:
                lines += ["### Thinking (\(entry.text.count) chars)", "", fence(entry.text), ""]
            case .toolCall:
                lines += ["### Calling \(entry.title)", "", fence(entry.text), ""]
            case .toolResult:
                lines += ["### \(entry.title) (\(entry.text.count) chars)", "", fence(entry.text), ""]
            case .answer:
                lines += ["## Assistant", "", entry.text, ""]
            case .failure:
                lines += ["> **\(entry.title)**: " + entry.text.replacingOccurrences(of: "\n", with: "\n> "), ""]
            case .note:
                lines += ["_\(entry.title): \(entry.text)_", ""]
            }
            if let stats = entry.stats { lines += ["_\(stats)_", ""] }
        }
        return lines.joined(separator: "\n")
    }
}
