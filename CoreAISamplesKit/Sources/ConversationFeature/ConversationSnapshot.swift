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

    public static let currentFormatVersion = 3

    public static func supports(version: Int) -> Bool { (1...currentFormatVersion).contains(version) }

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
    public var revision: Int = 0
    public var modelLocation: String?
    public var evidence: ConversationEvidence?

    public var suppliedSourceIDs: Set<String> {
        var ids: Set<String> = []
        for (index, turn) in turns.enumerated() where turn.sourceRequest != nil {
            let end =
                index + 1 < turns.count
                ? min(turns[index + 1].start, transcript.entries.count) : transcript.entries.count
            if turn.start >= 0, turn.start < end, transcript.entries.indices.contains(turn.start),
                transcript.entries[turn.start].isPrompt
            {
                ids.formUnion(turn.sourceIDs ?? [])
            }
        }
        return ids
    }

    public init(_ state: ConversationState) {
        let steps = state.turns.flatMap(\.steps)
        formatVersion = Self.currentFormatVersion
        id = state.id
        title = state.title
        createdAt = state.createdAt
        updatedAt = state.updatedAt
        modelID = state.modelID
        modelLocation = state.modelLocation
        instructionsVersion = state.instructionsVersion
        toolsVersion = state.toolsVersion
        turns = state.turns
        transcript = state.transcript
        status =
            state.isRunning || state.lastFinish == .stopped || state.lastFinish == .failed ? .interrupted : .completed
        revision = state.archiveRevision
        evidence = state.evidence
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
        for entry in ConversationEntry.render(transcript, turns, live: false, mergeContinuations: false) {
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
        if let sources = evidence?.sources, !sources.records.isEmpty {
            lines += ["## Supplied sources", ""]
            for source in sources.records {
                lines += [
                    "### [\(source.id)] \(source.title)", "", "Document: " + source.document,
                    "Version (SHA-256): " + source.version, "",
                ]
                for passage in source.passages {
                    lines += ["Characters \(passage.offset)..<\(passage.end)", "", fence(passage.text), ""]
                }
            }
        }
        return lines.joined(separator: "\n")
    }
}
