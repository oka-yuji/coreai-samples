import Foundation

public struct ConversationEntry: Equatable, Identifiable, Sendable {
    public enum Kind: Equatable, Sendable {
        case user, thinking, answer, toolCall, toolResult, note, failure
    }

    public var id: String
    public var kind: Kind
    public var title: String
    public var text: String
    public var live: Bool
    public var stats: String?
    public var warning: Bool
    public var collapsed: Bool

    public init(
        id: String, kind: Kind, title: String, text: String, live: Bool = false, stats: String? = nil,
        warning: Bool = false, collapsed: Bool = false
    ) {
        self.id = id
        self.kind = kind
        self.title = title
        self.text = text
        self.live = live
        self.stats = stats
        self.warning = warning
        self.collapsed = collapsed
    }
}

extension ConversationEntry {
    static let noAnswerText = "(no answer text)"
    static let limitTitle = "Ended at the step limit"

    static func render(_ transcript: TranscriptRecord, _ turns: [ConversationTurn], live: Bool) -> [ConversationEntry] {
        turns.indices.flatMap { render(turn: $0, of: turns, in: transcript, live: live) }
    }

    private static func render(
        turn number: Int, of turns: [ConversationTurn], in transcript: TranscriptRecord, live: Bool
    ) -> [ConversationEntry] {
        let turn = turns[number]
        let policy = turn.policy
        let records = transcript.entries
        let end = number + 1 < turns.count ? min(turns[number + 1].start, records.count) : records.count
        let steps = transcript.steps(from: turn.start + 1, to: end)
        let cut = turn.finalization == .cutOff
        let answered = turn.finish != Finish.none && turn.finish != .stopped && turn.failure == nil && !cut
        var out = [ConversationEntry(id: "turn-\(number)", kind: .user, title: "You", text: turn.prompt)]
        var partial: String?
        for (index, step) in steps.enumerated() where !cut || index < policy.maxSteps {
            let trailing = index == steps.count - 1 && step.outputs == nil
            let generated = records[step.generated]
            let answer = generated.compactMap(\.response).joined().trimmingCharacters(in: .whitespacesAndNewlines)
            if let dropped = generated.compactMap(\.metadata).last?.truncatedToolOutputs, dropped > 0 {
                out.append(
                    ConversationEntry(
                        id: "context-\(number)-\(index)", kind: .note,
                        title: "Context full — dropped the oldest tool output",
                        text: "\(dropped) tool output(s) shortened to fit the context window."))
            }
            var shown = step.generated.flatMap { entries(records[$0], live: live && $0 == records.count - 1) }
            if trailing, turn.finish == .stopped, !cut {
                partial = answer
                shown.removeAll { $0.kind == .answer }
            }
            let line = turn.steps.first { $0.step == index + 1 }?.line
            if trailing, answered, answer.isEmpty {
                shown.append(
                    ConversationEntry(
                        id: "answer-\(number)", kind: .answer, title: "Assistant", text: noAnswerText, stats: line))
            } else if !shown.isEmpty {
                shown[shown.count - 1].stats = line
            }
            out += shown
            out += step.outputs.map { $0.flatMap { entries(records[$0], live: false) } } ?? []
        }
        if cut {
            let range = min(turn.start + 1, end)..<end
            let limit = steps.indices.contains(policy.maxSteps - 1) ? steps[policy.maxSteps - 1] : nil
            out.append(
                ConversationEntry(
                    id: "limit-\(number)", kind: .failure, title: limitTitle,
                    text: limitSummary(records, turn: range, last: limit, steps: policy.maxSteps)))
        } else if turn.finish == .stopped {
            let text = partial.flatMap { $0.isEmpty ? nil : $0 } ?? "(stopped before any answer)"
            out.append(ConversationEntry(id: "stopped-\(number)", kind: .failure, title: "Stopped", text: text))
        }
        if let failure = turn.failure {
            out.append(
                ConversationEntry(id: "failed-\(number)", kind: .failure, title: "Generation failed", text: failure))
        }
        if answered, turn.finish == .cap {
            out.append(
                ConversationEntry(
                    id: "cap-\(number)", kind: .failure, title: "Answer cut at the token limit (\(policy.maxTokens))",
                    text:
                        "The answer above stops where generation ran out of tokens. Raise Max tokens or ask for something shorter."
                ))
        }
        if policy.collapsesSteps {
            for index in out.indices where [.thinking, .toolCall, .toolResult].contains(out[index].kind) {
                out[index].collapsed = true
            }
        }
        return out
    }

    private static func entries(_ record: TranscriptRecord.Entry, live: Bool) -> [ConversationEntry] {
        switch record {
        case .instructions:
            return []
        case .prompt(let id, let text):
            return [ConversationEntry(id: id, kind: .user, title: "You", text: trimmed(text))]
        case .reasoning(let id, let text):
            let text = trimmed(text)
            return text.isEmpty
                ? [] : [ConversationEntry(id: id, kind: .thinking, title: "Thinking", text: text, live: live)]
        case .response(let id, let text, _):
            let text = trimmed(text)
            return text.isEmpty
                ? [] : [ConversationEntry(id: id, kind: .answer, title: "Assistant", text: text, live: live)]
        case .toolCalls(_, let calls, _):
            return calls.map {
                ConversationEntry(id: $0.id, kind: .toolCall, title: $0.name, text: $0.summary, live: live)
            }
        case .toolOutput(let id, let name, let text):
            return [ConversationEntry(id: "result-" + id, kind: .toolResult, title: name + " result", text: text)]
        }
    }

    private static func limitSummary(
        _ records: [TranscriptRecord.Entry], turn range: Range<Int>, last step: TranscriptStep?, steps limit: Int
    ) -> String {
        var outputs: [String: String] = [:]
        for case .toolOutput(let id, _, let text) in records[range] { outputs[id] = text }
        func done(_ call: ToolCallRecord) -> Bool {
            outputs[call.id].map { !$0.hasPrefix("error:") } ?? false
        }
        func list(_ items: [String]) -> String {
            var seen = Set<String>()
            let unique = items.filter { seen.insert($0).inserted }
            return unique.isEmpty ? "none" : unique.joined(separator: ", ")
        }
        let finished = records[range].flatMap(\.calls).filter(done)
        let open = step.map { records[$0.generated].flatMap(\.calls).filter { !done($0) } } ?? []
        return [
            "The turn reached its \(limit)-step limit before a final answer.",
            "Fetched: " + list(finished.filter { $0.name == "fetch_page" }.compactMap { $0.value("url") }),
            "Saved notes: " + list(finished.filter { $0.name == "write_note" }.compactMap { $0.value("title") }),
            "Still open: "
                + list(open.map { ([$0.name] + $0.argumentLines.prefix(1).map(\.value)).joined(separator: " ") }),
        ].joined(separator: "\n")
    }

    private static func trimmed(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
