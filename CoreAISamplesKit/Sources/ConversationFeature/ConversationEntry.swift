import Foundation

public struct ConversationEntry: Equatable, Identifiable, Sendable {
    public enum Kind: Equatable, Sendable {
        case user, thinking, answer, toolCall, toolResult, note, failure
    }

    public enum Detail: Equatable, Sendable {
        case stepLimit(steps: Int, fetched: [String], saved: [String], open: [String])
        case stopped(partial: Bool)
        case tokenLimit(Int)
        case cutToolCall(name: String?, note: Bool)
        case noAnswer
        case researchLimit(Int)
        case failure
    }

    public var id: String
    public var kind: Kind
    public var title: String
    public var text: String
    public var live: Bool
    public var stats: String?
    public var warning: Bool
    public var collapsed: Bool
    public var detail: Detail?

    public init(
        id: String, kind: Kind, title: String, text: String, live: Bool = false, stats: String? = nil,
        warning: Bool = false, collapsed: Bool = false, detail: Detail? = nil
    ) {
        self.id = id
        self.kind = kind
        self.title = title
        self.text = text
        self.live = live
        self.stats = stats
        self.warning = warning
        self.collapsed = collapsed
        self.detail = detail
    }
}

extension ConversationEntry {
    static let noAnswerText = "(no answer text)"
    static let limitTitle = "Ended at the step limit"

    static func render(
        _ transcript: TranscriptRecord, _ turns: [ConversationTurn], live: Bool,
        mergeContinuations: Bool = true
    ) -> [ConversationEntry] {
        var output: [ConversationEntry] = []
        for number in turns.indices {
            let rows = render(turn: number, of: turns, in: transcript, live: live)
            guard mergeContinuations, turns[number].kind == .continuation,
                let previous = output.lastIndex(where: { $0.kind == .answer && $0.detail != .noAnswer })
            else {
                output += rows
                continue
            }
            var answer = output.remove(at: previous)
            if number > 0, turns[number - 1].kind != .continuation {
                answer.text = transcript.entries.first(where: { $0.id == answer.id })?.response ?? answer.text
            }
            output.removeAll { $0.id == "cap-\(number - 1)" }
            let additions = rows.filter { $0.kind == .answer && $0.detail != .noAnswer }
            for addition in additions {
                answer.text += transcript.entries.first(where: { $0.id == addition.id })?.response ?? addition.text
                answer.live = addition.live
                if let stats = addition.stats {
                    answer.stats = [answer.stats, stats].compactMap { $0 }.joined(separator: "\n")
                }
            }
            output += rows.filter { [.thinking, .toolCall, .toolResult].contains($0.kind) }
            output.append(answer)
            output += rows.filter { [.failure, .note].contains($0.kind) }
        }
        return output
    }

    private static func render(
        turn number: Int, of turns: [ConversationTurn], in transcript: TranscriptRecord, live: Bool
    ) -> [ConversationEntry] {
        let turn = turns[number]
        let policy = turn.policy
        let records = transcript.entries
        let end = number + 1 < turns.count ? min(turns[number + 1].start, records.count) : records.count
        let steps: [TranscriptStep]
        if let finalStart = turn.finalAnswerStart {
            let boundary = min(max(finalStart, turn.start), end)
            let research = transcript.steps(from: turn.start + 1, to: boundary)
            steps =
                Array(research.prefix(max(0, policy.maxSteps - turn.offset - 1)))
                + transcript.steps(from: min(boundary + 1, end), to: end)
        } else {
            steps = transcript.steps(from: turn.start + 1, to: end)
        }
        let cut = turn.finalization == .cutOff
        let answered = turn.finish != Finish.none && turn.finish != .stopped && turn.failure == nil && !cut
        var out = [ConversationEntry(id: "turn-\(number)", kind: .user, title: "You", text: turn.prompt)]
        var partial: String?
        let allowed = policy.maxSteps - turn.offset
        for (index, step) in steps.enumerated() where !cut || index < allowed {
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
            var shown = step.generated.flatMap { index in
                index == turn.finalAnswerStart ? [] : entries(records[index], live: live && index == records.count - 1)
            }
            if trailing, turn.finish == .stopped, !cut {
                partial = answer
                shown.removeAll { $0.kind == .answer }
            }
            let stepNumber =
                turn.finalAnswerStart.map { step.generated.lowerBound > $0 } == true
                ? policy.maxSteps : turn.offset + index + 1
            let line = turn.steps.first { $0.step == stepNumber }?.line
            if trailing, answered, answer.isEmpty {
                shown.append(
                    ConversationEntry(
                        id: "answer-\(number)", kind: .answer, title: "Assistant", text: noAnswerText, stats: line,
                        detail: .noAnswer))
            } else if !shown.isEmpty {
                shown[shown.count - 1].stats = line
            }
            out += shown
            out += step.outputs.map { $0.flatMap { entries(records[$0], live: false) } } ?? []
        }
        if cut {
            let range = min(turn.start + 1, end)..<end
            let limit = steps.indices.contains(allowed - 1) ? steps[allowed - 1] : nil
            let summary = limitSummary(records, turn: range, last: limit)
            let detail = Detail.stepLimit(
                steps: policy.maxSteps, fetched: summary.fetched, saved: summary.saved, open: summary.open)
            out.append(
                ConversationEntry(
                    id: "limit-\(number)", kind: .failure, title: limitTitle, text: limitText(detail), detail: detail))
        } else if turn.finish == .stopped {
            let answer = partial.flatMap { $0.isEmpty ? nil : $0 }
            out.append(
                ConversationEntry(
                    id: "stopped-\(number)", kind: .failure, title: "Stopped",
                    text: answer ?? "(stopped before any answer)", detail: .stopped(partial: answer != nil)))
        }
        if let failure = turn.failure {
            out.append(
                ConversationEntry(
                    id: "failed-\(number)", kind: .failure, title: "Generation failed", text: failure,
                    detail: .failure))
        }
        if answered, turn.finish == .cap {
            let lastText = records[min(turn.start + 1, end)..<end].compactMap(\.response).last ?? ""
            if let cut = IncompleteToolCall.find(in: lastText) {
                let tool = cut.name ?? "tool"
                out.append(
                    ConversationEntry(
                        id: "cap-\(number)", kind: .failure,
                        title: cut.isNote ? "Note was not saved" : "Tool call was not executed",
                        text: cut.isNote
                            ? (tool == "write_note"
                                ? "The write_note call was cut at the token limit; the note was not saved."
                                : "The append_note call was cut at the token limit; the new part was not saved. Earlier parts are preserved.")
                            : "The tool call was cut at the token limit and was not executed.",
                        detail: .cutToolCall(name: cut.name, note: cut.isNote)))
            } else {
                out.append(
                    ConversationEntry(
                        id: "cap-\(number)", kind: .failure,
                        title: "Answer cut at the token limit (\(policy.maxTokens))",
                        text:
                            "The answer above stops where generation ran out of tokens. Raise Max tokens or ask for something shorter.",
                        detail: .tokenLimit(policy.maxTokens)))
            }
        }
        out += turn.notes.enumerated().map { index, note in
            ConversationEntry(
                id: "note-\(number)-\(index)", kind: .note, title: note.title, text: note.text, warning: note.warning)
        }
        if turn.finalAnswerStart != nil {
            let limit = policy.maxSteps - 1
            out.append(
                ConversationEntry(
                    id: "budget-\(number)", kind: .note, title: "Research limit reached",
                    text: "Research stopped after \(limit) steps. The answer uses the information already collected.",
                    warning: true, detail: .researchLimit(limit)))
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
        _ records: [TranscriptRecord.Entry], turn range: Range<Int>, last step: TranscriptStep?
    ) -> (fetched: [String], saved: [String], open: [String]) {
        var outputs: [String: String] = [:]
        for case .toolOutput(let id, _, let text) in records[range] { outputs[id] = text }
        func done(_ call: ToolCallRecord) -> Bool {
            outputs[call.id].map { !$0.hasPrefix("error:") } ?? false
        }
        func unique(_ items: [String]) -> [String] {
            var seen = Set<String>()
            return items.filter { seen.insert($0).inserted }
        }
        let finished = records[range].flatMap(\.calls).filter(done)
        let open = step.map { records[$0.generated].flatMap(\.calls).filter { !done($0) } } ?? []
        return (
            unique(finished.filter { $0.name == "fetch_page" }.compactMap { $0.value("url") }),
            unique(finished.filter { $0.name == "write_note" }.compactMap { $0.value("title") }),
            unique(open.map { ([$0.name] + $0.argumentLines.prefix(1).map(\.value)).joined(separator: " ") })
        )
    }

    private static func limitText(_ detail: Detail) -> String {
        guard case .stepLimit(let steps, let fetched, let saved, let open) = detail else { return "" }
        func list(_ items: [String]) -> String { items.isEmpty ? "none" : items.joined(separator: ", ") }
        return [
            "The turn reached its \(steps)-step limit before a final answer.",
            "Fetched: " + list(fetched),
            "Saved notes: " + list(saved),
            "Still open: " + list(open),
        ].joined(separator: "\n")
    }

    private static func trimmed(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
