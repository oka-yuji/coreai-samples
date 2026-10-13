import ConversationFeature
import CryptoKit
import Foundation
import FoundationModels

struct FetchRecord: Codable, Sendable, Equatable {
    var url: String
    var finalURL: String
    var sha256: String
    var characters: Int
}

struct FetchAttempt: Codable, Sendable, Equatable {
    var url: String
    var ok: Bool
}

struct LedgerCheck: Codable, Sendable, Equatable {
    var stage: String
    var citationsTotal: Int
    var citationsFetched: Int
    var citationsFailed: Int
    var citationsUnfetched: [String]
    var noteCitationsUnfetched: [String]
    var sourcesRequired: Bool
    var sourcesMissing: Bool

    var violation: Bool { !citationsUnfetched.isEmpty || !noteCitationsUnfetched.isEmpty || sourcesMissing }

    var reason: String {
        let parts = [
            (!citationsUnfetched.isEmpty, "unfetched"), (!noteCitationsUnfetched.isEmpty, "note-unfetched"),
            (sourcesMissing, "sources-missing"),
        ].filter(\.0).map(\.1)
        return parts.isEmpty ? "none" : parts.joined(separator: "+")
    }

    var unfetched: [String] {
        citationsUnfetched
            + noteCitationsUnfetched.filter { url in
                !citationsUnfetched.contains {
                    CitationGuard.guardNormalized($0) == CitationGuard.guardNormalized(url)
                }
            }
    }

    var logLine: String {
        "[guard] check stage=\(stage) citations_total=\(citationsTotal) citations_fetched=\(citationsFetched) "
            + "citations_failed=\(citationsFailed) citations_unfetched=\(AgentTools.jsonList(citationsUnfetched)) "
            + "note_citations_unfetched=\(AgentTools.jsonList(noteCitationsUnfetched)) "
            + "sources_required=\(sourcesRequired ? "yes" : "no") sources_missing=\(sourcesMissing)"
    }
}

struct CitationCheck {
    var urls: [(url: String, seen: Bool, overlap: Double)] = []
    var total: Int { urls.count }
    var seenCount: Int { urls.count { $0.seen } }
    var meanOverlap: Double {
        urls.isEmpty ? 0 : urls.map(\.overlap).reduce(0, +) / Double(urls.count)
    }
    var note: String {
        String(
            format: "%d/%d cited URLs were retrieved in this task · mean 8-gram overlap %.2f",
            seenCount, total, meanOverlap)
    }
}

@MainActor
final class CitationGuard {
    enum Verdict {
        case repair(String)
        case done(note: (text: String, warning: Bool)?)
    }

    static let title = "Citation guard"
    static let citationsTitle = "Citations"
    nonisolated static let noAnswerText = "(no answer text)"
    nonisolated static let verifiedLabel = "Cited sources fetched"
    nonisolated static let heldLabel = "Could not confirm that the cited sources were fetched"
    static let citationSourceLimit = 200_000

    private let preferences: ConversationPreferences
    private let tools: ToolRunner
    private(set) var active = false
    private(set) var checks: [LedgerCheck] = []
    private(set) var outcome: String?
    private(set) var reason = ""
    private(set) var answerBeforeRepair: String?
    private(set) var guardedAnswer: String?
    private(set) var repairTurns = 0
    private(set) var repairSeconds: Double?
    private(set) var repairPromptTokens: Int?
    private(set) var repairGeneratedTokens: Int?
    private(set) var repairPending = false
    private var repairStarted: Date?
    private var repairUsage: LanguageModelSession.Usage?
    private var notesCommitted = false

    init(preferences: ConversationPreferences, tools: ToolRunner) {
        self.preferences = preferences
        self.tools = tools
    }

    func begin(active: Bool) {
        self.active = active
        checks = []
        outcome = nil
        reason = ""
        answerBeforeRepair = nil
        guardedAnswer = nil
        repairTurns = 0
        repairSeconds = nil
        repairPromptTokens = nil
        repairGeneratedTokens = nil
        repairPending = false
        repairStarted = nil
        repairUsage = nil
        notesCommitted = false
    }

    func settle(_ end: TurnEnd, step: Int, maxSteps: Int, stopped: Bool, usage: LanguageModelSession.Usage?)
        -> Verdict
    {
        if repairPending { return settleRepair(end, usage: usage) }
        let check = ledgerCheck(stage: "answer", answer: end.answer)
        checks = [check]
        guard active else { return .done(note: nil) }
        if end.outcome != TurnEnd.answered {
            finishIncompleteNotes(reason: end.outcome)
        } else if !check.violation {
            outcome = "verified"
            reason = "none"
        } else if let blocker = Self.repairBlocker(
            stepsUsed: step, maxSteps: maxSteps, deadline: preferences.deadline, now: Date(), stopped: stopped)
        {
            outcome = "held"
            reason = blocker
        } else {
            answerBeforeRepair = end.answer
            repairTurns = 1
            repairPending = true
            repairStarted = Date()
            repairUsage = usage
            tools.repairFetchLimit = ToolRunner.repairFetchLimit
            AgentTools.log("[guard] repair turn: \(check.reason)")
            return .repair(AgentPromptBuilder.repairPrompt(unfetched: check.unfetched))
        }
        if end.outcome == TurnEnd.answered { finishNotes(commit: outcome == "verified") }
        return .done(note: note())
    }

    private func settleRepair(_ end: TurnEnd, usage: LanguageModelSession.Usage?) -> Verdict {
        repairPending = false
        tools.repairFetchLimit = nil
        repairSeconds = repairStarted.map { Date().timeIntervalSince($0) }
        if let before = repairUsage, let usage {
            repairPromptTokens = usage.input.totalTokenCount - before.input.totalTokenCount
            repairGeneratedTokens = usage.output.totalTokenCount - before.output.totalTokenCount
        }
        if end.outcome == TurnEnd.answered {
            let check = ledgerCheck(stage: "repaired", answer: end.answer)
            checks.append(check)
            outcome = check.violation ? "held" : "repaired"
            reason = check.violation ? "re-violation" : checks[0].reason
            guardedAnswer = end.answer
        } else {
            outcome = "held"
            reason =
                end.outcome == "stopped" && preferences.deadline.map({ Date() >= $0 }) == true
                ? "deadline" : "repair-" + end.outcome
            guardedAnswer = answerBeforeRepair
        }
        finishNotes(commit: outcome == "repaired")
        return .done(note: note())
    }

    func finalAnswer(_ entries: [ConversationEntry]) -> String {
        guardedAnswer ?? entries.last { $0.kind == .answer }?.text ?? ""
    }

    private func ledgerCheck(stage: String, answer: String) -> LedgerCheck {
        Self.ledgerCheck(
            stage: stage, answer: answer, notes: tools.notesWritten.map(\.text), ledger: tools.ledger,
            attempts: tools.fetchAttempts, sourcesRequired: preferences.sourcesRequired,
            usedWebTools: tools.webToolCalls > 0)
    }

    func finishIncompleteNotes(reason: String) {
        let check = Self.ledgerCheck(
            stage: "notes", answer: "", notes: tools.notesWritten.map(\.text), ledger: tools.ledger,
            attempts: tools.fetchAttempts, sourcesRequired: false, usedWebTools: false)
        outcome = "skipped"
        self.reason =
            check.noteCitationsUnfetched.isEmpty
            ? reason : reason + ": note-unfetched: " + check.noteCitationsUnfetched.joined(separator: ", ")
        finishNotes(commit: check.noteCitationsUnfetched.isEmpty)
    }

    func finishNotes(commit: Bool) {
        let drafts = tools.draftOrder
        guard !drafts.isEmpty else { return }
        if commit {
            notesCommitted =
                AgentTools.commitDrafts(drafts, notesDirectory: tools.notesDirectory).count == drafts.count
        } else {
            AgentTools.holdDrafts(
                drafts, notesDirectory: tools.notesDirectory, reason: "\(outcome ?? "-"): \(reason)")
        }
    }

    private func note() -> (text: String, warning: Bool)? {
        Self.guardNoteText(
            outcome: outcome, reason: reason, check: checks.last,
            drafts: !tools.draftOrder.isEmpty && !notesCommitted)
    }

    func saveRunMetrics(session: ConversationSession) async {
        var record = MetricsLog.agent("agent-run", model: session.modelName)
        record.agentSession = session.id
        record.agentTurn = session.turnIndex
        record.agentStep = session.step
        record.agentEffort = session.effort
        record.agentInstructions = preferences.citeFetched ? "cite-fetched" : "frozen"
        record.citationGuard = active
        record.ledgerChecks = checks
        record.fetchLedger = tools.ledger
        record.fetchAttempts = tools.fetchAttempts
        record.guardOutcome = outcome
        record.guardReason = outcome == nil ? nil : reason
        if repairTurns > 0 {
            record.repairTurns = repairTurns
            record.repairFetches = tools.repairFetches
            record.repairSeconds = repairSeconds
            record.repairPromptTokens = repairPromptTokens
            record.repairGeneratedTokens = repairGeneratedTokens
        }
        if let cassette = preferences.cassette {
            record.cassetteHits = await cassette.hits
            record.cassetteMisses = await cassette.misses
            record.cassetteRecords = await cassette.records
        }
        MetricsLog.save(record)
    }

    static func guardNoteText(outcome: String?, reason: String, check: LedgerCheck?, drafts: Bool) -> (
        text: String, warning: Bool
    )? {
        let fetched =
            "\(check?.citationsFetched ?? 0)/\(check?.citationsTotal ?? 0) cited URLs were fetched in this task"
            + ((check?.citationsFailed ?? 0) > 0 ? " · \(check?.citationsFailed ?? 0) could not be fetched" : "")
        let kept = drafts ? " · notes kept as drafts in \(AgentTools.draftFolderName)" : ""
        switch outcome {
        case "verified", "repaired":
            let allFetched = (check?.citationsFetched ?? 0) > 0 && check?.citationsFetched == check?.citationsTotal
            return (
                (allFetched ? verifiedLabel + " · " : "")
                    + (outcome == "repaired" ? "after one repair turn · " : "") + fetched, false
            )
        case "held":
            let unfetched = check?.unfetched ?? []
            return (
                heldLabel + " · " + reason
                    + (unfetched.isEmpty ? "" : " · not fetched: " + unfetched.joined(separator: ", ")) + kept, true
            )
        default:
            return drafts ? ("The task ended without a complete answer (\(reason))" + kept, true) : nil
        }
    }

    static func checkCitations(_ answer: String, pages: [String: String]) async -> CitationCheck? {
        let (body, urls) = splitSources(answer)
        guard !urls.isEmpty else { return nil }
        var lookup: [String: String] = [:]
        for (url, text) in pages { lookup[normalized(url)] = text }
        var check = CitationCheck()
        for url in urls {
            let source = lookup[normalized(url)]
            var overlap = 0.0
            if let source { overlap = await overlapOffMain(body, String(source.prefix(citationSourceLimit))) }
            check.urls.append((url, source != nil, overlap))
            AgentTools.log(
                String(format: "[cite] url=%@ seen=%@ overlap=%.2f", url, source == nil ? "no" : "yes", overlap))
        }
        return check
    }

    @concurrent nonisolated static func overlapOffMain(_ body: String, _ source: String) async -> Double {
        gramOverlap(body, source)
    }

    nonisolated static func splitSources(_ answer: String) -> (body: String, urls: [String]) {
        let lines = answer.split(separator: "\n", omittingEmptySubsequences: false)
        guard
            let start = lines.lastIndex(where: { line in
                let head = line.trimmingCharacters(in: CharacterSet(charactersIn: " \t*#-")).lowercased()
                return head.hasPrefix("sources") || head.hasPrefix("\u{51FA}\u{5178}")
                    || head.hasPrefix("\u{53C2}\u{8003}")
            })
        else { return (answer, []) }
        return (lines[..<start].joined(separator: "\n"), urls(in: lines[start...].joined(separator: "\n")))
    }

    nonisolated static func urls(in text: String) -> [String] {
        var found: [String] = []
        for match in text.matches(of: /https?:\/\/[^\s<>"'()\[\]]+/) {
            var url = String(match.0)
            while let last = url.last,
                ".,;:\u{3001}\u{3002}\u{00BB}\u{201D}".contains(last)
            { url.removeLast() }
            if !url.isEmpty, !found.contains(url) { found.append(url) }
        }
        return found
    }

    nonisolated static func normalized(_ url: String) -> String {
        var text = url.lowercased()
        while text.hasSuffix("/") { text.removeLast() }
        return text
    }

    nonisolated static func repairBlocker(stepsUsed: Int, maxSteps: Int, deadline: Date?, now: Date, stopped: Bool)
        -> String?
    {
        if stopped { return "stopped" }
        if stepsUsed >= maxSteps { return "no-steps-left" }
        if let deadline, now >= deadline { return "deadline" }
        return nil
    }

    nonisolated static func fetchRecord(_ call: AgentToolCall, _ output: AgentToolOutput) -> FetchRecord? {
        guard call.name == "fetch_page", let final = output.finalURL, !output.text.isEmpty else { return nil }
        let digest = SHA256.hash(data: Data(output.text.utf8)).map { String(format: "%02x", $0) }.joined()
        return FetchRecord(
            url: (call.value("url") ?? "").trimmingCharacters(in: .whitespacesAndNewlines), finalURL: final,
            sha256: digest,
            characters: output.text.count)
    }

    nonisolated static func ledgerCheck(
        stage: String, answer: String, notes: [String], ledger: [FetchRecord], attempts: [FetchAttempt] = [],
        sourcesRequired: Bool?, usedWebTools: Bool
    ) -> LedgerCheck {
        let fetched = Set(ledger.flatMap { [guardNormalized($0.url), guardNormalized($0.finalURL)] })
        let failed = Set(attempts.filter { !$0.ok }.map { guardNormalized($0.url) }).subtracting(fetched)
        let cited = guardURLs(in: answer)
        let confirmed = cited.filter { fetched.contains(guardNormalized($0)) }
        let tried = fetched.union(failed)
        let unfetched = cited.filter { !tried.contains(guardNormalized($0)) }
        var noteUnfetched: [String] = []
        for url in notes.flatMap(guardURLs(in:))
        where !tried.contains(guardNormalized(url))
            && !noteUnfetched.contains(where: { guardNormalized($0) == guardNormalized(url) })
        {
            noteUnfetched.append(url)
        }
        let everyTryFailed = !attempts.isEmpty && attempts.allSatisfy { !$0.ok }
        let required = sourcesRequired ?? ((usedWebTools || !cited.isEmpty) && !everyTryFailed)
        return LedgerCheck(
            stage: stage, citationsTotal: cited.count, citationsFetched: confirmed.count,
            citationsFailed: cited.count - confirmed.count - unfetched.count, citationsUnfetched: unfetched,
            noteCitationsUnfetched: noteUnfetched, sourcesRequired: required,
            sourcesMissing: required && confirmed.isEmpty)
    }

    nonisolated static func guardURLs(in text: String) -> [String] {
        var found: [String] = []
        var seen = Set<String>()
        var rest = text[...]
        while let hit = rest.firstMatch(of: /https?:\/\//.ignoresCase()) {
            var url = String(hit.output)
            var open: [Character] = []
            var cursor = hit.range.upperBound
            scan: while cursor < rest.endIndex {
                let character = rest[cursor]
                if character.isWhitespace || "<>\"'`{}|\\^".contains(character)
                    || (!character.isASCII && (character.isPunctuation || character.isSymbol))
                {
                    break
                }
                switch character {
                case "(", "[":
                    open.append(character)
                case ")", "]":
                    guard open.last == (character == ")" ? "(" : "[") else { break scan }
                    open.removeLast()
                default:
                    break
                }
                url.append(character)
                cursor = rest.index(after: cursor)
            }
            rest = rest[cursor...]
            while let last = url.last, ".,;:!?*([".contains(last) { url.removeLast() }
            guard url.count > hit.output.count, seen.insert(guardNormalized(url)).inserted else { continue }
            found.append(url)
        }
        return found
    }

    nonisolated static func guardNormalized(_ url: String) -> String {
        var text = url
        if let hash = text.firstIndex(of: "#") { text = String(text[..<hash]) }
        if let scheme = text.range(of: "://") {
            let authorityEnd = text[scheme.upperBound...].firstIndex { "/?".contains($0) } ?? text.endIndex
            text = text[..<authorityEnd].lowercased() + percentNormalized(String(text[authorityEnd...]))
        }
        while text.hasSuffix("/"), !text.hasSuffix("://") { text.removeLast() }
        return text
    }

    nonisolated static func percentNormalized(_ text: String) -> String {
        var out = ""
        var bytes: [UInt8] = []
        var written = ""
        func flush() {
            if !bytes.isEmpty { out += String(bytes: bytes, encoding: .utf8) ?? written }
            bytes = []
            written = ""
        }
        var index = text.startIndex
        while index < text.endIndex {
            if text[index] == "%", let end = text.index(index, offsetBy: 3, limitedBy: text.endIndex),
                let byte = UInt8(text[text.index(after: index)..<end], radix: 16),
                text[text.index(after: index)..<end].allSatisfy(\.isHexDigit)
            {
                let escape = String(text[index..<end])
                if [0x2F, 0x3F, 0x23, 0x25].contains(byte) {
                    flush()
                    out += escape.uppercased()
                } else {
                    bytes.append(byte)
                    written += escape
                }
                index = end
            } else {
                flush()
                out.append(text[index])
                index = text.index(after: index)
            }
        }
        flush()
        return out
    }

    nonisolated static func gramOverlap(_ body: String, _ source: String, size: Int = 8) -> Double {
        func grams(_ text: String) -> [String] {
            let characters = Array(text.lowercased().filter { !$0.isWhitespace })
            guard characters.count >= size else { return [] }
            return (0...(characters.count - size)).map { String(characters[$0..<($0 + size)]) }
        }
        let needles = grams(body)
        guard !needles.isEmpty else { return 0 }
        let haystack = Set(grams(source))
        return Double(needles.count { haystack.contains($0) }) / Double(needles.count)
    }
}
