import ConversationFeature
import ConversationRuntime
import CryptoKit
import Foundation
import FoundationModels
import Synchronization

@MainActor
final class ToolRunner {
    enum Decision: Equatable, Sendable {
        case run(finalize: Bool)
        case refuse(String)
        case cancelled
    }

    private enum Slot: Sendable {
        case waiting(CheckedContinuation<Decision, Never>)
        case settled(Decision)
    }

    nonisolated static let repairFetchLimitText = "error: citation repair fetch limit reached"
    static let fetchOriginRefusal =
        "error: fetch_page can open only URLs from the user's messages, search results, or pages fetched in this conversation. Run web_search first, or use a URL the user wrote."
    static let repairFetchLimit = 3

    weak var session: ConversationSession?
    var guardActive = false
    var repairFetchLimit: Int?
    private(set) var repairFetches = 0
    private(set) var ledger: [FetchRecord] = []
    private(set) var fetchAttempts: [FetchAttempt] = []
    private(set) var seenPages: [String: String] = [:]
    private(set) var seenURLs: [String]?
    private(set) var notesWritten: [(key: String, text: String)] = []
    private(set) var draftOrder: [String] = []
    private(set) var webToolCalls = 0
    var requests: [String] = []
    private var noteIDs: [String] = []
    private var notePartsByID: [String: [String]] = [:]
    private var noteReceipts: [String: NoteReceipt] = [:]
    private var noteIDsByTitle: [String: String] = [:]
    private var taskNotesDirectory: URL?
    private(set) var sources = ConversationSources()
    private(set) var suppliedSourceIDs: Set<String> = []
    let preferences: ConversationPreferences
    private let executor: ((AgentToolCall) async -> AgentToolOutput)?
    private nonisolated let slots = Mutex<[String: Slot]>([:])

    init(preferences: ConversationPreferences, executor: ((AgentToolCall) async -> AgentToolOutput)? = nil) {
        self.preferences = preferences
        self.executor = executor
    }

    var notesDirectory: URL { taskNotesDirectory ?? preferences.notesDirectory }
    var stopped: Bool { session?.isStopping == true || Task.isCancelled }

    var noteTextsForValidation: [String] { notesWritten.map(\.text) + notePartsByID.values.flatMap { $0 } }

    var evidence: ConversationEvidence {
        ConversationEvidence(
            pages: ledger.map {
                .init(url: $0.url, finalURL: $0.finalURL, sha256: $0.sha256, characters: $0.characters)
            },
            attempts: fetchAttempts.map { .init(url: $0.url, ok: $0.ok, refused: $0.refused) }, seen: seenPages,
            noteIDs: noteIDs,
            taskWebCalls: webToolCalls, sources: sources.records.isEmpty ? nil : sources, seenURLs: seenURLs)
    }

    func restoreEvidence(
        _ evidence: ConversationEvidence, notesDirectory: URL? = nil, suppliedSourceIDs: Set<String> = [],
        requests: [String] = []
    ) {
        ledger = evidence.pages.map {
            FetchRecord(url: $0.url, finalURL: $0.finalURL, sha256: $0.sha256, characters: $0.characters)
        }
        fetchAttempts = evidence.attempts.map { FetchAttempt(url: $0.url, ok: $0.ok, refused: $0.refused) }
        seenPages = evidence.seen
        seenURLs = evidence.seenURLs
        noteIDs = evidence.noteIDs ?? []
        notesWritten = []
        draftOrder = []
        noteIDsByTitle = [:]
        notePartsByID = [:]
        noteReceipts = [:]
        taskNotesDirectory = notesDirectory
        webToolCalls = evidence.taskWebCalls ?? evidence.attempts.count
        sources = evidence.sources ?? ConversationSources()
        self.suppliedSourceIDs = suppliedSourceIDs
        self.requests = requests
    }

    func markSourcesSupplied(_ ids: [String]) { suppliedSourceIDs.formUnion(ids) }

    func registerSources(_ records: [ConversationSource]) throws -> [String] {
        let ids = try sources.register(records)
        let groups = Dictionary(grouping: records) { $0.document + "\n" + $0.version }
        for key in groups.keys.sorted() {
            guard let group = groups[key], let source = group.first,
                ["http", "https"].contains(URL(string: source.document)?.scheme?.lowercased())
            else { continue }
            let text = group.flatMap(\.passages).map(\.text).joined(separator: "\n\n")
            let hash = SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
            ledger.append(
                FetchRecord(
                    url: source.document, finalURL: source.document, sha256: hash, characters: text.count))
            see([source.document: text])
        }
        return ids
    }

    func beginTask(guardActive: Bool, continuing: Bool = false) {
        self.guardActive = guardActive
        repairFetchLimit = nil
        repairFetches = 0
        if !continuing {
            webToolCalls = 0
            notesWritten = []
            noteIDs = []
            noteIDsByTitle = [:]
            notePartsByID = [:]
            noteReceipts = [:]
            draftOrder = []
            taskNotesDirectory = preferences.notesDirectory
        }
    }

    func prepareContinuation() async throws {
        for id in noteIDs {
            if session?.isStopping == true { return }
            let note = try await AgentTools.notes.read(noteID: id, directory: notesDirectory)
            recordNote(note, replacingTitle: false)
        }
    }

    func finishNotes(commit: Bool, reason: String) async throws -> [String] {
        guard commit else {
            try await AgentTools.notes.hold(noteIDs: draftOrder, reason: reason, directory: notesDirectory)
            return []
        }
        var warnings: [String] = []
        for id in noteIDs {
            guard !stopped else { throw CancellationError() }
            guard let checked = noteReceipts[id] else { throw NoteStorageError.missing }
            let note = try await AgentTools.notes.publish(checked, directory: notesDirectory)
            recordNote(note, replacingTitle: false)
            if let warning = note.warning { warnings.append(warning) }
        }
        return warnings
    }

    func tools(fileOperations: Bool, splitNotes: Bool = true) -> [any Tool] {
        AgentPromptBuilder.tools(fileOperations: fileOperations, splitNotes: splitNotes) { [weak self] call in
            await self?.call(call) ?? ConversationFeature.endedRefusal
        }
    }

    private func call(_ call: AgentToolCall) async -> String {
        guard let record = session?.request(call) else { return ConversationFeature.endedRefusal }
        let decision = await decision(for: record.id)
        let start = Date()
        var ran = false
        let text = await Self.output(for: decision) {
            ran = true
            return await execute(call)
        }
        if !ran { log(call, AgentToolOutput(text: text), start: start, end: Date()) }
        return text
    }

    static func output(for decision: Decision, run: () async -> String) async -> String {
        switch decision {
        case .refuse(let reason):
            return reason
        case .cancelled:
            return ConversationFeature.endedRefusal
        case .run(let finalize):
            let output = await run()
            return finalize ? ConversationFeature.finalized(output) : output
        }
    }

    nonisolated func decision(for id: String) async -> Decision {
        await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                let early: Decision? = slots.withLock { slots in
                    if case .settled(let decision)? = slots[id] {
                        slots[id] = nil
                        return decision
                    }
                    slots[id] = .waiting(continuation)
                    return nil
                }
                if let early { continuation.resume(returning: early) }
            }
        } onCancel: {
            resolve(id, .cancelled)
        }
    }

    nonisolated func resolve(_ id: String, _ decision: Decision) {
        let waiting: CheckedContinuation<Decision, Never>? = slots.withLock { slots in
            switch slots[id] {
            case .waiting(let continuation)?:
                slots[id] = nil
                return continuation
            case .settled?:
                return nil
            case nil:
                slots[id] = .settled(decision)
                return nil
            }
        }
        waiting?.resume(returning: decision)
    }

    nonisolated var waitingCount: Int {
        slots.withLock { slots in slots.values.count { if case .waiting = $0 { true } else { false } } }
    }

    func drain() {
        let waiting: [CheckedContinuation<Decision, Never>] = slots.withLock { slots in
            let waiting = slots.values.compactMap { slot -> CheckedContinuation<Decision, Never>? in
                if case .waiting(let continuation) = slot { continuation } else { nil }
            }
            slots = [:]
            return waiting
        }
        for continuation in waiting { continuation.resume(returning: .cancelled) }
    }

    func execute(_ call: AgentToolCall) async -> String {
        var settings = preferences.toolSettings
        settings.notesDirectory = notesDirectory
        await session?.snapshotTokens()
        guard !stopped else { return ConversationFeature.endedRefusal }
        if let session, !session.allowedTools.contains(call.name) {
            return TurnPolicy.webTools.contains(call.name) ? TurnPolicy.searchRefusal : ConversationFeature.endedRefusal
        }
        let start = Date()
        let output: AgentToolOutput
        let permitted = admitted(call)
        let refused = permitted == nil
        let call = permitted ?? call
        if refused {
            output = AgentToolOutput(text: Self.fetchOriginRefusal)
        } else if call.name == "fetch_page", !reserveRepairFetch() {
            output = AgentToolOutput(text: Self.repairFetchLimitText)
        } else if let executor {
            output = await executor(call)
        } else {
            output = await AgentTools.run(call, settings: settings)
        }
        record(call, output, refused: refused, start: start, end: Date())
        return output.text
    }

    func allowsFetch(_ url: String) -> Bool {
        skipsOriginCheck || origin(of: url) != nil
    }

    private var skipsOriginCheck: Bool {
        preferences.cassette?.mode == .replay || (executor != nil && preferences.replaysRecordedTools)
    }

    private func admitted(_ call: AgentToolCall) -> AgentToolCall? {
        guard call.name == "fetch_page", !skipsOriginCheck else { return call }
        guard let origin = origin(of: call.value("url") ?? "") else { return nil }
        var admitted = call
        admitted.arguments = call.arguments.map { $0.name == "url" ? AgentArgument(name: "url", value: origin) : $0 }
        return admitted
    }

    private func origin(of url: String) -> String? {
        let written = url.trimmingCharacters(in: .whitespacesAndNewlines)
        let origins =
            requests.filter { $0 != ConversationHost.continuePrompt }.flatMap(CitationGuard.guardURLs(in:))
            + (seenURLs ?? Self.adding(seenPages, to: []))
        if let same = origins.first(where: { $0.utf8.elementsEqual(written.utf8) }) { return same }
        let target = CitationGuard.guardNormalized(written)
        return origins.first { CitationGuard.guardNormalized($0) == target }
    }

    private func see(_ seen: [String: String]) {
        guard !seen.isEmpty else { return }
        seenURLs = Self.adding(seen, to: seenURLs ?? Self.adding(seenPages, to: []))
        seenPages.merge(seen) { _, new in new }
    }

    private static func adding(_ seen: [String: String], to urls: [String]) -> [String] {
        var urls = urls
        var known = Set(urls)
        for (key, text) in seen.sorted(by: { $0.key < $1.key }) {
            for url in [key] + CitationGuard.guardURLs(in: text) where known.insert(url).inserted { urls.append(url) }
        }
        return urls
    }

    private func reserveRepairFetch() -> Bool {
        guard let limit = repairFetchLimit else { return true }
        guard repairFetches < limit else { return false }
        repairFetches += 1
        return true
    }

    private func recordNote(_ note: NoteReceipt, replacingTitle: Bool) {
        let title = AgentTools.noteFileName(note.title)
        if replacingTitle, let previous = noteIDsByTitle[title], previous != note.id {
            noteIDs.removeAll { $0 == previous }
            notePartsByID[previous] = nil
            noteReceipts[previous] = nil
            notesWritten.removeAll { $0.key == previous }
            draftOrder.removeAll { $0 == previous }
        }
        noteIDsByTitle[title] = note.id
        notePartsByID[note.id] = note.parts
        noteReceipts[note.id] = note
        if !noteIDs.contains(note.id) { noteIDs.append(note.id) }
        let text = note.title + "\n" + note.content
        if let index = notesWritten.firstIndex(where: { $0.key == note.id }) {
            notesWritten[index].text = text
        } else {
            notesWritten.append((note.id, text))
        }
        draftOrder.removeAll { $0 == note.id }
        if note.draft { draftOrder.append(note.id) }
    }

    private func record(_ call: AgentToolCall, _ output: AgentToolOutput, refused: Bool, start: Date, end: Date) {
        see(output.seen)
        if call.name == "fetch_page" {
            let page = CitationGuard.fetchRecord(call, output)
            if let page { ledger.append(page) }
            fetchAttempts.append(
                FetchAttempt(
                    url: (call.value("url") ?? "").trimmingCharacters(in: .whitespacesAndNewlines), ok: page != nil,
                    refused: refused ? true : nil))
        }
        if !refused, call.name == "web_search" || call.name == "fetch_page" { webToolCalls += 1 }
        if let note = output.note {
            recordNote(note, replacingTitle: call.name == "write_note" && repairFetchLimit != nil)
        }
        session?.updateEvidence(evidence)
        log(call, output, start: start, end: end)
    }

    private func log(_ call: AgentToolCall, _ output: AgentToolOutput, start: Date, end: Date) {
        guard let session, session.mode == .work, session.recordsMetrics else { return }
        let style = Date.ISO8601FormatStyle(includingFractionalSeconds: true)
        var record = MetricsLog.agent("agent-tool", model: session.modelName)
        record.agentSession = session.id
        record.agentTurn = session.turnIndex
        record.agentStep = session.step
        record.agentEffort = session.effort
        record.toolName = call.name
        record.toolArg = (call.value("url") ?? call.value("query") ?? call.value("title")).map {
            String($0.prefix(200))
        }
        record.toolStart = start.ISO8601Format(style)
        record.toolEnd = end.ISO8601Format(style)
        record.toolSeconds = end.timeIntervalSince(start)
        record.toolResultChars = output.text.count
        record.toolError = output.text.hasPrefix("error:")
        MetricsLog.save(record)
    }
}
