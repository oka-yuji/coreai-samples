import ConversationFeature
import ConversationRuntime
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
    static let repairFetchLimit = 3

    weak var session: ConversationSession?
    var guardActive = false
    var repairFetchLimit: Int?
    private(set) var repairFetches = 0
    private(set) var ledger: [FetchRecord] = []
    private(set) var fetchAttempts: [FetchAttempt] = []
    private(set) var seenPages: [String: String] = [:]
    private(set) var notesWritten: [(key: String, text: String)] = []
    private(set) var draftOrder: [String] = []
    private(set) var webToolCalls = 0
    private var noteIDs: [String] = []
    private var notePartsByID: [String: [String]] = [:]
    private var noteReceipts: [String: NoteReceipt] = [:]
    private var noteIDsByTitle: [String: String] = [:]
    private var taskNotesDirectory: URL?
    let preferences: ConversationPreferences
    private nonisolated let slots = Mutex<[String: Slot]>([:])

    init(preferences: ConversationPreferences) {
        self.preferences = preferences
    }

    var notesDirectory: URL { taskNotesDirectory ?? preferences.notesDirectory }
    var stopped: Bool { session?.stopRequested == true || Task.isCancelled }

    var noteTextsForValidation: [String] { notesWritten.map(\.text) + notePartsByID.values.flatMap { $0 } }

    var evidence: ConversationEvidence {
        ConversationEvidence(
            pages: ledger.map {
                .init(url: $0.url, finalURL: $0.finalURL, sha256: $0.sha256, characters: $0.characters)
            },
            attempts: fetchAttempts.map { .init(url: $0.url, ok: $0.ok) }, seen: seenPages, noteIDs: noteIDs,
            taskWebCalls: webToolCalls)
    }

    func restoreEvidence(_ evidence: ConversationEvidence, notesDirectory: URL? = nil) {
        ledger = evidence.pages.map {
            FetchRecord(url: $0.url, finalURL: $0.finalURL, sha256: $0.sha256, characters: $0.characters)
        }
        fetchAttempts = evidence.attempts.map { FetchAttempt(url: $0.url, ok: $0.ok) }
        seenPages = evidence.seen
        noteIDs = evidence.noteIDs ?? []
        notesWritten = []
        draftOrder = []
        noteIDsByTitle = [:]
        notePartsByID = [:]
        noteReceipts = [:]
        taskNotesDirectory = notesDirectory
        webToolCalls = evidence.taskWebCalls ?? evidence.attempts.count
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
            if session?.stopRequested == true { return }
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
        let start = Date()
        let output: AgentToolOutput
        if call.name == "fetch_page", !reserveRepairFetch() {
            output = AgentToolOutput(text: Self.repairFetchLimitText)
        } else {
            output = await AgentTools.run(call, settings: settings)
        }
        record(call, output, start: start, end: Date())
        return output.text
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

    private func record(_ call: AgentToolCall, _ output: AgentToolOutput, start: Date, end: Date) {
        seenPages.merge(output.seen) { _, new in new }
        if call.name == "fetch_page" {
            let page = CitationGuard.fetchRecord(call, output)
            if let page { ledger.append(page) }
            fetchAttempts.append(
                FetchAttempt(
                    url: (call.value("url") ?? "").trimmingCharacters(in: .whitespacesAndNewlines), ok: page != nil))
        }
        if call.name == "web_search" || call.name == "fetch_page" { webToolCalls += 1 }
        if let note = output.note {
            recordNote(note, replacingTitle: call.name == "write_note" && repairFetchLimit != nil)
        }
        session?.updateEvidence(evidence)
        log(call, output, start: start, end: end)
    }

    private func log(_ call: AgentToolCall, _ output: AgentToolOutput, start: Date, end: Date) {
        guard let session, session.mode == .work else { return }
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
