import ConversationFeature
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
    private var draftNames: [String: String] = [:]
    let preferences: ConversationPreferences
    private nonisolated let slots = Mutex<[String: Slot]>([:])

    init(preferences: ConversationPreferences) {
        self.preferences = preferences
    }

    var notesDirectory: URL { preferences.notesDirectory }

    func beginTask(guardActive: Bool) {
        self.guardActive = guardActive
        repairFetchLimit = nil
        repairFetches = 0
        webToolCalls = 0
        notesWritten = []
        draftNames = [:]
        draftOrder = []
    }

    func tools(fileOperations: Bool) -> [any Tool] {
        AgentPromptBuilder.tools(fileOperations: fileOperations) { [weak self] call in
            await self?.call(call) ?? ConversationFeature.endedRefusal
        }
    }

    private func call(_ call: AgentToolCall) async -> String {
        guard let record = session?.request(call) else { return ConversationFeature.endedRefusal }
        switch await decision(for: record.id) {
        case .refuse(let reason):
            return reason
        case .cancelled:
            return ConversationFeature.endedRefusal
        case .run(let finalize):
            let output = await execute(call)
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
        let settings = preferences.toolSettings
        await session?.snapshotTokens()
        let start = Date()
        let output: AgentToolOutput
        if call.name == "fetch_page", !reserveRepairFetch() {
            output = AgentToolOutput(text: Self.repairFetchLimitText)
        } else if call.name == "write_note", let draft = writeDraft(call) {
            output = draft
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

    private func writeDraft(_ call: AgentToolCall) -> AgentToolOutput? {
        guard guardActive else { return nil }
        let title = call.value("title") ?? ""
        let content = call.value("content") ?? ""
        guard !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return AgentToolOutput(text: "error: write_note needs a content parameter")
        }
        let key = AgentTools.noteFileName(title)
        do {
            let name = try AgentTools.writeDraft(
                title: title, content: content, notesDirectory: notesDirectory,
                replacing: repairFetchLimit != nil ? draftNames[key] : nil)
            draftNames[key] = name
            if !draftOrder.contains(name) { draftOrder.append(name) }
            return AgentToolOutput(
                text: AgentTools.savedText(
                    notesDirectory.appending(path: name).path(percentEncoded: false),
                    stable: preferences.stableNotePaths))
        } catch let failure as AgentToolMessage {
            return AgentToolOutput(text: failure.text)
        } catch {
            return AgentToolOutput(text: "error: write_note failed (\(error))")
        }
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
        if call.name == "write_note", output.text.hasPrefix("saved to ") {
            let text = (call.value("title") ?? "") + "\n" + (call.value("content") ?? "")
            if let index = notesWritten.firstIndex(where: { $0.key == output.text }) {
                notesWritten[index].text = text
            } else {
                notesWritten.append((output.text, text))
            }
        }
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
