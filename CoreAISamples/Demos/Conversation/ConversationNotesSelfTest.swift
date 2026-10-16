import ConversationFeature
import ConversationRuntime
import Foundation

@MainActor
enum ConversationNotesSelfTest {
    static func run(_ check: (String, Bool, String) -> Void) async {
        let root = URL.temporaryDirectory.appending(path: "notes-selftest-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        do {
            for active in [false, true] {
                let preferences = ConversationPreferences(persists: false)
                preferences.notesFolderPath = root.appending(path: active ? "on" : "off").path
                let tools = ToolRunner(preferences: preferences)
                tools.beginTask(guardActive: active)
                _ = await tools.execute(
                    AgentToolCall(
                        name: "write_note",
                        arguments: [
                            AgentArgument(name: "title", value: "Stopped"),
                            AgentArgument(name: "content", value: "Body"),
                        ]))
                let guarder = CitationGuard(tools: tools)
                guarder.begin(active: active)
                _ = await guarder.settle(
                    TurnEnd(kind: .ended(stopReason: "stopToken"), outcome: TurnEnd.answered, answer: "Done"),
                    step: 1, maxSteps: 8, stopped: true, usage: nil)
                check(
                    "notes/stoppedWithGuard-" + String(active),
                    tools.draftOrder.count == 1
                        && !FileManager.default.fileExists(
                            atPath: tools.notesDirectory.appending(path: "Stopped.md").path),
                    "drafts=\(tools.draftOrder.count) outcome=\(guarder.outcome ?? "-")")
            }
            let preferences = ConversationPreferences(persists: false)
            preferences.sourcesRequired = false
            let original = root.appending(path: "original")
            let note = try await AgentTools.notes.write(
                title: "Continue", fileName: "Continue.md", content: "First", directory: original)
            let tools = ToolRunner(preferences: preferences)
            tools.restoreEvidence(
                ConversationEvidence(noteIDs: [note.id], taskWebCalls: 1), notesDirectory: original)
            preferences.notesFolderPath = root.appending(path: "different").path
            tools.beginTask(guardActive: true, continuing: true)
            try await tools.prepareContinuation()
            check(
                "notes/restoredContinuationKeepsTask",
                tools.notesDirectory == original && tools.webToolCalls == 1
                    && tools.noteTextsForValidation.contains("First"),
                "folder=\(tools.notesDirectory.lastPathComponent)")
            let newPart = try await AgentTools.notes.append(
                noteID: note.id, part: 2, content: "Unchecked https://example.com/missing", directory: original)
            let guarder = CitationGuard(tools: tools)
            guarder.begin(active: true)
            _ = await guarder.settle(
                TurnEnd(kind: .ended(stopReason: "stopToken"), outcome: TurnEnd.answered, answer: "Done"),
                step: 1, maxSteps: 1, stopped: false, usage: nil)
            check(
                "notes/unseenAppendIsNotPublished",
                guarder.outcome == "held" && guarder.reason.hasPrefix("note-save-failed:") && newPart.draft
                    && !FileManager.default.fileExists(atPath: original.appending(path: "Continue.md").path),
                "outcome=\(guarder.outcome ?? "-")")
        } catch {
            check("notes/integration", false, error.localizedDescription)
        }
    }
}
