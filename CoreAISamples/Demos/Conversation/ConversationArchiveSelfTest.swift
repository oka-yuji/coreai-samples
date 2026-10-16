import ConversationFeature
import ConversationRuntime
import Foundation
import Observation

@MainActor
enum ConversationArchiveSelfTest {
    static func run(_ check: (String, Bool, String) -> Void) async {
        let directory = URL.temporaryDirectory.appending(path: "conversation-app-test-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        do {
            let archive = ConversationArchive(directory: directory)
            let now = Date(timeIntervalSince1970: 1_700_000_000)
            var state = ConversationState(
                id: "restored", now: now, settings: ConversationSettings(maxTokens: 512), modelID: "saved-model",
                instructionsVersion: "cite-fetched", toolsVersion: "web_search,fetch_page,write_note")
            state.modelLocation = "/tmp/conversation-app-test-model"
            _ = ConversationFeature.reduce(&state, .send("Remember blue", mode: .work, now: now))
            _ = ConversationFeature.reduce(
                &state,
                .transcriptChanged(
                    run: 1,
                    TranscriptRecord(entries: [
                        .instructions(id: "i", text: "Keep context"), .prompt(id: "p", text: "Remember blue"),
                        .response(id: "a", text: "Remembered", metadata: StepMetadata(stopReason: "stopToken")),
                    ])))
            state.evidence = ConversationEvidence(
                pages: [
                    .init(
                        url: "https://example.com/old", finalURL: "https://example.com/new", sha256: "digest",
                        characters: 4)
                ],
                attempts: [.init(url: "https://example.com/old", ok: true)], seen: ["https://example.com/old": "Blue"])
            _ = ConversationFeature.reduce(&state, .generationEnded(run: 1, stopReason: "stopToken"))
            let snapshot = ConversationSnapshot(state)
            try await archive.save(snapshot, expectedRevision: 0, writer: UUID())
            let preferences = ConversationPreferences(persists: false)
            preferences.resetForHeadless()
            preferences.braveAPIKey = "not-part-of-the-archive"
            let host = ConversationHost(chat: ChatViewModel(), preferences: preferences, archive: archive)
            host.store.send(.restoreLatestRequested)
            for await restoring in Observations({ host.state.isRestoring }) where !restoring { break }
            host.restorePreferences()
            check(
                "conversation/startupWithoutModel",
                host.chat.phase == .empty && host.state.id == "restored" && host.state.entries.count == 2
                    && host.runner.session.awaitingRestoredModel && !host.canSend,
                "phase=\(host.chat.phase) entries=\(host.state.entries.count)")
            check(
                "conversation/restoreSettingsAndEvidence",
                preferences.maxTokens == 512 && preferences.braveAPIKey == "not-part-of-the-archive"
                    && host.runner.tools.evidence == state.evidence,
                "tokens=\(String(describing: preferences.maxTokens)) pages=\(host.runner.tools.ledger.count)")
            host.runner.citationGuard.begin(active: true)
            _ = host.runner.citationGuard.settle(
                TurnEnd(
                    kind: .ended(stopReason: "stopToken"), outcome: TurnEnd.answered,
                    answer: "https://example.org/missing"),
                step: 1, maxSteps: 8, stopped: false, usage: nil)
            host.chat.externallyBusy = true
            await host.runner.run(.cancelRepair) { _ in }
            check(
                "conversation/quitCancelsRejectedRepair",
                !host.runner.citationGuard.repairPending && !host.chat.externallyBusy,
                "pending=\(host.runner.citationGuard.repairPending) busy=\(host.chat.externallyBusy)")
            let store = host.store
            let matched = host.prepareModelSwitch(to: URL(filePath: "/tmp/conversation-app-test-model"))
            check(
                "conversation/matchingModelKeepsRestoredHistory",
                matched && host.store === store && host.state.id == "restored", "id=\(host.state.id)")
            host.store.send(.library(.refresh))
            for await loading in Observations({ host.state.library.loading }) where loading == nil { break }
            host.renameConversation("restored", title: "Renamed archive")
            for await mutating in Observations({ host.state.library.isMutating }) where !mutating { break }
            let renamed = try await archive.load(id: "restored")
            let title: String
            if case .snapshot(let snapshot) = renamed { title = snapshot.title } else { title = "" }
            check(
                "conversation/sidebarRename",
                host.state.title == "Renamed archive" && title == "Renamed archive"
                    && host.state.archiveRevision == snapshot.revision + 1, "title=\(title)")
            host.startNew()
            for await mutating in Observations({ host.state.library.isMutating }) where !mutating { break }
            let empty = host.state.turns.isEmpty && host.state.entries.isEmpty
            let opened = host.openConversation("restored")
            for await restoring in Observations({ host.state.isRestoring }) where !restoring { break }
            check(
                "conversation/sidebarNewAndSwitch",
                empty && opened && host.state.id == "restored"
                    && host.state.title == "Renamed archive" && host.state.entries.count == 2, "id=\(host.state.id)")
            _ = host.openConversation("missing")
            for await restoring in Observations({ host.state.isRestoring }) where !restoring { break }
            check(
                "conversation/missingHistoryPreservesSelection",
                host.state.id == "restored" && host.canManageHistory
                    && host.state.library.error != nil, "id=\(host.state.id)")
            let reopenedStore = host.store
            let different = host.prepareModelSwitch(to: URL(filePath: "/tmp/conversation-app-test-other"))
            check(
                "conversation/differentModelStartsNewWithoutDeletingHistory",
                different && host.store !== reopenedStore
                    && host.state.entries.isEmpty
                    && FileManager.default.fileExists(atPath: directory.appending(path: "restored.json").path),
                "id=\(host.state.id)")
        } catch {
            check("conversation/archiveIntegration", false, error.localizedDescription)
        }
    }
}
