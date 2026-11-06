import ConversationFeature
import ConversationRuntime
import Foundation
import ModelLibraryStorage
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
                id: "restored", now: now, settings: ConversationSettings(maxTokens: 512, fileOperations: true),
                modelID: "saved-model",
                instructionsVersion: "cite-fetched", toolsVersion: "web_search,fetch_page,write_note,move_note")
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
                attempts: [.init(url: "https://example.com/old", ok: true)], seen: ["https://example.com/old": "Blue"],
                taskWebCalls: 1)
            _ = ConversationFeature.reduce(&state, .generationEnded(run: 1, stopReason: "stopToken"))
            let snapshot = ConversationSnapshot(state)
            try await launchArchives(snapshot, check: check)
            try await archive.save(snapshot, expectedRevision: 0, writer: UUID())
            let stoppedRunner = ConversationRunner(
                id: "before-generation", chat: ChatViewModel(), preferences: ConversationPreferences(persists: false),
                archive: archive)
            stoppedRunner.session.stop(run: 1)
            stoppedRunner.session.prepareRun(1)
            let stopped = await stoppedRunner.session.generate(
                run: 1, prompt: "Never generate", policy: TurnPolicy(.chat, settings: ConversationSettings()),
                finalize: false, repair: false
            ) { _ in }
            check(
                "conversation/stopBeforeGeneration",
                stopped.outcome == "stopped" && stoppedRunner.session.stopRequested, "outcome=\(stopped.outcome)")
            let preferences = ConversationPreferences(persists: false)
            preferences.resetForHeadless()
            let host = ConversationHost(chat: ChatViewModel(), preferences: preferences, archive: archive)
            host.store.send(.restoreLatestRequested)
            for await restoring in Observations({ host.state.isRestoring }) where !restoring { break }
            check(
                "conversation/startupWithoutModel",
                host.chat.phase == .empty && host.state.id == "restored" && host.state.entries.count == 2
                    && host.runner.session.awaitingRestoredModel && !host.canSend,
                "phase=\(host.chat.phase) entries=\(host.state.entries.count)")
            check(
                "conversation/restoreSettingsAndEvidence",
                preferences.maxTokens == 512 && !preferences.fileOperations
                    && host.runner.tools.evidence == state.evidence,
                "tokens=\(String(describing: preferences.maxTokens)) pages=\(host.runner.tools.ledger.count)")
            host.runner.citationGuard.begin(active: true)
            _ = await host.runner.citationGuard.settle(
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
            var legacyState = state
            legacyState.modelLocation = ModelStorage.directory(for: "okayuji/Qwen3.8-27B-coreai-4bit").path
            let legacyArchive = ConversationArchive(directory: directory.appending(path: "migration"))
            try await legacyArchive.save(ConversationSnapshot(legacyState), expectedRevision: 0, writer: UUID())
            let migrated = ConversationHost(chat: ChatViewModel(), preferences: preferences, archive: legacyArchive)
            migrated.store.send(.restoreRequested(id: legacyState.id))
            for await restoring in Observations({ migrated.state.isRestoring }) where !restoring { break }
            let restoredStore = migrated.store
            let managed = MacModelLibraryServices.root.appending(path: "artifacts")
                .appending(path: LanguageModelArtifacts.qwenID).appending(path: LanguageModelArtifacts.qwenRevision)
            let kept = migrated.prepareModelSwitch(to: managed)
            check(
                "conversation/legacyModelKeepsRestoredHistory",
                kept && migrated.matchesSavedModel(managed) && migrated.store === restoredStore
                    && migrated.state.entries.count == 2 && migrated.runner.session.awaitingRestoredModel,
                "matched=\(migrated.matchesSavedModel(managed)) entries=\(migrated.state.entries.count)")
            var terminationEvents: [String] = []
            let terminating = ConversationHost(
                chat: ChatViewModel(), preferences: preferences, archive: legacyArchive,
                termination: { terminationEvents.append($0 ? "finished" : "failed") },
                resumeModels: { terminationEvents.append("resumed") },
                modelUnloaded: { terminationEvents.append("unloaded") },
                releaseModels: { terminationEvents.append("released") })
            await terminating.runner.run(.terminationReady(true)) { _ in }
            terminating.cancelTermination()
            check(
                "conversation/terminationCleanupAndResume",
                terminationEvents == ["unloaded", "released", "finished", "resumed"],
                terminationEvents.joined(separator: ","))
            terminationEvents = []
            await terminating.runner.run(.terminationReady(false)) { _ in }
            terminating.cancelTermination(resumeModels: false)
            check(
                "conversation/failedCloseKeepsHiddenDemosStopped", terminationEvents == ["failed"],
                terminationEvents.joined(separator: ","))
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

    private static func launchArchives(
        _ snapshot: ConversationSnapshot, check: (String, Bool, String) -> Void
    ) async throws {
        let ordinary = await ConversationRunner.makeArchive(arguments: ["demo"]).directory
        check(
            "conversation/ordinaryArchive", ordinary == ConversationRunner.conversationsDirectory(),
            "normal launch keeps its archive")
        let launches = [
            ["demo", "--selftest"], ["demo", "--agent-e2e"],
            ["demo", "--gui-e2e", "/tmp/gui-output"], ["demo", "--gui-e2e=/tmp/gui-output", "--agent"],
        ]
        var directories = Set<URL>()
        for arguments in launches {
            let archive = ConversationRunner.makeArchive(arguments: arguments)
            let directory = await archive.directory
            let second = await ConversationRunner.makeArchive(arguments: arguments).directory
            check(
                "conversation/diagnosticArchiveIsolation",
                directory != ordinary && second != directory && directories.insert(directory).inserted,
                "each diagnostic launch owns a separate archive")
            guard directory != ordinary else { continue }
            defer { try? FileManager.default.removeItem(at: directory) }
            try await archive.save(snapshot, expectedRevision: 0, writer: UUID(), selectAfterSave: true)
            let restored = try await archive.latest()
            let markdown = try String(contentsOf: directory.appending(path: "\(snapshot.id).md"), encoding: .utf8)
            let selected = try JSONDecoder().decode(
                String.self, from: Data(contentsOf: directory.appending(path: ".selection")))
            check(
                "conversation/diagnosticArchiveExports",
                restored == .snapshot(snapshot) && selected == snapshot.id && markdown == snapshot.markdown(),
                "diagnostics retain restoration and Markdown export in their own archive")
        }
    }
}
