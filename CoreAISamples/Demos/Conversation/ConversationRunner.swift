import ConversationFeature
import ConversationRuntime
import FeatureStore
import Foundation

typealias ConversationStore = Store<ConversationFeature>

@MainActor
final class ConversationRunner {
    static let defaultArchive = ConversationArchive(directory: conversationsDirectory())
    let session: ConversationSession
    let tools: ToolRunner
    let citationGuard: CitationGuard
    private let chat: ChatViewModel
    private let archive: ConversationArchive
    private let termination: (Bool) -> Void
    private let writer = UUID()
    private var savedRevisions: [String: Int] = [:]

    init(
        id: String, chat: ChatViewModel, preferences: ConversationPreferences,
        archive: ConversationArchive = ConversationRunner.defaultArchive,
        termination: @escaping (Bool) -> Void = { _ in }
    ) {
        let tools = ToolRunner(preferences: preferences)
        session = ConversationSession(id: id, chat: chat, preferences: preferences, tools: tools)
        citationGuard = CitationGuard(tools: tools)
        tools.session = session
        self.tools = tools
        self.chat = chat
        self.archive = archive
        self.termination = termination
    }

    func run(_ effect: ConversationEffect, _ send: (ConversationAction) -> Void) async {
        switch effect {
        case .library(let effect):
            await runLibrary(effect, send: send)
        case .generate(let run, let prompt, let policy, let finalize):
            await generate(run: run, prompt: prompt, policy: policy, finalize: finalize, send: send)
        case .runTool(let call, let finalize):
            tools.resolve(call.id, .run(finalize: finalize))
        case .refuseTool(let call, let reason):
            tools.resolve(call.id, .refuse(reason))
        case .stop(let run):
            session.stop(run: run)
        case .save(let snapshot):
            await save(snapshot, send: send)
        case .restore(let id):
            do {
                let archived = try await archive.load(id: id)
                guard !Task.isCancelled else { return }
                try await archive.select(id: id)
                guard !Task.isCancelled else { return }
                send(.archiveRestored(id: id, install(archived)))
            } catch { send(.restoreFailed(id: id, message: error.localizedDescription)) }
        case .restoreLatest:
            do {
                let archived = try await archive.latest()
                guard !Task.isCancelled else { return }
                send(.latestRestored(archived.map(install)))
            } catch { send(.restoreFailed(id: nil, message: error.localizedDescription)) }
        case .cancelRepair:
            citationGuard.cancelRepair()
            chat.externallyBusy = false
        case .terminationReady(let success):
            termination(success)
        }
    }

    private func generate(
        run: Int, prompt: String, policy: TurnPolicy, finalize: Bool, send: (ConversationAction) -> Void
    ) async {
        chat.externallyBusy = true
        defer { if !citationGuard.repairPending { chat.externallyBusy = false } }
        let repair = citationGuard.repairPending
        if !repair {
            let guarding = policy.mode == .work && policy.citationGuard
            tools.beginTask(guardActive: guarding)
            citationGuard.begin(active: guarding)
        }
        let end = await session.generate(
            run: run, prompt: prompt, policy: policy, finalize: finalize, repair: repair, send: send)
        send(.evidenceChanged(run: run, tools.evidence))
        guard policy.mode == .work else {
            send(end.action(run))
            return
        }
        let verdict = citationGuard.settle(
            end, step: session.step, maxSteps: policy.maxSteps, stopped: session.stopRequested, usage: session.usage)
        switch verdict {
        case .repair(let prompt):
            send(end.action(run))
            send(.sendRepair(prompt, now: Date()))
        case .done(let note):
            if let note {
                send(.guardNote(run: run, title: CitationGuard.title, text: note.text, warning: note.warning))
            }
            await citationGuard.saveRunMetrics(session: session)
            send(end.action(run))
        }
    }

    private func save(_ snapshot: ConversationSnapshot, send: (ConversationAction) -> Void) async {
        do {
            let firstSave = savedRevisions[snapshot.id] == nil
            session.adoptID(snapshot.id)
            try await archive.save(
                snapshot, expectedRevision: savedRevisions[snapshot.id] ?? 0, writer: writer,
                selectAfterSave: firstSave)
            savedRevisions[snapshot.id] = max(savedRevisions[snapshot.id] ?? 0, snapshot.revision)
            session.adoptID(snapshot.id)
            send(.archiveSaved(id: snapshot.id, revision: snapshot.revision))
        } catch {
            send(.archiveFailed(id: snapshot.id, revision: snapshot.revision, message: error.localizedDescription))
        }
    }

    private func install(_ archived: ArchivedConversation) -> ArchivedConversation {
        guard case .snapshot(let snapshot) = archived else {
            if case .readOnly(let id, _, _) = archived {
                session.reset(id: id)
                tools.restoreEvidence(ConversationEvidence())
                citationGuard.begin(active: false)
            }
            return archived
        }
        do {
            let restored = try session.restore(snapshot)
            savedRevisions[snapshot.id] = snapshot.revision
            tools.restoreEvidence(snapshot.evidence ?? ConversationEvidence())
            return .snapshot(restored)
        } catch {
            return .readOnly(id: snapshot.id, markdown: snapshot.markdown(), reason: error.localizedDescription)
        }
    }

    private func runLibrary(_ effect: ConversationLibraryEffect, send: (ConversationAction) -> Void) async {
        let ticket: Int
        switch effect {
        case .list(let value), .rename(let value, _, _, _, _), .trash(let value, _, _, _, _), .selectNew(let value),
            .repairExport(let value, _):
            ticket = value
        }
        do {
            switch effect {
            case .list:
                send(.library(.loaded(ticket: ticket, try await archive.list())))
            case .rename(_, let id, let title, let expected, let now):
                let renamed = try await archive.rename(
                    id: id, title: title, expectedRevision: expected, now: now, writer: writer)
                let snapshot = renamed.snapshot
                if id == session.id { savedRevisions[id] = snapshot.revision }
                send(.library(.renamed(ticket: ticket, snapshot, warning: renamed.exportError)))
            case .trash(_, let id, let expected, let replacementID, let now):
                _ = try await archive.trash(id: id, expectedRevision: expected)
                var warning: String?
                if id == session.id {
                    session.reset(id: replacementID)
                    tools.restoreEvidence(ConversationEvidence())
                    citationGuard.begin(active: false)
                    do { try await archive.select(id: nil) } catch { warning = error.localizedDescription }
                }
                send(
                    .library(.deleted(ticket: ticket, id: id, replacementID: replacementID, now: now, warning: warning))
                )
            case .repairExport(_, let id):
                try await archive.repairExport(id: id)
                send(.library(.exportRepaired(ticket: ticket)))
            case .selectNew:
                try await archive.select(id: nil)
                send(.library(.selected(ticket: ticket)))
            }
        } catch {
            send(.library(.failed(ticket: ticket, message: error.localizedDescription)))
        }
    }

    static func conversationsDirectory() -> URL {
        ModelStorage.root.appending(path: "conversations", directoryHint: .isDirectory)
    }

    static func write(_ snapshot: ConversationSnapshot, to directory: URL) -> ConversationAction? {
        guard snapshot.status == .completed else { return nil }
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try Data(snapshot.markdown().utf8).write(
                to: directory.appending(path: "\(snapshot.id).md"), options: .atomic)
            return .saved(id: snapshot.id)
        } catch {
            AgentTools.log("[conversation] transcript save failed: \(error)")
            return .failed(id: snapshot.id, message: error.localizedDescription)
        }
    }

    static func transcriptsDirectory() -> URL {
        ModelStorage.root.appending(path: "agent-transcripts", directoryHint: .isDirectory)
    }
}
