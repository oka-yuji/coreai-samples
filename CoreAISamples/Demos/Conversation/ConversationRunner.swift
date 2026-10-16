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
                switch install(archived) {
                case .snapshot(let snapshot): send(.restored(snapshot))
                case .readOnly(_, _, let reason): send(.failed(id: id, message: reason))
                }
            } catch { send(.failed(id: id, message: error.localizedDescription)) }
        case .restoreLatest:
            do {
                let archived = try await archive.latest()
                guard !Task.isCancelled else { return }
                send(.latestRestored(archived.map(install)))
            } catch { send(.failed(id: session.id, message: error.localizedDescription)) }
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
            try await archive.save(snapshot, expectedRevision: savedRevisions[snapshot.id] ?? 0, writer: writer)
            savedRevisions[snapshot.id] = max(savedRevisions[snapshot.id] ?? 0, snapshot.revision)
            session.adoptID(snapshot.id)
            send(.archiveSaved(id: snapshot.id, revision: snapshot.revision))
        } catch {
            send(.archiveFailed(id: snapshot.id, revision: snapshot.revision, message: error.localizedDescription))
        }
    }

    private func install(_ archived: ArchivedConversation) -> ArchivedConversation {
        guard case .snapshot(let snapshot) = archived else { return archived }
        do {
            let restored = try session.restore(snapshot)
            savedRevisions[snapshot.id] = snapshot.revision
            tools.restoreEvidence(snapshot.evidence ?? ConversationEvidence())
            return .snapshot(restored)
        } catch {
            return .readOnly(id: snapshot.id, markdown: snapshot.markdown(), reason: error.localizedDescription)
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
