import ConversationFeature
import FeatureStore
import Foundation

typealias ConversationStore = Store<ConversationFeature>

@MainActor
final class ConversationRunner {
    let session: ConversationSession
    let tools: ToolRunner
    let citationGuard: CitationGuard
    private let chat: ChatViewModel

    init(id: String, chat: ChatViewModel, preferences: ConversationPreferences) {
        let tools = ToolRunner(preferences: preferences)
        session = ConversationSession(id: id, chat: chat, preferences: preferences, tools: tools)
        citationGuard = CitationGuard(tools: tools)
        tools.session = session
        self.tools = tools
        self.chat = chat
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
            save(snapshot, send: send)
        case .restore(let id):
            send(.failed(id: id, message: "Opening a saved conversation is not available yet."))
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

    private func save(_ snapshot: ConversationSnapshot, send: (ConversationAction) -> Void) {
        guard snapshot.status == .completed else { return }
        let directory = Self.transcriptsDirectory()
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try Data(snapshot.markdown().utf8).write(
                to: directory.appending(path: "\(snapshot.id).md"), options: .atomic)
            send(.saved(id: snapshot.id))
        } catch {
            send(
                .failed(id: snapshot.id, message: "The conversation could not be saved: \(error.localizedDescription)"))
        }
    }

    static func transcriptsDirectory() -> URL {
        ModelStorage.root.appending(path: "agent-transcripts", directoryHint: .isDirectory)
    }
}
