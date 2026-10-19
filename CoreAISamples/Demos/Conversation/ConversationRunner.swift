import ConversationFeature
import ConversationRuntime
import FeatureStore
import Foundation
import FoundationModels
import RAGContext

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
    private let sourceTokenCounter: (@Sendable (String) -> Int)?
    private let retrieveSources: @Sendable (String, RAGSettings, @Sendable (String) -> Int) async -> RAGRetrieval
    private let writer = UUID()
    private var savedRevisions: [String: Int] = [:]
    private var retrievingRun: Int?
    private var retrievalStop: AsyncStream<Void>.Continuation?
    private(set) var lastRetrieval: RAGRetrieval?
    private var sourceResult: GenerationStats?
    var lastGeneration: GenerationStats? { sourceResult ?? session.lastGeneration }

    init(
        id: String, chat: ChatViewModel, preferences: ConversationPreferences,
        archive: ConversationArchive = ConversationRunner.defaultArchive,
        termination: @escaping (Bool) -> Void = { _ in },
        sessionFactory: ((ToolRunner) -> LanguageModelSession)? = nil, recordsMetrics: Bool = true,
        toolExecutor: ((AgentToolCall) async -> AgentToolOutput)? = nil,
        sourceTokenCounter: (@Sendable (String) -> Int)? = nil,
        retrieveSources: @escaping @Sendable (String, RAGSettings, @Sendable (String) -> Int) async -> RAGRetrieval = {
            question, settings, count in
            await RAGRetriever.gather(question: question, settings: settings, count: count)
        }
    ) {
        let tools = ToolRunner(preferences: preferences, executor: toolExecutor)
        session = ConversationSession(
            id: id, chat: chat, preferences: preferences, tools: tools,
            nativeSession: sessionFactory?(tools), recordsMetrics: recordsMetrics)
        citationGuard = CitationGuard(tools: tools)
        tools.session = session
        self.tools = tools
        self.chat = chat
        self.archive = archive
        self.termination = termination
        self.sourceTokenCounter = sourceTokenCounter
        self.retrieveSources = retrieveSources
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
            if retrievingRun == run {
                retrievalStop?.yield(())
                retrievalStop?.finish()
            }
        case .stopForBudget(let run):
            session.stopForBudget(run: run)
        case .settleBudgetStop(let run, let policy):
            chat.externallyBusy = true
            let verdict = await citationGuard.settle(
                TurnEnd(kind: .stopped, outcome: "step-limit", answer: ""), step: session.step,
                maxSteps: policy.maxSteps, stopped: true, usage: session.usage)
            switch verdict {
            case .done(let note):
                if let note {
                    send(
                        .guardNote(
                            run: run, title: policy.citationGuard ? CitationGuard.title : "Notes",
                            text: note.text, warning: note.warning))
                }
            case .repair:
                await citationGuard.cancelRepair()
            }
            if policy.mode == .work { await citationGuard.saveRunMetrics(session: session) }
            chat.externallyBusy = false
            send(.stopped(run: run))
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
            await citationGuard.cancelRepair()
            chat.externallyBusy = false
        case .terminationReady(let success):
            termination(success)
        }
    }

    private func generate(
        run: Int, prompt: String, policy: TurnPolicy, finalize: Bool, send: (ConversationAction) -> Void
    ) async {
        session.prepareRun(run)
        chat.externallyBusy = true
        defer { if !citationGuard.repairPending { chat.externallyBusy = false } }
        let repair = citationGuard.repairPending
        if !repair && !policy.answerOnly {
            let guarding = policy.mode == .work && policy.citationGuard
            tools.beginTask(guardActive: guarding, continuing: policy.continuing)
            citationGuard.begin(
                active: guarding, suppliedSourcesRequired: policy.sourceRequest != nil || policy.sourceIDs != nil)
            lastRetrieval = nil
            sourceResult = nil
        }
        var prompt = prompt
        var sourceIDs: [String] = []
        if let request = policy.sourceRequest {
            let count: @Sendable (String) -> Int
            if let sourceTokenCounter {
                count = sourceTokenCounter
            } else if let model = chat.model {
                count = model.countTokens
            } else {
                send(.generationFailed(run: run, message: "No model is loaded."))
                return
            }
            guard session.prepareSources(run: run, send: send) else {
                send(.generationFailed(run: run, message: "No model is loaded."))
                return
            }
            var settings = RAGSettings.fromDefaults()
            settings.web = request.web
            settings.local = request.local
            settings.sourceTokenBudget = request.budget
            settings.braveAPIKey = tools.preferences.braveAPIKey
            let question = prompt
            let capturedSettings = settings
            let retrieve = retrieveSources
            let retrieval = await collectSources(run: run) {
                await retrieve(question, capturedSettings, count)
            }
            guard !session.stopRequested, !Task.isCancelled, let retrieval else {
                sourceResult = GenerationStats(stopReason: "cancelled", sources: 0)
                send(.stopped(run: run))
                return
            }
            lastRetrieval = retrieval
            for note in retrieval.notes {
                send(.guardNote(run: run, title: "Sources", text: note, warning: true))
            }
            guard !retrieval.sources.isEmpty else {
                sourceResult = GenerationStats(
                    answer: RAGContext.noSourcesReply, stopReason: "noSources",
                    sources: 0, sourceTokens: 0, retrievalSeconds: retrieval.seconds)
                send(.generationFailed(run: run, message: RAGContext.noSourcesReply))
                return
            }
            do {
                guard
                    let prepared = try tools.sources.prepare(
                        question: prompt, sources: retrieval.conversationSources(), today: RAGContext.today(),
                        budget: RAGContext.promptTokenBudget,
                        count: { count(ConversationFeature.finalized(policy.restrictingSearch(in: $0))) },
                        mode: policy.mode)
                else {
                    let message = "The question and sources exceed the RAG prompt budget."
                    sourceResult = GenerationStats(error: message, sources: 0, retrievalSeconds: retrieval.seconds)
                    send(.generationFailed(run: run, message: message))
                    return
                }
                let ids = try tools.registerSources(prepared.sources)
                sourceIDs = ids
                send(
                    .sourcesCollected(
                        run: run, ids: ids, evidence: tools.evidence,
                        lines: tools.sources.citations(ids)))
                prompt = prepared.prompt
                lastRetrieval?.sources = Array(retrieval.sources.prefix(prepared.sources.count))
                let sourceTokens = lastRetrieval?.sources.map { count(RAGContext.measuredBlock($0)) }
                    .reduce(0, +)
                lastRetrieval?.sourceTokens = sourceTokens ?? 0
                sourceResult = GenerationStats(
                    sources: ids.count, sourceTokens: sourceTokens,
                    ragPromptTokens: prepared.tokens, retrievalSeconds: retrieval.seconds)
            } catch {
                send(.generationFailed(run: run, message: error.localizedDescription))
                return
            }
        }
        if policy.continuing && !policy.answerOnly {
            do { try await tools.prepareContinuation() } catch {
                send(.guardNote(run: run, title: "Notes", text: error.localizedDescription, warning: true))
            }
        }
        send(.evidenceChanged(run: run, tools.evidence))
        let end = await session.generate(
            run: run, prompt: prompt, policy: policy, finalize: finalize, repair: repair,
            sourceStats: sourceResult, send: send)
        sourceResult = nil
        if session.didSendPrompt { tools.markSourcesSupplied(sourceIDs) }
        send(.evidenceChanged(run: run, tools.evidence))
        if end.outcome == "step-limit" {
            send(.budgetStopped(run: run))
            return
        }
        var checked = end
        checked.answer = policy.answerPrefix + end.answer
        let verdict = await citationGuard.settle(
            checked, step: session.step, maxSteps: policy.maxSteps, stopped: session.stopRequested, usage: session.usage
        )
        switch verdict {
        case .repair(let prompt):
            send(end.action(run))
            send(.sendRepair(prompt, now: Date()))
        case .done(let note):
            if let note {
                send(
                    .guardNote(
                        run: run,
                        title: policy.citationGuard
                            ? CitationGuard.title
                            : citationGuard.checks.last?.local != nil ? "Source citations" : "Notes", text: note.text,
                        warning: note.warning))
            }
            if policy.mode == .work { await citationGuard.saveRunMetrics(session: session) }
            send(end.action(run))
        }
    }

    func collectSources(run: Int, gather: @escaping @Sendable () async -> RAGRetrieval) async -> RAGRetrieval? {
        guard !session.stopRequested, !Task.isCancelled else { return nil }
        let (stops, stop) = AsyncStream<Void>.makeStream()
        retrievalStop = stop
        retrievingRun = run
        defer {
            retrievalStop = nil
            retrievingRun = nil
            stop.finish()
        }
        return await withTaskGroup(of: RAGRetrieval?.self) { group in
            group.addTask { await gather() }
            group.addTask {
                for await _ in stops { return nil }
                return nil
            }
            let result = await group.next() ?? nil
            group.cancelAll()
            return result
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
        lastRetrieval = nil
        sourceResult = nil
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
            if let settings = snapshot.turns.last?.settings { tools.preferences.restore(settings) }
            tools.preferences.citeFetched = snapshot.instructionsVersion.hasPrefix("cite-fetched")
            savedRevisions[snapshot.id] = snapshot.revision
            tools.restoreEvidence(
                snapshot.evidence ?? ConversationEvidence(), notesDirectory: tools.preferences.notesDirectory,
                suppliedSourceIDs: snapshot.suppliedSourceIDs)
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
