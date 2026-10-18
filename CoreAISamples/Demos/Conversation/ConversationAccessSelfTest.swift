import ConversationFeature
import ConversationRuntime
import Foundation
import FoundationModels
import RAGContext
import Synchronization

@MainActor
enum ConversationAccessSelfTest {
    static func run(_ check: (String, Bool, String) -> Void) async {
        for scenario in AccessProbe.Scenario.allCases {
            let root = URL.temporaryDirectory.appending(path: "access-selftest-" + UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let preferences = ConversationPreferences(persists: false)
            preferences.resetForHeadless()
            preferences.citationGuard = true
            if scenario == .workLimit { preferences.maxSteps = 2 }
            preferences.notesFolderPath = root.appending(path: "notes").path
            let chat = ChatViewModel()
            let archive = ConversationArchive(directory: root.appending(path: "conversations"))
            let reference = AccessStoreReference()
            let probe = AccessProbe(scenario: scenario) { reference.store?.send(.stopRequested) }
            let runner = ConversationRunner(
                id: "access", chat: chat, preferences: preferences, archive: archive,
                sessionFactory: { tools in
                    LanguageModelSession(model: AccessModel(probe: probe), tools: tools.tools(fileOperations: false))
                }, recordsMetrics: false,
                toolExecutor: { call in
                    probe.values.withLock { $0.executed.append(call.name) }
                    if call.name == "write_note" {
                        do {
                            let content = scenario.sources ? "Verified note [S1]." : "Note from the conversation."
                            let note = try await AgentTools.notes.write(
                                title: "Access note", fileName: "Access.md", content: content,
                                directory: preferences.notesDirectory)
                            return AgentToolOutput(text: "Saved draft", note: note)
                        } catch { return AgentToolOutput(text: "error: " + error.localizedDescription) }
                    }
                    return AgentToolOutput(
                        text: "Web fixture", seen: ["https://example.com/source": "Web fixture"],
                        finalURL: "https://example.com/source")
                }, sourceTokenCounter: { $0.utf8.count },
                retrieveSources: { _, settings, _ in
                    let attempt = probe.values.withLock { values in
                        values.selections.append([settings.web, settings.local])
                        return values.selections.count
                    }
                    if scenario == .stopSources {
                        await probe.stop()
                        for await _ in probe.gate.stream {}
                        return RAGRetrieval()
                    }
                    if scenario == .emptySources || scenario == .retrySources && attempt == 1 { return RAGRetrieval() }
                    if scenario == .failedSources { return RAGRetrieval(notes: ["Fixture retrieval failed."]) }
                    let url = settings.web ? "https://example.com/source" : "file:///manual.md"
                    return RAGRetrieval(
                        sources: RAGContext.documentSources(
                            title: "Fixture", url: url, text: "Verified source text."))
                })
            let updates = AsyncStream<ConversationState>.makeStream()
            let store = ConversationStore(
                initialState: ConversationState(
                    id: "access", now: Date(), mode: scenario.mode, settings: preferences.settings)
            ) { effect, send in
                await runner.run(effect) { action in
                    send(action)
                    if let state = reference.store?.state { updates.continuation.yield(state) }
                }
            }
            reference.store = store
            func send() {
                if scenario.sources {
                    store.send(
                        .sendSources(
                            "Read and summarize", .init(web: scenario.web, local: !scenario.web, budget: 3_000),
                            now: Date()))
                } else {
                    store.send(.send("Answer and save if in Work", mode: scenario.mode, now: Date()))
                }
            }
            send()
            var ended = 0
            for await state in updates.stream {
                guard !state.isRunning, state.persistence == .saved else { continue }
                ended += 1
                if scenario == .retrySources && ended == 1 { send() } else { break }
            }
            updates.continuation.finish()
            let state = store.state
            let values = probe.values.withLock { $0 }
            let failed = scenario == .emptySources || scenario == .failedSources
            let stopped = scenario == .stopSources
            let answered = !failed && !stopped
            let expected: [String] =
                !answered || scenario.mode == .chat
                ? []
                : scenario.web ? ["fetch_page", "write_note"] : ["write_note"]
            let sourceSelection =
                values.selections.allSatisfy { $0 == [scenario.web, !scenario.web] }
                && values.selections.count == (scenario == .retrySources ? 2 : scenario.sources ? 1 : 0)
            let finish =
                answered
                ? state.lastFinish == .eos : stopped ? state.lastFinish == .stopped : state.lastFinish == .failed
            let savedNote = FileManager.default.fileExists(
                atPath: preferences.notesDirectory.appending(path: "Access.md").path)
            let valid =
                finish && state.mode == scenario.mode && !chat.externallyBusy
                && values.executed == expected && sourceSelection
                && (!answered ? values.requests == 0 : values.requests > 0)
                && (scenario.mode == .chat ? values.disallowed == values.requests : true)
                && (answered && scenario.mode == .work ? savedNote : !savedNote)
            check(
                "conversation/access-" + scenario.rawValue, valid,
                "finish=\(String(describing: state.lastFinish)) mode=\(state.mode) requests=\(values.requests) tools=\(values.executed) retrievals=\(values.selections) note=\(savedNote)"
            )
        }
        verifySettings(check)
        await verifySourceGuard(check)
    }

    private static func verifySourceGuard(_ check: (String, Bool, String) -> Void) async {
        for variant in 0...2 {
            let preferences = ConversationPreferences(persists: false)
            preferences.resetForHeadless()
            preferences.sourcesRequired = variant == 1 ? true : nil
            let tools = ToolRunner(preferences: preferences)
            do {
                let ids = try tools.registerSources([
                    ConversationSource(
                        document: "https://example.com/source", version: String(repeating: "a", count: 64),
                        title: "Source", passages: [SourcePassage(offset: 0, text: "Verified")])
                ])
                var evidence = tools.evidence
                evidence.taskWebCalls = 1
                tools.restoreEvidence(evidence, suppliedSourceIDs: Set(ids))
                let guarder = CitationGuard(tools: tools)
                guarder.begin(active: true, suppliedSourcesRequired: true)
                let answer = "Verified [S1]." + (variant == 2 ? " https://unfetched.example/item" : "")
                _ = await guarder.settle(
                    TurnEnd(kind: .ended(stopReason: "stopToken"), outcome: TurnEnd.answered, answer: answer),
                    step: 1, maxSteps: 1, stopped: false, usage: nil)
                check(
                    "conversation/access-sourceGuard-\(variant)",
                    guarder.outcome == (variant == 0 ? "verified" : "held"),
                    "outcome=\(guarder.outcome ?? "none") reason=\(guarder.reason)")
            } catch { check("conversation/access-sourceGuard-\(variant)", false, error.localizedDescription) }
        }
    }

    private static func verifySettings(_ check: (String, Bool, String) -> Void) {
        let chat = ChatViewModel()
        let preferences = ConversationPreferences(persists: false)
        preferences.resetForHeadless()
        let root = URL.temporaryDirectory.appending(path: "access-settings-" + UUID().uuidString)
        let host = ConversationHost(chat: chat, preferences: preferences, archive: ConversationArchive(directory: root))
        defer { try? FileManager.default.removeItem(at: root) }
        let off = host.turnSettings.webSearchEnabled == false
        chat.ragEnabled = true
        chat.ragWeb = false
        chat.ragLocal = false
        let invalid = !chat.hasValidSourceSelection && !host.canSend
        chat.ragLocal = true
        let docs = chat.hasValidSourceSelection && host.turnSettings.webSearchEnabled == false
        host.store.send(.modeChanged(.work))
        chat.ragWeb = true
        check(
            "conversation/access-settings",
            off && invalid && docs && host.turnSettings.webSearchEnabled == true && host.state.mode == .work,
            "off=\(off) invalid=\(invalid) docs=\(docs)")
    }
}

@MainActor
private final class AccessStoreReference {
    weak var store: ConversationStore?
}

private nonisolated final class AccessProbe: Sendable {
    enum Scenario: String, CaseIterable, Sendable {
        case chatOff, workOff, chatDocs, workDocs, workWeb, workLimit, emptySources, failedSources, stopSources,
            retrySources
        var sources: Bool { self != .chatOff && self != .workOff }
        var web: Bool { self == .workWeb }
        var mode: ConversationMode { self == .chatOff || self == .chatDocs ? .chat : .work }
    }
    struct Values: Sendable {
        var requests = 0
        var disallowed = 0
        var selections: [[Bool]] = []
        var executed: [String] = []
    }
    let scenario: Scenario
    let values = Mutex(Values())
    let gate = AsyncStream<Void>.makeStream()
    let stop: @MainActor @Sendable () -> Void

    init(scenario: Scenario, stop: @escaping @MainActor @Sendable () -> Void) {
        self.scenario = scenario
        self.stop = stop
    }
}

private nonisolated struct AccessModel: LanguageModel {
    struct Executor: LanguageModelExecutor {
        typealias Model = AccessModel
        init(configuration: UUID) {}

        func respond(
            to request: LanguageModelExecutorGenerationRequest, model: Model,
            streamingInto channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            try Task.checkCancellation()
            let id = model.probe.values.withLock { values in
                values.requests += 1
                if request.generationOptions.toolCallingMode?.kind == .disallowed { values.disallowed += 1 }
                return "call-\(values.requests)"
            }
            let scenario = model.probe.scenario
            let entries = Array(request.transcript)
            let start = entries.lastIndex { if case .prompt = $0 { true } else { false } } ?? 0
            let outputs = entries.dropFirst(start + 1).count { if case .toolOutput = $0 { true } else { false } }
            let sequence =
                scenario.mode == .chat
                ? []
                : scenario.web
                    ? ["fetch_page", "write_note"] : ["web_search", "fetch_page", "write_note"]
            let calls = scenario == .workLimit ? ["write_note", "fetch_page"] : sequence
            if outputs < calls.count && request.generationOptions.toolCallingMode?.kind != .disallowed {
                let name = calls[outputs]
                let arguments =
                    name == "write_note"
                    ? #"{"title":"Access note","content":"Verified note [S1]."}"#
                    : name == "fetch_page" ? #"{"url":"https://example.com/source"}"# : #"{"query":"fixture"}"#
                await channel.send(
                    .toolCalls(
                        action: .toolCall(id: id, name: name, action: .appendArguments(arguments, tokenCount: 1))))
                await channel.send(
                    .toolCalls(
                        action: .toolCall(id: id, name: name, action: .updateMetadata(["stopReason": "stopToken"]))))
            } else {
                let text = scenario.sources ? "Verified answer [S1]." : "Answer from the conversation."
                await channel.send(.response(action: .appendText(text, tokenCount: 1)))
                await channel.send(.response(action: .updateMetadata(["stopReason": "stopToken"])))
            }
        }
    }
    let probe: AccessProbe
    let executorConfiguration = UUID()
    let capabilities = LanguageModelCapabilities([.reasoning, .toolCalling])
}
