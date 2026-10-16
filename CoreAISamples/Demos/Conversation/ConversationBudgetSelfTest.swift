import ConversationFeature
import ConversationRuntime
import Foundation
import FoundationModels
import Synchronization

@MainActor
enum ConversationBudgetSelfTest {
    static func run(_ check: (String, Bool, String) -> Void) async {
        verifyRestoredMarker(check)
        for scenario in BudgetProbe.Scenario.allCases {
            let root = URL.temporaryDirectory.appending(path: "budget-selftest-" + UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let preferences = ConversationPreferences(persists: false)
            preferences.resetForHeadless()
            preferences.chatMaxSteps = scenario == .research ? 16 : 2
            let work = scenario == .workFinal || scenario == .workRepairStop
            preferences.maxSteps = scenario == .workRepairStop ? 3 : 2
            preferences.citationGuard = work
            preferences.sourcesRequired = false
            preferences.notesFolderPath = root.appending(path: "notes").path
            let reference = BudgetStoreReference()
            let probe = BudgetProbe(scenario: scenario) {
                if scenario == .stopFinal { reference.store?.send(.stopRequested) }
            }
            let chat = ChatViewModel()
            let runner = ConversationRunner(
                id: "budget", chat: chat, preferences: preferences,
                archive: ConversationArchive(directory: root.appending(path: "conversations")),
                sessionFactory: { tools in
                    LanguageModelSession(model: BudgetModel(probe: probe), tools: tools.tools(fileOperations: false))
                }, recordsMetrics: false,
                toolExecutor: { call in
                    probe.values.withLock { $0.executed.append(call.name) }
                    if call.name == "write_note" {
                        do {
                            let note = try await AgentTools.notes.write(
                                title: "Draft", fileName: "Draft.md", content: "Verified draft text",
                                directory: preferences.notesDirectory)
                            return AgentToolOutput(text: "Saved draft", note: note)
                        } catch { return AgentToolOutput(text: "error: " + error.localizedDescription) }
                    }
                    let url = "https://example.com/verified"
                    return AgentToolOutput(
                        text: "Verified source", seen: call.name == "fetch_page" ? [url: "Verified source"] : [:],
                        finalURL: call.name == "fetch_page" ? url : nil)
                })
            let updates = AsyncStream<ConversationState>.makeStream()
            let store = ConversationStore(
                initialState: ConversationState(id: "budget", now: Date(), settings: preferences.settings)
            ) { effect, send in
                await runner.run(effect) { action in
                    send(action)
                    guard let store = reference.store else { return }
                    if scenario == .stopBudget || scenario == .workRepairStop, store.state.finalization == .cutOff,
                        !store.state.stopRequested
                    {
                        store.send(.stopRequested)
                    }
                    updates.continuation.yield(store.state)
                }
            }
            reference.store = store
            store.send(.send("Research the fixture", mode: work ? .work : .chat, now: Date()))
            var completed = 0
            for await state in updates.stream {
                guard !state.isRunning, state.persistence == .saved else { continue }
                completed += 1
                if scenario == .research && completed == 1 {
                    store.send(.send("Follow up on the fixture", mode: .chat, now: Date()))
                } else {
                    break
                }
            }
            updates.continuation.finish()
            let state = store.state
            let values = probe.values.withLock { $0 }
            let valid: Bool
            switch scenario {
            case .research:
                valid =
                    completed == 2 && state.turns.count == 2 && state.turns.allSatisfy { $0.finish == .eos }
                    && values.executed == ["web_search", "fetch_page", "web_search", "fetch_page"]
                    && values.finalRequests == 0
            case .answer:
                valid =
                    state.lastFinish == .eos && values.finalRequests == 1 && values.executed == ["web_search"]
                    && values.retainedOutputs > 0 && state.entries.filter { $0.kind == .user }.count == 1
                    && state.entries.contains { $0.kind == .answer && $0.text == "Bounded final answer" }
            case .failFinal, .emptyFinal, .toolInFinal:
                valid =
                    !state.isRunning && values.finalRequests == 1 && values.executed == ["web_search"]
                    && state.entries.contains { $0.kind == .failure || $0.detail == .noAnswer }
            case .stopFinal:
                valid =
                    state.lastFinish == .stopped && values.finalRequests == 1
                    && values.executed == ["web_search"]
            case .stopBudget:
                valid =
                    state.lastFinish == .stopped && values.finalRequests == 0
                    && values.executed == ["web_search"]
            case .workFinal:
                valid =
                    state.lastFinish == .eos && values.finalRequests == 1 && values.executed == ["write_note"]
                    && FileManager.default.fileExists(
                        atPath: preferences.notesDirectory.appending(path: "Draft.md").path)
                    && !runner.citationGuard.repairPending && !chat.externallyBusy
            case .workRepairStop:
                valid =
                    state.lastFinish == .stopped && values.finalRequests == 0 && values.executed == ["write_note"]
                    && !runner.citationGuard.repairPending && !chat.externallyBusy
                    && state.turns.count == 2 && !runner.tools.draftOrder.isEmpty
                    && state.entries.contains { $0.kind == .note && $0.warning }
                    && !FileManager.default.fileExists(
                        atPath: preferences.notesDirectory.appending(path: "Draft.md").path)
            }
            check(
                "conversation/budget-" + scenario.rawValue, valid,
                "finish=\(String(describing: state.lastFinish)) final=\(values.finalRequests) tools=\(values.executed) error=\(state.turns.last?.failure ?? "none") requests=\(values.requests)"
            )
        }
    }

    private static func verifyRestoredMarker(_ check: (String, Bool, String) -> Void) {
        let preferences = ConversationPreferences(persists: false)
        preferences.resetForHeadless()
        let chat = ChatViewModel()
        let tools = ToolRunner(preferences: preferences)
        let session = ConversationSession(
            id: "restored-budget", chat: chat, preferences: preferences, tools: tools, recordsMetrics: false)
        var state = ConversationState(
            id: "restored-budget", now: Date(), instructionsVersion: "cite-fetched-v3",
            toolsVersion: AgentTools.names.dropLast().joined(separator: ","))
        _ = ConversationFeature.reduce(&state, .send("Research", mode: .chat, now: Date()))
        state.transcript = TranscriptRecord(entries: [
            .instructions(id: "rules", text: "Use the verified material."),
            .prompt(id: "user", text: "Research"),
            .toolCalls(
                id: "call",
                calls: [ToolCallRecord(id: "pending", name: "web_search", arguments: #"{"query":"fixture"}"#)],
                metadata: nil),
            .prompt(id: "internal", text: ConversationFeature.budgetAnswerPrompt),
            .response(id: "answer", text: "Summary", metadata: StepMetadata(stopReason: "stopToken")),
        ])
        state.turns[0].start = 1
        state.turns[0].finalAnswerStart = 3
        state.turns[0].finish = .eos
        state.isRunning = false
        do {
            let restored = try session.restore(ConversationSnapshot(state))
            let marker = restored.turns[0].finalAnswerStart
            check(
                "conversation/budget-restoreMarker",
                marker == 4 && restored.transcript.entries[4].id == "internal"
                    && !restored.markdown().contains(ConversationFeature.budgetAnswerPrompt),
                "marker=\(String(describing: marker))")
        } catch { check("conversation/budget-restoreMarker", false, error.localizedDescription) }
    }
}

@MainActor
private final class BudgetStoreReference {
    weak var store: ConversationStore?
}

private nonisolated final class BudgetProbe: Sendable {
    enum Scenario: String, CaseIterable, Sendable {
        case research, answer, failFinal, emptyFinal, toolInFinal, stopFinal, stopBudget, workFinal, workRepairStop
    }

    struct Values: Sendable {
        var executed: [String] = []
        var requests = 0
        var finalRequests = 0
        var retainedOutputs = 0
    }

    let scenario: Scenario
    let values = Mutex(Values())
    let onFinal: @MainActor @Sendable () -> Void
    let gate = AsyncStream<Void>.makeStream()

    init(scenario: Scenario, onFinal: @escaping @MainActor @Sendable () -> Void) {
        self.scenario = scenario
        self.onFinal = onFinal
    }
}

private nonisolated struct BudgetModel: LanguageModel {
    struct Executor: LanguageModelExecutor {
        typealias Model = BudgetModel
        init(configuration: UUID) {}

        func respond(
            to request: LanguageModelExecutorGenerationRequest, model: Model,
            streamingInto channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            try Task.checkCancellation()
            let final = request.generationOptions.toolCallingMode?.kind == .disallowed
            let count = model.probe.values.withLock { values in
                values.requests += 1
                if final {
                    values.finalRequests += 1
                    values.retainedOutputs = request.transcript.count {
                        if case .toolOutput = $0 { true } else { false }
                    }
                }
                return values.requests
            }
            if final {
                await model.probe.onFinal()
                switch model.probe.scenario {
                case .failFinal: throw BudgetFailure.expected
                case .stopFinal:
                    for await _ in model.probe.gate.stream {}
                    try Task.checkCancellation()
                case .toolInFinal: break
                default:
                    let text = model.probe.scenario == .emptyFinal ? "" : "Bounded final answer"
                    await channel.send(.response(action: .appendText(text, tokenCount: text.isEmpty ? 0 : 1)))
                    await channel.send(.response(action: .updateMetadata(["stopReason": "stopToken"])))
                    return
                }
            }
            let entries = Array(request.transcript)
            let start = entries.lastIndex { if case .prompt = $0 { true } else { false } } ?? 0
            let outputs = entries.dropFirst(start + 1).count { if case .toolOutput = $0 { true } else { false } }
            let prompts = entries.count { if case .prompt = $0 { true } else { false } }
            if model.probe.scenario == .workRepairStop && prompts == 1 && outputs > 0 {
                await channel.send(
                    .response(action: .appendText("Claim https://unfetched.example/item", tokenCount: 1)))
                await channel.send(.response(action: .updateMetadata(["stopReason": "stopToken"])))
                return
            }
            if model.probe.scenario == .research && outputs >= 2 {
                await channel.send(.response(action: .appendText("Verified answer", tokenCount: 1)))
                await channel.send(.response(action: .updateMetadata(["stopReason": "stopToken"])))
                return
            }
            let writing =
                (model.probe.scenario == .workFinal || model.probe.scenario == .workRepairStop)
                && prompts == 1 && outputs == 0
            let name =
                writing ? "write_note" : model.probe.scenario == .research && outputs == 1 ? "fetch_page" : "web_search"
            let arguments =
                writing
                ? #"{"title":"Draft","content":"Verified draft text"}"#
                : name == "fetch_page"
                    ? #"{"url":"https://example.com/verified","query":"fixture"}"# : #"{"query":"fixture"}"#
            let id = "call-\(count)"
            await channel.send(
                .toolCalls(
                    action: .toolCall(
                        id: id, name: name, action: .appendArguments(arguments, tokenCount: 1))))
            await channel.send(
                .toolCalls(
                    action: .toolCall(
                        id: id, name: name, action: .updateMetadata(["stopReason": "stopToken"]))))
        }
    }

    let probe: BudgetProbe
    let executorConfiguration = UUID()
    let capabilities = LanguageModelCapabilities([.reasoning, .toolCalling])
}

private enum BudgetFailure: Error {
    case expected
}
