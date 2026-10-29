import ConversationFeature
import ConversationRuntime
import Foundation
import FoundationModels
import Observation
import RAGContext
import Synchronization

@MainActor
enum RAGSectionSelfTest {
    static func run(_ check: (String, Bool, String) -> Void) async {
        for scenario in SectionProbe.Scenario.allCases {
            await run(scenario, check: check)
        }
    }

    private static func run(_ scenario: SectionProbe.Scenario, check: (String, Bool, String) -> Void) async {
        let root = URL.temporaryDirectory.appending(path: "rag-section-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let probe = SectionProbe(scenario: scenario)
        defer { probe.finish() }
        let preferences = ConversationPreferences(persists: false)
        preferences.resetForHeadless()
        let raw =
            "<main><h1>Manual</h1><h2>Alpha</h2><p>Wait ten minutes.</p><h2>Beta</h2><p>Wait thirty minutes.</p></main>"
        let text = AgentTools.plainText(fromHTML: raw)
        let page = RAGPage(raw: raw, text: text, finalURL: "https://rag.example/sections", format: .html)
        let inputs = RAGInputs(
            local: { _ in [] },
            search: { _, _ in [.init(title: "Manual", url: page.finalURL, snippet: "Manual")] },
            read: { _ in page })
        let model = SectionFixtureModel(probe: probe)
        let runner = ConversationRunner(
            id: "sections", chat: ChatViewModel(), preferences: preferences,
            archive: ConversationArchive(directory: root),
            sessionFactory: { tools in LanguageModelSession(model: model, tools: tools.tools(fileOperations: false)) },
            recordsMetrics: false,
            toolExecutor: { _ in
                probe.values.withLock { $0.tools += 1 }
                return AgentToolOutput(text: "Unexpected tool")
            },
            sourceTokenCounter: { $0.count },
            retrieveSources: { question, original, count in
                var settings = original
                settings.strategy = .structureLLM
                settings.recordsDiagnostics = true
                return await RAGRetriever.gather(question: question, settings: settings, count: count, inputs: inputs)
            },
            selectionFactory: {
                RAGSectionSelector(
                    makeSession: { LanguageModelSession(model: model, tools: [], instructions: $0) },
                    countInput: { instructions, prompt in
                        if scenario == .inputBudget { return 3_000 }
                        if scenario == .earlyStop {
                            probe.started.continuation.yield(())
                            for await _ in probe.release.stream { break }
                        }
                        return instructions.count + prompt.count
                    },
                    wait: { duration in
                        if scenario == .timeout {
                            for await _ in probe.timerReady.stream { break }

                        } else {
                            try await Task.sleep(for: duration)
                        }
                    }, stopping: { probe.stopRequested.continuation.yield(()) },
                    finishExecution: {
                        for await _ in probe.finished.stream { break }
                    })
            })
        var initial = ConversationState(id: "sections", now: Date(), mode: .work, settings: preferences.settings)
        initial.settings.webSearchEnabled = true
        let store = ConversationStore(initialState: initial, run: runner.run)
        let question = "Alpha waiting time"
        store.send(.sendSources(question, .init(web: true, local: false, budget: 3_000), now: Date()))
        if scenario == .earlyStop {
            for await _ in probe.started.stream { break }
            store.send(.stopRequested)
        }
        if scenario == .stop || scenario == .timeout {
            for await _ in probe.started.stream { break }
            if scenario == .stop { store.send(.stopRequested) }
            for await _ in probe.stopRequested.stream { break }
            let held = probe.values.withLock { $0.answers == 0 && $0.active == 1 }
            check(
                "rag-section/join-before-answer/" + scenario.rawValue,
                held && store.state.isRunning, "active=\(probe.values.withLock { $0.active })")
            probe.release.continuation.yield(())
            probe.release.continuation.finish()
        }
        for await settled in Observations({ !store.state.isRunning && store.state.persistence == .saved })
        where settled { break }
        let trace = runner.lastRetrieval?.selectionTrace
        let values = probe.values.withLock { $0 }
        let expectedAnswers = [.none, .stop, .earlyStop, .fatal].contains(scenario) ? 0 : 1
        let expectedCalls = [.inputBudget, .earlyStop].contains(scenario) ? 0 : 1
        let expectedStatus: String =
            switch scenario {
            case .valid: "selected"
            case .none: "none"
            case .stop, .earlyStop: "cancelled"
            case .fatal: "failed"
            default: "fallback"
            }
        check(
            "rag-section/" + scenario.rawValue,
            values.answers == expectedAnswers && values.selections == expectedCalls && values.tools == 0
                && values.maximumActive == (expectedCalls + expectedAnswers > 0 ? 1 : 0) && values.optionsValid
                && values.historyClean
                && trace?.status == expectedStatus,
            "answers=\(values.answers), selections=\(values.selections), trace=\(String(describing: trace))")
        check(
            "rag-section/cost/" + scenario.rawValue,
            trace != nil && (trace?.seconds ?? -1) >= 0
                && (expectedCalls == 0 || trace?.usageStatus == "unavailable"
                    || trace?.inputTokens != nil && trace?.outputTokens != nil),
            "status=\(trace?.status ?? "missing")")
        if scenario == .valid {
            check(
                "rag-section/original-source",
                runner.lastRetrieval?.effectiveStrategy == "structure-llm"
                    && runner.lastRetrieval?.bodySources.first?.text.contains("ten minutes") == true
                    && runner.tools.suppliedSourceIDs == ["S1"]
                    && runner.citationGuard.checks.last?.local?.violation == false,
                "selected=\(trace?.selectedIDs ?? [])")
        }
        if scenario == .stop {
            store.send(.sendSources(question, .init(web: true, local: false, budget: 3_000), now: Date()))
            for await settled in Observations({ !store.state.isRunning && store.state.persistence == .saved })
            where settled { break }
            check(
                "rag-section/restart-after-stop",
                probe.values.withLock { $0.answers == 1 && $0.selections == 2 && $0.maximumActive == 1 }
                    && store.state.lastFinish == .eos, "finish=\(store.state.lastFinish)")
        }
    }
}

private final class SectionProbe: Sendable {
    enum Scenario: String, CaseIterable, Sendable {
        case valid, invalid, none, inputBudget, timeout, stop, earlyStop, fatal
    }
    struct Values: Sendable {
        var selections = 0
        var answers = 0
        var tools = 0
        var active = 0
        var maximumActive = 0
        var optionsValid = true
        var historyClean = true
    }
    let scenario: Scenario
    let values = Mutex(Values())
    let started = AsyncStream<Void>.makeStream()
    let timerReady = AsyncStream<Void>.makeStream()
    let stopRequested = AsyncStream<Void>.makeStream()
    let release = AsyncStream<Void>.makeStream()
    let finished = AsyncStream<Void>.makeStream()

    init(scenario: Scenario) { self.scenario = scenario }

    func finish() {
        started.continuation.finish()
        timerReady.continuation.finish()
        stopRequested.continuation.finish()
        release.continuation.finish()
        finished.continuation.finish()
    }
}

private nonisolated struct SectionFixtureModel: LanguageModel {
    enum Failure: Error { case unavailable }
    struct Executor: LanguageModelExecutor {
        typealias Model = SectionFixtureModel
        init(configuration: UUID) {}

        func respond(
            to request: LanguageModelExecutorGenerationRequest, model: Model,
            streamingInto channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            let instructions = request.transcript.compactMap { entry -> String? in
                if case .instructions(let value) = entry {
                    return value.segments.map { if case .text(let text) = $0 { text.content } else { "" } }.joined()
                }
                return nil
            }.joined()
            let selecting = instructions.contains("Choose at most")
            let probe = model.probe
            let call = probe.values.withLock { state in
                state.active += 1
                state.maximumActive = max(state.maximumActive, state.active)
                if selecting {
                    state.selections += 1
                    state.optionsValid =
                        state.optionsValid && request.enabledToolDefinitions.isEmpty
                        && request.generationOptions.maximumResponseTokens == 128
                        && request.generationOptions.samplingMode?.kind == .greedy
                        && request.generationOptions.toolCallingMode?.kind == .disallowed
                        && request.contextOptions.reasoningLevel == .custom("off")
                } else {
                    state.answers += 1
                    let prompts = request.transcript.compactMap { entry -> String? in
                        if case .prompt(let value) = entry {
                            return value.segments.map { if case .text(let text) = $0 { text.content } else { "" } }
                                .joined()
                        }
                        return nil
                    }
                    state.historyClean = state.historyClean && !prompts.contains { $0.contains("\"candidates\"") }
                }
                return state.selections
            }
            defer {
                probe.values.withLock { $0.active -= 1 }
                if selecting { probe.finished.continuation.yield(()) }
            }
            if selecting, call == 1, probe.scenario == .stop || probe.scenario == .timeout {
                probe.started.continuation.yield(())
                probe.timerReady.continuation.yield(())
                await withTaskCancellationShield {
                    for await _ in probe.release.stream { break }
                }
                try Task.checkCancellation()
            }
            if selecting && probe.scenario == .fatal { throw Failure.unavailable }
            let text =
                selecting
                ? probe.scenario == .none ? "NONE" : probe.scenario == .invalid ? "C99" : "C1"
                : "Ten minutes [S1]."
            await channel.send(.response(action: .appendText(text, tokenCount: 4)))
            await channel.send(.response(action: .updateMetadata(["stopReason": "stopToken"])))
        }
    }
    let probe: SectionProbe
    let executorConfiguration = UUID()
    let capabilities = LanguageModelCapabilities([.reasoning, .toolCalling])
}
