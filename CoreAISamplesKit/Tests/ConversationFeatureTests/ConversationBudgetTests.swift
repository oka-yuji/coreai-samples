import FeatureStore
import Foundation
import Testing

@testable import ConversationFeature

struct ConversationBudgetTests {
    private let now = Date(timeIntervalSince1970: 100)

    @Test("Chat has a configurable sixteen-step default independent of Work and clamps new budgets")
    func budgets() {
        #expect(TurnPolicy(.chat, settings: ConversationSettings()).maxSteps == 16)
        let settings = ConversationSettings(maxSteps: 5, chatMaxSteps: 23)
        #expect(TurnPolicy(.chat, settings: settings).maxSteps == 23)
        #expect(TurnPolicy(.work, settings: settings).maxSteps == 5)
        #expect(TurnPolicy(.chat, settings: ConversationSettings(chatMaxSteps: 0)).maxSteps == 1)
        #expect(TurnPolicy(.chat, settings: ConversationSettings(chatMaxSteps: 500)).maxSteps == 64)
    }

    @Test("Chat can search, fetch a page and answer, then starts the next question with a fresh budget")
    func searchFetchAnswer() {
        let now = Date(timeIntervalSince1970: 100)
        var state = ConversationState(id: "budget", now: now, settings: ConversationSettings(webSearchEnabled: true))
        _ = ConversationFeature.reduce(&state, .send("Find the release date", mode: .chat, now: now))
        let search = ToolCallRecord(id: "search", name: "web_search", arguments: "{}")
        let fetch = ToolCallRecord(id: "fetch", name: "fetch_page", arguments: "{}")
        var record = TranscriptRecord(entries: [
            .prompt(id: "p1", text: "Find the release date"),
            .toolCalls(id: "c1", calls: [search], metadata: nil),
            .toolOutput(id: "search", name: "web_search", text: "Found the official page"),
            .toolCalls(id: "c2", calls: [fetch], metadata: nil),
        ])
        _ = ConversationFeature.reduce(&state, .transcriptChanged(run: 1, record))
        let fetchCommand = ConversationFeature.reduce(&state, .toolRequested(run: 1, fetch))
        #expect(fetchCommand == [.run(.runTool(fetch, finalize: false), id: nil)])
        record.entries += [
            .toolOutput(id: "fetch", name: "fetch_page", text: "Verified the date"),
            .response(id: "answer", text: "The verified date", metadata: StepMetadata(stopReason: "stopToken")),
        ]
        _ = ConversationFeature.reduce(&state, .transcriptChanged(run: 1, record))
        _ = ConversationFeature.reduce(&state, .generationEnded(run: 1, stopReason: "stopToken"))
        #expect(state.lastFinish == .eos && state.finalization != .cutOff)
        #expect(state.entries.contains { $0.kind == .answer && $0.text == "The verified date" })
        _ = ConversationFeature.reduce(&state, .send("Compare the price", mode: .chat, now: now + 1))
        #expect(state.runID == 2 && state.step == 1 && state.turns.last?.offset == 0)
        #expect(state.transcript == record)
    }

    @Test("A research cutoff reserves exactly one answer-only generation and hides its internal prompt")
    func finalAnswer() throws {
        var state = cutOff()
        #expect(!state.stopRequested && state.status == .finalizing)
        let original = state.transcript
        let commands = ConversationFeature.reduce(&state, .budgetStopped(run: 1))
        let policy = try #require(state.turns.last?.policy)
        #expect(policy.answerOnly && policy.allowedTools.isEmpty && policy.maxSteps == 2)
        #expect(
            commands == [
                .run(
                    .generate(
                        run: 2, prompt: ConversationFeature.budgetAnswerPrompt, policy: policy, finalize: false),
                    id: .generation)
            ])
        #expect(state.transcript == original && state.turns.count == 1)
        var record = original
        record.entries.append(.toolOutput(id: "t", name: "web_search", text: ConversationFeature.endedRefusal))
        _ = ConversationFeature.reduce(&state, .transcriptChanged(run: 2, record))
        _ = ConversationFeature.reduce(&state, .finalAnswerStarted(run: 2, start: record.entries.count))
        record.entries += [
            .prompt(id: "internal", text: ConversationFeature.budgetAnswerPrompt),
            .response(id: "answer", text: "Here is what was verified", metadata: StepMetadata(stopReason: "stopToken")),
        ]
        _ = ConversationFeature.reduce(&state, .transcriptChanged(run: 2, record))
        _ = ConversationFeature.reduce(&state, .generationEnded(run: 2, stopReason: "stopToken"))
        #expect(state.lastFinish == .eos && !state.isRunning)
        #expect(state.entries.filter { $0.kind == .user }.map(\.text) == ["Research"])
        #expect(state.entries.contains { $0.kind == .answer && $0.text == "Here is what was verified" })
        #expect(!ConversationSnapshot(state).markdown().contains(ConversationFeature.budgetAnswerPrompt))
    }

    @Test("User cancellation and termination prevent an automatic final answer", arguments: [false, true])
    func cancelsHandoff(terminating: Bool) {
        var state = cutOff()
        _ = ConversationFeature.reduce(&state, terminating ? .terminationRequested : .stopRequested)
        let commands = ConversationFeature.reduce(&state, .budgetStopped(run: 1))
        #expect(state.isRunning && state.turns.last?.finalAnswerStart == nil)
        #expect(!commands.contains { if case .run(.generate, _) = $0 { true } else { false } })
        #expect(commands.contains { if case .run(.settleBudgetStop, _) = $0 { true } else { false } })
        _ = ConversationFeature.reduce(&state, .stopped(run: state.runID))
        #expect(!state.isRunning)
    }

    @Test("The final answer ignores old events, can be stopped, and never starts another automatic answer")
    func boundedAndCancellable() {
        var state = cutOff()
        _ = ConversationFeature.reduce(&state, .budgetStopped(run: 1))
        let finalizing = state
        #expect(ConversationFeature.reduce(&state, .stopped(run: 1)).isEmpty)
        #expect(ConversationFeature.reduce(&state, .generationEnded(run: 1, stopReason: "stopToken")).isEmpty)
        #expect(state == finalizing)
        #expect(ConversationFeature.reduce(&state, .stopRequested) == [.run(.stop(run: 2), id: .stop)])
        _ = ConversationFeature.reduce(&state, .stopped(run: 2))
        #expect(!state.isRunning && state.lastFinish == .stopped)

        state = cutOff()
        _ = ConversationFeature.reduce(&state, .budgetStopped(run: 1))
        let call = ToolCallRecord(id: "unexpected", name: "fetch_page", arguments: "{}")
        let denied = ConversationFeature.reduce(&state, .toolRequested(run: 2, call))
        #expect(denied.contains(.run(.stopForBudget(run: 2), id: .stop)))
        let ended = ConversationFeature.reduce(&state, .budgetStopped(run: 2))
        #expect(state.isRunning && state.runID == 3)
        #expect(!ended.contains { if case .run(.generate, _) = $0 { true } else { false } })
        _ = ConversationFeature.reduce(&state, .stopped(run: 3))
        #expect(!state.isRunning)
    }

    @Test(
        "Final-answer failures, empty output and token limits retain the research without another retry",
        arguments: [0, 1, 2])
    func finalFailure(_ ending: Int) {
        var state = cutOff()
        _ = ConversationFeature.reduce(&state, .budgetStopped(run: 1))
        let original = state.transcript
        let action: ConversationAction =
            ending == 0
            ? .generationFailed(run: 2, message: "Unavailable")
            : .generationEnded(run: 2, stopReason: ending == 1 ? "stopToken" : "maxTokens")
        let commands = ConversationFeature.reduce(&state, action)
        #expect(!state.isRunning && state.transcript == original)
        #expect(state.turns.last?.finalAnswerStart != nil)
        #expect(!commands.contains { if case .run(.generate, _) = $0 { true } else { false } })
    }

    @Test("Old Chat archives retain their actual two-step limit while new turns store the chosen budget")
    func legacyBudget() throws {
        var state = ConversationState(id: "legacy", now: now)
        _ = ConversationFeature.reduce(&state, .send("Question", mode: .chat, now: now))
        let snapshot = ConversationSnapshot(state)
        #expect(snapshot.turns.first?.stepLimit == 16)
        let data = try JSONEncoder().encode(snapshot)
        var object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        var turns = try #require(object["turns"] as? [[String: Any]])
        turns[0].removeValue(forKey: "stepLimit")
        object["turns"] = turns
        object["formatVersion"] = 3
        let legacy = try JSONDecoder().decode(
            ConversationSnapshot.self, from: JSONSerialization.data(withJSONObject: object))
        #expect(legacy.turns.first?.policy.maxSteps == 2)
        #expect(legacy.turns.first?.settings.chatMaxSteps == nil)
        #expect(TurnPolicy(.chat, settings: legacy.turns[0].settings).maxSteps == 16)
    }

    @Test(
        "A continuation of the final answer keeps tools disabled, while a research continuation preserves its citation prefix"
    )
    func continuationPermissionsAndPrefix() throws {
        var state = cutOff()
        _ = ConversationFeature.reduce(&state, .budgetStopped(run: 1))
        state.transcript.entries += [
            .prompt(id: "internal", text: ConversationFeature.budgetAnswerPrompt),
            .response(id: "partial", text: "Partial answer", metadata: StepMetadata(stopReason: "maxTokens")),
        ]
        _ = ConversationFeature.reduce(&state, .generationEnded(run: 2, stopReason: "maxTokens"))
        state.persistence = .saved
        let commands = ConversationFeature.reduce(&state, .continueRequested(now: now + 1))
        guard case .run(.generate(_, _, let continuation, _), _)? = commands.first else {
            Issue.record("Missing continuation")
            return
        }
        #expect(continuation.allowedTools.isEmpty && continuation.maxSteps == 1)

        var work = ConversationState(
            id: "prefix", now: now, settings: ConversationSettings(maxSteps: 1, webSearchEnabled: true))
        _ = ConversationFeature.reduce(&work, .send("Research", mode: .work, now: now))
        let prefix = "Unverified https://unfetched.example/item "
        let record = TranscriptRecord(entries: [
            .prompt(id: "p", text: "Research"),
            .response(id: "a", text: prefix, metadata: StepMetadata(stopReason: "maxTokens")),
        ])
        _ = ConversationFeature.reduce(&work, .transcriptChanged(run: 1, record))
        _ = ConversationFeature.reduce(&work, .generationEnded(run: 1, stopReason: "maxTokens"))
        work.persistence = .saved
        _ = ConversationFeature.reduce(&work, .continueRequested(now: now + 1))
        let call = ToolCallRecord(id: "t", name: "web_search", arguments: "{}")
        var next = record
        next.entries += [.prompt(id: "continued", text: "Continue"), .toolCalls(id: "c", calls: [call], metadata: nil)]
        _ = ConversationFeature.reduce(&work, .transcriptChanged(run: 2, next))
        _ = ConversationFeature.reduce(&work, .toolRequested(run: 2, call))
        let final = ConversationFeature.reduce(&work, .budgetStopped(run: 2))
        guard case .run(.generate(_, _, let policy, _), _)? = final.first else {
            Issue.record("Missing final answer")
            return
        }
        #expect(policy.answerPrefix == prefix)
    }

    @Test("New turns save the normalized budget instead of an out-of-range settings value")
    func normalizedArchive() {
        var state = ConversationState(id: "normalized", now: now, settings: ConversationSettings(chatMaxSteps: 500))
        _ = ConversationFeature.reduce(&state, .send("Question", mode: .chat, now: now))
        #expect(state.turns.last?.stepLimit == 64 && state.turns.last?.settings.chatMaxSteps == 64)
    }

    private func cutOff() -> ConversationState {
        var state = ConversationState(
            id: "budget", now: now, settings: ConversationSettings(chatMaxSteps: 1, webSearchEnabled: true))
        _ = ConversationFeature.reduce(&state, .send("Research", mode: .chat, now: now))
        let call = ToolCallRecord(id: "t", name: "web_search", arguments: "{}")
        let transcript = TranscriptRecord(entries: [
            .prompt(id: "p", text: "Research"), .toolCalls(id: "c", calls: [call], metadata: nil),
        ])
        _ = ConversationFeature.reduce(&state, .transcriptChanged(run: 1, transcript))
        _ = ConversationFeature.reduce(&state, .toolRequested(run: 1, call))
        return state
    }
}
