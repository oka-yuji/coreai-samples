import ConversationFeature
import Foundation
import Testing

struct ConversationStatusTests {
    @Test(
        "The status line names the running step, the tool calls, the stop, the step limit, the failures and the last step"
    )
    func statusLineFollowsTheTurn() {
        #expect(ConversationState(id: "empty", now: launch).statusLine.isEmpty)

        let search = tool("t1", "web_search", #"{"query":"swift"}"#)
        var running = started(.work)
        #expect(running.statusLine == "Step 1 of 8: generating…")
        _ = reduce(&running, .transcriptChanged(run: 1, turn("Question", [calls("c1", search)])))
        #expect(running.statusLine == "Step 1: running web_search…")
        _ = reduce(&running, .stopRequested)
        #expect(running.statusLine == "Stopping…")
        _ = reduce(&running, .stopped(run: 1))
        #expect(running.statusLine == "Stopped after 1 step(s).")

        let fetch = tool("t2", "fetch_page", #"{"url":"https://swift.org"}"#)
        var limited = started(.chat, settings: ConversationSettings(chatMaxSteps: 2))
        let record = turn(
            "Question", [calls("c1", search), output(search, "Results"), calls("c2", fetch), output(fetch, "Page")])
        _ = reduce(&limited, .transcriptChanged(run: 1, record), .stopped(run: 1))
        #expect(limited.statusLine == "Stopped at the 2-step limit.")

        var overflow = started(.work)
        _ = reduce(&overflow, .generationFailed(run: 1, message: ConversationFeature.contextFullPrefix + ": too long"))
        #expect(overflow.statusLine == "Context overflow at step 1." && overflow.contextFull)

        var failed = started(.work)
        _ = reduce(&failed, .generationFailed(run: 1, message: "boom"))
        #expect(failed.statusLine == "Failed at step 1." && !failed.contextFull)

        var finished = started(.work)
        _ = reduce(
            &finished, .transcriptChanged(run: 1, turn("Question", [answer("a", "Done")])),
            .stepFinished(run: 1, stats(1)), .generationEnded(run: 1, stopReason: "stopToken"))
        #expect(finished.statusLine == stats(1).line)
    }

    @Test("The context-end notice appears only after a reply cut at the token limit with less room than the limit")
    func contextEndNoticeNeedsACutReplyShortOfRoom() {
        func ended(_ stopReason: String, reserved: Int?) -> ConversationState {
            var state = started(.work)
            let reply = TranscriptRecord.Entry.response(
                id: "a", text: "Long", metadata: StepMetadata(stopReason: stopReason, reservedTokens: reserved))
            _ = reduce(
                &state, .transcriptChanged(run: 1, turn("Question", [reply])),
                .generationEnded(run: 1, stopReason: stopReason))
            return state
        }
        #expect(ended("maxTokens", reserved: 300).contextEndReserved == 300)
        #expect(!ConversationFeature.continueEligibility(ended("maxTokens", reserved: 300)))
        #expect(ended("maxTokens", reserved: 9_000).contextEndReserved == nil)
        #expect(ended("maxTokens", reserved: nil).contextEndReserved == nil)
        #expect(ended("stopToken", reserved: 300).contextEndReserved == nil)
        #expect(started(.work).contextEndReserved == nil)
    }

    @Test("A refused save is cleared by the next send, and a token limit set in the settings applies to Chat too")
    func refusedSaveClearsAndChatTakesTheTokenLimit() {
        var state = started(.chat)
        let note = tool("t1", "write_note", #"{"title":"Plan","content":"Draft"}"#)
        _ = reduce(&state, .toolRequested(run: 1, note), .generationEnded(run: 1, stopReason: "stopToken"))
        #expect(state.refusedSave && !state.isRunning)
        _ = reduce(&state, .send("Next", mode: .chat, now: launch))
        #expect(!state.refusedSave && state.isRunning)

        let tuned = ConversationSettings(maxSteps: 5, maxTokens: 1_536)
        #expect(TurnPolicy(.chat, settings: tuned).maxTokens == 1_536)
        #expect(TurnPolicy(.chat, settings: ConversationSettings()).maxTokens == 2_048)
    }

    @Test("Rows the app writes carry a detail the screen can translate, while their text stays in English")
    func appRowsCarryADetail() {
        let search = tool("t1", "web_search", #"{"query":"swift"}"#)
        let fetch = tool("t2", "fetch_page", #"{"url":"https://swift.org"}"#)
        var limited = started(.chat, settings: ConversationSettings(chatMaxSteps: 2))
        let record = turn(
            "Question", [calls("c1", search), output(search, "Results"), calls("c2", fetch), output(fetch, "Page")])
        _ = reduce(&limited, .transcriptChanged(run: 1, record), .stopped(run: 1))
        let limit = limited.entries.last
        #expect(
            limit?.detail == .stepLimit(steps: 2, fetched: ["https://swift.org"], saved: [], open: [])
                && limit?.text
                    == "The turn reached its 2-step limit before a final answer.\nFetched: https://swift.org\n"
                    + "Saved notes: none\nStill open: none"
        )

        var partial = started(.work)
        _ = reduce(
            &partial, .transcriptChanged(run: 1, turn("Question", [answer("a", "Half an")])), .stopRequested,
            .stopped(run: 1))
        #expect(partial.entries.last?.detail == .stopped(partial: true) && partial.entries.last?.text == "Half an")
        var empty = started(.work)
        _ = reduce(&empty, .stopRequested, .stopped(run: 1))
        #expect(
            empty.entries.last?.detail == .stopped(partial: false)
                && empty.entries.last?.text == "(stopped before any answer)")

        var cut = started(.work)
        _ = reduce(
            &cut, .transcriptChanged(run: 1, turn("Question", [answer("a", "Long", stop: "maxTokens")])),
            .generationEnded(run: 1, stopReason: "maxTokens"))
        #expect(cut.entries.last?.detail == .tokenLimit(4_096))
    }

    @Test("The context-full message is written and read back in one place, and other messages give no numbers")
    func contextFullMessageRoundTrips() {
        let message = ConversationFeature.contextFullMessage(needed: 17_000, limit: 16_384)
        #expect(message.hasPrefix(ConversationFeature.contextFullPrefix))
        let numbers = ConversationFeature.contextFullNumbers(message)
        #expect(numbers?.needed == 17_000 && numbers?.limit == 16_384)
        #expect(ConversationFeature.contextFullNumbers("boom") == nil)
        #expect(ConversationFeature.contextFullNumbers(ConversationFeature.contextFullPrefix + ": too long") == nil)
    }

    @Test("The citation guard follows the setting in Work and is always off in Chat")
    func citationGuardFollowsTheMode() {
        #expect(TurnPolicy(.work, settings: ConversationSettings()).citationGuard)
        #expect(!TurnPolicy(.work, settings: ConversationSettings(citationGuard: false)).citationGuard)
        #expect(!TurnPolicy(.chat, settings: ConversationSettings(citationGuard: true)).citationGuard)
    }

    @Test(
        "A stop followed by a generation end or a failure, and a step-limit cut followed by an end, still close the turn"
    )
    func stopsAndCutsCloseTheTurnWhateverEndArrives() {
        var ended = started(.work)
        _ = reduce(&ended, .stopRequested, .generationEnded(run: 1, stopReason: "stopToken"))
        #expect(!ended.isRunning && !ended.stopRequested)
        _ = reduce(&ended, .send("Next", mode: .work, now: launch))
        #expect(ended.isRunning)

        let search = tool("t1", "web_search", #"{"query":"swift"}"#)
        let fetch = tool("t2", "fetch_page", #"{"url":"https://swift.org"}"#)
        var cut = started(.chat, settings: ConversationSettings(chatMaxSteps: 2))
        let record = turn(
            "Question", [calls("c1", search), output(search, "Results"), calls("c2", fetch), output(fetch, "Page")])
        _ = reduce(&cut, .transcriptChanged(run: 1, record), .generationEnded(run: 1, stopReason: "stopToken"))
        #expect(!cut.isRunning && cut.finalization == .cutOff)
        #expect(cut.entries.last?.title == "Ended at the step limit")
        #expect(cut.statusLine == "Stopped at the 2-step limit.")

        var failed = started(.work)
        _ = reduce(&failed, .stopRequested, .generationFailed(run: 1, message: "boom"))
        #expect(!failed.isRunning && !failed.stopRequested && failed.entries.last?.title == "Generation failed")
    }

    @Test("The status is a value the screen can translate, and its English line is what the headless runs print")
    func statusIsAValue() {
        var state = started(.work)
        #expect(state.status == .generating(step: 1, limit: 8) && state.statusLine == "Step 1 of 8: generating…")
        let search = tool("t1", "web_search", #"{"query":"swift"}"#)
        _ = reduce(&state, .transcriptChanged(run: 1, turn("Question", [calls("c1", search)])))
        #expect(state.status == .running(step: 1, tools: ["web_search"]))
        var failed = started(.work)
        _ = reduce(&failed, .generationFailed(run: 1, message: "boom"))
        #expect(failed.status == .failed(step: 1) && failed.status?.line == "Failed at step 1.")
        #expect(ConversationState(id: "empty", now: launch).status == nil)
    }

    @Test(
        "Only the record being generated is live, tool outputs never are, nothing is after the turn, and Chat folds results"
    )
    func liveAndCollapsedRows() {
        var state = started(.work)
        let partial = turn(
            "Question", [.reasoning(id: "r", text: "Plan"), .response(id: "a", text: "Partial", metadata: nil)])
        _ = reduce(&state, .transcriptChanged(run: 1, partial))
        #expect(state.entries.map(\.live) == [false, false, true])

        let search = tool("t1", "web_search", #"{"query":"swift"}"#)
        var calling = started(.work)
        _ = reduce(&calling, .transcriptChanged(run: 1, turn("Question", [calls("c1", search)])))
        #expect(calling.entries.last?.kind == .toolCall && calling.entries.last?.live == true)
        let answered = turn("Question", [calls("c1", search), output(search, "Hits")])
        _ = reduce(&calling, .transcriptChanged(run: 1, answered))
        #expect(calling.entries.last?.kind == .toolResult && calling.entries.last?.live == false)

        _ = reduce(&state, .generationEnded(run: 1, stopReason: "stopToken"))
        #expect(state.entries.allSatisfy { !$0.live })

        var chat = started(.chat)
        _ = reduce(&chat, .transcriptChanged(run: 1, turn("Question", [calls("c1", search), output(search, "Hits")])))
        let results = chat.entries.filter { $0.kind == .toolResult }
        #expect(!results.isEmpty && results.allSatisfy { $0.collapsed })
    }
}
