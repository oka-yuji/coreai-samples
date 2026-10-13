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
        var limited = started(.chat)
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
        #expect(
            ended("maxTokens", reserved: 300).contextEndNotice
                == "The reply stopped at the end of the context window (300 tokens were left for it). Start a new conversation if the next message does not fit."
        )
        #expect(ended("maxTokens", reserved: 9_000).contextEndNotice == nil)
        #expect(ended("maxTokens", reserved: nil).contextEndNotice == nil)
        #expect(ended("stopToken", reserved: 300).contextEndNotice == nil)
        #expect(started(.work).contextEndNotice == nil)
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
        var limited = started(.chat)
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
}
