import ConversationFeature
import Foundation
import Testing

struct ConversationEndingTests {
    @Test("A generation that ends without a known stop reason is shown as a failure and offers no continue")
    func reportsAMissingStopReasonAsAFailure() {
        let record = turn("Question", [answer("a", "An answer")])
        var state = started(.chat)
        _ = reduce(&state, .transcriptChanged(run: 1, record), .generationEnded(run: 1, stopReason: nil))
        #expect(state.lastFinish == .failed && !state.isRunning)
        #expect(
            state.entries.last
                == ConversationEntry(
                    id: "failed-0", kind: .failure, title: "Generation failed",
                    text: "Generation ended without a stop reason"))
        #expect(!ConversationFeature.continueEligibility(state))

        var unknown = started(.chat)
        _ = reduce(&unknown, .transcriptChanged(run: 1, record), .generationEnded(run: 1, stopReason: "contextLimit"))
        #expect(unknown.lastFinish == .failed)
        #expect(unknown.entries.last?.text == "Generation ended with stop reason: contextLimit")
    }

    @Test("A failed generation ends the turn with its message, and a late failure or one from another run is ignored")
    func endsTheTurnOnAFailedGeneration() throws {
        var state = started(.work)
        let record = turn("Question", [.reasoning(id: "r", text: "Reading the pages")])
        let failed = reduce(
            &state, .transcriptChanged(run: 1, record), .generationFailed(run: 1, message: "context overflow"))
        let saved = try #require(snapshot(failed[1]))
        #expect(saved.status == .completed && saved.turns.last?.finish == .failed)
        #expect(saved.turns.last?.failure == "context overflow")
        #expect(!state.isRunning && state.lastFinish == .failed)
        #expect(
            state.entries.last
                == ConversationEntry(
                    id: "failed-0", kind: .failure, title: "Generation failed", text: "context overflow"))

        let ended = state
        #expect(reduce(&state, .generationFailed(run: 1, message: "late")) == [[]])
        #expect(state == ended)

        _ = reduce(&state, .send("Try again", mode: .work, now: launch + 60))
        let running = state
        let stale = reduce(
            &state, .generationFailed(run: 1, message: "stale"), .generationFailed(run: 3, message: "unknown run"))
        #expect(stale == [[], []])
        #expect(state == running && state.isRunning)
    }

    @Test("An interrupted snapshot is restored with its last turn marked as stopped and offers no continue")
    func restoresAnInterruptedTurnAsStopped() throws {
        var source = started(.work, id: "old")
        let search = tool("t1", "web_search", #"{"query":"swift"}"#)
        let record = turn("Question", [calls("c1", search), output(search, "results")])
        let progress = reduce(&source, .transcriptChanged(run: 1, record), .stepFinished(run: 1, stats(1, calls: 1)))
        let interrupted = try #require(snapshot(progress[1]))
        #expect(interrupted.status == .interrupted && interrupted.turns.last?.finish == Finish.none)

        var state = ConversationState(id: "c1", now: launch)
        _ = reduce(&state, .restoreRequested(id: "old"), .restored(interrupted))
        #expect(state.id == "old" && state.turns.last?.finish == .stopped)
        #expect(!state.isRunning && state.step == 0)
        #expect(state.entries.last?.title == "Stopped")
        #expect(!ConversationFeature.continueEligibility(state))
    }

    @Test("A step whose tool outputs were shortened shows the context-full note on screen and in the Markdown export")
    func showsTheContextFullNote() throws {
        var state = ConversationState(id: "c1", now: launch)
        var entries: [TranscriptRecord.Entry] = [
            .instructions(id: "i", text: "Rules"), .prompt(id: "p1", text: "Hello"), answer("a1", "Hi."),
        ]
        _ = reduce(
            &state, .send("Hello", mode: .chat, now: launch),
            .transcriptChanged(run: 1, TranscriptRecord(entries: entries)),
            .generationEnded(run: 1, stopReason: "stopToken"))
        let shortened = StepMetadata(stopReason: "stopToken", truncatedToolOutputs: 1)
        entries += [
            .prompt(id: "p2", text: "Summarize the pages"),
            .response(id: "a2", text: "A short summary.", metadata: shortened),
        ]
        let ended = reduce(
            &state, .send("Summarize the pages", mode: .work, now: launch + 60),
            .transcriptChanged(run: 2, TranscriptRecord(entries: entries)),
            .generationEnded(run: 2, stopReason: "stopToken"))
        let note = ConversationEntry(
            id: "context-1-0", kind: .note, title: "Context full — dropped the oldest tool output",
            text: "1 tool output(s) shortened to fit the context window.")
        #expect(state.entries.contains(note))
        #expect(state.entries.map(\.id) == ["turn-0", "a1", "turn-1", "context-1-0", "a2"])
        let saved = try #require(snapshot(ended[2]))
        #expect(
            saved.markdown().contains(
                "\n_Context full — dropped the oldest tool output: 1 tool output(s) shortened to fit the context window._\n"
            ))
    }

    @Test("A turn that ends with only thinking shows the no-answer placeholder, and a stop before any answer says so")
    func showsThePlaceholdersForMissingAnswers() {
        let thinking = turn("Question", [.reasoning(id: "r", text: "Let me think")])
        var answered = started(.chat)
        _ = reduce(&answered, .transcriptChanged(run: 1, thinking), .generationEnded(run: 1, stopReason: "stopToken"))
        #expect(
            answered.entries.last
                == ConversationEntry(id: "answer-0", kind: .answer, title: "Assistant", text: "(no answer text)"))
        #expect(answered.entries.last?.stats == nil)

        var stopped = started(.chat)
        _ = reduce(&stopped, .transcriptChanged(run: 1, thinking), .stopRequested, .stopped(run: 1))
        #expect(stopped.entries.last?.text == "(stopped before any answer)")
        #expect(stopped.entries.last?.title == "Stopped")
    }
}
