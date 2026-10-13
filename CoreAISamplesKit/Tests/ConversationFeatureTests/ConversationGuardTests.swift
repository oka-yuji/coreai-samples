import ConversationFeature
import FeatureStore
import Foundation
import Testing

struct ConversationGuardTests {
    @Test("A guard note goes at the end of the turn's entries and is saved, and a note for another run is ignored")
    func guardNotesGoAtTheEndOfTheTurn() throws {
        var state = started(.work)
        _ = reduce(&state, .transcriptChanged(run: 1, turn("Question", [answer("a", "Answer")])))
        let running = reduce(
            &state,
            .guardNote(run: 1, title: "Citations", text: "1/1 cited URLs were retrieved in this task", warning: false))
        #expect(try #require(snapshot(running[0])).status == .interrupted)
        #expect(state.entries.map(\.kind) == [.user, .answer, .note])

        _ = reduce(&state, .generationEnded(run: 1, stopReason: "maxTokens"))
        let ended = reduce(
            &state, .guardNote(run: 1, title: "Citation guard", text: "Could not confirm", warning: true),
            .guardNote(run: 2, title: "Late", text: "ignored", warning: false))
        #expect(try #require(snapshot(ended[0])).status == .completed)
        #expect(ended[1] == [])
        #expect(
            state.entries.map(\.title) == [
                "You", "Assistant", "Answer cut at the token limit (4096)", "Citations", "Citation guard",
            ])
        #expect(
            state.entries.last
                == ConversationEntry(
                    id: "note-0-1", kind: .note, title: "Citation guard", text: "Could not confirm", warning: true))
        #expect(ConversationSnapshot(state).markdown().hasSuffix("_Citation guard: Could not confirm_\n"))
    }

    @Test(
        "A repair turn continues the step count of the turn before it, keeps the Work policy and leaves the refused save alone"
    )
    func repairTurnContinuesTheStepCount() throws {
        var chat = started(.chat)
        _ = reduce(
            &chat, .transcriptChanged(run: 1, turn("Question", [answer("a", "Hi")])),
            .generationEnded(run: 1, stopReason: "stopToken"))
        #expect(reduce(&chat, .sendRepair("Fix the sources", now: launch)) == [[]])
        var busy = started(.work)
        #expect(reduce(&busy, .sendRepair("Fix the sources", now: launch)) == [[]])

        let settings = ConversationSettings(maxSteps: 4)
        var state = started(.work, settings: settings, prompt: "Research")
        let search = tool("t1", "web_search", #"{"query":"swift"}"#)
        var entries: [TranscriptRecord.Entry] = [
            .instructions(id: "i", text: "Rules"), .prompt(id: "p1", text: "Research"), calls("c1", search),
            output(search, "results"), answer("a1", "Answer"),
        ]
        _ = reduce(
            &state, .transcriptChanged(run: 1, TranscriptRecord(entries: entries)),
            .stepFinished(run: 1, stats(1, calls: 1)), .stepFinished(run: 1, stats(2)),
            .generationEnded(run: 1, stopReason: "stopToken"))
        #expect(state.step == 2 && !state.isRunning)

        state.refusedSave = true
        let policy = TurnPolicy(.work, settings: settings)
        let commands = reduce(&state, .sendRepair("Fix the sources", now: launch + 60))
        #expect(
            commands == [
                [.run(.generate(run: 2, prompt: "Fix the sources", policy: policy, finalize: false), id: .generation)]
            ])
        let repair = try #require(state.turns.last)
        #expect(repair.kind == .repair && repair.mode == .work && repair.offset == 2 && repair.start == 5)
        #expect(state.step == 3 && state.refusedSave && state.title == "Research" && state.updatedAt == launch + 60)

        let fetch = tool("t2", "fetch_page", #"{"url":"https://a.example/a"}"#)
        let further = tool("t3", "fetch_page", #"{"url":"https://b.example/b"}"#)
        entries += [.prompt(id: "p2", text: "Fix the sources"), calls("c2", fetch)]
        _ = reduce(&state, .transcriptChanged(run: 2, TranscriptRecord(entries: entries)))
        #expect(state.step == 3 && state.finalization == .normal)
        #expect(reduce(&state, .toolRequested(run: 2, fetch)) == [[.run(.runTool(fetch, finalize: true), id: nil)]])

        entries += [output(fetch, ConversationFeature.finalized("page a")), calls("c3", further)]
        _ = reduce(
            &state, .transcriptChanged(run: 2, TranscriptRecord(entries: entries)),
            .stepFinished(run: 2, stats(3, calls: 1)))
        #expect(state.step == 4 && state.finalization == .lastStep)
        let ended = reduce(&state, .toolRequested(run: 2, further))
        #expect(
            ended == [
                [
                    .run(.refuseTool(further, reason: ConversationFeature.endedRefusal), id: nil),
                    .run(.stop(run: 2), id: .stop),
                ]
            ])
        _ = reduce(&state, .stopped(run: 2))
        #expect(
            state.entries.map(\.id) == [
                "turn-0", "t1", "result-t1", "a1", "turn-1", "t2", "result-t2", "t3", "limit-1",
            ])
        #expect(state.entries.first { $0.id == "t2" }?.stats == stats(3, calls: 1).line)
        #expect(
            state.entries.last?.text == """
                The turn reached its 4-step limit before a final answer.
                Fetched: https://a.example/a
                Saved notes: none
                Still open: fetch_page https://b.example/b
                """)
    }
}
