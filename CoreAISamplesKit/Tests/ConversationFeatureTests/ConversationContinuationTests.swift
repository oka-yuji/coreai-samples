import ConversationFeature
import Foundation
import Testing

struct ConversationContinuationTests {
    @Test("Continue is explicit, adds a user turn and joins only the displayed answer")
    func continuesWithoutRewritingHistory() throws {
        var state = started(.work)
        var record = turn("Question", [answer("a1", "The answer is inter", stop: "maxTokens")])
        _ = reduce(&state, .transcriptChanged(run: 1, record), .generationEnded(run: 1, stopReason: "maxTokens"))
        #expect(state.turns.count == 1 && !state.isRunning)
        _ = reduce(&state, .archiveSaved(id: state.id, revision: state.archiveRevision), .modeChanged(.chat))
        let continuation = reduce(&state, .continueRequested(now: launch + 1))
        #expect(
            continuation.flatMap { $0 }.contains {
                if case .run(.generate(_, _, let policy, _), _) = $0 {
                    return policy.answerPrefix == "The answer is inter"
                }
                return false
            })
        #expect(state.isRunning && state.mode == .work && state.turns.count == 2)
        #expect(
            state.turns.last?.prompt == ConversationFeature.continuationPrompt
                && state.turns.last?.kind == .continuation)
        #expect(state.turns.last?.policy.continuing == true)
        record.entries += [.prompt(id: "p2", text: ConversationFeature.continuationPrompt), answer("a2", "esting.")]
        _ = reduce(&state, .transcriptChanged(run: 2, record), .generationEnded(run: 2, stopReason: "stopToken"))
        #expect(state.entries.filter { $0.kind == .answer }.map(\.text) == ["The answer is interesting."])
        #expect(state.entries.filter { $0.kind == .user }.count == 1)
        #expect(!state.entries.contains { $0.id == "cap-0" })
        let snapshot = ConversationSnapshot(state)
        #expect(snapshot.transcript == record && snapshot.turns.count == 2)
        #expect(snapshot.markdown().contains("## You\n\n" + ConversationFeature.continuationPrompt))
        #expect(try JSONDecoder().decode(ConversationSnapshot.self, from: JSONEncoder().encode(snapshot)) == snapshot)
    }

    @Test("An unfinished note call is reported as an unsaved note and cannot be continued as an answer")
    func cutNote() {
        var state = started(.work)
        let text = "<tool_call>\n<function=write_note>\n<parameter=content>Partial note"
        _ = reduce(
            &state, .transcriptChanged(run: 1, turn("Question", [answer("a", text, stop: "maxTokens")])),
            .generationEnded(run: 1, stopReason: "maxTokens"))
        let failure = state.entries.last { $0.kind == .failure }
        #expect(failure?.title == "Note was not saved")
        #expect(failure?.text == "The write_note call was cut at the token limit; the note was not saved.")
        #expect(!ConversationFeature.continueEligibility(state))
        #expect(!state.entries.contains { if case .tokenLimit = $0.detail { true } else { false } })
    }

    @Test(
        "Partial control tags and unfinished thinking never offer Continue",
        arguments: [
            "Answer <tool_", "Answer <tool_call", "Answer </tool_", "Answer </tool_call", "Answer <think>",
            "Answer </thi",
        ])
    func refusesOpenSyntax(text: String) {
        var state = started(.chat)
        _ = reduce(
            &state, .transcriptChanged(run: 1, turn("Question", [answer("a", text, stop: "maxTokens")])),
            .generationEnded(run: 1, stopReason: "maxTokens"))
        #expect(!ConversationFeature.continueEligibility(state))
    }

    @Test("Old archive versions stay supported and unknown future versions stay unsupported")
    func versions() {
        #expect(ConversationSnapshot.supports(version: 1) && ConversationSnapshot.supports(version: 2))
        #expect(!ConversationSnapshot.supports(version: 0) && !ConversationSnapshot.supports(version: 3))
    }
}
