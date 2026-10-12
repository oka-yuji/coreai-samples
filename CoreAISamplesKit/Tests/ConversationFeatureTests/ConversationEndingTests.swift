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
}
