import Foundation
import FoundationModels
import Testing

@testable import CoreAIBackend

@Suite struct ReloadPolicyTests {
    private func prompt(_ id: String) -> Transcript.Entry {
        .prompt(.init(id: id, segments: [.text(.init(content: "a"))]))
    }

    private func observe(_ state: inout ReloadState, _ transcript: Transcript) -> Bool {
        state.observe(Runtime.reloadConversation(transcript))
    }

    @Test(
        "Ten completed reads wait for the next new conversation and a footprint limit also applies within a conversation"
    )
    func reloadTriggers() {
        var state = ReloadState()
        let policy = LocalLanguageModel.ReloadPolicy(readInterval: 10, footprintLimitBytes: 100)
        for _ in 0..<10 {
            #expect(!state.shouldReload(policy: policy, isNewConversation: true, footprint: 99))
            state.completedRead()
        }
        #expect(!state.shouldReload(policy: policy, isNewConversation: false, footprint: 99))
        #expect(state.shouldReload(policy: policy, isNewConversation: true, footprint: 99))
        #expect(state.shouldReload(policy: policy, isNewConversation: false, footprint: 100))
        #expect(state.shouldReload(policy: policy, isNewConversation: false, footprint: 101))
        state.didLoad()
        #expect(state.completedReads == 0)
        #expect(!state.shouldReload(policy: policy, isNewConversation: true, footprint: 99))
        for _ in 0..<11 { state.completedRead() }
        #expect(
            !state.shouldReload(
                policy: .init(readInterval: 0, footprintLimitBytes: 0), isNewConversation: true, footprint: .max))
    }

    @Test("The default enables the ten-read interval and the calibrated footprint threshold")
    func calibratedDefault() {
        #expect(LocalLanguageModel.ReloadPolicy() == .init(readInterval: 10, footprintLimitBytes: 18_000_000_000))
    }

    @Test("An initial prompt is new once even when its first request fails and the same prompt is retried")
    func retryIsNotNew() {
        var state = ReloadState()
        let request = Transcript(entries: [prompt("initial")])
        let observed1 = observe(&state, request)
        #expect(observed1)
        let observed2 = observe(&state, request)
        #expect(!observed2)
        #expect(state.completedReads == 0)
    }

    @Test("Internal tool rounds, preserved reasoning and prior responses are conversation continuations")
    func historyPreventsNewConversation() throws {
        let response = Transcript.Entry.response(.init(segments: [.text(.init(content: "b"))]))
        let reasoning = Transcript.Entry.reasoning(.init(segments: [.text(.init(content: "b"))]))
        let calls = Transcript.Entry.toolCalls(
            .init([
                .init(id: "call", toolName: "read", arguments: try GeneratedContent(json: "{}"))
            ]))
        let output = Transcript.Entry.toolOutput(
            .init(id: "output", toolName: "read", segments: [.text(.init(content: "b"))]))
        for history in [[response], [reasoning], [calls, output]] {
            var state = ReloadState()
            let observed3 = observe(&state, Transcript(entries: [prompt("initial")] + history))
            #expect(!observed3)
        }
    }

    @Test("Trimmed history and switching to a known session cannot create another new conversation")
    func trimmedAndSwitchedHistory() {
        var state = ReloadState()
        let observed4 = observe(&state, Transcript(entries: [prompt("a1")]))
        #expect(observed4)
        let observed5 = observe(&state, Transcript(entries: [prompt("b1")]))
        #expect(observed5)
        let observed6 = observe(&state, Transcript(entries: [prompt("a1"), prompt("a2")]))
        #expect(!observed6)
        let observed7 = observe(&state, Transcript(entries: [prompt("a2")]))
        #expect(!observed7)
        let observed8 = observe(&state, Transcript(entries: [prompt("b1")]))
        #expect(!observed8)
    }

    @Test("Token sharing, checkpoint loss and tool-output cuts cannot affect the transcript identity rule")
    func identityDoesNotUseEngineState() {
        var state = ReloadState()
        let input = Transcript(entries: [prompt("same-id")])
        let observed9 = observe(&state, input)
        #expect(observed9)
        for _ in 0..<3 {
            state.didLoad()
            let observed10 = observe(&state, input)
            #expect(!observed10)
        }
    }
}
