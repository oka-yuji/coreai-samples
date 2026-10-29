import Foundation
import FoundationModels
import Testing

@_spi(Verification) @testable import CoreAIBackend

@Suite("Tool-free prompt budget")
struct ToolFreeTokenCountTests {
    @Test("A measured value shares the loaded runtime and retains partial failed prefill counts")
    func measuredFailure() async throws {
        let fixture = try await ReloadTestModel.make(policy: .init(readInterval: 0), limit: 512)
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        let measured = fixture.model.recordingMeasurements(.summary)
        fixture.control.state.withLock { $0.failReadAfterTokens = 1 }
        let session = LanguageModelSession(model: measured)
        await #expect(throws: (any Error).self) {
            try await session.respond(to: "aaaa", options: .init(maximumResponseTokens: 1))
        }
        let record = try #require(await measured.lastRequestMeasurement())
        #expect(record.outcome == .failed)
        #expect(session.usage.input.totalTokenCount == 0)
        #expect(record.promptTokens == 4 && record.prefilledTokens == 1 && record.sampledTokens == 0)
        #expect(fixture.control.loads == 1)
        _ = try await fixture.request()
        #expect(await fixture.model.lastRequestMeasurement() == nil)
        #expect(fixture.control.loads == 1)
    }

    @Test("The preview includes instructions and the native template without reading the engine")
    func previewMatchesNative() async throws {
        let tokenizer =
            #"{"added_tokens":[{"id":3,"content":"<|im_end|>","special":true},{"id":4,"content":"<think>","special":true}],"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]}}"#
        let fixture = try await ReloadTestModel.make(policy: .init(readInterval: 0), limit: 512, tokenizer: tokenizer)
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        let before = fixture.control.events
        let instructions = "a"
        let prompt = "b"
        let count = try await fixture.model.countToolFreePromptTokens(instructions: instructions, prompt: prompt)
        #expect(fixture.control.events == before)
        let session = LanguageModelSession(model: fixture.model, tools: [], instructions: instructions)
        let answer = try await session.respond(
            to: prompt,
            options: .init(samplingMode: .greedy, maximumResponseTokens: 1, toolCallingMode: .disallowed),
            contextOptions: .init(reasoningLevel: .custom("off")))
        #expect(count == answer.usage.input.totalTokenCount)
        #expect(fixture.control.loads == 1)
    }
}
