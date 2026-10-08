import Foundation
import FoundationModels
import Testing

@_spi(Verification) @testable import CoreAIBackend

extension ModelTestSettings {
    static func reloadPolicy() throws -> LocalLanguageModel.ReloadPolicy {
        let environment = ProcessInfo.processInfo.environment
        var policy = LocalLanguageModel.ReloadPolicy()
        if let value = environment["COREAI_TEST_RELOAD_INTERVAL"] {
            policy.readInterval = try #require(Int(value))
            try #require(policy.readInterval >= 0)
        }
        if let value = environment["COREAI_TEST_RELOAD_FOOTPRINT_BYTES"] {
            policy.footprintLimitBytes = try #require(UInt64(value))
        }
        return policy
    }
}

extension ModelIntegrationTests {
    @Test(
        "An interval of two reads waits through five turns and reloads only when the next conversation starts",
        .enabled(if: !bundlePaths.isEmpty), arguments: bundlePaths)
    func intervalWaitsForANewConversation(path: String) async throws {
        let model = try await LocalLanguageModel(
            bundle: URL(filePath: path), contextLimit: 16_384, reloadPolicy: .init(readInterval: 2))
        let session = LanguageModelSession(model: model)
        for _ in 0..<5 {
            _ = try await session.respond(
                to: "Name a fruit.", options: .init(samplingMode: .greedy, maximumResponseTokens: 16),
                contextOptions: ModelTestSettings.context)
            guard case .response(let response) = session.transcript.last else {
                Issue.record("No final response")
                return
            }
            #expect(response.metadata["reloadSeconds"] == nil)
        }
        let next = LanguageModelSession(model: model)
        let response = try await next.respond(
            to: "Name a color.", options: .init(samplingMode: .greedy, maximumResponseTokens: 16),
            contextOptions: ModelTestSettings.context)
        #expect(try ModelTestSettings.metadata(next, "reloadSeconds", as: Double.self) >= 0)
        #expect(response.usage.input.cachedTokenCount == 0)
        #expect(
            try ModelTestSettings.metadata(next, "prefillTokens", as: Int.self) == response.usage.input.totalTokenCount)
        try await ModelTestSettings.checkState(model, label: "new-conversation-reload")
    }

    @Test(
        "A footprint reload preserves the complete prompt, logical sharing and next-token logits",
        .enabled(if: !bundlePaths.isEmpty), arguments: bundlePaths)
    func thresholdPreservesThePrompt(path: String) async throws {
        struct Record {
            let tokens: [Int32]
            let logits: [Float16]
            let shared: Int
        }
        func run(_ threshold: UInt64) async throws -> Record {
            let model = try await LocalLanguageModel(
                bundle: URL(filePath: path), contextLimit: 16_384,
                reloadPolicy: .init(readInterval: 0, footprintLimitBytes: threshold))
            let session = LanguageModelSession(model: model)
            _ = try await session.respond(
                to: "Remember the word river.", options: .init(samplingMode: .greedy, maximumResponseTokens: 8),
                contextOptions: ModelTestSettings.context)
            let response = try await session.respond(
                to: "What word did I ask you to remember?",
                options: .init(samplingMode: .greedy, maximumResponseTokens: 1),
                contextOptions: ModelTestSettings.context)
            if threshold > 0 {
                #expect(try ModelTestSettings.metadata(session, "reloadSeconds", as: Double.self) >= 0)
                #expect(response.usage.input.cachedTokenCount == 0)
                #expect(
                    try ModelTestSettings.metadata(session, "prefillTokens", as: Int.self)
                        == response.usage.input.totalTokenCount)
            }
            let state = await model.processedState()
            #expect(state.tokens.count == response.usage.input.totalTokenCount)
            return Record(
                tokens: state.tokens, logits: state.logits,
                shared: try ModelTestSettings.metadata(session, "sharedTokens", as: Int.self))
        }
        let control = try await run(0)
        let reloaded = try await run(1)
        #expect(reloaded.tokens == control.tokens)
        #expect(reloaded.shared == control.shared)
        try #require(reloaded.logits.count == control.logits.count && reloaded.logits.count > 1)
        #expect(StateCheck(current: reloaded.logits, reread: control.logits).passed)
    }

    @Test(
        "Reloading before a tool-result step preserves its logical shared-token count",
        .enabled(if: !bundlePaths.isEmpty), arguments: bundlePaths)
    func thresholdPreservesToolSharing(path: String) async throws {
        func run(_ threshold: UInt64) async throws -> (tokens: [Int32], shared: Int) {
            let model = try await LocalLanguageModel(
                bundle: URL(filePath: path), contextLimit: 16_384,
                reloadPolicy: .init(readInterval: 0, footprintLimitBytes: threshold))
            let session = LanguageModelSession(model: model, tools: [WeatherTool()])
            let response = try await session.respond(
                to: "What is the weather in Tokyo right now? Use the weather tool.",
                options: .init(samplingMode: .greedy, maximumResponseTokens: 128),
                contextOptions: ModelTestSettings.context)
            #expect(session.transcript.filter { if case .toolOutput = $0 { true } else { false } }.count == 1)
            if threshold > 0 {
                #expect(try ModelTestSettings.metadata(session, "reloadSeconds", as: Double.self) >= 0)
                #expect(response.usage.input.cachedTokenCount == 0)
            }
            let tokens = await model.processedTokens()
            let shared = try ModelTestSettings.metadata(session, "sharedTokens", as: Int.self)
            try await ModelTestSettings.checkState(model, label: "tool-reload-\(threshold)")
            return (tokens, shared)
        }
        let control = try await run(0)
        let reloaded = try await run(1)
        #expect(reloaded.shared == control.shared)
    }

    @Test(
        "Footprint reloads preserve EOS and token-limit state and recover after streamed cancellation",
        .enabled(if: !bundlePaths.isEmpty), arguments: bundlePaths)
    func thresholdCancellationAndState(path: String) async throws {
        let model = try await LocalLanguageModel(
            bundle: URL(filePath: path), contextLimit: 16_384,
            reloadPolicy: .init(readInterval: 0, footprintLimitBytes: 1))
        let finished = LanguageModelSession(model: model)
        let response = try await finished.respond(
            to: "What is the capital of France? Answer with one word.",
            options: .init(samplingMode: .greedy, maximumResponseTokens: 128), contextOptions: ModelTestSettings.context
        )
        #expect(try ModelTestSettings.metadata(finished, "stopReason", as: String.self) == "stopToken")
        #expect(
            await model.processedTokens().count == response.usage.input.totalTokenCount
                + response.usage.output.totalTokenCount)
        try await ModelTestSettings.checkState(model, label: "reload-eos")
        let limited = LanguageModelSession(model: model)
        let short = try await limited.respond(
            to: "Explain the history of the ocean in detail.",
            options: .init(samplingMode: .greedy, maximumResponseTokens: 1), contextOptions: ModelTestSettings.context)
        #expect(try ModelTestSettings.metadata(limited, "stopReason", as: String.self) == "maxTokens")
        #expect(await model.processedTokens().count == short.usage.input.totalTokenCount)
        try await ModelTestSettings.checkState(model, label: "reload-limit")
        let cancelled = LanguageModelSession(model: model)
        let question = "Write a long story about a dragon."
        let prompt = try model.runtime.tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": question]], tools: nil,
            additionalContext: Runtime.templateContext(ModelTestSettings.context.reasoningLevel)
        ).map(Int32.init)
        _ = try await cancelAfterFirstFragment(cancelled, prompt: question, maximumTokens: 512)
        let partial = await model.processedTokens()
        if !partial.isEmpty {
            #expect(partial.starts(with: prompt))
            try await ModelTestSettings.checkState(model, label: "reload-cancel")
        }
        let recovered = try await cancelled.respond(
            to: "Name a fruit.", options: .init(samplingMode: .greedy, maximumResponseTokens: 32),
            contextOptions: ModelTestSettings.context)
        #expect(!recovered.content.isEmpty)
        #expect(try ModelTestSettings.metadata(cancelled, "reloadSeconds", as: Double.self) >= 0)
    }
}
