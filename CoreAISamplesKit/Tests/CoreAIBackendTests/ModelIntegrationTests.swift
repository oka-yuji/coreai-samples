import Foundation
import FoundationModels
import Testing

@_spi(Verification) @testable import CoreAIBackend

@Suite(.serialized, .timeLimit(.minutes(30)))
struct ModelIntegrationTests {}

extension ModelIntegrationTests {
    @Test(
        "Non-language-model bundles are refused", .enabled(if: !nonLanguageModelPaths.isEmpty),
        arguments: nonLanguageModelPaths)
    func engineRefusesABundleThatIsNotALanguageModel(path: String) async throws {
        let bundle = try ModelBundle(url: URL(filePath: path))
        let width = try #require(Engine.mainOutputWidth(bundle.modelURL))
        #expect(width != bundle.vocabSize)
        await #expect(throws: BundleError.notLanguageModel(outputWidth: width, vocabSize: bundle.vocabSize)) {
            try await LocalLanguageModel(bundle: URL(filePath: path))
        }
    }

    @Test(
        "The engine restores saved state and rejects context overflow", .enabled(if: !bundlePaths.isEmpty),
        arguments: bundlePaths)
    func modelRunsAndRestores(path: String) async throws {
        let bundle = try ModelBundle(url: URL(filePath: path))
        #expect(Engine.mainOutputWidth(bundle.modelURL) == bundle.vocabSize)
        let tokenizer = try await bundle.loadTokenizer()
        let engine = try await Engine(bundle: bundle)
        let stop = Set([tokenizer.eosTokenId].compactMap { $0.map(Int32.init) })
        let prefix = tokenizer.encode(text: "The following is a short conversation about geography.\n").map(Int32.init)
        let suffix = try tokenizer.applyChatTemplate(messages: [
            ["role": "user", "content": "Name three rivers in Europe."]
        ])
        .map(Int32.init)

        try await engine.prefill(prefix)
        let point = engine.checkpoint()
        #expect((point.stateBytes > 0) == engine.isHybrid)
        let first = try await engine.generate(suffix, maxTokens: 24, stopTokens: stop)
        try engine.restore(point)
        let second = try await engine.generate(suffix, maxTokens: 24, stopTokens: stop)
        #expect(first == second)

        engine.reset()
        try await engine.prefill(prefix + suffix, usePrefillFunction: false)
        var viaMain = [engine.greedyToken()]
        while viaMain.count < 16 {
            try await engine.step(try #require(viaMain.last))
            viaMain.append(engine.greedyToken())
        }
        #expect(viaMain.allSatisfy { (0..<Int32(engine.vocabSize)).contains($0) })

        let small = try await Engine(bundle: bundle, contextLimit: 8)
        await #expect(throws: EngineError.contextOverflow(needed: prefix.count, limit: 8)) {
            try await small.prefill(prefix)
        }
    }

    func texts(_ session: LanguageModelSession) -> String {
        session.transcript.map { entry -> String in
            let (kind, segments): (String, [Transcript.Segment]) =
                switch entry {
                case .prompt(let prompt): ("prompt", prompt.segments)
                case .reasoning(let reasoning): ("reasoning", reasoning.segments)
                case .response(let response): ("response", response.segments)
                default: ("other", [])
                }
            return kind + ": " + segments.map { if case .text(let text) = $0 { text.content } else { "" } }.joined()
        }.joined(separator: "\n")
    }

    @Test(
        "Sessions reuse state, honor sampling and recover after cancellation", .enabled(if: !bundlePaths.isEmpty),
        arguments: bundlePaths)
    func sessionReusesStopsSamplesAndCancels(path: String) async throws {
        let model = try await LocalLanguageModel(bundle: URL(filePath: path))
        func stopReason(_ session: LanguageModelSession) -> String? {
            guard case .response(let response) = session.transcript.last else { return nil }
            return try? response.metadata["stopReason"]?.value(String.self)
        }
        func prefilled(_ session: LanguageModelSession) -> Int? {
            guard case .response(let response) = session.transcript.last else { return nil }
            return try? response.metadata["prefillTokens"]?.value(Int.self)
        }

        let session = LanguageModelSession(model: model, instructions: "You are a concise assistant.")
        for (turn, prompt) in [
            "My name is Taro. Remember it.", "What is the capital of France? Answer with one word.", "What is my name?",
        ].enumerated() {
            let response = try await session.respond(to: prompt, options: GenerationOptions(maximumResponseTokens: 512))
            let usage = response.usage
            #expect(stopReason(session) == "stopToken")
            #expect(usage.input.cachedTokenCount + (prefilled(session) ?? -1) == usage.input.totalTokenCount)
            #expect((usage.input.cachedTokenCount > 0) == (turn > 0))
            #expect(await model.runtime.processed.count == usage.input.totalTokenCount + usage.output.totalTokenCount)
        }

        let direct = try await LanguageModelSession(model: model).respond(
            to: "What is the capital of France? Answer with one word.",
            options: GenerationOptions(maximumResponseTokens: 64),
            contextOptions: ContextOptions(reasoningLevel: .custom("off")))
        #expect(direct.usage.output.reasoningTokenCount == 0)
        #expect(direct.content.contains("Paris"))

        let limited = LanguageModelSession(model: model)
        let short = try await limited.respond(
            to: "Explain why the sky is blue.", options: GenerationOptions(maximumResponseTokens: 8))
        #expect(stopReason(limited) == "maxTokens")
        #expect(short.usage.output.totalTokenCount == 8)
        #expect(await model.runtime.processed.count == short.usage.input.totalTokenCount + 7)

        func sample(_ mode: GenerationOptions.SamplingMode) async throws -> String {
            let options = GenerationOptions(samplingMode: mode, maximumResponseTokens: 24)
            let session = LanguageModelSession(model: model)
            try await session.respond(to: "Write a short poem about the sea.", options: options)
            return texts(session)
        }
        let greedy = try await sample(.greedy)
        #expect(try await sample(.greedy) == greedy)
        let topK = try await sample(.random(top: 40, seed: 7))
        #expect(try await sample(.random(top: 40, seed: 7)) == topK)
        #expect(try await sample(.random(top: 40, seed: 7)) != sample(.random(top: 40, seed: 8)))
        let topP = try await sample(.random(probabilityThreshold: 0.95, seed: 7))
        #expect(try await sample(.random(probabilityThreshold: 0.95, seed: 7)) == topP)
        #expect(
            try await sample(.random(probabilityThreshold: 0.95, seed: 7))
                != sample(.random(probabilityThreshold: 0.95, seed: 8)))

        let cancelled = LanguageModelSession(model: model)
        _ = try await cancelAfterFirstFragment(
            cancelled, prompt: "Write a long story about a dragon.", maximumTokens: 2048)
        let after = try await cancelled.respond(
            to: "What is the capital of Japan? Answer with one word.",
            options: GenerationOptions(maximumResponseTokens: 512))
        #expect(cancelled.transcript.filter { if case .prompt = $0 { true } else { false } }.count == 1)
        #expect(stopReason(cancelled) == "stopToken")
        #expect(!after.content.isEmpty, "\(texts(cancelled))")
        #expect(
            await model.runtime.processed.count == after.usage.input.totalTokenCount
                + after.usage.output.totalTokenCount)

        let tiny = try await LocalLanguageModel(bundle: URL(filePath: path), contextLimit: 16)
        await #expect {
            try await LanguageModelSession(model: tiny).respond(to: "Tell me about the history of computing in detail.")
        } throws: { error in
            guard case LanguageModelError.contextSizeExceeded(let exceeded) = error else { return false }
            return exceeded.contextSize == 16 && exceeded.tokenCount > 16
        }
        #expect(await tiny.runtime.processed.isEmpty)
    }

    struct WeatherTool: Tool {
        let name = "get_weather"
        let description = "Get the current weather for a city."

        @Generable
        struct Arguments {
            @Guide(description: "The city name.")
            var city: String
        }

        func call(arguments: Arguments) async throws -> String {
            "Sunny, 22 C in \(arguments.city)."
        }
    }

    @Test(
        "Tool sessions preserve history and recover after cancellation", .enabled(if: !bundlePaths.isEmpty),
        arguments: bundlePaths)
    func sessionCallsToolsReusesAndRecovers(path: String) async throws {
        let model = try await LocalLanguageModel(bundle: URL(filePath: path))
        func metadata(_ session: LanguageModelSession, _ key: String) -> Int? {
            guard case .response(let response) = session.transcript.last else { return nil }
            return try? response.metadata[key]?.value(Int.self)
        }

        let session = LanguageModelSession(model: model, tools: [WeatherTool()])
        try await session.respond(
            to: "What is the weather in Tokyo right now?", options: GenerationOptions(maximumResponseTokens: 512))
        let calls = session.transcript.flatMap { entry -> [Transcript.ToolCall] in
            if case .toolCalls(let calls) = entry { Array(calls) } else { [] }
        }
        #expect(calls.map(\.toolName) == ["get_weather"])
        #expect(try calls.first?.arguments.value(String.self, forProperty: "city") == "Tokyo")
        #expect(session.transcript.contains { if case .toolOutput = $0 { true } else { false } })
        #expect(texts(session).contains("22"), "\(texts(session))")
        let shared = try #require(metadata(session, "sharedTokens"))
        let prefilled = try #require(metadata(session, "prefillTokens"))
        #expect(shared > 200 && prefilled < 64, "shared \(shared) prefilled \(prefilled)")
        #expect(metadata(session, "splicedGenerations") != nil)

        let tokenizer = model.runtime.tokenizer
        func encode(_ text: String) -> [Int32] { tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init) }
        let parts = ["\u{30EC}\u{30A4}", "\u{30C8}\u{30EC}\u{30FC}", "\u{30B7}\u{30F3}\u{30B0}"]
        let written = parts.flatMap(encode)
        let rerendered = encode(parts.joined())
        #expect(written != rerendered)
        let restored = ReusePlan.splice([0] + rerendered + [1], processed: [0] + written, generations: [(1, written)]) {
            tokenizer.decode(tokens: $0)
        }
        #expect(restored.prompt == [0] + written + [1] && restored.spliced == 1)

        let question = "What is the capital of Japan?"
        let allowed = try await LanguageModelSession(model: model, tools: [WeatherTool()]).respond(
            to: question, options: GenerationOptions(maximumResponseTokens: 16, toolCallingMode: .allowed))
        let disallowed = try await LanguageModelSession(model: model, tools: [WeatherTool()]).respond(
            to: question, options: GenerationOptions(maximumResponseTokens: 16, toolCallingMode: .disallowed))
        let plain = try await LanguageModelSession(model: model).respond(
            to: question, options: GenerationOptions(maximumResponseTokens: 16))
        #expect(disallowed.usage.input.totalTokenCount == plain.usage.input.totalTokenCount)
        #expect(allowed.usage.input.totalTokenCount > disallowed.usage.input.totalTokenCount + 100)
        let processed = await model.runtime.processed
        await #expect {
            try await LanguageModelSession(model: model, tools: [WeatherTool()]).respond(
                to: question, options: GenerationOptions(maximumResponseTokens: 16, toolCallingMode: .required))
        } throws: { error in
            guard case LanguageModelError.unsupportedCapability(let unsupported) = error else { return false }
            return unsupported.capability == .guidedGeneration
        }
        #expect(await model.runtime.processed == processed)

        let cut = LanguageModelSession(model: model, tools: [WeatherTool()])
        try await cut.respond(
            to: "What is the weather in Tokyo right now?", options: GenerationOptions(maximumResponseTokens: 8))
        #expect(!cut.transcript.contains { if case .toolCalls = $0 { true } else { false } })
        guard case .response(let last) = cut.transcript.last else {
            Issue.record("no response entry")
            return
        }
        #expect(try last.metadata["stopReason"]?.value(String.self) == "maxTokens")

        let resumed = LanguageModelSession(model: model, instructions: "You are a concise assistant.")
        try await resumed.respond(
            to: "My name is Taro. Remember it.", options: GenerationOptions(maximumResponseTokens: 256))
        _ = try await cancelAfterFirstFragment(
            resumed, prompt: "Write a long story about a dragon.", maximumTokens: 2048)
        let after = try await resumed.respond(
            to: "What is my name?", options: GenerationOptions(maximumResponseTokens: 256))
        #expect(after.usage.input.cachedTokenCount > 0, "\(after.usage)")
    }
}
