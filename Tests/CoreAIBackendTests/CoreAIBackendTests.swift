import Foundation
import FoundationModels
import Testing

@testable import CoreAIBackend

@Test func readsMetadata() throws {
    let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let json = """
        {"metadata_version": "0.2", "kind": "llm", "name": "tiny",
         "assets": {"main": "tiny.aimodel"},
         "language": {"tokenizer": "x/tiny", "vocab_size": 151936, "max_context_length": 8192,
                      "embedded_tokenizer": true, "function_map": {"main": ["main"]}}}
        """
    try Data(json.utf8).write(to: folder.appending(path: "metadata.json"))

    let bundle = try ModelBundle(url: folder)
    #expect(bundle.name == "tiny")
    #expect(bundle.modelURL == folder.appending(path: "tiny.aimodel"))
    #expect(bundle.tokenizerURL == folder.appending(path: "tokenizer"))
    #expect(bundle.vocabSize == 151936)
    #expect(bundle.maxContextLength == 8192)

    try Data(#"{"name": "tiny", "assets": {"main": "tiny.aimodel"}}"#.utf8).write(to: folder.appending(path: "metadata.json"))
    #expect(throws: DecodingError.self) { try ModelBundle(url: folder) }
}

@Test func growsCapacityByDoublingUpToTheLimit() {
    #expect(Engine.grownCapacity(current: 256, needed: 256, limit: 8192) == 256)
    #expect(Engine.grownCapacity(current: 256, needed: 257, limit: 8192) == 512)
    #expect(Engine.grownCapacity(current: 256, needed: 3000, limit: 8192) == 4096)
    #expect(Engine.grownCapacity(current: 4096, needed: 5000, limit: 6000) == 6000)
    #expect(Engine.grownCapacity(current: 1, needed: 7, limit: 16384) == 8)
}

func contiguous(_ width: Int) -> @Sendable ([Int]) -> (strides: [Int], byteCount: Int) {
    { shape in
        let strides = shape.indices.map { shape[($0 + 1)...].reduce(1, *) }
        return (strides, shape.reduce(1, *) * width)
    }
}

@Test func growingKeepsEveryRow() {
    var state = StateBuffer(
        name: "keyCache", scalarType: .float16, sequenceDim: 3, shape: [2, 1, 3, 4, 5], layout: contiguous(2))
    state.withMutableBytes { bytes in
        for i in 0..<(2 * 3 * 4 * 5) { bytes.storeBytes(of: UInt16(i + 1), toByteOffset: i * 2, as: UInt16.self) }
    }

    state.grow(to: 8)

    #expect(state.shape == [2, 1, 3, 8, 5])
    #expect(state.strides == [120, 120, 40, 5, 1])
    #expect(state.byteCount == 2 * 3 * 8 * 5 * 2)
    state.withBytes { bytes in
        for block in 0..<6 {
            for row in 0..<4 {
                for column in 0..<5 {
                    let value = bytes.load(fromByteOffset: (block * 40 + row * 5 + column) * 2, as: UInt16.self)
                    #expect(value == UInt16(block * 20 + row * 5 + column + 1))
                }
            }
        }
    }
}

@Test func checkpointRestoresFixedStatesAndRejectsAnOverwrittenPrefix() throws {
    let kv = StateBuffer(name: "keyCache", scalarType: .bfloat16, sequenceDim: 1, shape: [2, 4, 3], layout: contiguous(2))
    let fixed = StateBuffer(
        name: "recurrentStates", scalarType: .bfloat16, sequenceDim: nil, shape: [2, 6], layout: contiguous(2))
    var context = Context(buffers: [kv, fixed], capacity: 4, limit: 64)
    func fill(_ value: Int32) {
        context.buffers[1].withMutableBytes { _ = memset($0, value, 24) }
    }
    func fixedBytes() -> [UInt8] {
        context.buffers[1].withBytes { Array(UnsafeRawBufferPointer(start: $0, count: 24)) }
    }
    #expect(fixedBytes() == Array(repeating: 0, count: 24))

    context.tokens = [1, 2, 3]
    fill(7)
    let saved = context.checkpoint()
    #expect(saved.tokens == [1, 2, 3])
    #expect(saved.stateBytes == 24)

    context.tokens += [4, 5]
    fill(9)
    try context.restore(saved)
    #expect(context.tokens == [1, 2, 3])
    #expect(fixedBytes() == Array(repeating: 7, count: 24))

    context.tokens = [1, 2, 8]
    #expect(throws: EngineError.checkpointMismatch) { try context.restore(saved) }

    context.reset()
    #expect(context.tokens.isEmpty)
    #expect(fixedBytes() == Array(repeating: 0, count: 24))
    #expect(throws: EngineError.checkpointMismatch) { try context.restore(saved) }

    var attentionOnly = Context(buffers: [kv], capacity: 4, limit: 64)
    attentionOnly.tokens = [5, 6, 7, 8]
    let point = attentionOnly.checkpoint()
    attentionOnly.tokens = [5, 6, 7, 8, 9, 10]
    try attentionOnly.restore(Engine.Checkpoint(tokens: [5, 6], fixedStates: []))
    #expect(attentionOnly.tokens == [5, 6])
    #expect(point.stateBytes == 0)
    #expect(throws: EngineError.checkpointMismatch) { try attentionOnly.restore(point) }
    #expect(throws: EngineError.contextOverflow(needed: 65, limit: 64)) { try attentionOnly.checkRoom(63) }
}

@Test func greedyPicksTheFirstMaximum() {
    func pick(_ values: [Float16]) -> Int32 {
        var scratch = [Float](repeating: 0, count: values.count)
        return values.withUnsafeBufferPointer { Engine.argmax($0, scratch: &scratch) }
    }
    #expect(pick([1, 3, 2]) == 1)
    #expect(pick([-3, -1, -2]) == 1)
    #expect(pick([0, 5, 5, 1]) == 1)
    #expect(pick([2, 0, 2]) == 0)

    var values = (0..<248_320).map { Float16(Float(($0 * 7919) % 1000) / 100) }
    values[1000] = 12
    values[200_000] = 12
    #expect(pick(values) == 1000)
}

@Test func reusePlanGoesBackToTheRightPoint() {
    typealias Stop = ReusePlan.Stop
    var plan = ReusePlan(processed: [1, 2, 3, 9, 9], saved: [], prompt: [1, 2, 3, 4, 5], hybrid: false, afterUser: 4)
    #expect(plan.keep == 3)
    #expect(plan.stops == [Stop(position: 5, role: .generationStart)])

    plan = ReusePlan(processed: [1, 2, 3, 4, 5, 6], saved: [], prompt: [1, 2, 3], hybrid: false, afterUser: 2)
    #expect(plan.keep == 2)
    #expect(plan.stops == [Stop(position: 3, role: .generationStart)])
    plan = ReusePlan(processed: [1, 2, 3, 4, 5, 6], saved: [3], prompt: [1, 2, 3], hybrid: false, afterUser: 2)
    #expect(plan.keep == 3)
    #expect(plan.stops.isEmpty)

    plan = ReusePlan(processed: [1, 2, 3, 4, 5, 6, 7], saved: [2, 6], prompt: [1, 2, 3, 4, 9, 9, 9, 9], hybrid: true, afterUser: 6)
    #expect(plan.keep == 2)
    #expect(plan.stops == [Stop(position: 6, role: .afterUser), Stop(position: 8, role: .generationStart)])

    let preamble = (0..<300).map(Int32.init)
    plan = ReusePlan(processed: preamble + [7, 7], saved: [302], prompt: preamble + [8, 9, 10], hybrid: true, afterUser: 302)
    #expect(plan.keep == 0)
    #expect(plan.stops == [
        Stop(position: 300, role: .divergence), Stop(position: 302, role: .afterUser), Stop(position: 303, role: .generationStart),
    ])
    plan = ReusePlan(processed: preamble + [8, 9, 10, 11], saved: [300], prompt: preamble + [5, 6], hybrid: true, afterUser: 301)
    #expect(plan.keep == 300)
    #expect(plan.stops == [Stop(position: 301, role: .afterUser), Stop(position: 302, role: .generationStart)])
    plan = ReusePlan(processed: Array(preamble[..<200]) + [7], saved: [], prompt: preamble, hybrid: true, afterUser: 299)
    #expect(plan.keep == 0)
    #expect(plan.stops == [Stop(position: 299, role: .afterUser), Stop(position: 300, role: .generationStart)])

    plan = ReusePlan(processed: [1, 2, 3], saved: [], prompt: [1, 2, 3, 4, 5], hybrid: true, afterUser: 4)
    #expect(plan.keep == 3)
    #expect(plan.stops == [Stop(position: 4, role: .afterUser), Stop(position: 5, role: .generationStart)])

    plan = ReusePlan(processed: [1, 2, 3, 4, 5, 6], saved: [3, 4], prompt: [1, 2, 3, 4], hybrid: true, afterUser: 3)
    #expect(plan.keep == 4)
    #expect(plan.stops.isEmpty)
    plan = ReusePlan(processed: [1, 2, 3, 4, 5, 6], saved: [3], prompt: [1, 2, 3, 4], hybrid: true, afterUser: 3)
    #expect(plan.keep == 3)
    #expect(plan.stops == [Stop(position: 4, role: .generationStart)])
}

func merged(_ parts: [ThinkParser.Part]) -> [ThinkParser.Part] {
    parts.reduce(into: []) { result, part in
        switch (result.last, part) {
        case (.reasoning(let a)?, .reasoning(let b)): result[result.count - 1] = .reasoning(a + b)
        case (.response(let a)?, .response(let b)): result[result.count - 1] = .response(a + b)
        default: result.append(part)
        }
    }
}

@Test func thinkParserSeparatesReasoningFromTheAnswer() {
    var parser = ThinkParser(reasoning: false)
    var parts: [ThinkParser.Part] = []
    for chunk in ["<th", "ink>", "\n", "Let me", " think.", "\n</", "think>", "\n\n", "Paris", "."] {
        parts += parser.feed(chunk)
    }
    #expect(merged(parts + parser.finish()) == [.reasoning("Let me think.\n"), .response("Paris.")])

    parser = ThinkParser(reasoning: true)
    #expect(merged(parser.feed("Hmm") + parser.feed("</think>\n\nOK") + parser.finish()) == [.reasoning("Hmm"), .response("OK")])

    parser = ThinkParser(reasoning: false)
    #expect(parser.feed("Paris") == [.response("Paris")])
    #expect(parser.feed(" is <thi") == [.response(" is ")])
    #expect(parser.finish() == [.response("<thi")])

    #expect(ThinkParser.opensReasoning("<|im_start|>assistant\n<think>\n"))
    #expect(!ThinkParser.opensReasoning("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    #expect(!ThinkParser.opensReasoning("<|im_start|>assistant\n"))
}

@Test func transcriptKeepsReasoningForTheTemplate() throws {
    func text(_ content: String) -> [Transcript.Segment] { [.text(.init(content: content))] }
    let transcript = Transcript(entries: [
        .instructions(.init(segments: text("Be brief."), toolDefinitions: [])),
        .prompt(.init(segments: text("Hi"))),
        .reasoning(.init(segments: text("The user greets."))),
        .response(.init(segments: text("Hello."))),
        .prompt(.init(segments: text("Bye"))),
    ])
    let messages = try Runtime.messages(transcript)
    #expect(messages.map { $0["role"] as? String } == ["system", "user", "assistant", "user"])
    #expect(messages.map { $0["content"] as? String } == ["Be brief.", "Hi", "Hello.", "Bye"])
    #expect(messages[2]["reasoning_content"] as? String == "The user greets.")
    #expect(messages[1]["reasoning_content"] == nil)

    #expect(Runtime.templateContext(nil).isEmpty)
    #expect(Runtime.templateContext(.light)["reasoning_effort"] as? String == "low")
    #expect(Runtime.templateContext(.moderate)["reasoning_effort"] as? String == "medium")
    #expect(Runtime.templateContext(.deep)["reasoning_effort"] as? String == "xhigh")
    #expect(Runtime.templateContext(.custom("off"))["enable_thinking"] as? Bool == false)
}

@Test func samplerHonorsModeTemperatureAndSeed() {
    let logits: [Float16] = [1, 4, 3, 0, 2]
    func draws(_ mode: Sampler.Mode, temperature: Float = 1, seed: UInt64?, count: Int = 400) -> [Int32] {
        var sampler = Sampler(mode: mode, temperature: temperature, seed: seed)
        return (0..<count).map { _ in logits.withUnsafeBufferPointer { sampler.next($0) } }
    }
    #expect(Set(draws(.greedy, seed: nil)) == [1])
    #expect(Set(draws(.topK(5), temperature: 0, seed: 1)) == [1])
    #expect(Set(draws(.topK(1), seed: 1)) == [1])
    #expect(Set(draws(.topK(2), seed: 1)) == [1, 2])
    #expect(Set(draws(.topP(0.5), seed: 1)) == [1])
    #expect(Set(draws(.topP(0.8), seed: 1)) == [1, 2])
    #expect(Set(draws(.full, seed: 1, count: 4000)) == [0, 1, 2, 3, 4])
    #expect(draws(.topK(3), seed: 7) == draws(.topK(3), seed: 7))
    #expect(draws(.topK(3), seed: 7) != draws(.topK(3), seed: 8))
    #expect(draws(.topP(0.95), seed: 7) == draws(.topP(0.95), seed: 7))
    #expect(draws(.topP(0.95), seed: 7) != draws(.topP(0.95), seed: 8))
}

let bundlePaths = (ProcessInfo.processInfo.environment["COREAI_TEST_BUNDLES"] ?? "")
    .split(separator: ":").map(String.init)

@Test(.enabled(if: !bundlePaths.isEmpty), arguments: bundlePaths)
func modelRunsAndRestores(path: String) async throws {
    let bundle = try ModelBundle(url: URL(filePath: path))
    let tokenizer = try await bundle.loadTokenizer()
    let engine = try await Engine(bundle: bundle)
    let stop = Set([tokenizer.eosTokenId].compactMap { $0.map(Int32.init) })
    let prefix = tokenizer.encode(text: "The following is a short conversation about geography.\n").map(Int32.init)
    let suffix = try tokenizer.applyChatTemplate(messages: [["role": "user", "content": "Name three rivers in Europe."]])
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
        try await engine.step(viaMain.last!)
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

@Test(.enabled(if: !bundlePaths.isEmpty), arguments: bundlePaths)
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
    for (turn, prompt) in ["My name is Taro. Remember it.", "What is the capital of France? Answer with one word.", "What is my name?"].enumerated() {
        let response = try await session.respond(to: prompt, options: GenerationOptions(maximumResponseTokens: 512))
        let usage = response.usage
        #expect(stopReason(session) == "stopToken")
        #expect(usage.input.cachedTokenCount + (prefilled(session) ?? -1) == usage.input.totalTokenCount)
        #expect((usage.input.cachedTokenCount > 0) == (turn > 0))
        #expect(await model.runtime.processed.count == usage.input.totalTokenCount + usage.output.totalTokenCount)
    }

    let direct = try await LanguageModelSession(model: model).respond(
        to: "What is the capital of France? Answer with one word.", options: GenerationOptions(maximumResponseTokens: 64),
        contextOptions: ContextOptions(reasoningLevel: .custom("off")))
    #expect(direct.usage.output.reasoningTokenCount == 0)
    #expect(direct.content.contains("Paris"))

    let limited = LanguageModelSession(model: model)
    let short = try await limited.respond(to: "Explain why the sky is blue.", options: GenerationOptions(maximumResponseTokens: 8))
    #expect(stopReason(limited) == "maxTokens")
    #expect(short.usage.output.totalTokenCount == 8)
    #expect(await model.runtime.processed.count == short.usage.input.totalTokenCount + 7)

    func sample(_ mode: GenerationOptions.SamplingMode) async throws -> String {
        let options = GenerationOptions(samplingMode: mode, maximumResponseTokens: 24)
        let session = LanguageModelSession(model: model)
        try await session.respond(to: "Write a short poem about the sea.", options: options)
        return texts(session)
    }
    #expect(try await sample(.greedy) == sample(.greedy))
    #expect(try await sample(.random(top: 40, seed: 7)) == sample(.random(top: 40, seed: 7)))
    #expect(try await sample(.random(top: 40, seed: 7)) != sample(.random(top: 40, seed: 8)))
    #expect(try await sample(.random(probabilityThreshold: 0.95, seed: 7)) == sample(.random(probabilityThreshold: 0.95, seed: 7)))
    #expect(try await sample(.random(probabilityThreshold: 0.95, seed: 7)) != sample(.random(probabilityThreshold: 0.95, seed: 8)))

    let cancelled = LanguageModelSession(model: model)
    let task = Task { try await cancelled.respond(to: "Write a long story about a dragon.", options: GenerationOptions(maximumResponseTokens: 2048)) }
    try await Task.sleep(for: .milliseconds(300))
    task.cancel()
    _ = try? await task.value
    let after = try await cancelled.respond(to: "What is the capital of Japan? Answer with one word.", options: GenerationOptions(maximumResponseTokens: 512))
    #expect(cancelled.transcript.filter { if case .prompt = $0 { true } else { false } }.count == 1)
    #expect(stopReason(cancelled) == "stopToken")
    #expect(!after.content.isEmpty, "\(texts(cancelled))")
    #expect(await model.runtime.processed.count == after.usage.input.totalTokenCount + after.usage.output.totalTokenCount)

    let tiny = try await LocalLanguageModel(bundle: URL(filePath: path), contextLimit: 16)
    await #expect {
        try await LanguageModelSession(model: tiny).respond(to: "Tell me about the history of computing in detail.")
    } throws: { error in
        guard case LanguageModelError.contextSizeExceeded(let exceeded) = error else { return false }
        return exceeded.contextSize == 16 && exceeded.tokenCount > 16
    }
    #expect(await tiny.runtime.processed.isEmpty)
}
