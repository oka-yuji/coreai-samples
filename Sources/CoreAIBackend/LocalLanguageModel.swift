import Foundation
import FoundationModels
import Tokenizers

public struct LocalLanguageModel: LanguageModel {
    public struct Executor: LanguageModelExecutor {
        public typealias Model = LocalLanguageModel

        public init(configuration: URL) {}

        public nonisolated(nonsending) func respond(
            to request: LanguageModelExecutorGenerationRequest,
            model: LocalLanguageModel,
            streamingInto channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            try await model.runtime.respond(to: request, streamingInto: channel)
        }
    }

    public let capabilities: LanguageModelCapabilities
    public let executorConfiguration: URL
    public let loadStats: Engine.LoadStats
    let runtime: Runtime

    public init(bundle url: URL, contextLimit: Int? = nil) async throws {
        runtime = try await Runtime(bundle: ModelBundle(url: url), contextLimit: contextLimit)
        capabilities = LanguageModelCapabilities(runtime.thinks ? [.reasoning] : [])
        executorConfiguration = url
        loadStats = runtime.loadStats
    }
}

actor Runtime {
    nonisolated let loadStats: Engine.LoadStats
    nonisolated let thinks: Bool
    private let core: EngineBox
    private let tokenizer: any Tokenizer
    private let stopTokens: Set<Int32>
    private var checkpoints: [CheckpointRole: Engine.Checkpoint] = [:]
    private var busy = false
    private var waiting: [CheckedContinuation<Void, Never>] = []

    init(bundle: ModelBundle, contextLimit: Int?) async throws {
        tokenizer = try await bundle.loadTokenizer()
        core = EngineBox(try await Engine(bundle: bundle, contextLimit: contextLimit))
        loadStats = core.engine.loadStats
        thinks = tokenizer.convertTokenToId(ThinkParser.open) != nil
        stopTokens = Set([tokenizer.eosTokenId, tokenizer.convertTokenToId("<|endoftext|>")].compactMap { $0.map(Int32.init) })
    }

    var processed: [Int32] { core.engine.tokens }

    func respond(
        to request: LanguageModelExecutorGenerationRequest,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        if !request.enabledToolDefinitions.isEmpty {
            throw LanguageModelError.unsupportedCapability(.init(capability: .toolCalling, debugDescription: "Tool calling is not supported yet."))
        }
        if request.schema != nil {
            throw LanguageModelError.unsupportedCapability(.init(capability: .guidedGeneration, debugDescription: "Guided generation is not supported."))
        }
        let messages = try Self.messages(request.transcript)
        let context = Self.templateContext(request.contextOptions.reasoningLevel)
        let prompt = try render(messages, generationPrompt: true, context: context)
        let afterUser = zip(try render(messages, generationPrompt: false, context: context), prompt).prefix { $0 == $1 }.count
        guard !prompt.isEmpty, prompt.count < core.engine.contextLimit else {
            throw LanguageModelError.contextSizeExceeded(
                .init(contextSize: core.engine.contextLimit, tokenCount: prompt.count, debugDescription: "The prompt has \(prompt.count) tokens; the context limit is \(core.engine.contextLimit)."))
        }
        let opensReasoning = ThinkParser.opensReasoning(tokenizer.decode(tokens: prompt[afterUser...].map(Int.init)))
        var sampler = Self.sampler(request.generationOptions)
        let limit = min(request.generationOptions.maximumResponseTokens ?? .max, core.engine.contextLimit - prompt.count)

        await lock()
        defer { unlock() }
        let start = core.engine.stats.prefillTokens
        let plan = try await read(prompt, afterUser: afterUser)
        let prefilled = core.engine.stats.prefillTokens - start

        var parser = ThinkParser(reasoning: opensReasoning)
        var pending: [Int] = []
        var produced = 0
        var reasoningTokens = 0
        var stopReason = "maxTokens"
        while produced < limit {
            try Task.checkCancellation()
            let token = core.engine.withLogits { sampler.next($0) }
            if stopTokens.contains(token) {
                stopReason = "stopToken"
                break
            }
            produced += 1
            pending.append(Int(token))
            let text = tokenizer.decode(tokens: pending)
            let complete = !text.hasSuffix("\u{FFFD}")
            if complete { pending.removeAll() }
            let before = parser.reasoning
            let parts = parser.feed(complete ? text : "")
            let counted = before || parser.reasoning
            if counted { reasoningTokens += 1 }
            await send(parts, counted: counted, to: channel)
            if produced < limit { try await compute { try await core.engine.step(token) } }
        }
        await send(parser.feed(tokenizer.decode(tokens: pending)) + parser.finish(), counted: nil, to: channel)
        await channel.send(.response(action: .updateMetadata(["stopReason": stopReason, "prefillTokens": prefilled, "sharedTokens": plan.shared])))
        await channel.send(.response(action: .updateUsage(
            input: .init(totalTokenCount: prompt.count, cachedTokenCount: plan.keep),
            output: .init(totalTokenCount: produced, reasoningTokenCount: reasoningTokens))))
    }

    private func read(_ prompt: [Int32], afterUser: Int) async throws -> ReusePlan {
        let plan = ReusePlan(
            processed: core.engine.tokens, saved: checkpoints.values.map(\.tokens.count), prompt: prompt,
            hybrid: core.engine.isHybrid, afterUser: afterUser)
        checkpoints = checkpoints.filter { $0.value.tokens.count <= plan.keep }
        if let point = checkpoints.values.first(where: { $0.tokens.count == plan.keep }) {
            try core.engine.restore(point)
        } else if plan.keep == 0 {
            core.engine.reset()
        } else if plan.keep < core.engine.tokens.count {
            try core.engine.restore(Engine.Checkpoint(tokens: Array(core.engine.tokens.prefix(plan.keep)), fixedStates: []))
        }
        var cursor = plan.keep
        for stop in plan.stops {
            if stop.position > cursor {
                let segment = Array(prompt[cursor..<stop.position])
                try await compute { try await core.engine.prefill(segment) }
                cursor = stop.position
            }
            checkpoints[stop.role] = core.engine.checkpoint()
        }
        precondition(core.engine.tokens == prompt)
        return plan
    }

    private func compute(_ body: () async throws -> Void) async throws {
        do {
            try await body()
        } catch {
            core.engine.reset()
            checkpoints.removeAll()
            throw error
        }
    }

    private func send(_ parts: [ThinkParser.Part], counted: Bool?, to channel: LanguageModelExecutorGenerationChannel) async {
        var tokenCount = counted == nil ? 0 : 1
        for part in parts {
            switch part {
            case .reasoning(let text): await channel.send(.reasoning(action: .appendText(text, tokenCount: tokenCount)))
            case .response(let text): await channel.send(.response(action: .appendText(text, tokenCount: tokenCount)))
            }
            tokenCount = 0
        }
        guard parts.isEmpty, let counted else { return }
        await channel.send(counted ? .reasoning(action: .appendText("", tokenCount: 1)) : .response(action: .appendText("", tokenCount: 1)))
    }

    private func render(_ messages: [Message], generationPrompt: Bool, context: [String: any Sendable]) throws -> [Int32] {
        try tokenizer.applyChatTemplate(
            messages: messages, chatTemplate: nil, addGenerationPrompt: generationPrompt, truncation: false, maxLength: nil,
            tools: nil, additionalContext: context.isEmpty ? nil : context
        ).map(Int32.init)
    }

    private func lock() async {
        if busy {
            await withCheckedContinuation { waiting.append($0) }
        } else {
            busy = true
        }
    }

    private func unlock() {
        if waiting.isEmpty { busy = false } else { waiting.removeFirst().resume() }
    }

    static func messages(_ transcript: Transcript) throws -> [Message] {
        var messages: [Message] = []
        var reasoning: String?
        for entry in transcript {
            func text(_ segments: [Transcript.Segment]) throws -> String {
                try segments.map { segment in
                    guard case .text(let text) = segment else {
                        throw LanguageModelError.unsupportedTranscriptContent(
                            .init(unsupportedContent: [entry], debugDescription: "Only text segments are supported."))
                    }
                    return text.content
                }.joined(separator: "\n")
            }
            switch entry {
            case .instructions(let instructions):
                let content = try text(instructions.segments)
                if !content.isEmpty { messages.append(["role": "system", "content": content]) }
            case .prompt(let prompt):
                if let pending = reasoning { messages.append(["role": "assistant", "content": "", "reasoning_content": pending]) }
                reasoning = nil
                messages.append(["role": "user", "content": try text(prompt.segments)])
            case .reasoning(let thought):
                reasoning = (reasoning ?? "") + (try text(thought.segments))
            case .response(let response):
                var message: Message = ["role": "assistant", "content": try text(response.segments)]
                if let reasoning { message["reasoning_content"] = reasoning }
                reasoning = nil
                messages.append(message)
            default:
                throw LanguageModelError.unsupportedTranscriptContent(
                    .init(unsupportedContent: [entry], debugDescription: "Tool calls are not supported yet."))
            }
        }
        return messages
    }

    static func templateContext(_ level: ContextOptions.ReasoningLevel?) -> [String: any Sendable] {
        switch level {
        case .light?: ["reasoning_effort": "low"]
        case .moderate?: ["reasoning_effort": "medium"]
        case .deep?: ["reasoning_effort": "xhigh"]
        case .custom("off")?: ["enable_thinking": false]
        case .custom(let effort)?: ["reasoning_effort": effort]
        default: [:]
        }
    }

    static func sampler(_ options: GenerationOptions) -> Sampler {
        let temperature = Float(options.temperature ?? 1)
        switch options.samplingMode?.kind {
        case .randomTopK(let k, let seed)?: return Sampler(mode: .topK(k), temperature: temperature, seed: seed)
        case .randomProbabilityThreshold(let p, let seed)?: return Sampler(mode: .topP(p), temperature: temperature, seed: seed)
        case nil where options.temperature != nil: return Sampler(mode: .full, temperature: temperature)
        default: return Sampler(mode: .greedy)
        }
    }
}

final class EngineBox: @unchecked Sendable {
    let engine: Engine

    init(_ engine: Engine) {
        self.engine = engine
    }
}
