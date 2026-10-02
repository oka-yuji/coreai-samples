import Foundation
import FoundationModels
import Jinja
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
        capabilities = LanguageModelCapabilities(runtime.thinks ? [.reasoning, .toolCalling] : [.toolCalling])
        executorConfiguration = url
        loadStats = runtime.loadStats
    }

    public func processedTokens() async -> [Int32] {
        await runtime.processed
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
        if request.schema != nil {
            throw LanguageModelError.unsupportedCapability(.init(capability: .guidedGeneration, debugDescription: "Guided generation is not supported."))
        }
        let tools = (request.generationOptions.toolCallingMode?.kind == .disallowed ? [] : request.enabledToolDefinitions).map(ToolSchema.init)
        let messages = try Self.messages(request.transcript)
        var context = Self.templateContext(request.contextOptions.reasoningLevel)
        if !tools.isEmpty { context["tools"] = Value.array(tools.map(\.spec)) }
        let prompt = try render(messages, generationPrompt: true, context: context)
        let afterUser = try messages.lastIndex { $0["role"] as? String == "user" }.map {
            zip(try render(Array(messages[...$0]), generationPrompt: false, context: context), prompt).prefix { $0 == $1 }.count
        } ?? 0
        guard !prompt.isEmpty, prompt.count < core.engine.contextLimit else {
            throw LanguageModelError.contextSizeExceeded(
                .init(contextSize: core.engine.contextLimit, tokenCount: prompt.count, debugDescription: "The prompt has \(prompt.count) tokens; the context limit is \(core.engine.contextLimit)."))
        }
        let opensReasoning = ThinkParser.opensReasoning(tokenizer.decode(tokens: prompt[afterUser...].map(Int.init)))
        var sampler = Self.sampler(request.generationOptions)
        let limit = min(request.generationOptions.maximumResponseTokens ?? .max, core.engine.contextLimit - prompt.count)

        await lock()
        defer { unlock() }
        try Task.checkCancellation()
        let clock = ContinuousClock()
        let start = clock.now
        let before = core.engine.stats.prefillTokens
        let plan = try await read(prompt, afterUser: afterUser)
        let prefilled = core.engine.stats.prefillTokens - before
        let readEnd = clock.now

        var output = Output(channel: channel, calls: tools.isEmpty ? nil : ToolCallParser(types: Dictionary(tools.map { ($0.name, $0.types) }) { a, _ in a }))
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
            let wasReasoning = parser.reasoning
            let parts = parser.feed(complete ? text : "")
            let counted = wasReasoning || parser.reasoning
            if counted { reasoningTokens += 1 }
            await output.send(parts, reasoning: counted)
            if produced < limit { try await compute { try await core.engine.step(token) } }
        }
        await output.send(parser.feed(tokenizer.decode(tokens: pending)) + parser.finish(), reasoning: false, last: true)
        let metadata: [String: any ConvertibleToGeneratedContent] = [
            "stopReason": stopReason, "prefillTokens": prefilled, "sharedTokens": plan.shared,
            "prefillSeconds": (readEnd - start) / .seconds(1), "decodeSeconds": (clock.now - readEnd) / .seconds(1),
        ]
        if let call = output.lastCall {
            await channel.send(.toolCalls(action: .toolCall(id: call.id, name: call.name, action: .updateMetadata(metadata))))
        } else {
            await channel.send(.response(action: .updateMetadata(metadata)))
        }
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
            if stop.role == .afterUser { checkpoints[.previousAfterUser] = checkpoints[.afterUser] ?? checkpoints[.previousAfterUser] }
            checkpoints[stop.role] = core.engine.checkpoint()
        }
        guard core.engine.tokens == prompt else {
            core.engine.reset()
            checkpoints.removeAll()
            throw EngineError.checkpointMismatch
        }
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
        var order: [String: [String]] = [:]
        var reply: Message?
        func flush() {
            if let reply { messages.append(reply) }
            reply = nil
        }
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
                flush()
                for tool in instructions.toolDefinitions { order[tool.name] = ToolSchema(tool).order }
                let content = try text(instructions.segments)
                if !content.isEmpty { messages.append(["role": "system", "content": content]) }
            case .prompt(let prompt):
                flush()
                messages.append(["role": "user", "content": try text(prompt.segments)])
            case .reasoning(let thought):
                var message = reply ?? ["role": "assistant", "content": ""]
                message["reasoning_content"] = (message["reasoning_content"] as? String ?? "") + (try text(thought.segments))
                reply = message
            case .response(let response):
                var message = reply ?? ["role": "assistant", "content": ""]
                message["content"] = (message["content"] as? String ?? "") + (try text(response.segments))
                reply = message
            case .toolCalls(let calls):
                var message = reply ?? ["role": "assistant", "content": ""]
                message["tool_calls"] = (message["tool_calls"] as? [Value] ?? []) + calls.map { call in
                    ["type": "function", "function": ["name": .string(call.toolName), "arguments": Self.value(call.arguments, order: order[call.toolName] ?? [])]]
                }
                reply = message
            case .toolOutput(let output):
                flush()
                messages.append(["role": "tool", "content": try text(output.segments)])
            @unknown default:
                throw LanguageModelError.unsupportedTranscriptContent(
                    .init(unsupportedContent: [entry], debugDescription: "Unknown transcript entry."))
            }
        }
        flush()
        return messages
    }

    static func value(_ content: GeneratedContent, order: [String] = []) -> Value {
        switch content.kind {
        case .null: .null
        case .bool(let flag): .boolean(flag)
        case .number(let number): Int(exactly: number).map(Value.int) ?? .double(number)
        case .string(let string): .string(string)
        case .array(let items): .array(items.map { value($0) })
        case .structure(let properties, let keys):
            .object(OrderedDictionary(uniqueKeysWithValues: (order.filter(keys.contains) + keys.filter { !order.contains($0) }).map {
                (ObjectKey.string($0), value(properties[$0] ?? GeneratedContent(kind: .null)))
            }))
        @unknown default: .null
        }
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

struct Output {
    let channel: LanguageModelExecutorGenerationChannel
    var calls: ToolCallParser?
    var lastCall: (id: String, name: String)?
    private var unreported = 0

    init(channel: LanguageModelExecutorGenerationChannel, calls: ToolCallParser?) {
        self.channel = channel
        self.calls = calls
    }

    mutating func send(_ parts: [ThinkParser.Part], reasoning: Bool, last: Bool = false) async {
        if !last { unreported += 1 }
        var events: [ToolCallParser.Event] = []
        for part in parts {
            switch part {
            case .reasoning(let text): await channel.send(.reasoning(action: .appendText(text, tokenCount: take())))
            case .response(let text): events += calls?.feed(text) ?? [.text(text)]
            }
        }
        if last { events += calls?.finish() ?? [] }
        for event in events {
            switch event {
            case .text(let text):
                await channel.send(.response(action: .appendText(text, tokenCount: take())))
                lastCall = nil
            case .call(let name, let arguments):
                let id = UUID().uuidString
                await channel.send(.toolCalls(action: .toolCall(id: id, name: name, action: .appendArguments(arguments, tokenCount: take()))))
                lastCall = (id, name)
            }
        }
        if reasoning, unreported > 0 { await channel.send(.reasoning(action: .appendText("", tokenCount: take()))) }
    }

    private mutating func take() -> Int {
        defer { unreported = 0 }
        return unreported
    }
}

struct ToolSchema {
    static let leadingKeys = ["type", "description", "enum", "items", "properties", "required"]
    static let droppedKeys: Set = ["x-order", "title", "additionalProperties"]

    let name: String
    let spec: Value
    let order: [String]
    let types: [String: String]

    init(_ definition: Transcript.ToolDefinition) {
        let parameters = (try? JSONEncoder().encode(definition.parameters))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } ?? [:]
        let properties = parameters["properties"] as? [String: Any] ?? [:]
        name = definition.name
        order = parameters["x-order"] as? [String] ?? properties.keys.sorted()
        types = properties.compactMapValues { ($0 as? [String: Any])?["type"] as? String }
        spec = [
            "type": "function",
            "function": ["name": .string(definition.name), "description": .string(definition.description), "parameters": Self.value(parameters)],
        ]
    }

    static func value(_ json: Any, order: [String]? = nil) -> Value {
        switch json {
        case let object as [String: Any]:
            let first = order ?? leadingKeys
            let rest = object.keys.filter { !first.contains($0) && (order != nil || !droppedKeys.contains($0)) }.sorted()
            return .object(OrderedDictionary(uniqueKeysWithValues: (first.filter { object[$0] != nil } + rest).map { key in
                let children = order == nil && key == "properties" ? object["x-order"] as? [String] ?? [] : nil
                return (ObjectKey.string(key), value(object[key]!, order: children))
            }))
        case let array as [Any]: return .array(array.map { value($0) })
        case let string as String: return .string(string)
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() { return .boolean(number.boolValue) }
            return Int(exactly: number.doubleValue).map(Value.int) ?? .double(number.doubleValue)
        default: return .null
        }
    }
}

final class EngineBox: @unchecked Sendable {
    let engine: Engine

    init(_ engine: Engine) {
        self.engine = engine
    }
}
