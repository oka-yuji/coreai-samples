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

    public enum OverflowPolicy: Sendable {
        case error
        case dropOldToolOutputs
    }

    @_spi(Verification) public struct VerificationSettings: Sendable {
        public var reusesState = true
        public var keepsHistoryReasoning = true
        public var templateVariables: [String: Bool] = [:]
        public var measurement = GenerationMeasurement.Level.off

        public init() {}
    }

    public let capabilities: LanguageModelCapabilities
    public let executorConfiguration: URL
    public let loadStats: Engine.LoadStats
    let runtime: Runtime

    public init(
        bundle url: URL, contextLimit: Int? = nil, overflow: OverflowPolicy = .error,
        reloadPolicy: ReloadPolicy = .init()
    ) async throws {
        try await self.init(
            bundle: url, contextLimit: contextLimit, overflow: overflow, reloadPolicy: reloadPolicy,
            verification: VerificationSettings())
    }

    @_spi(Verification) public init(
        bundle url: URL, contextLimit: Int?, overflow: OverflowPolicy, reloadPolicy: ReloadPolicy = .init(),
        verification: VerificationSettings
    ) async throws {
        let runtime = try await Runtime(
            bundle: ModelBundle(url: url), contextLimit: contextLimit, overflow: overflow, settings: verification,
            reloadPolicy: reloadPolicy)
        self.init(runtime: runtime, bundle: url)
    }

    init(runtime: Runtime, bundle url: URL) {
        self.runtime = runtime
        capabilities = LanguageModelCapabilities(runtime.thinks ? [.reasoning, .toolCalling] : [.toolCalling])
        executorConfiguration = url
        loadStats = runtime.loadStats
    }

    public func processedTokens() async -> [Int32] {
        await runtime.processed
    }

    public func preparationUpdates() -> AsyncStream<PreparationProgress> {
        runtime.preparationProgress.stream()
    }

    public func countTokens(_ text: String) -> Int {
        runtime.tokenizer.encode(text: text, addSpecialTokens: false).count
    }

    @_spi(Verification) public func processedState() async -> (tokens: [Int32], logits: [Float16]) {
        await runtime.state()
    }

    @_spi(Verification) public func lastGeneratedTokens() async -> [Int32] {
        await runtime.generatedTokens()
    }

    @_spi(Verification) public func lastRequestMeasurement() async -> GenerationMeasurement? {
        await runtime.measurement()
    }

    @_spi(Verification) public func resetAndRead(_ tokens: [Int32]) async throws -> [Float16] {
        try await runtime.resetAndRead(tokens)
    }
}

actor Runtime {
    static let rememberedCalls = 512
    static let rememberedGenerations = 32

    nonisolated let loadStats: Engine.LoadStats
    nonisolated let preparationProgress = PreparationProgressHub()
    nonisolated let thinks: Bool
    private var core: EngineBox?
    private let bundle: ModelBundle
    private let contextLimit: Int
    private let reloadPolicy: LocalLanguageModel.ReloadPolicy
    private let makeEngine: @Sendable (ModelBundle, Int) async throws -> EngineBox
    private let footprint: @Sendable () throws -> UInt64
    private var reloadState = ReloadState()
    private var unloadedTokens: [Int32]?
    private var beganReading = false
    private var lastGeneration: [Int32] = []
    private var lastMeasurement: GenerationMeasurement?
    nonisolated let tokenizer: any Tokenizer
    private let stopTokens: Set<Int32>
    private let sanitizer: ControlTokenSanitizer
    private let overflow: LocalLanguageModel.OverflowPolicy
    private let settings: LocalLanguageModel.VerificationSettings
    private var cuts: [String: ToolOutputCut] = [:]
    private var written: OrderedDictionary<String, [String]> = [:]
    private var generations: [(start: Int, tokens: [Int32])] = []
    private var checkpoints: [CheckpointRole: Engine.Checkpoint] = [:]
    private var busy = false
    private var waiting: [CheckedContinuation<Void, Never>] = []

    init(
        bundle: ModelBundle, contextLimit: Int?, overflow: LocalLanguageModel.OverflowPolicy,
        settings: LocalLanguageModel.VerificationSettings,
        reloadPolicy: LocalLanguageModel.ReloadPolicy = .init(),
        makeEngine: @escaping @Sendable (ModelBundle, Int) async throws -> EngineBox = {
            EngineBox(try await Engine(bundle: $0, contextLimit: $1))
        },
        footprint: @escaping @Sendable () throws -> UInt64 = ProcessFootprint.bytes
    ) async throws {
        guard reloadPolicy.readInterval >= 0 else { throw ReloadError.negativeInterval(reloadPolicy.readInterval) }
        self.bundle = bundle
        self.contextLimit = min(contextLimit ?? bundle.maxContextLength, bundle.maxContextLength)
        self.reloadPolicy = reloadPolicy
        self.makeEngine = makeEngine
        self.footprint = footprint
        sanitizer = try ControlTokenSanitizer(tokens: bundle.addedTokenStrings())
        tokenizer = try await bundle.loadTokenizer()
        let loaded = try await makeEngine(bundle, self.contextLimit)
        core = loaded
        self.overflow = overflow
        self.settings = settings
        loadStats = loaded.engine.loadStats
        thinks = tokenizer.convertTokenToId(ThinkParser.open) != nil
        stopTokens = Set(
            [tokenizer.eosTokenId, tokenizer.convertTokenToId("<|endoftext|>")].compactMap { $0.map(Int32.init) })
    }

    var processed: [Int32] {
        get async {
            await lock()
            defer { unlock() }
            return core?.engine.tokens ?? []
        }
    }

    func state() async -> (tokens: [Int32], logits: [Float16]) {
        await lock()
        defer { unlock() }
        return (core?.engine.tokens ?? [], core?.engine.hasLogits == true ? core?.engine.logits() ?? [] : [])
    }

    func generatedTokens() async -> [Int32] {
        await lock()
        defer { unlock() }
        return lastGeneration
    }

    func measurement() async -> GenerationMeasurement? {
        await lock()
        defer { unlock() }
        return lastMeasurement
    }

    func resetAndRead(_ tokens: [Int32]) async throws -> [Float16] {
        await lock()
        defer { unlock() }
        return try await reread(tokens, core: currentCore())
    }

    private func reread(_ tokens: [Int32], core: EngineBox) async throws -> [Float16] {
        core.engine.reset()
        checkpoints.removeAll()
        guard !tokens.isEmpty else {
            unloadedTokens = nil
            return []
        }
        try await compute(core: core) { try await core.engine.prefill(tokens, usePrefillFunction: true) }
        unloadedTokens = nil
        return core.engine.logits()
    }

    private func currentCore() throws -> EngineBox {
        guard let core else { throw EngineError.unloaded }
        return core
    }

    private func reloadIfNeeded(isNewConversation: Bool) async throws -> Double? {
        try Task.checkCancellation()
        let bytes = reloadPolicy.footprintLimitBytes > 0 ? try footprint() : 0
        guard
            core == nil
                || reloadState.shouldReload(
                    policy: reloadPolicy, isNewConversation: isNewConversation, footprint: bytes)
        else {
            return nil
        }
        let start = ContinuousClock.now
        if unloadedTokens == nil { unloadedTokens = core?.engine.tokens }
        checkpoints.removeAll()
        core?.engine.unload()
        core = nil
        core = try await withTaskCancellationShield { try await makeEngine(bundle, contextLimit) }
        reloadState.didLoad()
        let elapsed = (ContinuousClock.now - start) / .seconds(1)
        try Task.checkCancellation()
        return elapsed
    }

    func respond(
        to request: LanguageModelExecutorGenerationRequest,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        let tools = try Self.validatedTools(for: request)
        await lock()
        beganReading = false
        lastGeneration = []
        lastMeasurement = nil
        defer {
            if beganReading { reloadState.completedRead() }
            unlock()
        }
        let isNewConversation = reloadState.observe(Self.reloadConversation(request.transcript))
        try Task.checkCancellation()
        var context = Self.templateContext(request.contextOptions.reasoningLevel, variables: settings.templateVariables)
        if !tools.isEmpty { context["tools"] = Value.array(tools.map(\.spec)) }
        let reserve =
            overflow == .dropOldToolOutputs
            ? min(request.generationOptions.maximumResponseTokens ?? 1024, contextLimit / 2) : nil

        let (messages, rendered, cuts) = try Self.fit(
            request.transcript, remembered: self.cuts, written: written, keepsReasoning: settings.keepsHistoryReasoning,
            limit: contextLimit,
            reserve: reserve, sanitizer: sanitizer
        ) {
            try render($0, generationPrompt: true, context: context)
        }
        let truncated = Self.remember(cuts, in: &self.cuts)
        let lastUser = messages.lastIndex { $0["role"] as? String == "user" }
        let afterUser =
            try lastUser.map {
                zip(try render(Array(messages[...$0]), generationPrompt: false, context: context), rendered).prefix {
                    $0 == $1
                }.count
            } ?? 0
        let previous = unloadedTokens ?? core?.engine.tokens ?? []
        let hasUnloadedHistory = unloadedTokens != nil
        var splicing = generations
        func splice(_ tokens: ArraySlice<Int32>) -> (prompt: [Int32], spliced: Int) {
            ReusePlan.splice(Array(tokens), processed: previous, generations: splicing) {
                tokenizer.decode(tokens: $0)
            }
        }
        var (prompt, spliced) = splice(rendered[...])
        if prompt.count >= contextLimit { (splicing, prompt, spliced) = ([], rendered, 0) }
        let boundary = splice(rendered.prefix(afterUser)).prompt.count
        let reloadSeconds = try await reloadIfNeeded(isNewConversation: isNewConversation)
        let sharedOverride =
            reloadSeconds != nil || hasUnloadedHistory
            ? zip(settings.reusesState ? previous : [], prompt).prefix { $0 == $1 }.count : nil
        try await generate(
            request: request, channel: channel, core: currentCore(), tools: tools, messages: messages, context: context,
            rendered: rendered, prompt: prompt, afterUser: boundary, previous: previous, splicing: splicing,
            spliced: spliced, truncated: truncated, reloadSeconds: reloadSeconds, sharedOverride: sharedOverride)
    }

    private func generate(
        request: LanguageModelExecutorGenerationRequest, channel: LanguageModelExecutorGenerationChannel,
        core: EngineBox, tools: [ToolSchema], messages: [Message], context: [String: any Sendable],
        rendered: [Int32], prompt: [Int32], afterUser: Int, previous: [Int32],
        splicing: [(start: Int, tokens: [Int32])], spliced: Int, truncated: Int, reloadSeconds: Double?,
        sharedOverride: Int?
    ) async throws {
        var sampler = Self.sampler(request.generationOptions)
        let lastUser = messages.lastIndex { $0["role"] as? String == "user" }
        let opensReasoning = ThinkParser.opensReasoning(after: prompt) { tokenizer.decode(tokens: $0) }
        let limit = min(request.generationOptions.maximumResponseTokens ?? .max, contextLimit - prompt.count)
        let clock = ContinuousClock()
        let start = clock.now
        var probe = GenerationProbe(
            settings.measurement, requestID: request.id, promptTokens: prompt.count, reloadSeconds: reloadSeconds,
            start: start)
        defer { lastMeasurement = probe?.record }
        core.engine.measuresPreparation = probe != nil
        let before = core.engine.stats.prefillTokens
        let hub = preparationProgress
        let transcriptID = request.transcript.first?.id ?? request.id.uuidString
        let requestID = request.id
        var generated: [Int32] = []
        do {
            let plan = try await read(
                prompt, core: core, afterUser: afterUser,
                progress: { processed, total in
                    hub.publish(
                        PreparationProgress(
                            requestID: requestID, transcriptID: transcriptID,
                            processedTokens: processed, totalTokens: total))
                },
                questionStart: {
                    guard let lastUser else { return 0 }
                    var replaced = Array(messages[...lastUser])
                    replaced[lastUser]["content"] =
                        (messages[lastUser]["content"] as? String ?? "").hasPrefix("A") ? "B" : "A"
                    let question = zip(try render(replaced, generationPrompt: false, context: context), rendered)
                        .prefix { $0 == $1 }.count
                    return ReusePlan.splice(
                        Array(rendered.prefix(question)), processed: previous, generations: splicing
                    ) {
                        tokenizer.decode(tokens: $0)
                    }.prompt.count
                })
            unloadedTokens = nil
            let prefilled = core.engine.stats.prefillTokens - before
            let readEnd = clock.now
            probe?.read(until: readEnd, prefilledTokens: prefilled, prepareSeconds: core.engine.stats.prepareSeconds)

            var output = Output(
                channel: channel,
                calls: tools.isEmpty
                    ? nil : ToolCallParser(types: Dictionary(tools.map { ($0.name, $0.types) }) { a, _ in a }))
            defer {
                lastGeneration = generated
                for call in output.written { written[call.id] = call.keys }
                written.removeFirst(max(0, written.count - Self.rememberedCalls))
                if !generated.isEmpty { generations.append((prompt.count, generated)) }
                generations.removeFirst(max(0, generations.count - Self.rememberedGenerations))
            }
            var parser = ThinkParser(reasoning: opensReasoning)
            var pending: [Int] = []
            var reasoningTokens = 0
            var stopReason = "maxTokens"
            while generated.count < limit {
                try Task.checkCancellation()
                let token = core.engine.withLogits { sampler.next($0) }
                probe?.sampled()
                if stopTokens.contains(token) {
                    stopReason = "stopToken"
                    break
                }
                generated.append(token)
                pending.append(Int(token))
                let text = tokenizer.decode(tokens: pending)
                let complete = !text.hasSuffix("\u{FFFD}")
                if complete { pending.removeAll() }
                let wasReasoning = parser.reasoning
                let parts = parser.feed(complete ? text : "")
                probe?.decoded()
                let counted = wasReasoning || parser.reasoning
                if counted { reasoningTokens += 1 }
                let sent = await output.send(parts, reasoning: counted)
                probe?.sent(sent)
                if generated.count < limit {
                    try await compute(core: core) { try await core.engine.step(token) }
                    probe?.stepped()
                }
            }
            let drained = await output.send(
                parser.feed(tokenizer.decode(tokens: pending)) + parser.finish(), reasoning: false, last: true)
            let end = clock.now
            probe?.finish(
                .init(rawValue: stopReason), drained: drained, until: end, sampledTokens: generated.count,
                prepareSeconds: core.engine.stats.prepareSeconds)
            let shared = sharedOverride ?? plan.shared
            var metadata: [String: any ConvertibleToGeneratedContent] = [
                "stopReason": stopReason, "prefillTokens": prefilled, "sharedTokens": shared,
                "truncatedToolOutputs": truncated,
                "reservedTokens": contextLimit - prompt.count, "splicedGenerations": spliced,
                "prefillSeconds": (readEnd - start) / .seconds(1), "decodeSeconds": (end - readEnd) / .seconds(1),
            ]
            if let reloadSeconds { metadata["reloadSeconds"] = reloadSeconds }
            if let call = output.lastCall {
                await channel.send(
                    .toolCalls(action: .toolCall(id: call.id, name: call.name, action: .updateMetadata(metadata))))
            } else {
                await channel.send(.response(action: .updateMetadata(metadata)))
            }
            await channel.send(
                .response(
                    action: .updateUsage(
                        input: .init(totalTokenCount: prompt.count, cachedTokenCount: plan.keep),
                        output: .init(totalTokenCount: generated.count, reasoningTokenCount: reasoningTokens))))
        } catch {
            probe?.fail(
                error, sampledTokens: generated.count, prefilledTokens: core.engine.stats.prefillTokens - before,
                prepareSeconds: core.engine.stats.prepareSeconds)
            throw error
        }
    }

    private func read(
        _ prompt: [Int32], core: EngineBox, afterUser: Int,
        progress: @Sendable (Int, Int) -> Void, questionStart: () throws -> Int
    ) async throws
        -> ReusePlan
    {
        let plan = try Self.plan(
            prompt, processed: core.engine.tokens, saved: checkpoints.values.map(\.tokens.count),
            hybrid: core.engine.isHybrid,
            reusing: settings.reusesState, afterUser: afterUser, questionStart: questionStart)
        checkpoints = checkpoints.filter { $0.value.tokens.count <= plan.keep }
        if let point = checkpoints.values.first(where: { $0.tokens.count == plan.keep }) {
            try core.engine.restore(point)
        } else if plan.keep == 0 {
            core.engine.reset()
        } else if plan.keep < core.engine.tokens.count {
            try core.engine.restore(
                Engine.Checkpoint(tokens: Array(core.engine.tokens.prefix(plan.keep)), fixedStates: []))
        }
        var cursor = plan.keep
        let total = prompt.count
        progress(cursor, total)
        for stop in plan.stops {
            if stop.position > cursor {
                let segment = Array(prompt[cursor..<stop.position])
                let start = cursor
                beganReading = true
                try await compute(core: core) {
                    try await core.engine.prefill(segment, usePrefillFunction: true) {
                        progress(start + $0, total)
                    }
                }
                cursor = stop.position
            }
            if stop.role == .afterUser {
                checkpoints[.previousAfterUser] = checkpoints[.afterUser] ?? checkpoints[.previousAfterUser]
            }
            checkpoints[stop.role] = core.engine.checkpoint()
        }
        guard core.engine.tokens == prompt else {
            core.engine.reset()
            checkpoints.removeAll()
            throw EngineError.checkpointMismatch
        }
        generations.removeAll { !prompt.dropFirst($0.start).starts(with: $0.tokens) }
        return plan
    }

    private func compute(core: EngineBox, _ body: () async throws -> Void) async throws {
        do {
            try await body()
        } catch {
            core.engine.reset()
            checkpoints.removeAll()
            throw error
        }
    }

    private func render(_ messages: [Message], generationPrompt: Bool, context: [String: any Sendable]) throws
        -> [Int32]
    {
        try tokenizer.applyChatTemplate(
            messages: messages, chatTemplate: nil, addGenerationPrompt: generationPrompt, truncation: false,
            maxLength: nil,
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

    static func tools(_ request: LanguageModelExecutorGenerationRequest) throws -> [ToolSchema] {
        switch request.generationOptions.toolCallingMode?.kind {
        case .required?:
            throw LanguageModelError.unsupportedCapability(
                .init(
                    capability: .guidedGeneration,
                    debugDescription:
                        "This runtime cannot force a tool call (toolCallingMode .required needs constrained decoding). Use .allowed."
                ))
        case .disallowed?: return []
        default: return request.enabledToolDefinitions.map(ToolSchema.init)
        }
    }

    static func plan(
        _ prompt: [Int32], processed: [Int32], saved: [Int], hybrid: Bool, reusing: Bool, afterUser: Int,
        questionStart: () throws -> Int
    ) rethrows -> ReusePlan {
        try ReusePlan(
            processed: reusing ? processed : [], saved: saved, prompt: prompt, hybrid: hybrid, afterUser: afterUser,
            questionStart: questionStart)
    }

    static func remember(_ cuts: [String: ToolOutputCut], in memory: inout [String: ToolOutputCut]) -> Int {
        let changed = cuts.filter { memory[$0.key] != $0.value }.count
        memory.merge(cuts) { $1 }
        if memory.count > rememberedCalls { memory = cuts }
        return changed
    }

    static func fit(
        _ transcript: Transcript, remembered: [String: ToolOutputCut],
        written: OrderedDictionary<String, [String]> = [:],
        keepsReasoning: Bool = true, limit: Int, reserve: Int?, sanitizer: ControlTokenSanitizer = .init(tokens: []),
        render: ([Message]) throws -> [Int32]
    ) throws -> (messages: [Message], prompt: [Int32], cuts: [String: ToolOutputCut]) {
        let ids = Set(transcript.map(\.id))
        var cuts = remembered.filter { ids.contains($0.key) }
        let groups = toolOutputGroups(transcript)
        while true {
            let messages = try Self.messages(
                transcript, cuts: cuts, written: written, keepsReasoning: keepsReasoning, sanitizer: sanitizer)
            let prompt = try render(messages)
            if let reserve, prompt.count + reserve > limit, let next = ToolOutputCut.shrink(groups, cuts) {
                cuts = next
                continue
            }
            guard !prompt.isEmpty, prompt.count < limit else {
                throw LanguageModelError.contextSizeExceeded(
                    .init(
                        contextSize: limit, tokenCount: prompt.count,
                        debugDescription: "The prompt has \(prompt.count) tokens; the context limit is \(limit)."))
            }
            return (messages, prompt, cuts)
        }
    }

    static func toolOutputGroups(_ transcript: Transcript) -> [[(id: String, count: Int)]] {
        var groups: [[(id: String, count: Int)]] = []
        var previous = false
        for entry in transcript {
            guard case .toolOutput(let output) = entry else {
                previous = false
                continue
            }
            let count = output.segments.map { if case .text(let text) = $0 { text.content } else { "" } }.joined(
                separator: "\n"
            ).count
            if previous {
                groups[groups.count - 1].append((entry.id, count))
            } else {
                groups.append([(entry.id, count)])
            }
            previous = true
        }
        return groups
    }

    static func messages(
        _ transcript: Transcript, cuts: [String: ToolOutputCut] = [:],
        written: OrderedDictionary<String, [String]> = [:],
        keepsReasoning: Bool = true, sanitizer: ControlTokenSanitizer = .init(tokens: [])
    ) throws -> [Message] {
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
                messages.append(["role": "user", "content": sanitizer.sanitized(try text(prompt.segments))])
            case .reasoning(let thought):
                guard keepsReasoning else { continue }
                var message = reply ?? ["role": "assistant", "content": ""]
                message["reasoning_content"] =
                    (message["reasoning_content"] as? String ?? "") + (try text(thought.segments))
                reply = message
            case .response(let response):
                var message = reply ?? ["role": "assistant", "content": ""]
                message["content"] = (message["content"] as? String ?? "") + (try text(response.segments))
                reply = message
            case .toolCalls(let calls):
                var message = reply ?? ["role": "assistant", "content": ""]
                message["tool_calls"] =
                    (message["tool_calls"] as? [Value] ?? [])
                    + calls.map { call in
                        let arguments = Self.value(
                            call.arguments, order: (written[call.id] ?? []) + (order[call.toolName] ?? []))
                        return [
                            "type": "function", "function": ["name": .string(call.toolName), "arguments": arguments],
                        ]
                    }
                reply = message
            case .toolOutput(let output):
                flush()
                let content = try text(output.segments)
                messages.append([
                    "role": "tool", "content": sanitizer.sanitized(cuts[entry.id]?.apply(content) ?? content),
                ])
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
            .object(
                OrderedDictionary(
                    (order.filter(keys.contains) + keys).map {
                        (ObjectKey.string($0), value(properties[$0] ?? GeneratedContent(kind: .null)))
                    }, uniquingKeysWith: { first, _ in first }))
        @unknown default: .null
        }
    }

    static func templateContext(_ level: ContextOptions.ReasoningLevel?, variables: [String: Bool] = [:]) -> [String:
        any Sendable]
    {
        let context: [String: any Sendable] =
            switch level {
            case .light?: ["reasoning_effort": "low"]
            case .moderate?: ["reasoning_effort": "medium"]
            case .deep?: ["reasoning_effort": "xhigh"]
            case .custom("off")?: ["enable_thinking": false]
            case .custom(let effort)?: ["reasoning_effort": effort]
            default: [:]
            }
        return variables.reduce(into: context) { $0[$1.key] = $1.value }
    }

    static func sampler(_ options: GenerationOptions) -> Sampler {
        let temperature = Float(options.temperature ?? 1)
        switch options.samplingMode?.kind {
        case .randomTopK(let k, let seed)?: return Sampler(mode: .topK(k), temperature: temperature, seed: seed)
        case .randomProbabilityThreshold(let p, let seed)?:
            return Sampler(mode: .topP(p), temperature: temperature, seed: seed)
        case nil where options.temperature != nil: return Sampler(mode: .full, temperature: temperature)
        default: return Sampler(mode: .greedy)
        }
    }
}

struct Output {
    let channel: LanguageModelExecutorGenerationChannel
    var calls: ToolCallParser?
    var lastCall: (id: String, name: String)?
    var written: [(id: String, keys: [String])] = []
    private var unreported = 0

    init(channel: LanguageModelExecutorGenerationChannel, calls: ToolCallParser?) {
        self.channel = channel
        self.calls = calls
    }

    mutating func send(_ parts: [ThinkParser.Part], reasoning: Bool, last: Bool = false) async -> Int {
        if !last { unreported += 1 }
        var sent = 0
        var events: [ToolCallParser.Event] = []
        for part in parts {
            switch part {
            case .reasoning(let text):
                await channel.send(.reasoning(action: .appendText(text, tokenCount: take())))
                sent += 1
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
                written.append((id, ToolCallParser.members(arguments)?.map(\.key) ?? []))
                await channel.send(
                    .toolCalls(
                        action: .toolCall(id: id, name: name, action: .appendArguments(arguments, tokenCount: take()))))
                lastCall = (id, name)
            }
        }
        if reasoning, unreported > 0 {
            await channel.send(.reasoning(action: .appendText("", tokenCount: take())))
            sent += 1
        }
        return sent + events.count
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
        let parameters =
            (try? JSONEncoder().encode(definition.parameters))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } ?? [:]
        let properties = parameters["properties"] as? [String: Any] ?? [:]
        name = definition.name
        order = parameters["x-order"] as? [String] ?? properties.keys.sorted()
        types = properties.compactMapValues { ($0 as? [String: Any])?["type"] as? String }
        spec = [
            "type": "function",
            "function": [
                "name": .string(definition.name), "description": .string(definition.description),
                "parameters": Self.value(parameters),
            ],
        ]
    }

    static func value(_ json: Any, order: [String]? = nil) -> Value {
        switch json {
        case let object as [String: Any]:
            let first = order ?? leadingKeys
            let rest = object.keys.filter { !first.contains($0) && (order != nil || !droppedKeys.contains($0)) }
                .sorted()
            return .object(
                OrderedDictionary(
                    uniqueKeysWithValues: (first.filter { object[$0] != nil } + rest).map { key in
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
    let engine: any RuntimeEngine

    init(_ engine: any RuntimeEngine) {
        self.engine = engine
    }
}
