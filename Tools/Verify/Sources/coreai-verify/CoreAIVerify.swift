import ArgumentParser
import ControlRuntime
import CoreAILanguageModels
import Darwin
import Foundation
import FoundationModels
import Synchronization

@_spi(Verification) import struct CoreAIBackend.GenerationMeasurement

@main
struct CoreAIVerify: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "coreai-verify",
        abstract: "Drive the official Core AI runtime and the own engine, and print one JSON line per measurement.",
        subcommands: [
            Chat.self, ToolRoundTrip.self, Bench.self, Compare.self, Rewind.self, Speed.self, SpeedRun.self,
            OCRBlock.self,
        ]
    )
}

extension Engine: ExpressibleByArgument {}

enum RuntimeKind: String, ExpressibleByArgument {
    case official
    case own
}

struct ModelOptions: ParsableArguments {
    @Option(help: "Model bundle folder.")
    var model: String

    @Option(help: "Runtime: official (coreai-models) or own (CoreAIBackend).")
    var runtime: RuntimeKind = .official

    @Option(help: "Inference engine of the official runtime: default or sequential.")
    var engine: Engine = .default

    @Flag(help: "Print the runtime's own log lines to stderr.")
    var verbose = false

    @Option(help: "Context limit in tokens of the own runtime.")
    var contextLimit: Int?

    @Flag(help: "On overflow, let the own runtime drop old tool outputs instead of returning an error.")
    var dropOldToolOutputs = false

    func validate() throws {
        if runtime == .official, contextLimit != nil || dropOldToolOutputs {
            throw ValidationError("--context-limit and --drop-old-tool-outputs apply to --runtime own only.")
        }
    }

    func load(_ experiment: ExperimentOptions? = nil, reload: ReloadOptions? = nil) async throws -> any LanguageModel {
        if runtime == .own {
            return try await loadOwnModel(
                URL(filePath: model), contextLimit: contextLimit,
                overflow: dropOldToolOutputs ? .dropOldToolOutputs : .error,
                experiment: experiment, reloadPolicy: reload?.policy ?? .init())
        }
        if verbose { CLILogger.level = 1 }
        let clock = ContinuousClock()
        let start = clock.now
        let model = try await ControlRuntime.makeModel(bundle: URL(filePath: self.model), engine: engine)
        let loadStart = clock.now
        try await model.load()
        try emit(LoadRecord(initS: seconds(loadStart - start), loadS: seconds(clock.now - loadStart), memory: .now()))
        return model
    }
}

struct Chat: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Stream several turns through one LanguageModelSession."
    )

    @OptionGroup var options: ModelOptions

    @Option(help: "A user turn. Repeat to send several turns in order.")
    var prompt: [String]

    @Option(help: "Maximum response tokens per turn.")
    var maxTokens = 512

    @Option(help: "Session instructions.")
    var instructions: String?

    @Flag(help: "Send each prompt in a new session instead of one shared session.")
    var freshSession = false

    @Option(help: "Start a new conversation after this many turns. Zero keeps one conversation.")
    var newConversationEvery = 0

    @Option(
        help:
            "Wait five seconds after this turn and record memory before any state check. Repeat to select turns. Own runtime only."
    )
    var memorySampleTurn: [Int] = []

    @OptionGroup var experiment: ExperimentOptions

    @OptionGroup var reload: ReloadOptions

    func validate() throws {
        try experiment.validate(options.runtime)
        try reload.validate(options.runtime)
        guard newConversationEvery >= 0 else {
            throw ValidationError("--new-conversation-every must be zero or greater.")
        }
        guard !freshSession || newConversationEvery == 0 else {
            throw ValidationError("Choose either --fresh-session or --new-conversation-every.")
        }
        guard memorySampleTurn.isEmpty || options.runtime == .own else {
            throw ValidationError("--memory-sample-turn requires --runtime own.")
        }
        guard Set(memorySampleTurn).count == memorySampleTurn.count else {
            throw ValidationError("--memory-sample-turn values must be unique.")
        }
        guard memorySampleTurn.allSatisfy({ $0 > 0 && $0 <= prompt.count }) else {
            throw ValidationError("--memory-sample-turn must refer to an existing turn, starting at one.")
        }
    }

    func run() async throws {
        let model = try await options.load(experiment, reload: reload)
        let snapshot = tokenSnapshot(model)
        let generated = generatedTokenSnapshot(model)
        let measured = measurementSnapshot(model, experiment)
        var measuredID: UUID?
        var session = LanguageModelSession(model: model, instructions: instructions)
        for (index, text) in prompt.enumerated() {
            if index > 0, freshSession || (newConversationEvery > 0 && index.isMultiple(of: newConversationEvery)) {
                session = LanguageModelSession(model: model, instructions: instructions)
            }
            let result = await stream(session, from: .now) {
                session.streamResponse(
                    to: text, options: GenerationOptions(maximumResponseTokens: maxTokens),
                    contextOptions: experiment.contextOptions)
            }
            let response = joinedText(result.entries, where: responseSegments)
            let reasoning = joinedText(result.entries, where: reasoningSegments)
            let processed = await snapshot?()
            let generatedTokens = result.error == nil ? await generated?() : nil
            let latest = await measured?()
            let measurement = latest?.requestID == measuredID ? nil : latest
            measuredID = latest?.requestID ?? measuredID
            try emit(
                TurnRecord(
                    runtime: options.runtime.rawValue,
                    turn: index + 1,
                    promptChars: text.count,
                    response: response,
                    reasoning: reasoning,
                    responseChars: response.count,
                    reasoningChars: reasoning.count,
                    stopReason: result.metadata("stopReason", as: String.self),
                    prefillTokens: result.metadata("prefillTokens", as: Int.self),
                    sharedTokens: result.metadata("sharedTokens", as: Int.self),
                    reloadSeconds: result.metadata("reloadSeconds", as: Double.self),
                    ttftS: result.ttft,
                    firstAnswerS: result.firstAnswer,
                    totalS: result.total,
                    outputTokens: result.usage.outputTotal,
                    decodeTokS: result.decodeRate,
                    snapshots: result.snapshots,
                    transcriptChanges: result.changes,
                    usage: result.usage,
                    entries: result.marks,
                    memory: .now(),
                    processedTokens: processed,
                    generatedTokens: generatedTokens,
                    measurement: measurement,
                    error: result.error
                ))
            if memorySampleTurn.contains(index + 1) {
                try emit(try await sampleMemory(turn: index + 1, retaining: model))
            }
            if experiment.checkState, result.metadata("reloadSeconds", as: Double.self) != nil {
                try await checkState(model)
            }
        }
        if experiment.checkState { try await checkState(model) }
    }
}

enum ToolMode: String, ExpressibleByArgument {
    case allowed
    case required
    case disallowed

    var mode: GenerationOptions.ToolCallingMode {
        switch self {
        case .allowed: .allowed
        case .required: .required
        case .disallowed: .disallowed
        }
    }
}

struct ToolRoundTrip: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "tools",
        abstract: "Ask questions that may need tools; print one record per model step and one per question."
    )

    @OptionGroup var options: ModelOptions

    @Option(help: "A question. Each one runs in a new session unless --same-session is set.")
    var prompt: [String] = [
        "What is the weather in Tokyo right now?",
        "What is the weather in Paris, and what is 17 plus 25? Use the tools for both.",
        "What is the capital of Japan?",
    ]

    @Option(help: "Maximum response tokens per model call.")
    var maxTokens = 512

    @Option(help: "Text file used as the session instructions.")
    var instructionsFile: String?

    @Option(help: "Folder of <name>.txt pages served by a read_page tool, used instead of get_weather and add.")
    var pages: String?

    @Flag(help: "Send every question in one session.")
    var sameSession = false

    @Option(help: "Tool calling mode: allowed, required or disallowed.")
    var toolMode: ToolMode?

    @Option(help: "Cancel this question (1-based) after --cancel-after seconds.")
    var cancel: Int?

    @Option(help: "Seconds before the cancellation.")
    var cancelAfter = 1.0

    @OptionGroup var experiment: ExperimentOptions

    @OptionGroup var reload: ReloadOptions

    func validate() throws {
        try experiment.validate(options.runtime)
        try reload.validate(options.runtime)
        if experiment.measure != nil { throw ValidationError("--measure applies to chat only.") }
    }

    func run() async throws {
        let model = try await options.load(experiment, reload: reload)
        let snapshot = tokenSnapshot(model)
        let log = ToolLog(snapshot: snapshot)
        let instructions = try instructionsFile.map { try String(contentsOfFile: $0, encoding: .utf8) }
        let tools: [any Tool] =
            pages.map { [PageTool(folder: $0, log: log)] } ?? [WeatherTool(log: log), AddTool(log: log)]
        var session = LanguageModelSession(model: model, tools: tools, instructions: instructions)
        for (index, text) in prompt.enumerated() {
            if !sameSession, index > 0 {
                session = LanguageModelSession(model: model, tools: tools, instructions: instructions)
            }
            let start = log.restart()
            let options = GenerationOptions(maximumResponseTokens: maxTokens, toolCallingMode: toolMode?.mode)
            let result = await stream(session, from: start, cancelAfter: cancel == index + 1 ? cancelAfter : nil) {
                session.streamResponse(to: text, options: options, contextOptions: experiment.contextOptions)
            }
            let (invocations, snapshots) = log.take()
            let final = await snapshot?()
            let steps = StepRecord.split(
                result, runtime: self.options.runtime.rawValue, question: index + 1, snapshots: snapshots, final: final)
            for step in steps { try emit(step) }
            try emit(
                ToolRecord(
                    question: index + 1,
                    prompt: text,
                    toolCalled: !invocations.isEmpty,
                    calls: steps.flatMap { $0.calls },
                    invocations: invocations,
                    steps: steps.count,
                    cancelled: cancel == index + 1,
                    response: joinedText(result.entries.suffix(1), where: responseSegments),
                    ttftS: result.ttft,
                    firstAnswerS: result.firstAnswer,
                    totalS: result.total,
                    transcriptChanges: result.changes,
                    usage: result.usage,
                    entries: result.marks,
                    transcript: session.transcript.map(kind),
                    memory: .now(),
                    error: result.error
                ))
        }
        if experiment.checkState { try await checkState(model) }
    }
}

struct Bench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Time a reload, then stream one fixed prompt three times, each in a new session."
    )

    static let fixedPrompt =
        "Write a detailed essay of at least 1500 words about the history of computing, from the abacus to modern AI. Use multiple sections with headings."

    @OptionGroup var options: ModelOptions

    func run() async throws {
        let model = try await options.load()
        if let official = model as? CoreAILanguageModel {
            official.unload()
            let clock = ContinuousClock()
            let start = clock.now
            try await official.load()
            try emit(ReloadRecord(loadS: seconds(clock.now - start), memory: .now()))
        }
        for run in 1...3 {
            let session = LanguageModelSession(model: model)
            let result = await stream(session, from: .now) {
                session.streamResponse(to: Self.fixedPrompt, options: GenerationOptions(maximumResponseTokens: 1024))
            }
            let response = joinedText(result.entries, where: responseSegments)
            let reasoning = joinedText(result.entries, where: reasoningSegments)
            try emit(
                BenchRecord(
                    runtime: options.runtime.rawValue,
                    run: run,
                    response: response,
                    reasoning: reasoning,
                    responseChars: response.count,
                    reasoningChars: reasoning.count,
                    stopReason: result.metadata("stopReason", as: String.self),
                    prefillTokens: result.metadata("prefillTokens", as: Int.self),
                    sharedTokens: result.metadata("sharedTokens", as: Int.self),
                    ttftS: result.ttft,
                    firstAnswerS: result.firstAnswer,
                    totalS: result.total,
                    outputTokens: result.usage.outputTotal,
                    decodeTokS: result.decodeRate,
                    snapshots: result.snapshots,
                    transcriptChanges: result.changes,
                    usage: result.usage,
                    memory: .now(),
                    error: result.error
                ))
        }
    }
}

struct WeatherTool: Tool {
    let name = "get_weather"
    let description = "Get the current weather for a city."
    let log: ToolLog

    @Generable
    struct Arguments {
        @Guide(description: "The city name.")
        var city: String
    }

    func call(arguments: Arguments) async throws -> String {
        let started = await log.begin()
        defer { log.record(name, started) }
        return "Sunny, 22 C in \(arguments.city)."
    }
}

struct AddTool: Tool {
    let name = "add"
    let description = "Add two integers and return the sum."
    let log: ToolLog

    @Generable
    struct Arguments {
        @Guide(description: "The first integer.")
        var a: Int
        @Guide(description: "The second integer.")
        var b: Int
    }

    func call(arguments: Arguments) async throws -> String {
        let started = await log.begin()
        defer { log.record(name, started) }
        return String(arguments.a + arguments.b)
    }
}

struct PageTool: Tool {
    let name = "read_page"
    let description = "Read a page of the office handbook by its name and return the full text of the page."
    let folder: String
    let log: ToolLog

    @Generable
    struct Arguments {
        @Guide(description: "The page name, for example index.")
        var name: String
    }

    func call(arguments: Arguments) async throws -> String {
        let started = await log.begin()
        defer { log.record(name, started) }
        let url = URL(filePath: folder).appending(path: arguments.name + ".txt")
        if let page = try? String(contentsOf: url, encoding: .utf8) { return page }
        let names = (try? FileManager.default.contentsOfDirectory(atPath: folder)) ?? []
        let available = names.filter { $0.hasSuffix(".txt") }.map { String($0.dropLast(4)) }.sorted()
        return "No page named \(arguments.name). Available pages: \(available.joined(separator: ", "))."
    }
}

final class ToolLog: Sendable {
    private let snapshot: (@Sendable () async -> [Int32])?
    private let state = Mutex<(start: ContinuousClock.Instant, events: [Invocation], snapshots: [[Int32]])>(
        (.now, [], []))

    init(snapshot: (@Sendable () async -> [Int32])?) {
        self.snapshot = snapshot
    }

    func restart() -> ContinuousClock.Instant {
        state.withLock { state in
            state = (.now, [], [])
            return state.start
        }
    }

    func begin() async -> ContinuousClock.Instant {
        let now = ContinuousClock.now
        if let snapshot {
            let tokens = await snapshot()
            state.withLock { if $0.snapshots.last != tokens { $0.snapshots.append(tokens) } }
        }
        return now
    }

    func record(_ name: String, _ started: ContinuousClock.Instant) {
        let end = ContinuousClock.now
        state.withLock {
            $0.events.append(Invocation(name: name, t: seconds(started - $0.start), endT: seconds(end - $0.start)))
        }
    }

    func take() -> (events: [Invocation], snapshots: [[Int32]]) {
        state.withLock { ($0.events, $0.snapshots) }
    }
}

final class TranscriptWatch: Sendable {
    private let start: ContinuousClock.Instant
    private let state: Mutex<(seen: Int, marks: [EntryMark], changes: Int, stopped: Bool)>

    init(start: ContinuousClock.Instant, seen: Int) {
        self.start = start
        state = Mutex((seen, [], 0, false))
    }

    func observe(_ kinds: [String], usage: Usage) -> Bool {
        let t = seconds(ContinuousClock.now - start)
        return state.withLock { state in
            guard !state.stopped else { return false }
            state.changes += 1
            state.marks += kinds.dropFirst(state.seen).map { EntryMark(t: t, kind: $0, usage: usage) }
            state.seen = max(state.seen, kinds.count)
            return true
        }
    }

    func stop() -> (marks: [EntryMark], changes: Int) {
        state.withLock { state in
            state.stopped = true
            return (state.marks, state.changes)
        }
    }
}

struct StreamResult {
    var ttft: Double?
    var firstAnswer: Double?
    var total = 0.0
    var snapshots = 0
    var changes = 0
    var marks: [EntryMark] = []
    var entries: [Transcript.Entry] = []
    var usage = Usage()
    var error: String?

    var decodeRate: Double? {
        guard let ttft, usage.outputTotal > 1, total > ttft else { return nil }
        return Double(usage.outputTotal - 1) / (total - ttft)
    }

    func metadata<Value: ConvertibleFromGeneratedContent>(_ key: String, as type: Value.Type) -> Value? {
        let last = entries.last { if case .response = $0 { true } else { false } }
        guard case .response(let response) = last else { return nil }
        return try? response.metadata[key]?.value(type)
    }
}

func stream(
    _ session: LanguageModelSession,
    from start: ContinuousClock.Instant,
    cancelAfter: Double? = nil,
    _ makeStream: () -> LanguageModelSession.ResponseStream<String>
) async -> StreamResult {
    let base = session.transcript.count
    let before = session.usage
    let watch = TranscriptWatch(start: start, seen: base)
    let (armed, arm) = AsyncStream<Void>.makeStream()
    let observer = Task {
        for await (kinds, usage) in Observations({ (session.transcript.map(kind), Usage(session.usage, since: before)) }
        ) {
            arm.finish()
            if !watch.observe(kinds, usage: usage) { break }
        }
    }
    for await _ in armed {}
    let responses = makeStream()
    let consumer = Task { () -> (first: Double?, snapshots: Int, error: String?) in
        var first: Double?
        var snapshots = 0
        do {
            for try await _ in responses {
                if first == nil { first = seconds(ContinuousClock.now - start) }
                snapshots += 1
            }
            return (first, snapshots, Task.isCancelled ? "cancelled" : nil)
        } catch {
            let mirror = Mirror(reflecting: error)
            guard mirror.displayStyle == .enum, let name = mirror.children.first?.label else {
                return (first, snapshots, String(describing: error))
            }
            return (first, snapshots, "\(type(of: error)).\(name): \(error)")
        }
    }
    if let cancelAfter {
        Task {
            try? await Task.sleep(for: .seconds(cancelAfter))
            consumer.cancel()
        }
    }
    var result = StreamResult()
    (result.firstAnswer, result.snapshots, result.error) = await consumer.value
    result.total = seconds(ContinuousClock.now - start)
    if cancelAfter != nil { try? await Task.sleep(for: .milliseconds(300)) }
    (result.marks, result.changes) = watch.stop()
    observer.cancel()
    result.ttft = result.marks.first { $0.kind != "prompt" }?.t
    result.entries = Array(session.transcript.dropFirst(base))
    result.usage = Usage(session.usage, since: before)
    return result
}

func kind(_ entry: Transcript.Entry) -> String {
    switch entry {
    case .instructions: "instructions"
    case .prompt: "prompt"
    case .toolCalls(let calls): "toolCalls(\(calls.map(\.toolName).joined(separator: ",")))"
    case .toolOutput(let output): "toolOutput(\(output.toolName))"
    case .response: "response"
    case .reasoning: "reasoning"
    @unknown default: "unknown"
    }
}

func responseSegments(_ entry: Transcript.Entry) -> [Transcript.Segment]? {
    if case .response(let response) = entry { response.segments } else { nil }
}

func reasoningSegments(_ entry: Transcript.Entry) -> [Transcript.Segment]? {
    if case .reasoning(let reasoning) = entry { reasoning.segments } else { nil }
}

func joinedText(
    _ entries: some Sequence<Transcript.Entry>,
    where segments: (Transcript.Entry) -> [Transcript.Segment]?
) -> String {
    entries.compactMap(segments).joined().compactMap { segment -> String? in
        if case .text(let text) = segment { text.content } else { nil }
    }.joined()
}

func seconds(_ duration: Duration) -> Double {
    Double(duration.components.seconds) + Double(duration.components.attoseconds) * 1e-18
}

func emit(_ record: some Encodable) throws {
    FileHandle.standardOutput.write(try JSONEncoder.records.encode(record) + Data("\n".utf8))
}

extension JSONEncoder {
    static var records: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }
}

struct Memory: Encodable {
    var status = "ok"
    var error: String?
    var capturedAt = Date.now
    let unit = "B"
    var rssBytes: UInt64?
    var footprintBytes: UInt64?
    var graphicsFootprintBytes: UInt64?

    static func now() -> Memory {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return failed(code: result) }
        return Memory(
            rssBytes: info.resident_size, footprintBytes: info.phys_footprint,
            graphicsFootprintBytes: UInt64(max(0, info.ledger_tag_graphics_footprint))
                + UInt64(max(0, info.ledger_tag_graphics_footprint_compressed)))
    }

    static func failed(code: kern_return_t) -> Memory {
        Memory(status: "failed", error: "task_info returned \(code)")
    }
}

struct Usage: Encodable {
    var inputTotal = 0
    var inputCached = 0
    var outputTotal = 0
    var outputReasoning = 0

    static func - (a: Usage, b: Usage) -> Usage {
        Usage(
            inputTotal: a.inputTotal - b.inputTotal, inputCached: a.inputCached - b.inputCached,
            outputTotal: a.outputTotal - b.outputTotal, outputReasoning: a.outputReasoning - b.outputReasoning)
    }
}

extension Usage {
    init(_ now: LanguageModelSession.Usage, since before: LanguageModelSession.Usage) {
        inputTotal = now.input.totalTokenCount - before.input.totalTokenCount
        inputCached = now.input.cachedTokenCount - before.input.cachedTokenCount
        outputTotal = now.output.totalTokenCount - before.output.totalTokenCount
        outputReasoning = now.output.reasoningTokenCount - before.output.reasoningTokenCount
    }
}

struct EntryMark: Encodable {
    let t: Double
    let kind: String
    let usage: Usage
}

struct Invocation: Encodable {
    let name: String
    let t: Double
    let endT: Double
}

struct CallRecord: Encodable {
    let name: String
    let arguments: String
    let keys: [String]
}

struct StepRecord: Encodable {
    var record = "step"
    let runtime: String
    let question: Int
    let step: Int
    let startS: Double
    let ttftS: Double
    var entries: [String] = []
    var calls: [CallRecord] = []
    var reasoning = ""
    var response = ""
    var usage = Usage()
    var stopReason: String?
    var prefillTokens: Int?
    var sharedTokens: Int?
    var reloadSeconds: Double?
    var prefillSeconds: Double?
    var decodeSeconds: Double?
    var truncatedToolOutputs: Int?
    var reservedTokens: Int?
    var processedTokens: [Int32]?

    static func split(_ result: StreamResult, runtime: String, question: Int, snapshots: [[Int32]], final: [Int32]?)
        -> [StepRecord]
    {
        var steps: [StepRecord] = []
        var current: StepRecord?
        var boundary = 0.0
        var counted = Usage()
        for (index, mark) in result.marks.enumerated() where mark.kind != "prompt" {
            if mark.kind.hasPrefix("toolOutput") {
                if var step = current {
                    step.usage = mark.usage - counted
                    counted = mark.usage
                    steps.append(step)
                    current = nil
                }
                boundary = mark.t
                continue
            }
            var step =
                current
                ?? StepRecord(
                    runtime: runtime, question: question, step: steps.count + 1, startS: boundary,
                    ttftS: mark.t - boundary)
            step.entries.append(mark.kind)
            if index < result.entries.count { step.read(result.entries[index]) }
            current = step
        }
        if var step = current {
            step.usage = result.usage - counted
            steps.append(step)
        }
        for index in steps.indices {
            steps[index].processedTokens =
                index == steps.count - 1 ? final : index < snapshots.count ? snapshots[index] : nil
        }
        return steps
    }

    mutating func read(_ entry: Transcript.Entry) {
        var metadata: [String: GeneratedContent] = [:]
        switch entry {
        case .reasoning(let reasoning):
            self.reasoning += joinedText([entry], where: reasoningSegments)
            metadata = reasoning.metadata
        case .response(let response):
            self.response += joinedText([entry], where: responseSegments)
            metadata = response.metadata
        case .toolCalls(let toolCalls):
            calls += toolCalls.map { call in
                let keys: [String] = if case .structure(_, let keys) = call.arguments.kind { keys } else { [] }
                return CallRecord(name: call.toolName, arguments: call.arguments.jsonString, keys: keys)
            }
            metadata = toolCalls.last?.metadata ?? [:]
        default: break
        }
        guard metadata["stopReason"] != nil else { return }
        stopReason = try? metadata["stopReason"]?.value(String.self)
        prefillTokens = try? metadata["prefillTokens"]?.value(Int.self)
        sharedTokens = try? metadata["sharedTokens"]?.value(Int.self)
        reloadSeconds = try? metadata["reloadSeconds"]?.value(Double.self)
        prefillSeconds = try? metadata["prefillSeconds"]?.value(Double.self)
        decodeSeconds = try? metadata["decodeSeconds"]?.value(Double.self)
        truncatedToolOutputs = try? metadata["truncatedToolOutputs"]?.value(Int.self)
        reservedTokens = try? metadata["reservedTokens"]?.value(Int.self)
    }
}

struct LoadRecord: Encodable {
    var record = "load"
    let initS: Double
    let loadS: Double
    let memory: Memory
}

struct ReloadRecord: Encodable {
    var record = "reload"
    let loadS: Double
    let memory: Memory
}

struct TurnRecord: Encodable {
    var record = "turn"
    let runtime: String
    let turn: Int
    let promptChars: Int
    let response: String
    let reasoning: String
    let responseChars: Int
    let reasoningChars: Int
    let stopReason: String?
    let prefillTokens: Int?
    let sharedTokens: Int?
    let reloadSeconds: Double?
    let ttftS: Double?
    let firstAnswerS: Double?
    let totalS: Double
    let outputTokens: Int
    let decodeTokS: Double?
    let snapshots: Int
    let transcriptChanges: Int
    let usage: Usage
    let entries: [EntryMark]
    let memory: Memory
    let processedTokens: [Int32]?
    let generatedTokens: [Int32]?
    let measurement: GenerationMeasurement?
    let error: String?
}

struct ToolRecord: Encodable {
    var record = "tools"
    let question: Int
    let prompt: String
    let toolCalled: Bool
    let calls: [CallRecord]
    let invocations: [Invocation]
    let steps: Int
    let cancelled: Bool
    let response: String
    let ttftS: Double?
    let firstAnswerS: Double?
    let totalS: Double
    let transcriptChanges: Int
    let usage: Usage
    let entries: [EntryMark]
    let transcript: [String]
    let memory: Memory
    let error: String?
}

struct BenchRecord: Encodable {
    var record = "bench"
    let runtime: String
    let run: Int
    let response: String
    let reasoning: String
    let responseChars: Int
    let reasoningChars: Int
    let stopReason: String?
    let prefillTokens: Int?
    let sharedTokens: Int?
    let ttftS: Double?
    let firstAnswerS: Double?
    let totalS: Double
    let outputTokens: Int
    let decodeTokS: Double?
    let snapshots: Int
    let transcriptChanges: Int
    let usage: Usage
    let memory: Memory
    let error: String?
}
