import ArgumentParser
import ControlRuntime
import CoreAILanguageModels
import Darwin
import Foundation
import FoundationModels
import Synchronization

@main
struct CoreAIVerify: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "coreai-verify",
        abstract: "Drive the official Core AI runtime and the own engine, and print one JSON line per measurement.",
        subcommands: [Chat.self, ToolRoundTrip.self, Bench.self, Compare.self, Rewind.self, Speed.self, SpeedRun.self]
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

    func load() async throws -> any LanguageModel {
        if runtime == .own { return try await loadOwnModel(URL(filePath: model)) }
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

    func run() async throws {
        let model = try await options.load()
        var session = LanguageModelSession(model: model, instructions: instructions)
        for (index, text) in prompt.enumerated() {
            if freshSession, index > 0 {
                session = LanguageModelSession(model: model, instructions: instructions)
            }
            let result = await stream(session, from: .now) {
                session.streamResponse(to: text, options: GenerationOptions(maximumResponseTokens: maxTokens))
            }
            let response = joinedText(result.entries, where: responseSegments)
            let reasoning = joinedText(result.entries, where: reasoningSegments)
            try emit(TurnRecord(
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
                error: result.error
            ))
        }
    }
}

struct ToolRoundTrip: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "tools",
        abstract: "Ask questions that need zero, one, or two tool calls, each in a new session."
    )

    @OptionGroup var options: ModelOptions

    @Option(help: "A question. Each one runs in a new session.")
    var prompt: [String] = [
        "What is the weather in Tokyo right now?",
        "What is the weather in Paris, and what is 17 plus 25? Use the tools for both.",
        "What is the capital of Japan?",
    ]

    @Option(help: "Maximum response tokens per model call.")
    var maxTokens = 512

    func run() async throws {
        let model = try await options.load()
        for (index, text) in prompt.enumerated() {
            let log = ToolLog(start: .now)
            let session = LanguageModelSession(model: model, tools: [WeatherTool(log: log), AddTool(log: log)])
            let result = await stream(session, from: log.start) {
                session.streamResponse(to: text, options: GenerationOptions(maximumResponseTokens: maxTokens))
            }
            let calls = result.entries.flatMap { entry -> [CallRecord] in
                guard case .toolCalls(let calls) = entry else { return [] }
                return calls.map { CallRecord(name: $0.toolName, arguments: $0.arguments.jsonString) }
            }
            let invocations = log.events
            let outputs = result.entries.map { if case .toolOutput = $0 { true } else { false } }
            try emit(ToolRecord(
                question: index + 1,
                prompt: text,
                toolCalled: !invocations.isEmpty,
                calls: calls,
                invocations: invocations,
                roundTrips: zip(outputs, outputs.dropFirst() + [false]).filter { $0 && !$1 }.count,
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
            try emit(BenchRecord(
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
        log.record(name)
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
        log.record(name)
        return String(arguments.a + arguments.b)
    }
}

final class ToolLog: Sendable {
    let start: ContinuousClock.Instant
    private let storage = Mutex<[Invocation]>([])

    init(start: ContinuousClock.Instant) {
        self.start = start
    }

    func record(_ name: String) {
        let event = Invocation(name: name, t: seconds(ContinuousClock.now - start))
        storage.withLock { $0.append(event) }
    }

    var events: [Invocation] {
        storage.withLock { $0 }
    }
}

final class TranscriptWatch: Sendable {
    private let start: ContinuousClock.Instant
    private let state: Mutex<(seen: Int, marks: [EntryMark], changes: Int, stopped: Bool)>

    init(start: ContinuousClock.Instant, seen: Int) {
        self.start = start
        state = Mutex((seen, [], 0, false))
    }

    func observe(_ kinds: [String]) -> Bool {
        let t = seconds(ContinuousClock.now - start)
        return state.withLock { state in
            guard !state.stopped else { return false }
            state.changes += 1
            state.marks += kinds.dropFirst(state.seen).map { EntryMark(t: t, kind: $0) }
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
    _ makeStream: () -> LanguageModelSession.ResponseStream<String>
) async -> StreamResult {
    let base = session.transcript.count
    let before = session.usage
    let watch = TranscriptWatch(start: start, seen: base)
    let (armed, arm) = AsyncStream<Void>.makeStream()
    let observer = Task {
        for await kinds in Observations({ session.transcript.map(kind) }) {
            arm.finish()
            if !watch.observe(kinds) { break }
        }
    }
    for await _ in armed {}
    var result = StreamResult()
    do {
        for try await _ in makeStream() {
            if result.firstAnswer == nil { result.firstAnswer = seconds(ContinuousClock.now - start) }
            result.snapshots += 1
        }
    } catch {
        result.error = String(describing: error)
    }
    result.total = seconds(ContinuousClock.now - start)
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
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    FileHandle.standardOutput.write(try encoder.encode(record) + Data("\n".utf8))
}

struct Memory: Encodable {
    let rssBytes: UInt64
    let footprintBytes: UInt64

    static func now() -> Memory {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return Memory(rssBytes: 0, footprintBytes: 0) }
        return Memory(rssBytes: info.resident_size, footprintBytes: info.phys_footprint)
    }
}

struct Usage: Encodable {
    var inputTotal = 0
    var inputCached = 0
    var outputTotal = 0
    var outputReasoning = 0
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
}

struct Invocation: Encodable {
    let name: String
    let t: Double
}

struct CallRecord: Encodable {
    let name: String
    let arguments: String
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
    let error: String?
}

struct ToolRecord: Encodable {
    var record = "tools"
    let question: Int
    let prompt: String
    let toolCalled: Bool
    let calls: [CallRecord]
    let invocations: [Invocation]
    let roundTrips: Int
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
