import ArgumentParser
@_spi(Verification) import CoreAIBackend
import CoreAILanguageModels
import Darwin
import Foundation
import FoundationModels
import Tokenizers

let rewindQuestions = [
    "According to the notes, which floor is Room 42 on? Answer briefly.",
    "How many seats does Room 17 have? Answer briefly.",
    "Which rooms on floor 2 have a projector? Answer briefly.",
]

struct Compare: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract:
            "Feed the same tokens to the own engine and the official sequential engine; compare greedy tokens and logits."
    )

    @Option(help: "Model bundle folder.")
    var model: String

    @Option(help: "A prompt as name=text. Repeat to send several.")
    var prompt: [String] = []

    @Option(
        help:
            "Prompt tokens as name=path, path being a JSON array of token IDs; append @n to prefill the own engine in two parts split at n."
    )
    var tokens: [String] = []

    @Option(help: "Maximum generated tokens per prompt.")
    var maxTokens = 1024

    @Option(help: "Text file sent as the system message before each prompt.")
    var system: String?

    @Flag(
        help:
            "Prefill the own engine in two parts split at the end of the user turn, as the own runtime does for hybrid models."
    )
    var turnBoundary = false

    func run() async throws {
        let url = URL(filePath: model)
        let systemText = try system.map { try String(contentsOfFile: $0, encoding: .utf8) }
        let (tokenizer, own) = try await loadOwn(url)
        let clock = ContinuousClock()
        let start = clock.now
        let official = try await EngineFactory.createEngine(
            bundle: LanguageModelBundle(at: url), options: EngineOptions(variant: "coreai-sequential"))
        try emit(OfficialLoadRecord(engine: "sequential", loadS: (clock.now - start) / .seconds(1), memory: .now()))
        let stop = stopTokens(tokenizer)
        for entry in prompt {
            let name = String(entry.prefix { $0 != "=" })
            let user = String(entry.dropFirst(name.count + 1))
            let input = try chatInput(tokenizer, system: systemText, user: user)
            let boundary = try chatInput(tokenizer, system: systemText, user: user, generationPrompt: false)
            let split = zip(boundary, input).prefix { $0 == $1 }.count
            for usePrefillFunction in turnBoundary ? [true] : [true, false] {
                own.reset()
                try await official.reset(to: 0)
                var record = try await lockstep(
                    own: own, official: official, input: input, usePrefillFunction: usePrefillFunction,
                    split: turnBoundary ? split : nil, maxTokens: maxTokens, stop: stop)
                record.prompt = name
                record.text = tokenizer.decode(tokens: record.tokens.map(Int.init))
                try emitComparison(record)
            }
        }
        for entry in tokens {
            let name = String(entry.prefix { $0 != "=" })
            let spec = entry.dropFirst(name.count + 1)
            let path = String(spec.prefix { $0 != "@" })
            let split = Int(spec.dropFirst(path.count + 1))
            let input = try JSONDecoder().decode([Int32].self, from: Data(contentsOf: URL(filePath: path)))
            own.reset()
            try await official.reset(to: 0)
            var record = try await lockstep(
                own: own, official: official, input: input, usePrefillFunction: true, split: split,
                maxTokens: maxTokens, stop: stop)
            record.prompt = name
            record.text = tokenizer.decode(tokens: record.tokens.map(Int.init))
            try emitComparison(record)
        }
    }

    func lockstep(
        own: CoreAIBackend.Engine, official: some InferenceEngine, input: [Int32], usePrefillFunction: Bool,
        split: Int?, maxTokens: Int, stop: Set<Int32>
    ) async throws -> CompareRecord {
        let sequence = try await official.generate(
            with: input, samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: maxTokens, includeLogits: true))
        var iterator = sequence.makeAsyncIterator()
        if let split {
            try await own.prefill(Array(input[..<split]))
            try await own.prefill(Array(input[split...]))
        } else {
            try await own.prefill(input, usePrefillFunction: usePrefillFunction)
        }
        var record = CompareRecord(
            prefill: split.map { "turn-boundary-\($0)" } ?? (usePrefillFunction ? "prefill-function" : "main-only"),
            promptTokens: input.count)
        while record.tokens.count < maxTokens, let reference = try await iterator.next() {
            let logits = own.logits()
            let referenceLogits = reference.logits ?? []
            guard record.observeLogits(own: logits, official: referenceLogits) else { break }
            let token = own.greedyToken()
            record.tokens.append(token)
            record.officialTokens.append(reference.tokenId)
            if token != reference.tokenId {
                record.firstMismatch = record.tokens.count - 1
                record.mismatch = Mismatch(own: top2(logits), official: top2(referenceLogits))
                record.stop = "mismatch"
                break
            }
            if stop.contains(token) {
                record.stop = "eos"
                break
            }
            if record.tokens.count < maxTokens { try await own.step(token) }
        }
        record.match = record.firstMismatch == nil
        return record
    }
}

struct Rewind: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract:
            "Save a point after a shared preamble, generate, restore, and generate again; time a full re-read for comparison."
    )

    @Option(help: "Model bundle folder.")
    var model: String

    @Option(help: "Text file sent as the system message: the shared preamble.")
    var preamble: String

    @Option(
        help:
            "A user question that follows the preamble. Repeat to send two or more; the shared tokens form the saved point."
    )
    var question: [String] = rewindQuestions

    @Option(help: "Tokens generated after each question (stop tokens are ignored).")
    var maxTokens = 64

    @Option(help: "Rounds of restores and of full re-reads.")
    var trials = 3

    func validate() throws {
        guard question.count >= 2 else { throw ValidationError("Pass two or more questions.") }
    }

    func run() async throws {
        let (tokenizer, engine) = try await loadOwn(URL(filePath: model))
        let notes = try String(contentsOfFile: preamble, encoding: .utf8)
        let inputs = try question.map { try chatInput(tokenizer, system: notes, user: $0) }
        let shared = inputs.dropFirst().map { zip($0, inputs[0]).prefix { $0.0 == $0.1 }.count }.min() ?? 0
        let prefix = Array(inputs[0].prefix(shared))
        let clock = ContinuousClock()

        _ = try await engine.generate(Array(prefix.prefix(16)), maxTokens: 4, stopTokens: [])
        engine.reset()
        engine.stats = .init()
        try await engine.prefill(prefix)
        let saveStart = clock.now
        let point = engine.checkpoint()
        try emit(
            PointRecord(
                prefixTokens: prefix.count, prefillS: engine.stats.prefillSeconds,
                saveS: (clock.now - saveStart) / .seconds(1), stateBytes: point.stateBytes, hybrid: engine.isHybrid,
                kvCapacity: engine.kvCapacity, memory: .now()))

        var first: [[Int32]] = Array(repeating: [], count: inputs.count)
        for trial in 1...trials {
            for (index, input) in inputs.enumerated() {
                let restoreStart = clock.now
                try engine.restore(point)
                let restore = (clock.now - restoreStart) / .seconds(1)
                engine.stats = .init()
                let output = try await engine.generate(
                    Array(input[prefix.count...]), maxTokens: maxTokens, stopTokens: [])
                if trial == 1 { first[index] = output }
                try emit(
                    RewindRecord(
                        kind: "restore", trial: trial, question: index, restoreS: restore,
                        inputTokens: input.count - prefix.count, prefillS: engine.stats.prefillSeconds,
                        matchesFirst: output == first[index], tokens: output,
                        text: tokenizer.decode(tokens: output.map(Int.init))))
            }
        }
        for (index, input) in inputs.enumerated() {
            engine.reset()
            try await engine.prefill(prefix)
            engine.stats = .init()
            let output = try await engine.generate(Array(input[prefix.count...]), maxTokens: maxTokens, stopTokens: [])
            try emit(
                RewindRecord(
                    kind: "fresh", trial: 1, question: index, restoreS: nil, inputTokens: input.count - prefix.count,
                    prefillS: engine.stats.prefillSeconds, matchesFirst: output == first[index], tokens: output,
                    text: nil))
        }
        for trial in 1...trials {
            for (index, input) in inputs.enumerated() {
                engine.reset()
                engine.stats = .init()
                let output = try await engine.generate(input, maxTokens: maxTokens, stopTokens: [])
                try emit(
                    RewindRecord(
                        kind: "full", trial: trial, question: index, restoreS: nil, inputTokens: input.count,
                        prefillS: engine.stats.prefillSeconds, matchesFirst: output == first[index], tokens: output,
                        text: nil))
            }
        }
    }
}

struct Speed: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract:
            "Alternate the own engine, the official sequential engine, and the official default engine, one process each."
    )

    @Option(help: "Model bundle folder.")
    var model: String

    @Option(help: "Text file sent as the system message before the question.")
    var preamble: String

    @Option(help: "Rounds; each round runs every engine once.")
    var rounds = 3

    @Option(help: "Seconds to rest after each process.")
    var pause = 0

    @Option(help: "Decode tokens per run (stop tokens are ignored).")
    var maxTokens = 1024

    @Option(help: "Engines to alternate: own, sequential, default.")
    var engine: [String] = ["own", "sequential", "default"]

    func run() async throws {
        guard let executable = Bundle.main.executableURL else { throw ExitCode.failure }
        var records: [SpeedRecord] = []
        for round in 1...rounds {
            for name in engine {
                let process = Process()
                let pipe = Pipe()
                process.executableURL = executable
                process.arguments = [
                    "speed-run", "--model", model, "--preamble", preamble, "--engine", name,
                    "--max-tokens", String(maxTokens), "--round", String(round),
                ]
                process.standardOutput = pipe
                try process.run()
                let output = pipe.fileHandleForReading.readDataToEndOfFile()
                process.waitUntilExit()
                guard process.terminationStatus == 0 else { throw ExitCode(process.terminationStatus) }
                let record = try JSONDecoder.snake.decode(SpeedRecord.self, from: output)
                records.append(record)
                try emit(record)
                if pause > 0 { try await Task.sleep(for: .seconds(pause)) }
            }
        }
        let own = records.filter { $0.engine == "own" }
        for name in engine {
            try emit(SpeedSummary(engine: name, runs: records.filter { $0.engine == name }, own: own))
        }
    }
}

struct SpeedRun: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "speed-run", shouldDisplay: false)

    @Option var model: String
    @Option var preamble: String
    @Option var engine: String
    @Option var maxTokens: Int
    @Option var round: Int
    @Option(name: .customLong("repeat")) var repeatCount = 1

    func validate() throws {
        guard repeatCount > 0 else { throw ValidationError("--repeat takes a count of one or more.") }
    }

    func run() async throws {
        let url = URL(filePath: model)
        let bundle = try ModelBundle(url: url)
        let tokenizer = try await bundle.loadTokenizer()
        let input = try chatInput(
            tokenizer, system: String(contentsOfFile: preamble, encoding: .utf8), user: rewindQuestions[0])
        let warmup = Array(input.prefix(16))
        let loadStart = ContinuousClock.now
        if engine == "own" {
            let own = try await CoreAIBackend.Engine(bundle: bundle)
            let loadEnd = ContinuousClock.now
            _ = try await own.generate(warmup, maxTokens: 8, stopTokens: [])
            try await measure(input, loadStart, loadEnd, own.loadStats) {
                own.reset()
            } generate: { onToken in
                let output = try await own.generate(input, maxTokens: maxTokens, stopTokens: [], onToken: onToken)
                return SpeedRecord.ownStopReason(generated: output.count, maxTokens: maxTokens)
            }
        } else {
            let official = try await EngineFactory.createEngine(
                bundle: LanguageModelBundle(at: url),
                options: EngineOptions(variant: engine == "sequential" ? "coreai-sequential" : nil))
            let loadEnd = ContinuousClock.now
            try await official.warmup(queryLength: 1, sampling: .greedy)
            _ = try await drain(official, warmup, 8) { _ in }
            try await measure(input, loadStart, loadEnd, nil) {
                try await official.reset(to: 0)
            } generate: {
                try await SpeedRecord.officialStopReason(drain(official, input, maxTokens, $0))
            }
        }
    }

    func measure(
        _ input: [Int32], _ loadStart: ContinuousClock.Instant, _ loadEnd: ContinuousClock.Instant,
        _ loadStats: CoreAIBackend.Engine.LoadStats?, reset: () async throws -> Void,
        generate: ((Int32) -> Void) async throws -> String
    ) async throws {
        for index in 1...repeatCount {
            try await reset()
            let start = ContinuousClock.now
            var first = start
            var last = start
            var generated = 0
            let stopReason = try await generate { _ in
                last = ContinuousClock.now
                if generated == 0 { first = last }
                generated += 1
            }
            var usage = rusage()
            let peakRssBytes: UInt64? = getrusage(RUSAGE_SELF, &usage) == 0 ? UInt64(usage.ru_maxrss) : nil
            let prefillS = (first - start) / .seconds(1)
            let decodeS = (last - first) / .seconds(1)
            let initial = index == 1
            try emit(
                SpeedRecord(
                    engine: engine, round: round, repeat: index,
                    loadS: initial ? (loadEnd - loadStart) / .seconds(1) : 0,
                    warmupS: initial ? (start - loadEnd) / .seconds(1) : 0,
                    specializeS: (loadStats?.specializeSeconds).map { initial ? $0 : 0 }, cached: loadStats?.cached,
                    promptTokens: input.count, prefillS: prefillS, prefillTokS: Double(input.count) / prefillS,
                    maxTokens: maxTokens, generatedTokens: generated, stopReason: stopReason, decodeS: decodeS,
                    decodeTokS: SpeedRecord.decodeRate(generated: generated, seconds: decodeS),
                    requestedCompute: loadStats?.requestedComputeUnit,
                    expectFrequentReshapes: loadStats?.expectsFrequentReshapes, peakRssBytes: peakRssBytes,
                    rssBytes: Memory.now().rssBytes))
        }
    }

    func drain(_ engine: some InferenceEngine, _ input: [Int32], _ count: Int, _ onToken: (Int32) -> Void) async throws
        -> StopReason?
    {
        let sequence = try await engine.generate(
            with: input, samplingConfiguration: .greedy, inferenceOptions: InferenceOptions(maxTokens: count))
        for try await output in sequence { onToken(output.tokenId) }
        return sequence.stopReason
    }
}

enum OnOff: String, ExpressibleByArgument {
    case on
    case off
}

enum HistoryReasoning: String, ExpressibleByArgument {
    case keep
    case drop
}

enum ReasoningChoice: String, ExpressibleByArgument {
    case `default`
    case low
    case off
}

enum MeasureChoice: String, CaseIterable, ExpressibleByArgument {
    case off
    case summary
    case trace

    var level: GenerationMeasurement.Level {
        switch self {
        case .off: .off
        case .summary: .summary
        case .trace: .trace
        }
    }
}

struct ExperimentOptions: ParsableArguments {
    @Option(help: "Reuse the processed state across requests: on (the default) or off. Own runtime only.")
    var reuse: OnOff?

    @Option(
        help:
            "Reasoning of earlier assistant entries in the rebuilt history: keep (the default) or drop. Own runtime only."
    )
    var historyReasoning: HistoryReasoning?

    @Option(
        help: "An extra chat template variable as name=true or name=false. Repeat to set several. Own runtime only.")
    var templateVar: [String] = []

    @Option(help: "Reasoning level of every request: default, low or off. Own runtime only.")
    var reasoning: ReasoningChoice?

    @Flag(
        help:
            "After the last prompt, read the processed tokens again from a reset engine and compare the next-token logits. Own runtime only."
    )
    var checkState = false

    @Option(
        help: "Per-request measurement in chat turn records: off (the default), summary or trace. Own runtime only.")
    var measure: MeasureChoice?

    func validate() throws {
        if templateVar.contains(where: { Self.variable($0) == nil }) {
            throw ValidationError("--template-var takes name=true or name=false.")
        }
    }

    func validate(_ runtime: RuntimeKind) throws {
        if runtime != .own,
            reuse != nil || historyReasoning != nil || !templateVar.isEmpty || reasoning != nil || checkState
                || measure != nil
        {
            throw ValidationError(
                "--reuse, --history-reasoning, --template-var, --reasoning, --check-state and --measure apply to --runtime own only."
            )
        }
    }

    var settings: LocalLanguageModel.VerificationSettings {
        var settings = LocalLanguageModel.VerificationSettings()
        settings.reusesState = reuse != .off
        settings.keepsHistoryReasoning = historyReasoning != .drop
        settings.templateVariables = Dictionary(templateVar.compactMap(Self.variable)) { $1 }
        settings.measurement = measure?.level ?? .off
        return settings
    }

    var contextOptions: ContextOptions {
        switch reasoning {
        case .low?: ContextOptions(reasoningLevel: .light)
        case .off?: ContextOptions(reasoningLevel: .custom("off"))
        default: ContextOptions()
        }
    }

    static func variable(_ entry: String) -> (String, Bool)? {
        let parts = entry.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
        guard parts.count == 2, !parts[0].isEmpty, let value = Bool(String(parts[1])) else { return nil }
        return (String(parts[0]), value)
    }
}

func loadOwnModel(
    _ url: URL, contextLimit: Int?, overflow: LocalLanguageModel.OverflowPolicy, experiment: ExperimentOptions?,
    reloadPolicy: LocalLanguageModel.ReloadPolicy
) async throws -> any LanguageModel {
    let clock = ContinuousClock()
    let start = clock.now
    let model = try await LocalLanguageModel(
        bundle: url, contextLimit: contextLimit, overflow: overflow, reloadPolicy: reloadPolicy,
        verification: experiment?.settings ?? .init())
    try emit(
        OwnModelLoadRecord(
            cached: model.loadStats.cached, specializeS: model.loadStats.specializeSeconds,
            initS: (clock.now - start) / .seconds(1), requestedCompute: model.loadStats.requestedComputeUnit,
            expectFrequentReshapes: model.loadStats.expectsFrequentReshapes, memory: .now()))
    return model
}

func checkState(_ model: any LanguageModel) async throws {
    guard let own = model as? LocalLanguageModel else { return }
    let (tokens, logits) = await own.processedState()
    guard !tokens.isEmpty, logits.count > 1 else {
        throw ValidationError("The own runtime has no processed state to verify.")
    }
    guard !logits.isEmpty else {
        try emit(StateRecord(tokens: tokens.count, error: "No logits for the processed tokens."))
        return
    }
    let start = ContinuousClock.now
    let reread = try await own.resetAndRead(tokens)
    let check = StateCheck(current: logits, reread: reread)
    try emit(
        StateRecord(
            tokens: tokens.count, current: top2(logits), reread: top2(reread),
            sameTop: check.currentTop == check.rereadTop,
            marginTicks: check.marginTicks, maxAbsDiff: check.maxAbsDiff, pass: check.passed,
            rereadS: (ContinuousClock.now - start) / .seconds(1)))
}

func generatedTokenSnapshot(_ model: any LanguageModel) -> (@Sendable () async -> [Int32])? {
    guard let own = model as? LocalLanguageModel else { return nil }
    return { await own.lastGeneratedTokens() }
}

func measurementSnapshot(_ model: any LanguageModel, _ experiment: ExperimentOptions)
    -> (@Sendable () async -> GenerationMeasurement?)?
{
    guard experiment.settings.measurement != .off, let own = model as? LocalLanguageModel else { return nil }
    return { await own.lastRequestMeasurement() }
}

func tokenSnapshot(_ model: any LanguageModel) -> (@Sendable () async -> [Int32])? {
    guard let own = model as? LocalLanguageModel else { return nil }
    return { await own.processedTokens() }
}

func loadOwn(_ url: URL) async throws -> (any Tokenizer, CoreAIBackend.Engine) {
    let bundle = try ModelBundle(url: url)
    let tokenizer = try await bundle.loadTokenizer()
    let engine = try await CoreAIBackend.Engine(bundle: bundle)
    try emit(
        OwnLoadRecord(
            cached: engine.loadStats.cached, specializeS: engine.loadStats.specializeSeconds,
            loadS: engine.loadStats.totalSeconds, prefillFunction: engine.hasPrefillFunction, hybrid: engine.isHybrid,
            contextLimit: engine.contextLimit, requestedCompute: engine.loadStats.requestedComputeUnit,
            expectFrequentReshapes: engine.loadStats.expectsFrequentReshapes, memory: .now()))
    return (tokenizer, engine)
}

func chatInput(_ tokenizer: any Tokenizer, system: String? = nil, user: String, generationPrompt: Bool = true) throws
    -> [Int32]
{
    let messages: [Message] =
        (system.map { [["role": "system", "content": $0]] } ?? []) + [["role": "user", "content": user]]
    return try tokenizer.applyChatTemplate(
        messages: messages, chatTemplate: nil, addGenerationPrompt: generationPrompt, truncation: false, maxLength: nil,
        tools: nil,
        additionalContext: nil
    ).map(Int32.init)
}

func stopTokens(_ tokenizer: any Tokenizer) -> Set<Int32> {
    Set([tokenizer.eosTokenId].compactMap { $0.map(Int32.init) })
}

func top2(_ logits: [Float16]) -> [Candidate] {
    var ranked: [(Int, Float)] = [(-1, -.infinity), (-1, -.infinity)]
    for (index, value) in logits.enumerated() where Float(value) > ranked[1].1 {
        ranked[1] = (index, Float(value))
        if ranked[1].1 > ranked[0].1 { ranked.swapAt(0, 1) }
    }
    return ranked.map { Candidate(token: Int32($0.0), logit: $0.1) }
}

extension JSONDecoder {
    static var snake: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }
}

struct OwnLoadRecord: Encodable {
    var record = "own-load"
    let cached: Bool
    let specializeS: Double
    let loadS: Double
    let prefillFunction: Bool
    let hybrid: Bool
    let contextLimit: Int
    let requestedCompute: String?
    let expectFrequentReshapes: Bool?
    var observedCompute = "unknown"
    let memory: Memory
}

struct OwnModelLoadRecord: Encodable {
    var record = "load"
    var runtime = "own"
    let cached: Bool
    let specializeS: Double
    let initS: Double
    let requestedCompute: String?
    let expectFrequentReshapes: Bool?
    var observedCompute = "unknown"
    let memory: Memory
}

struct OfficialLoadRecord: Encodable {
    var record = "official-load"
    let engine: String
    let loadS: Double
    let memory: Memory
}

struct Candidate: Encodable {
    let token: Int32
    let logit: Float
}

struct Mismatch: Encodable {
    let own: [Candidate]
    let official: [Candidate]
}

struct CompareRecord: Encodable {
    var record = "compare"
    var prompt = ""
    let prefill: String
    let promptTokens: Int
    var match = false
    var stop = "max"
    var firstMismatch: Int?
    var mismatch: Mismatch?
    var maxAbsDiff: Float = 0
    var identicalSteps = 0
    var tokens: [Int32] = []
    var officialTokens: [Int32] = []
    var text = ""
    var ownNonFiniteLogitCount = 0
    var officialNonFiniteLogitCount = 0
    var error: String?

    mutating func observeLogits(own: [Float16], official: [Float16]) -> Bool {
        switch LogitComparison(own: own, official: official) {
        case .finite(let identical, let difference):
            maxAbsDiff = max(maxAbsDiff, difference)
            identicalSteps += identical ? 1 : 0
            return true
        case .nonFinite(let ownCount, let officialCount):
            ownNonFiniteLogitCount += ownCount
            officialNonFiniteLogitCount += officialCount
            stop = "non-finite"
            error = "Non-finite logits: own=\(ownNonFiniteLogitCount), official=\(officialNonFiniteLogitCount)."
            return false
        }
    }
}

func emitComparison(_ record: CompareRecord) throws {
    try emit(record)
    if let error = record.error { throw ComparisonFailure(description: error) }
}

struct ComparisonFailure: Error, CustomStringConvertible {
    let description: String
}

struct StateRecord: Encodable {
    var record = "state-check"
    let tokens: Int
    var current: [Candidate] = []
    var reread: [Candidate] = []
    var sameTop: Bool?
    var marginTicks: Int?
    var maxAbsDiff: Float?
    var pass: Bool?
    var rereadS: Double?
    var error: String?
}

struct PointRecord: Encodable {
    var record = "point"
    let prefixTokens: Int
    let prefillS: Double
    let saveS: Double
    let stateBytes: Int
    let hybrid: Bool
    let kvCapacity: Int
    let memory: Memory
}

struct RewindRecord: Encodable {
    var record = "rewind"
    let kind: String
    let trial: Int
    let question: Int
    let restoreS: Double?
    let inputTokens: Int
    let prefillS: Double
    let matchesFirst: Bool
    let tokens: [Int32]
    let text: String?
}

struct SpeedRecord: Codable {
    var record = "speed"
    let engine: String
    let round: Int
    var `repeat` = 1
    let loadS: Double
    let warmupS: Double
    let specializeS: Double?
    let cached: Bool?
    let promptTokens: Int
    let prefillS: Double
    let prefillTokS: Double
    let maxTokens: Int
    let generatedTokens: Int
    let stopReason: String
    let decodeS: Double
    let decodeTokS: Double?
    let requestedCompute: String?
    let expectFrequentReshapes: Bool?
    var observedCompute = "unknown"
    let peakRssBytes: UInt64?
    let rssBytes: UInt64?

    static func decodeRate(generated: Int, seconds: Double) -> Double? {
        generated < 2 ? nil : Double(generated - 1) / seconds
    }

    static func ownStopReason(generated: Int, maxTokens: Int) -> String {
        generated < maxTokens ? "stopToken" : "maxTokens"
    }

    static func officialStopReason(_ reason: StopReason?) -> String {
        switch reason {
        case .maxTokens?: "maxTokens"
        case .eos?: "stopToken"
        case .stopSequence?: "stopSequence"
        case .cancelled?: "cancelled"
        case .error?: "error"
        case nil: "unknown"
        @unknown default: "unknown"
        }
    }
}

extension SpeedRecord {
    init(from decoder: any Swift.Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        record = try values.decode(String.self, forKey: .record)
        engine = try values.decode(String.self, forKey: .engine)
        round = try values.decode(Int.self, forKey: .round)
        self.repeat = try values.decodeIfPresent(Int.self, forKey: .repeat) ?? 1
        loadS = try values.decode(Double.self, forKey: .loadS)
        warmupS = try values.decode(Double.self, forKey: .warmupS)
        specializeS = try values.decodeIfPresent(Double.self, forKey: .specializeS)
        cached = try values.decodeIfPresent(Bool.self, forKey: .cached)
        promptTokens = try values.decode(Int.self, forKey: .promptTokens)
        prefillS = try values.decode(Double.self, forKey: .prefillS)
        prefillTokS = try values.decode(Double.self, forKey: .prefillTokS)
        maxTokens = try values.decode(Int.self, forKey: .maxTokens)
        generatedTokens = try values.decode(Int.self, forKey: .generatedTokens)
        stopReason = try values.decode(String.self, forKey: .stopReason)
        decodeS = try values.decode(Double.self, forKey: .decodeS)
        decodeTokS = try values.decodeIfPresent(Double.self, forKey: .decodeTokS)
        requestedCompute = try values.decodeIfPresent(String.self, forKey: .requestedCompute)
        expectFrequentReshapes = try values.decodeIfPresent(Bool.self, forKey: .expectFrequentReshapes)
        observedCompute = try values.decode(String.self, forKey: .observedCompute)
        peakRssBytes = try values.decodeIfPresent(UInt64.self, forKey: .peakRssBytes)
        rssBytes = try values.decodeIfPresent(UInt64.self, forKey: .rssBytes)
    }
}

struct SpeedSummary: Encodable {
    var record = "speed-summary"
    let engine: String
    let decodeTokS: Double?
    let decodeRuns: Int
    let prefillTokS: Double?
    let peakRssBytes: UInt64?
    let peakRssRuns: Int
    let ownDecodeRatio: Double?
    let ownPrefillRatio: Double?
}

extension SpeedSummary {
    init(engine: String, runs: [SpeedRecord], own: [SpeedRecord]) {
        let decode = runs.compactMap(\.decodeTokS)
        let peaks = runs.compactMap(\.peakRssBytes)
        let decodeTokS = Self.median(decode)
        let prefillTokS = Self.median(runs.map(\.prefillTokS))
        self.init(
            engine: engine, decodeTokS: decodeTokS, decodeRuns: decode.count, prefillTokS: prefillTokS,
            peakRssBytes: peaks.max(), peakRssRuns: peaks.count,
            ownDecodeRatio: Self.ratio(Self.median(own.compactMap(\.decodeTokS)), decodeTokS),
            ownPrefillRatio: Self.ratio(Self.median(own.map(\.prefillTokS)), prefillTokS))
    }

    static func median(_ values: [Double]) -> Double? {
        let sorted = values.sorted()
        return sorted.isEmpty ? nil : (sorted[(sorted.count - 1) / 2] + sorted[sorted.count / 2]) / 2
    }

    static func ratio(_ numerator: Double?, _ denominator: Double?) -> Double? {
        guard let numerator, let denominator else { return nil }
        return numerator / denominator
    }
}
