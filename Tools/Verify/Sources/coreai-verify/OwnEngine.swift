import ArgumentParser
import CoreAIBackend
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
        abstract: "Feed the same tokens to the own engine and the official sequential engine; compare greedy tokens and logits."
    )

    @Option(help: "Model bundle folder.")
    var model: String

    @Option(help: "A prompt as name=text. Repeat to send several.")
    var prompt: [String] = []

    @Option(help: "Prompt tokens as name=path, path being a JSON array of token IDs; append @n to prefill the own engine in two parts split at n.")
    var tokens: [String] = []

    @Option(help: "Maximum generated tokens per prompt.")
    var maxTokens = 1024

    @Option(help: "Text file sent as the system message before each prompt.")
    var system: String?

    @Flag(help: "Prefill the own engine in two parts split at the end of the user turn, as the own runtime does for hybrid models.")
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
                try emit(record)
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
                own: own, official: official, input: input, usePrefillFunction: true, split: split, maxTokens: maxTokens, stop: stop)
            record.prompt = name
            record.text = tokenizer.decode(tokens: record.tokens.map(Int.init))
            try emit(record)
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
            let token = own.greedyToken()
            let referenceLogits = reference.logits ?? []
            record.tokens.append(token)
            record.officialTokens.append(reference.tokenId)
            var identical = logits.count == referenceLogits.count
            var difference: Float = 0
            for (a, b) in zip(logits, referenceLogits) where a.bitPattern != b.bitPattern {
                identical = false
                difference = max(difference, abs(Float(a) - Float(b)))
            }
            record.maxAbsDiff = max(record.maxAbsDiff, difference)
            record.identicalSteps += identical ? 1 : 0
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
        abstract: "Save a point after a shared preamble, generate, restore, and generate again; time a full re-read for comparison."
    )

    @Option(help: "Model bundle folder.")
    var model: String

    @Option(help: "Text file sent as the system message: the shared preamble.")
    var preamble: String

    @Option(help: "A user question that follows the preamble. Repeat to send two or more; the shared tokens form the saved point.")
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
        try emit(PointRecord(
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
                let output = try await engine.generate(Array(input[prefix.count...]), maxTokens: maxTokens, stopTokens: [])
                if trial == 1 { first[index] = output }
                try emit(RewindRecord(
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
            try emit(RewindRecord(
                kind: "fresh", trial: 1, question: index, restoreS: nil, inputTokens: input.count - prefix.count,
                prefillS: engine.stats.prefillSeconds, matchesFirst: output == first[index], tokens: output, text: nil))
        }
        for trial in 1...trials {
            for (index, input) in inputs.enumerated() {
                engine.reset()
                engine.stats = .init()
                let output = try await engine.generate(input, maxTokens: maxTokens, stopTokens: [])
                try emit(RewindRecord(
                    kind: "full", trial: trial, question: index, restoreS: nil, inputTokens: input.count,
                    prefillS: engine.stats.prefillSeconds, matchesFirst: output == first[index], tokens: output, text: nil))
            }
        }
    }
}

struct Speed: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Alternate the own engine, the official sequential engine, and the official default engine, one process each."
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
            let runs = records.filter { $0.engine == name }
            try emit(SpeedSummary(
                engine: name,
                decodeTokS: median(runs.map(\.decodeTokS)),
                prefillTokS: median(runs.map(\.prefillTokS)),
                peakRssBytes: runs.map(\.peakRssBytes).max() ?? 0,
                ownDecodeRatio: median(own.map(\.decodeTokS)) / median(runs.map(\.decodeTokS)),
                ownPrefillRatio: median(own.map(\.prefillTokS)) / median(runs.map(\.prefillTokS))))
        }
    }

    func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        return sorted.isEmpty ? .nan : (sorted[(sorted.count - 1) / 2] + sorted[sorted.count / 2]) / 2
    }
}

struct SpeedRun: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "speed-run", shouldDisplay: false)

    @Option var model: String
    @Option var preamble: String
    @Option var engine: String
    @Option var maxTokens: Int
    @Option var round: Int

    func run() async throws {
        let url = URL(filePath: model)
        let bundle = try ModelBundle(url: url)
        let tokenizer = try await bundle.loadTokenizer()
        let input = try chatInput(
            tokenizer, system: String(contentsOfFile: preamble, encoding: .utf8), user: rewindQuestions[0])
        let warmup = Array(input.prefix(16))
        let clock = ContinuousClock()
        let loadStart = clock.now
        var first = loadStart
        var last = loadStart
        var start = loadStart
        var loadEnd = loadStart
        if engine == "own" {
            let own = try await CoreAIBackend.Engine(bundle: bundle)
            loadEnd = clock.now
            _ = try await own.generate(warmup, maxTokens: 8, stopTokens: [])
            own.reset()
            start = clock.now
            _ = try await own.generate(input, maxTokens: maxTokens, stopTokens: []) { _ in
                last = clock.now
                if first == loadStart { first = last }
            }
        } else {
            let official = try await EngineFactory.createEngine(
                bundle: LanguageModelBundle(at: url),
                options: EngineOptions(variant: engine == "sequential" ? "coreai-sequential" : nil))
            try await official.warmup(queryLength: 1, sampling: .greedy)
            loadEnd = clock.now
            try await drain(official, warmup, 8) { _ in }
            try await official.reset(to: 0)
            start = clock.now
            try await drain(official, input, maxTokens) { _ in
                last = clock.now
                if first == loadStart { first = last }
            }
        }
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)
        try emit(SpeedRecord(
            engine: engine, round: round, loadS: (loadEnd - loadStart) / .seconds(1), promptTokens: input.count,
            prefillS: (first - start) / .seconds(1), prefillTokS: Double(input.count) / ((first - start) / .seconds(1)),
            decodeTokens: maxTokens, decodeS: (last - first) / .seconds(1),
            decodeTokS: Double(maxTokens - 1) / ((last - first) / .seconds(1)),
            peakRssBytes: UInt64(usage.ru_maxrss), rssBytes: Memory.now().rssBytes))
    }

    func drain(_ engine: some InferenceEngine, _ input: [Int32], _ count: Int, _ onToken: (Int32) -> Void) async throws {
        let sequence = try await engine.generate(
            with: input, samplingConfiguration: .greedy, inferenceOptions: InferenceOptions(maxTokens: count))
        for try await output in sequence { onToken(output.tokenId) }
    }
}

func loadOwnModel(_ url: URL) async throws -> any LanguageModel {
    let clock = ContinuousClock()
    let start = clock.now
    let model = try await LocalLanguageModel(bundle: url)
    try emit(OwnModelLoadRecord(
        cached: model.loadStats.cached, specializeS: model.loadStats.specializeSeconds,
        initS: (clock.now - start) / .seconds(1), memory: .now()))
    return model
}

func tokenSnapshot(_ model: any LanguageModel) -> (@Sendable () async -> [Int32])? {
    guard let own = model as? LocalLanguageModel else { return nil }
    return { await own.processedTokens() }
}

func loadOwn(_ url: URL) async throws -> (any Tokenizer, CoreAIBackend.Engine) {
    let bundle = try ModelBundle(url: url)
    let tokenizer = try await bundle.loadTokenizer()
    let engine = try await CoreAIBackend.Engine(bundle: bundle)
    try emit(OwnLoadRecord(
        cached: engine.loadStats.cached, specializeS: engine.loadStats.specializeSeconds,
        loadS: engine.loadStats.totalSeconds, prefillFunction: engine.hasPrefillFunction, hybrid: engine.isHybrid,
        contextLimit: engine.contextLimit, memory: .now()))
    return (tokenizer, engine)
}

func chatInput(_ tokenizer: any Tokenizer, system: String? = nil, user: String, generationPrompt: Bool = true) throws -> [Int32] {
    let messages: [Message] = (system.map { [["role": "system", "content": $0]] } ?? []) + [["role": "user", "content": user]]
    return try tokenizer.applyChatTemplate(
        messages: messages, chatTemplate: nil, addGenerationPrompt: generationPrompt, truncation: false, maxLength: nil, tools: nil,
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
    let memory: Memory
}

struct OwnModelLoadRecord: Encodable {
    var record = "load"
    var runtime = "own"
    let cached: Bool
    let specializeS: Double
    let initS: Double
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
    let loadS: Double
    let promptTokens: Int
    let prefillS: Double
    let prefillTokS: Double
    let decodeTokens: Int
    let decodeS: Double
    let decodeTokS: Double
    let peakRssBytes: UInt64
    let rssBytes: UInt64
}

struct SpeedSummary: Encodable {
    var record = "speed-summary"
    let engine: String
    let decodeTokS: Double
    let prefillTokS: Double
    let peakRssBytes: UInt64
    let ownDecodeRatio: Double
    let ownPrefillRatio: Double
}
