import ArgumentParser
@_spi(Verification) import CoreAIBackend
import CryptoKit
import Foundation

extension CoreAIBackend.Engine: LogitsSource {}

struct Evaluate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "evaluate",
        abstract:
            "Generate greedily from the token sequences of input records with the own engine, and write the output records and the run record of the quality contract."
    )

    @Option(help: "Model bundle folder.")
    var model: String

    @Option(help: "Input records, one JSON object per line with id, input_ids, input_sha256 and max_new_tokens.")
    var inputs: String

    @Option(help: "The contract.json of the quality contract.")
    var contract: String

    @Option(help: "Name of the evaluation set, such as primary.")
    var set: String

    @Option(help: "Role of this run, one of the keys of roles in the contract.")
    var role: String

    @Option(help: "Repository commit written to run.json. The command does not run git.")
    var commit: String

    @Option(help: "Run folder: a new folder inside an existing folder, or the folder of the same run to resume.")
    var output: String

    @Option(help: "After this many minutes, end once the item in progress is written.")
    var maxMinutes: Double?

    @Option(help: "Run only the first N input records.")
    var limit: Int?

    func validate() throws {
        if let limit, limit <= 0 { throw ValidationError("--limit takes a positive value.") }
        if let maxMinutes, !(maxMinutes > 0 && maxMinutes.isFinite) {
            throw ValidationError("--max-minutes takes a positive number of minutes.")
        }
    }

    func run() async throws {
        let started = ContinuousClock.now
        let contractData = try Data(contentsOf: URL(filePath: contract))
        let terms = try JSONDecoder.snake.decode(EvaluationContract.self, from: contractData)
        guard terms.roles[role] != nil else {
            throw ValidationError("--role must be one of \(terms.roles.keys.sorted().joined(separator: ", ")).")
        }
        let inputData = try Data(contentsOf: URL(filePath: inputs))
        let bundle = try ModelBundle(url: URL(filePath: model))
        guard let mainHash = bundle.mainHash else {
            throw ValidationError("The bundle has no main.hash, so the run record could not name the asset.")
        }
        guard bundle.vocabSize >= 2 else { throw ValidationError("language.vocab_size must be at least 2.") }
        let items = try EvaluationInput.read(inputData, vocabulary: bundle.vocabSize)
        let folder = URL(filePath: output)
        _ = try Evaluation.stored(in: folder)
        let engine = try await CoreAIBackend.Engine(bundle: bundle)
        let stats = engine.loadStats
        let record = EvaluationRun(
            contract: .init(version: terms.version, sha256: hex(Data(SHA256.hash(data: contractData)))), set: set,
            role: role, asset: .init(name: bundle.name, mainHash: mainHash, sourceModelId: bundle.sourceModelID),
            implementation: .init(commit: commit),
            callShape: .init(
                prefillChunk: engine.prefillChunkSize, prefillFunction: engine.hasPrefillFunction,
                expectFrequentReshapes: stats.expectsFrequentReshapes, requestedCompute: stats.requestedComputeUnit),
            inputs: .init(sha256: hex(Data(SHA256.hash(data: inputData))), count: items.count), items: 0,
            started: .now,
            load: .init(cached: stats.cached, specializeS: stats.specializeSeconds, loadS: stats.totalSeconds))
        _ = try await Evaluation.run(
            items, limit: limit, in: folder, record: record, source: engine,
            stopTokens: Set(terms.generation.eosTokenId), started: started, maxMinutes: maxMinutes)
    }
}

struct EvaluationContract: Decodable {
    struct Generation: Decodable {
        let eosTokenId: [Int32]
    }

    let version: String
    let roles: [String: String]
    let generation: Generation
}

struct EvaluationRun: Codable, Equatable {
    struct Contract: Codable, Equatable {
        let version: String
        let sha256: String
    }

    struct Asset: Codable, Equatable {
        let name: String
        let mainHash: String
        let sourceModelId: String?
    }

    struct Implementation: Codable, Equatable {
        var command = "coreai-verify evaluate"
        let commit: String
        var osVersion = ProcessInfo.processInfo.operatingSystemVersion.text
        var osBuild = TypedDecideHost.sysctl("kern.osversion")
    }

    struct CallShape: Codable, Equatable {
        let prefillChunk: Int
        let prefillFunction: Bool
        let expectFrequentReshapes: Bool?
        let requestedCompute: String?
    }

    struct Inputs: Codable, Equatable {
        let sha256: String
        let count: Int
    }

    struct Load: Codable, Equatable {
        let cached: Bool
        let specializeS: Double
        let loadS: Double
    }

    let contract: Contract
    let set: String
    let role: String
    let asset: Asset
    let implementation: Implementation
    let callShape: CallShape
    let inputs: Inputs
    var items: Int
    let started: Date
    var finished: Date?
    let load: Load

    func differences(from stored: EvaluationRun) -> [String] {
        [
            ("contract", contract == stored.contract), ("set", set == stored.set), ("role", role == stored.role),
            ("asset.main_hash", asset.mainHash == stored.asset.mainHash), ("inputs", inputs == stored.inputs),
            ("call_shape", callShape == stored.callShape), ("implementation", implementation == stored.implementation),
        ].filter { !$0.1 }.map(\.0)
    }
}

extension OperatingSystemVersion {
    var text: String { "\(majorVersion).\(minorVersion).\(patchVersion)" }
}

struct EvaluationProgress: Encodable {
    var record = "evaluate"
    let id: String
    let inputTokens: Int
    let outputTokens: Int
    let stop: String
    let seconds: Double
}

enum Evaluation {
    private struct WrittenID: Decodable {
        let id: String
    }

    static func stored(in folder: URL) throws -> EvaluationRun? {
        var isFolder: ObjCBool = false
        guard FileManager.default.fileExists(atPath: folder.path, isDirectory: &isFolder) else {
            let parent = folder.deletingLastPathComponent()
            guard FileManager.default.fileExists(atPath: parent.path, isDirectory: &isFolder), isFolder.boolValue else {
                throw ValidationError("The folder that holds --output must exist: \(parent.path)")
            }
            return nil
        }
        let runURL = folder.appending(path: "run.json")
        guard isFolder.boolValue, FileManager.default.fileExists(atPath: runURL.path) else {
            throw ValidationError("--output exists but is not a run folder with run.json: \(folder.path)")
        }
        let decoder = JSONDecoder.snake
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(EvaluationRun.self, from: Data(contentsOf: runURL))
    }

    static func run(
        _ items: [EvaluationInput], limit: Int? = nil, in folder: URL, record: EvaluationRun,
        source: some LogitsSource, stopTokens: Set<Int32>, started: ContinuousClock.Instant = .now,
        maxMinutes: Double? = nil
    ) async throws -> EvaluationRun {
        let outputs = folder.appending(path: "outputs.jsonl")
        var current = record
        var done: Set<String> = []
        if let stored = try stored(in: folder) {
            let differing = record.differences(from: stored)
            guard differing.isEmpty else {
                throw ValidationError(
                    "--output holds another run; these differ from its run.json: \(differing.joined(separator: ", ")).")
            }
            done = try writtenIDs(in: outputs, known: Set(items.map(\.id)))
            current = stored
            current.items = done.count
            current.finished = nil
        } else {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
            try Data().write(to: outputs, options: .withoutOverwriting)
        }
        try write(current, in: folder)
        let file = try FileHandle(forWritingTo: outputs)
        defer { try? file.close() }
        try file.seekToEnd()
        for item in items.prefix(limit ?? items.count) where !done.contains(item.id) {
            let itemStarted = ContinuousClock.now
            let output = try await item.generate(with: source, stopTokens: stopTokens)
            try file.write(contentsOf: JSONEncoder.records.encode(output) + Data("\n".utf8))
            try file.synchronize()
            current.items += 1
            try emit(
                EvaluationProgress(
                    id: output.id, inputTokens: output.inputTokens, outputTokens: output.outputIds.count,
                    stop: output.error == nil ? output.stop : "error", seconds: seconds(.now - itemStarted)))
            if let maxMinutes, seconds(.now - started) >= maxMinutes * 60 { break }
        }
        current.finished = .now
        try write(current, in: folder)
        return current
    }

    static func writtenIDs(in outputs: URL, known: Set<String>) throws -> Set<String> {
        let newline = UInt8(ascii: "\n")
        var data = try Data(contentsOf: outputs)
        if let last = data.last, last != newline {
            let kept = data.lastIndex(of: newline).map { $0 + 1 } ?? 0
            let file = try FileHandle(forWritingTo: outputs)
            try file.truncate(atOffset: UInt64(kept))
            try file.close()
            data = data.prefix(kept)
        }
        let ids = try data.split(separator: newline).map { try JSONDecoder().decode(WrittenID.self, from: $0).id }
        let unknown = Set(ids).subtracting(known)
        guard unknown.isEmpty else {
            throw ValidationError("outputs.jsonl has ids that are not in the input records: \(unknown.sorted()).")
        }
        guard Set(ids).count == ids.count else { throw ValidationError("outputs.jsonl has an id more than once.") }
        return Set(ids)
    }

    static func write(_ record: EvaluationRun, in folder: URL) throws {
        let encoder = JSONEncoder.records
        encoder.outputFormatting.insert(.prettyPrinted)
        try encoder.encode(record).write(to: folder.appending(path: "run.json"), options: .atomic)
    }
}
