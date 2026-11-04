import ArgumentParser
import CoreAI
import CoreAIBackend
import CryptoKit
import Darwin
import Foundation

struct TypedDecide: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "typed-decide",
        abstract: "Answer typed questions with a typed-decision bundle on Core AI and write one JSON document."
    )

    @Option(help: "Typed-decision bundle folder.")
    var bundle: String

    @Option(help: "Input JSON: {\"rows\": [{\"id\", \"kind\", \"state\", \"question\", \"options\"}]}.")
    var input: String

    @Option(help: "Output JSON file. It must not exist yet.")
    var output: String

    @Option(help: "Run only the first N rows.")
    var limit: Int?

    @Flag(help: "Delete the specialization cache entry of the bundle's asset before loading.")
    var clearCache = false

    @Flag(help: "Sample the process memory every 100 ms while each row runs and record the largest RSS and footprint.")
    var sampleMemory = false

    @Option(help: "Before every row, wait up to this many seconds for a nominal thermal state and record the wait.")
    var waitNominalThermal: Int?

    func validate() throws {
        if let limit, limit <= 0 { throw ValidationError("--limit takes a positive value.") }
        if let waitNominalThermal, waitNominalThermal < 0 {
            throw ValidationError("--wait-nominal-thermal takes zero or more seconds.")
        }
    }

    func run() async throws {
        let started = Date.now
        let outputURL = URL(filePath: output)
        var isFolder: ObjCBool = false
        guard !FileManager.default.fileExists(atPath: outputURL.path),
            FileManager.default.fileExists(atPath: outputURL.deletingLastPathComponent().path, isDirectory: &isFolder),
            isFolder.boolValue
        else { throw ValidationError("The output file must not exist yet, and its folder must exist: \(output)") }
        let data = try Data(contentsOf: URL(filePath: input))
        let rows = try JSONDecoder().decode(TypedDecideInput.self, from: data).rows.prefix(limit ?? .max)
        let bundleURL = URL(filePath: bundle)
        if clearCache { try AIModelCache.default.deleteEntries(for: TypedDecisionBundle(url: bundleURL).assetURL) }
        let model = try await TypedDecisionModel.load(bundle: bundleURL)
        let bundleRecord = try TypedDecideBundleRecord(model)
        let clock = ContinuousClock()
        var records: [TypedDecideRecord] = []
        for (index, row) in rows.enumerated() {
            var thermal: ThermalStart?
            if let waitNominalThermal {
                thermal = try await ThermalGate(limit: .seconds(waitNominalThermal)).wait()
            }
            let rowStarted = Date.now
            let start = clock.now
            var record: TypedDecideRecord
            if sampleMemory {
                let sampled = try await sampleResident { try await Self.decide(row, model: model) }
                record = sampled.value
                record.residentBytesMax = sampled.resident
            } else {
                record = try await Self.decide(row, model: model)
            }
            if let thermal {
                record.setStart(thermal, at: rowStarted)
            }
            records.append(record)
            let summary =
                (record.error ?? "\(record.tokenCount ?? 0) tokens")
                + (thermal.map { " thermal \($0.stateName) after \($0.waitedSeconds) s" } ?? "")
            FileHandle.standardError.write(
                Data("[\(index + 1)/\(rows.count)] \(row.id) \(summary) \(seconds(clock.now - start)) s\n".utf8))
        }
        let document = TypedDecideDocument(
            bundle: bundleRecord, inputSha256: hex(Data(SHA256.hash(data: data))), started: started,
            host: TypedDecideHost(), rows: records)
        try JSONEncoder.records.encode(document).write(to: outputURL, options: .withoutOverwriting)
    }

    static func decide(_ row: TypedDecideInput.Row, model: TypedDecisionModel) async throws -> TypedDecideRecord {
        let request = TypedDecisionRequest(
            state: row.state, question: row.question, kind: row.kind, options: row.options)
        let ids: [Int]
        do {
            ids = try await model.tokenIDs(for: request)
        } catch TypedDecisionError.tooManyTokens(let count, _) {
            return TypedDecideRecord(id: row.id, kind: row.kind, error: "tooManyTokens", tokenCount: count)
        } catch TypedDecisionError.invalidOptions {
            return TypedDecideRecord(id: row.id, kind: row.kind, error: "invalidOptions")
        }
        let clock = ContinuousClock()
        let start = clock.now
        let answer = try await model.decide(request)
        let elapsed = seconds(clock.now - start)
        guard answer.tokenCount == ids.count else {
            throw ValidationError("\(row.id): the answer used \(answer.tokenCount) tokens, not \(ids.count).")
        }
        return TypedDecideRecord(
            id: row.id, kind: row.kind, tokenCount: ids.count, tokensSha256: tokensDigest(ids),
            capacity: model.capacity(forTokens: ids.count), logits: answer.logits.map(Double.init),
            probabilities: answer.probabilities.map(Double.init),
            probabilitiesT1: answer.probabilitiesT1.map(Double.init), top: answer.top, elapsedSeconds: elapsed)
    }
}

func tokensDigest(_ ids: [Int]) -> String {
    hex(Data(SHA256.hash(data: Data(ids.map(String.init).joined(separator: ",").utf8))))
}

struct TypedDecideInput: Decodable {
    struct Row: Decodable {
        let id: String
        let kind: TypedDecisionKind
        let state: String
        let question: String
        let options: [String]
    }

    let rows: [Row]
}

struct TypedDecideDocument: Encodable {
    var schema = "typed-decide/1"
    let bundle: TypedDecideBundleRecord
    let inputSha256: String
    let started: Date
    let host: TypedDecideHost
    let rows: [TypedDecideRecord]
}

struct TypedDecideBundleRecord: Encodable {
    struct LoadSeconds: Encodable {
        let bundle: Double
        let tokenizer: Double
        let head: Double
        let backbone: Double
    }

    let name: String
    let version: String
    let assetSha256: String
    let specialization: String
    let loadSeconds: LoadSeconds

    init(_ model: TypedDecisionModel) throws {
        let bundle = model.bundle
        let asset = Array((String(bundle.assetURL.path.dropFirst(bundle.url.path.count + 1)) + "/main.mlirb").utf8)
        let sums = try String(contentsOf: bundle.url.appending(path: "SHA256SUMS"), encoding: .utf8)
        guard let line = sums.split(separator: "\n").first(where: { Array($0.utf8.dropFirst(66)) == asset }) else {
            throw ValidationError("SHA256SUMS does not list the asset's main.mlirb.")
        }
        let stats = model.loadStats
        name = bundle.name
        version = bundle.version
        assetSha256 = String(line.prefix(64))
        specialization = stats.cachedSpecialization ? "cached" : "fresh"
        loadSeconds = LoadSeconds(
            bundle: stats.bundleSeconds, tokenizer: stats.tokenizerSeconds, head: stats.headSeconds,
            backbone: stats.backboneSeconds)
    }
}

struct TypedDecideHost: Encodable {
    var osBuild = Self.sysctl("kern.osversion")
    var machine = Self.sysctl("hw.model")

    static func sysctl(_ name: String) -> String {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "" }
        var buffer = [UInt8](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return "" }
        return String(decoding: buffer.prefix { $0 != 0 }, as: UTF8.self)
    }
}

struct TypedDecideRecord: Encodable {
    let id: String
    let kind: TypedDecisionKind
    var error: String?
    var tokenCount: Int?
    var tokensSha256: String?
    var capacity: Int?
    var logits: [Double]?
    var probabilities: [Double]?
    var probabilitiesT1: [Double]?
    var top: Int?
    var elapsedSeconds: Double?
    var residentBytesMax: ResidentMaximum?
    var started: Date?
    var thermalWaitSeconds: Double?
    var thermalStateAtStart: String?
}
