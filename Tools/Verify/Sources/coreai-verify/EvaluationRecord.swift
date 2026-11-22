import ArgumentParser
import Foundation

protocol LogitsSource {
    func reset()
    func prefill(_ input: [Int32], usePrefillFunction: Bool) async throws
    func step(_ token: Int32) async throws
    func logits() -> [Float16]
}

struct TopTwo: Encodable, Equatable {
    let first: Int32
    let firstProbability: Double
    let second: Int32
    let secondProbability: Double

    func encode(to encoder: any Encoder) throws {
        var entry = encoder.unkeyedContainer()
        try entry.encode(first)
        try entry.encode(firstProbability)
        try entry.encode(second)
        try entry.encode(secondProbability)
    }
}

enum LogitsRow: Equatable {
    case nonFinite(Int)
    case finite(TopTwo)

    init(_ logits: [Float16]) {
        let nonFinite = logits.lazy.filter { !$0.isFinite }.count
        guard nonFinite == 0 else {
            self = .nonFinite(nonFinite)
            return
        }
        var first = 0
        var second = -1
        for id in logits.indices.dropFirst() {
            if logits[id] > logits[first] {
                second = first
                first = id
            } else if second < 0 || logits[id] > logits[second] {
                second = id
            }
        }
        let maximum = Double(logits[first])
        let total = logits.reduce(0) { $0 + exp(Double($1) - maximum) }
        func probability(_ id: Int) -> Double { exp(Double(logits[id]) - maximum) / total }
        self = .finite(
            TopTwo(
                first: Int32(first), firstProbability: probability(first), second: Int32(second),
                secondProbability: probability(second)))
    }
}

struct OutputRecord: Encodable, Equatable {
    let id: String
    let inputSha256: String
    let inputTokens: Int
    var outputIds: [Int32] = []
    var stop = "limit"
    var stopToken: Int32?
    var top2: [TopTwo] = []
    var error: String?
    var nonFinite = 0
}

struct EvaluationInput: Decodable, Equatable {
    let id: String
    let inputIds: [Int32]
    let inputSha256: String
    let maxNewTokens: Int

    static func read(_ data: Data, vocabulary: Int) throws -> [EvaluationInput] {
        var items: [EvaluationInput] = []
        var seen: Set<String> = []
        for (index, line) in data.split(separator: UInt8(ascii: "\n")).enumerated() {
            let item: EvaluationInput
            do {
                item = try JSONDecoder.snake.decode(EvaluationInput.self, from: line)
            } catch {
                throw ValidationError("Input record \(index + 1) could not be read: \(error)")
            }
            guard seen.insert(item.id).inserted else {
                throw ValidationError("The input id \(item.id) appears more than once.")
            }
            guard !item.inputIds.isEmpty, item.inputIds.allSatisfy({ (0..<Int32(vocabulary)).contains($0) }) else {
                throw ValidationError(
                    "\(item.id): input_ids must hold at least one token id from 0 to \(vocabulary - 1).")
            }
            guard item.maxNewTokens > 0 else { throw ValidationError("\(item.id): max_new_tokens must be positive.") }
            guard tokensDigest(item.inputIds.map(Int.init)) == item.inputSha256 else {
                throw ValidationError("\(item.id): input_sha256 differs from the SHA-256 of input_ids.")
            }
            items.append(item)
        }
        guard !items.isEmpty else { throw ValidationError("The input file has no records.") }
        return items
    }

    func generate(with source: some LogitsSource, stopTokens: Set<Int32>) async throws -> OutputRecord {
        var record = OutputRecord(id: id, inputSha256: inputSha256, inputTokens: inputIds.count)
        source.reset()
        do {
            try await source.prefill(inputIds, usePrefillFunction: true)
            while true {
                switch LogitsRow(source.logits()) {
                case .nonFinite(let count):
                    record.nonFinite = count
                    record.error = "The logits had \(count) non-finite values at output position \(record.top2.count)."
                    return record
                case .finite(let top):
                    record.top2.append(top)
                    if stopTokens.contains(top.first) {
                        record.stop = "stop_token"
                        record.stopToken = top.first
                        return record
                    }
                    record.outputIds.append(top.first)
                    if record.outputIds.count >= maxNewTokens { return record }
                    try await source.step(top.first)
                }
            }
        } catch let error as CancellationError {
            throw error
        } catch {
            record.error = "\(type(of: error)): \(error)"
            return record
        }
    }
}
