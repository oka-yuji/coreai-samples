import Foundation
import GLiNERCore
import Testing

private let fixturesDirectory = URL(
    filePath: ProcessInfo.processInfo.environment["GLINER_FIXTURES_DIR"]
        ?? NSHomeDirectory() + "/Desktop/SelfApps/CoreLLM/models/coreai/gliner")
private let stageOutputsPath = "fixtures/g5_stage_outputs/stage_outputs_fp32.json"
private let swiftOutputPath = "work/g4_swift_out_fp32.json"
private let hasFixtures = [stageOutputsPath, swiftOutputPath].allSatisfy {
    FileManager.default.fileExists(atPath: fixturesDirectory.appending(path: $0).path(percentEncoded: false))
}

private struct StageOutputs: Decodable {
    struct Span: Decodable {
        let label: Int
        let start: Int
        let end: Int
        let probability: Float
    }

    struct Extraction: Decodable {
        let id: String
        let bucket: Int
        let nWords: Int
        let nq: Int
        let startMap: [Int]
        let endMap: [Int]
        let text: String
        let startLogits: [[Float]]
        let endLogits: [[Float]]
        let poolStart: [[Float]]
        let poolEnd: [[Float]]
        let nullLogits: [Float]
        let pairLogits: [[Float]]
        let candStart: [Int32]
        let candEnd: [Int32]
        let candMask: [Int32]
        let candPrior: [Float]
        let spans: [Span]
        let abstained: [Int]
    }

    struct Classification: Decodable {
        let id: String
        let bucket: Int
        let call: String
        let clsLogits: [Float]
        let probs: [Float]
    }

    let extractions: [Extraction]
    let classifications: [Classification]
}

private struct SwiftOutput: Decodable {
    struct Result: Decodable {
        let id: String
        let bucket: Int
        let call: String
        let candidates: [[Int]]?
        let probs: [Float]?
    }

    let results: [Result]
}

private func load<Value: Decodable>(_ type: Value.Type, _ path: String) throws -> Value {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(type, from: Data(contentsOf: fixturesDirectory.appending(path: path)))
}

private func padded(_ block: [[Float]], rows: Int, columns: Int, fill: Float = 100) -> Matrix {
    Matrix(
        rows: rows, columns: columns,
        values: (0..<rows).flatMap { row in
            (0..<columns).map { block.indices.contains(row) && block[row].indices.contains($0) ? block[row][$0] : fill }
        })
}

private func pool(of call: StageOutputs.Extraction) -> CandidatePool {
    let boundaries = call.bucket + 1
    let width = call.poolStart.first?.count ?? 0
    return CandidatePool(
        startLogits: padded(call.startLogits, rows: PreparedInput.labelCapacity, columns: boundaries),
        endLogits: padded(call.endLogits, rows: PreparedInput.labelCapacity, columns: boundaries),
        poolStart: padded(call.poolStart, rows: boundaries, columns: width),
        poolEnd: padded(call.poolEnd, rows: boundaries, columns: width),
        wordCount: call.nWords, labelCount: call.nq)
}

private func candidates(_ pool: CandidatePool) -> [[Int]] {
    pool.mask.indices.filter { pool.mask[$0] != 0 }.map { [Int(pool.starts[$0]), Int(pool.ends[$0])] }
}

@Suite(.enabled(if: hasFixtures, "GLINER_FIXTURES_DIR must hold the GL4 Swift output and the GL5 stage outputs"))
struct StageOutputParityTests {
    @Test("Candidate selection matches the candidate set and order of the GL4 Swift fixtures in all 160 calls")
    func candidatesMatchGL4() throws {
        let calls = try load(StageOutputs.self, stageOutputsPath).extractions
        let reference = try load(SwiftOutput.self, swiftOutputPath).results.filter { $0.call == "x" }
        let expected = Dictionary(uniqueKeysWithValues: reference.map { ("\($0.id) T=\($0.bucket)", $0.candidates) })
        try #require(calls.count == 160 && expected.count == 160)
        var mismatches: [String] = []
        for call in calls {
            let name = "\(call.id) T=\(call.bucket)"
            let pool = pool(of: call)
            if candidates(pool) != expected[name] ?? nil { mismatches.append("\(name): candidates") }
            if pool.starts != call.candStart || pool.ends != call.candEnd || pool.mask != call.candMask
                || pool.priors != call.candPrior
            {
                mismatches.append("\(name): pool arrays")
            }
        }
        #expect(mismatches.isEmpty, "\(mismatches.joined(separator: "\n"))")
    }

    @Test(
        "Decoding (threshold, abstention, overlap resolution, character offsets) matches the reference in all 160 calls"
    )
    func decodingMatchesReference() throws {
        let calls = try load(StageOutputs.self, stageOutputsPath).extractions
        try #require(calls.count == 160)
        var mismatches: [String] = []
        for call in calls {
            try #require(call.text.unicodeScalars.last == ".")
            let text = String(String.UnicodeScalarView(call.text.unicodeScalars.dropLast()))
            let words = zip(call.startMap, call.endMap).map { WordSpan(text: "", scalarRange: $0..<$1) }
            let input = try PreparedInput(
                text: text, schema: .entities((0..<call.nq).map { EntityLabel(name: "\($0)") }),
                tokenCapacities: [call.bucket], splitWords: { _ in words }, tokenize: { _ in [0] })
            let result = ExtractionResult(
                pairLogits: padded(call.pairLogits, rows: CandidatePool.size, columns: PreparedInput.labelCapacity),
                nullLogits: call.nullLogits + Array(repeating: 100, count: PreparedInput.labelCapacity - call.nq),
                pool: pool(of: call), input: input, text: text)
            let scalars = Array(text.unicodeScalars)
            let expected = ExtractionResult(
                spans: call.spans.map { span in
                    let range = min(span.start, scalars.count)..<min(span.end, scalars.count)
                    return ExtractedSpan(
                        labelIndex: span.label, scalarRange: range,
                        text: String(String.UnicodeScalarView(scalars[range])), probability: span.probability)
                },
                abstained: call.abstained, tokenCapacity: call.bucket)
            if result != expected { mismatches.append("\(call.id) T=\(call.bucket): \(result) != \(expected)") }
        }
        #expect(mismatches.isEmpty, "\(mismatches.joined(separator: "\n"))")
    }

    @Test("The classification softmax matches the reference probabilities within 1e-5 in all 258 calls")
    func softmaxMatchesReference() throws {
        let calls = try load(StageOutputs.self, stageOutputsPath).classifications
        let reference = try load(SwiftOutput.self, swiftOutputPath).results.filter { $0.call != "x" }
        let expected = Dictionary(
            uniqueKeysWithValues: reference.map { ("\($0.id) T=\($0.bucket) \($0.call)", $0.probs) })
        try #require(calls.count == 258 && expected.count == 258)
        var largest: Float = 0
        for call in calls {
            let probabilities = ClassificationResult(logits: call.clsLogits).probabilities
            let reference = try #require(expected["\(call.id) T=\(call.bucket) \(call.call)"] ?? nil)
            try #require(probabilities.count == reference.count && call.probs.count == reference.count)
            for (index, probability) in probabilities.enumerated() {
                largest = max(largest, abs(probability - reference[index]), abs(probability - call.probs[index]))
            }
        }
        print("GLiNER softmax: \(calls.count) calls, largest probability difference \(largest)")
        #expect(largest <= 1e-5, "largest difference \(largest)")
    }
}
