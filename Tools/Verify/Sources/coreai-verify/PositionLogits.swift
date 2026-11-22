import ArgumentParser
@_spi(Verification) import CoreAIBackend
import Foundation

struct PositionLogits: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "logits",
        abstract:
            "Read a token sequence with the own engine and write the logits of every position from a start position to a file."
    )

    @Option(help: "Model bundle folder.")
    var model: String

    @Option(help: "JSON array of token IDs.")
    var tokens: String

    @Option(
        help:
            "First position whose logits are written. The tokens before it are read but not written: with the prefill function when the bundle has one and the call shape rule keeps the prefill chunks, otherwise through main, which needs logits for a whole call."
    )
    var start: Int

    @Option(help: "Tokens per main call from the start position.")
    var chunk: Int

    @Option(
        help:
            "Output file of little-endian float16 logits shaped [positions, vocab]. It must not exist yet. The rows are written to the same path with .partial appended, which is renamed when the last row is written."
    )
    var output: String

    func validate() throws {
        guard chunk > 0 else { throw ValidationError("--chunk takes a positive value.") }
        try Self.validateTokens(try tokenIDs(), start: start)
        guard !FileManager.default.fileExists(atPath: output) else {
            throw ValidationError("The output file already exists: \(output)")
        }
    }

    static func validateTokens(_ ids: [Int32], start: Int) throws {
        guard !ids.isEmpty else { throw ValidationError("The token file has no tokens.") }
        guard ids.indices.contains(start) else {
            throw ValidationError("--start must be from 0 to \(ids.count - 1), a position in the token file.")
        }
    }

    func tokenIDs() throws -> [Int32] {
        try JSONDecoder().decode([Int32].self, from: Data(contentsOf: URL(filePath: tokens)))
    }

    static func pieces(_ range: Range<Int>, size: Int) -> [Range<Int>] {
        stride(from: range.lowerBound, to: range.upperBound, by: size).map {
            $0..<($0 + min(size, range.upperBound - $0))
        }
    }

    static func pieces(from lower: Int, lengths: [Int]) -> [Range<Int>] {
        var end = lower
        return lengths.map { length in
            end += length
            return (end - length)..<end
        }
    }

    static func prefixFunction(start: Int, lengths: [Int], chunkSize: Int, hasPrefillFunction: Bool) -> String? {
        guard start > 0 else { return nil }
        let unsplit = lengths == pieces(0..<start, size: chunkSize).map(\.count)
        return hasPrefillFunction && unsplit ? "prefill" : "main"
    }

    static func writeFile<Value>(at path: String, _ body: (URL) async throws -> Value) async throws -> Value {
        guard !FileManager.default.fileExists(atPath: path) else {
            throw ValidationError("The output file already exists: \(path)")
        }
        let partial = URL(filePath: path + ".partial")
        try Data().write(to: partial)
        do {
            let result = try await body(partial)
            try FileManager.default.moveItem(at: partial, to: URL(filePath: path))
            return result
        } catch {
            try? FileManager.default.removeItem(at: partial)
            throw error
        }
    }

    func run() async throws {
        let ids = try tokenIDs()
        try Self.validateTokens(ids, start: start)
        let bundle = try ModelBundle(url: URL(filePath: model))
        guard ids.count <= bundle.maxContextLength, ids.allSatisfy({ (0..<Int32(bundle.vocabSize)).contains($0) })
        else {
            throw ValidationError(
                "The token file must hold at most \(bundle.maxContextLength) IDs from 0 to \(bundle.vocabSize - 1).")
        }
        try emit(try await Self.writeFile(at: output) { try await write(ids, bundle: bundle, to: $0) })
    }

    func write(_ ids: [Int32], bundle: ModelBundle, to url: URL) async throws -> PositionLogitsRecord {
        let file = try FileHandle(forWritingTo: url)
        let clock = ContinuousClock()
        let loadStart = clock.now
        let engine = try await CoreAIBackend.Engine(bundle: bundle)
        let prefixStart = clock.now
        let lengths = engine.callLengths(from: 0, count: start)
        let prefixFunction = Self.prefixFunction(
            start: start, lengths: lengths, chunkSize: engine.prefillChunkSize,
            hasPrefillFunction: engine.hasPrefillFunction)
        let prefix = Self.pieces(from: 0, lengths: lengths)
        for piece in prefix {
            try await engine.readInOneCall(Array(ids[piece]), usePrefillFunction: prefixFunction == "prefill")
        }
        let scoringStart = clock.now
        let scored = Self.pieces(start..<ids.count, size: chunk)
        var nonFinite = 0
        for piece in scored {
            try await engine.readInOneCall(Array(ids[piece]))
            for row in 0..<piece.count {
                let values = try engine.logits(row: row)
                nonFinite += values.lazy.filter { !$0.isFinite }.count
                try values.map { $0.bitPattern.littleEndian }.withUnsafeBytes { try file.write(contentsOf: $0) }
            }
        }
        try file.close()
        let end = clock.now
        let stats = engine.loadStats
        return PositionLogitsRecord(
            tokenCount: ids.count, tokensSha256: tokensDigest(ids.map(Int.init)), start: start, chunk: chunk,
            shape: [ids.count - start, engine.vocabSize], prefixFunction: prefixFunction,
            prefixChunk: engine.prefillChunkSize, prefixCalls: prefix.count, mainCalls: scored.count,
            mainHash: bundle.mainHash, requestedCompute: stats.requestedComputeUnit,
            expectFrequentReshapes: stats.expectsFrequentReshapes, cached: stats.cached,
            cacheQueryError: stats.cacheQueryError, specializeS: stats.specializeSeconds,
            nonFiniteCount: nonFinite, loadS: seconds(prefixStart - loadStart),
            prefixS: seconds(scoringStart - prefixStart), scoringS: seconds(end - scoringStart))
    }
}
