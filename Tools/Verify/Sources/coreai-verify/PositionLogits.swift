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
            "First position whose logits are written. The tokens before it are read without logits, with the prefill function when the bundle has one."
    )
    var start: Int

    @Option(help: "Tokens per main call from the start position.")
    var chunk: Int

    @Option(help: "Output file of little-endian float16 logits shaped [positions, vocab]. It must not exist yet.")
    var output: String

    func validate() throws {
        guard chunk > 0 else { throw ValidationError("--chunk takes a positive value.") }
        let count = try tokenIDs().count
        guard count > 0 else { throw ValidationError("The token file has no tokens.") }
        guard (0..<count).contains(start) else {
            throw ValidationError("--start must be from 0 to \(count - 1), a position in the token file.")
        }
        guard !FileManager.default.fileExists(atPath: output) else {
            throw ValidationError("The output file already exists: \(output)")
        }
    }

    func tokenIDs() throws -> [Int32] {
        try JSONDecoder().decode([Int32].self, from: Data(contentsOf: URL(filePath: tokens)))
    }

    static func pieces(_ range: Range<Int>, size: Int) -> [Range<Int>] {
        stride(from: range.lowerBound, to: range.upperBound, by: size).map { $0..<min($0 + size, range.upperBound) }
    }

    func run() async throws {
        let ids = try tokenIDs()
        let bundle = try ModelBundle(url: URL(filePath: model))
        guard ids.count <= bundle.maxContextLength, ids.allSatisfy({ (0..<Int32(bundle.vocabSize)).contains($0) })
        else {
            throw ValidationError(
                "The token file must hold at most \(bundle.maxContextLength) IDs from 0 to \(bundle.vocabSize - 1).")
        }
        let url = URL(filePath: output)
        try Data().write(to: url, options: .withoutOverwriting)
        do {
            try await write(ids, bundle: bundle, to: url)
        } catch {
            try? FileManager.default.removeItem(at: url)
            throw error
        }
    }

    func write(_ ids: [Int32], bundle: ModelBundle, to url: URL) async throws {
        let file = try FileHandle(forWritingTo: url)
        let clock = ContinuousClock()
        let loadStart = clock.now
        let engine = try await CoreAIBackend.Engine(bundle: bundle)
        let prefixStart = clock.now
        let prefix = Self.pieces(0..<start, size: engine.prefillChunkSize)
        for piece in prefix { try await engine.readInOneCall(Array(ids[piece]), usePrefillFunction: true) }
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
        try emit(
            PositionLogitsRecord(
                tokenCount: ids.count, tokensSha256: tokensDigest(ids.map(Int.init)), start: start, chunk: chunk,
                shape: [ids.count - start, engine.vocabSize],
                prefixFunction: start == 0 ? nil : engine.hasPrefillFunction ? "prefill" : "main",
                prefixChunk: engine.prefillChunkSize, prefixCalls: prefix.count, mainCalls: scored.count,
                mainHash: bundle.mainHash, requestedCompute: stats.requestedComputeUnit,
                expectFrequentReshapes: stats.expectsFrequentReshapes, cached: stats.cached,
                cacheQueryError: stats.cacheQueryError, specializeS: stats.specializeSeconds,
                nonFiniteCount: nonFinite, loadS: seconds(prefixStart - loadStart),
                prefixS: seconds(scoringStart - prefixStart), scoringS: seconds(end - scoringStart)))
    }
}
