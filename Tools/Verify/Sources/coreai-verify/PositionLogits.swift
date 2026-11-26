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

    @Option(help: "Output file of little-endian float16 logits shaped [positions, vocab]. It must not exist yet.")
    var output: String

    @Option(
        name: .customLong("main-output"),
        help:
            "NAME=FILE. Write output NAME of main for the positions of the logits to FILE, as little-endian float16 shaped [positions, width]. FILE must not exist yet. Repeat for more outputs."
    )
    var mainOutputs: [String] = []

    @Option(
        name: .customLong("prefill-output"),
        help:
            "NAME=FILE. Write output NAME of the prefill function for the positions before the start to FILE, as little-endian float16 shaped [start, width]. Those positions must go through the prefill function. FILE must not exist yet. Repeat for more outputs."
    )
    var prefillOutputs: [String] = []

    func validate() throws {
        guard chunk > 0 else { throw ValidationError("--chunk takes a positive value.") }
        try Self.validateTokens(try tokenIDs(), start: start)
        guard !FileManager.default.fileExists(atPath: output) else {
            throw ValidationError("The output file already exists: \(output)")
        }
        let files =
            try Self.namedFiles(mainOutputs, option: "--main-output")
            + Self.namedFiles(prefillOutputs, option: "--prefill-output")
        guard prefillOutputs.isEmpty || start > 0 else {
            throw ValidationError("--prefill-output needs a start position after 0.")
        }
        for file in files where FileManager.default.fileExists(atPath: file.path) {
            throw ValidationError("The output file already exists: \(file.path)")
        }
    }

    static func namedFiles(_ arguments: [String], option: String) throws -> [(name: String, path: String)] {
        let files = try arguments.map { argument in
            guard let equals = argument.firstIndex(of: "="), equals > argument.startIndex,
                argument.index(after: equals) < argument.endIndex
            else { throw ValidationError("\(option) takes NAME=FILE, not \(argument).") }
            return (name: String(argument[..<equals]), path: String(argument[argument.index(after: equals)...]))
        }
        guard Set(files.map(\.name)).count == files.count else {
            throw ValidationError("\(option) names an output more than once.")
        }
        return files
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

    func run() async throws {
        let ids = try tokenIDs()
        try Self.validateTokens(ids, start: start)
        let mainFiles = try Self.namedFiles(mainOutputs, option: "--main-output")
        let prefillFiles = try Self.namedFiles(prefillOutputs, option: "--prefill-output")
        let bundle = try ModelBundle(url: URL(filePath: model))
        guard ids.count <= bundle.maxContextLength, ids.allSatisfy({ (0..<Int32(bundle.vocabSize)).contains($0) })
        else {
            throw ValidationError(
                "The token file must hold at most \(bundle.maxContextLength) IDs from 0 to \(bundle.vocabSize - 1).")
        }
        var created: [URL] = []
        do {
            for path in [output] + (mainFiles + prefillFiles).map(\.path) {
                let url = URL(filePath: path)
                try Data().write(to: url, options: .withoutOverwriting)
                created.append(url)
            }
            try await write(ids, bundle: bundle, main: mainFiles, prefill: prefillFiles)
        } catch {
            for url in created { try? FileManager.default.removeItem(at: url) }
            throw error
        }
    }

    func write(
        _ ids: [Int32], bundle: ModelBundle, main mainFiles: [(name: String, path: String)],
        prefill prefillFiles: [(name: String, path: String)]
    ) async throws {
        let file = try FileHandle(forWritingTo: URL(filePath: output))
        var mainWriters = try mainFiles.map { try OutputFileWriter(function: "main", name: $0.name, path: $0.path) }
        var prefillWriters = try prefillFiles.map {
            try OutputFileWriter(function: "prefill", name: $0.name, path: $0.path)
        }
        let clock = ContinuousClock()
        let loadStart = clock.now
        let engine = try await CoreAIBackend.Engine(bundle: bundle)
        let prefixStart = clock.now
        let lengths = engine.callLengths(from: 0, count: start)
        let prefixFunction = Self.prefixFunction(
            start: start, lengths: lengths, chunkSize: engine.prefillChunkSize,
            hasPrefillFunction: engine.hasPrefillFunction)
        guard prefillWriters.isEmpty || prefixFunction == "prefill" else {
            throw ValidationError("The positions before the start go through main, so --prefill-output has no rows.")
        }
        let prefix = Self.pieces(from: 0, lengths: lengths)
        for piece in prefix {
            try await engine.readInOneCall(
                Array(ids[piece]), usePrefillFunction: prefixFunction == "prefill",
                outputs: Set(prefillWriters.map(\.name)))
            for row in 0..<piece.count {
                for index in prefillWriters.indices {
                    try prefillWriters[index].append(engine.output(prefillWriters[index].name, row: row))
                }
            }
        }
        let scoringStart = clock.now
        let scored = Self.pieces(start..<ids.count, size: chunk)
        let names = Set(mainWriters.map(\.name))
        var nonFinite = 0
        var callTime = Duration.zero
        for piece in scored {
            let tokens = Array(ids[piece])
            let callStart = clock.now
            try await engine.readInOneCall(tokens, outputs: names)
            callTime += clock.now - callStart
            for row in 0..<piece.count {
                let values = try engine.logits(row: row)
                nonFinite += values.lazy.filter { !$0.isFinite }.count
                try values.map { $0.bitPattern.littleEndian }.withUnsafeBytes { try file.write(contentsOf: $0) }
                for index in mainWriters.indices {
                    try mainWriters[index].append(engine.output(mainWriters[index].name, row: row))
                }
            }
        }
        try file.close()
        let outputs = try (mainWriters + prefillWriters).map { try $0.finish() }
        let end = clock.now
        let stats = engine.loadStats
        try emit(
            PositionLogitsRecord(
                tokenCount: ids.count, tokensSha256: tokensDigest(ids.map(Int.init)), start: start, chunk: chunk,
                shape: [ids.count - start, engine.vocabSize],
                prefixFunction: prefixFunction,
                prefixChunk: engine.prefillChunkSize, prefixCalls: prefix.count, mainCalls: scored.count,
                mainHash: bundle.mainHash, requestedCompute: stats.requestedComputeUnit,
                expectFrequentReshapes: stats.expectsFrequentReshapes, cached: stats.cached,
                cacheQueryError: stats.cacheQueryError, specializeS: stats.specializeSeconds,
                nonFiniteCount: nonFinite, loadS: seconds(prefixStart - loadStart),
                prefixS: seconds(scoringStart - prefixStart), scoringS: seconds(end - scoringStart),
                scoringCallS: seconds(callTime), outputs: outputs.isEmpty ? nil : outputs))
    }
}
