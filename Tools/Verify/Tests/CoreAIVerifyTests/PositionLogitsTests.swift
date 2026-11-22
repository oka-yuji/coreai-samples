import ArgumentParser
import Foundation
import Testing

@testable import coreai_verify

@Suite(.timeLimit(.minutes(1))) struct PositionLogitsTests {
    private func commandLine(tokens: [Int32], start: Int, chunk: Int, existingOutput: Bool = false) throws -> (
        arguments: [String], folder: URL
    ) {
        let folder = FileManager.default.temporaryDirectory.appending(path: "position-logits-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        let input = folder.appending(path: "tokens.json")
        try JSONEncoder().encode(tokens).write(to: input)
        let output = folder.appending(path: "logits.f16")
        if existingOutput { try Data().write(to: output) }
        let arguments = [
            "--model", "/not-loaded", "--tokens", input.path, "--start=\(start)", "--chunk=\(chunk)", "--output",
            output.path,
        ]
        return (arguments, folder)
    }

    @Test("A token file, a start position inside it, a positive chunk length and a new output file are accepted")
    func acceptsValidArguments() throws {
        let (arguments, folder) = try commandLine(tokens: [151_644, 872, 198], start: 2, chunk: 4)
        defer { try? FileManager.default.removeItem(at: folder) }
        let command = try PositionLogits.parse(arguments)
        #expect(command.start == 2 && command.chunk == 4)
        #expect(try command.tokenIDs() == [151_644, 872, 198])
    }

    @Test(
        "An empty token list, a start position outside the tokens, a chunk length of zero or less and an existing output are rejected",
        arguments: [
            ([], 0, 1, false, "The token file has no tokens."),
            ([1, 2, 3], 3, 1, false, "--start must be from 0 to 2, a position in the token file."),
            ([1, 2, 3], -1, 1, false, "--start must be from 0 to 2, a position in the token file."),
            ([1, 2, 3], 0, 0, false, "--chunk takes a positive value."),
            ([1, 2, 3], 0, -4, false, "--chunk takes a positive value."),
            ([1, 2, 3], 0, 1, true, "The output file already exists: "),
        ] as [([Int32], Int, Int, Bool, String)])
    func rejectsInvalidArguments(tokens: [Int32], start: Int, chunk: Int, existingOutput: Bool, message: String) throws
    {
        let (arguments, folder) = try commandLine(
            tokens: tokens, start: start, chunk: chunk, existingOutput: existingOutput)
        defer { try? FileManager.default.removeItem(at: folder) }
        let error = #expect(throws: (any Error).self) { try PositionLogits.parse(arguments) }
        #expect(PositionLogits.message(for: try #require(error)).hasPrefix(message))
    }

    @Test(
        "The token check rejects an empty token list and a start position outside the tokens",
        arguments: [
            ([], 0, "The token file has no tokens."),
            ([1, 2, 3], 3, "--start must be from 0 to 2, a position in the token file."),
            ([1, 2, 3], -1, "--start must be from 0 to 2, a position in the token file."),
        ] as [([Int32], Int, String)])
    func tokenCheckRejectsMissingStart(tokens: [Int32], start: Int, message: String) {
        let error = #expect(throws: ValidationError.self) { try PositionLogits.validateTokens(tokens, start: start) }
        #expect(error?.message == message)
    }

    @Test("The token check accepts a start position inside the tokens", arguments: [0, 2])
    func tokenCheckAcceptsStart(start: Int) throws {
        try PositionLogits.validateTokens([1, 2, 3], start: start)
    }

    @Test("A token file that becomes shorter after validation is rejected before the output file is created")
    func runChecksTheTokensItReads() async throws {
        let (arguments, folder) = try commandLine(tokens: [1, 2, 3], start: 2, chunk: 4)
        defer { try? FileManager.default.removeItem(at: folder) }
        let command = try PositionLogits.parse(arguments)
        try JSONEncoder().encode([Int32(7)]).write(to: folder.appending(path: "tokens.json"))
        let error = await #expect(throws: ValidationError.self) { try await command.run() }
        #expect(error?.message == "--start must be from 0 to 0, a position in the token file.")
        #expect(!FileManager.default.fileExists(atPath: folder.appending(path: "logits.f16").path))
    }

    @Test("A range is split into calls of the chunk length, with a shorter last call")
    func splitsRangesIntoCalls() {
        #expect(PositionLogits.pieces(3..<10, size: 4) == [3..<7, 7..<10])
        #expect(PositionLogits.pieces(5..<8, size: 1) == [5..<6, 6..<7, 7..<8])
        #expect(PositionLogits.pieces(0..<3, size: 4096) == [0..<3])
        #expect(PositionLogits.pieces(0..<0, size: 4096).isEmpty)
    }

    @Test(
        "The prefix call lengths from the engine become consecutive ranges from position 0, and the previous chunk lengths give the previous pieces"
    )
    func prefixLengthsBecomeConsecutiveRanges() {
        #expect(PositionLogits.pieces(from: 0, lengths: [96, 4]) == [0..<96, 96..<100])
        #expect(PositionLogits.pieces(from: 0, lengths: [96, 7, 7, 2]) == [0..<96, 96..<103, 103..<110, 110..<112])
        #expect(PositionLogits.pieces(from: 0, lengths: []).isEmpty)
        for start in 0...300 {
            for size in [1, 8, 256, 4096] {
                let previous = PositionLogits.pieces(0..<start, size: size)
                #expect(PositionLogits.pieces(from: 0, lengths: previous.map(\.count)) == previous)
            }
        }
    }

    @Test(
        "The prefix goes through main when the call shape rule splits it differently from the prefill chunks, and keeps the prefill function otherwise",
        arguments: [
            (0, [], 4096, true, nil), (100, [96, 4], 4096, true, "main"), (112, [96, 7, 7, 2], 4096, true, "main"),
            (100, [100], 4096, true, "prefill"), (96, [96], 4096, true, "prefill"),
            (113, [113], 4096, true, "prefill"), (300, [256, 44], 256, true, "prefill"),
            (100, [100], 4096, false, "main"), (100, [96, 4], 4096, false, "main"),
        ] as [(Int, [Int], Int, Bool, String?)])
    func prefixFunctionFollowsTheSplit(
        start: Int, lengths: [Int], chunkSize: Int, hasPrefillFunction: Bool, expected: String?
    ) {
        #expect(
            PositionLogits.prefixFunction(
                start: start, lengths: lengths, chunkSize: chunkSize, hasPrefillFunction: hasPrefillFunction)
                == expected)
    }

    private func outputPath() throws -> (folder: URL, output: String) {
        let folder = FileManager.default.temporaryDirectory.appending(path: "position-logits-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        return (folder, folder.appending(path: "logits.f16").path(percentEncoded: false))
    }

    private func names(_ folder: URL) throws -> Set<String> {
        Set(try FileManager.default.contentsOfDirectory(atPath: folder.path(percentEncoded: false)))
    }

    @Test(
        "The rows go to a new partial file with a unique name in the same folder, and only the output remains after the writer returns"
    )
    func outputTakesItsNameWhenComplete() async throws {
        let (folder, output) = try outputPath()
        defer { try? FileManager.default.removeItem(at: folder) }
        let value = try await PositionLogits.writeFile(at: output) { partial in
            let name = partial.lastPathComponent
            #expect(name.hasPrefix("logits.f16.") && name.hasSuffix(".partial") && name != "logits.f16.partial")
            #expect(try names(folder) == [name])
            try Data([1, 2, 3]).write(to: partial)
            return 7
        }
        #expect(value == 7)
        #expect(try Data(contentsOf: URL(filePath: output)) == Data([1, 2, 3]))
        #expect(try names(folder) == ["logits.f16"])
    }

    @Test("An existing output is refused before anything is written, and it is left unchanged")
    func existingOutputIsRefused() async throws {
        let (folder, output) = try outputPath()
        defer { try? FileManager.default.removeItem(at: folder) }
        try Data([9]).write(to: URL(filePath: output))
        let error = await #expect(throws: ValidationError.self) {
            try await PositionLogits.writeFile(at: output) { _ in Issue.record("The writer must not run") }
        }
        #expect(error?.message == "The output file already exists: \(output)")
        #expect(try Data(contentsOf: URL(filePath: output)) == Data([9]))
        #expect(try names(folder) == ["logits.f16"])
    }

    @Test("Partial files of other runs and unrelated files with a partial suffix are neither overwritten nor removed")
    func otherPartialFilesAreKept() async throws {
        let (folder, output) = try outputPath()
        defer { try? FileManager.default.removeItem(at: folder) }
        let others = ["logits.f16.partial", "logits.f16.0123.partial", "tokens.json.partial"]
        for (index, name) in others.enumerated() {
            try Data([UInt8(index + 10)]).write(to: folder.appending(path: name))
        }
        try await PositionLogits.writeFile(at: output) { partial in
            #expect(!others.contains(partial.lastPathComponent))
            try Data([5]).write(to: partial)
        }
        #expect(try Data(contentsOf: URL(filePath: output)) == Data([5]))
        for (index, name) in others.enumerated() {
            #expect(try Data(contentsOf: folder.appending(path: name)) == Data([UInt8(index + 10)]))
        }
        #expect(try names(folder) == Set(others + ["logits.f16"]))
    }

    @Test(
        "An output that another run creates before this one finishes is kept, and this run fails and removes its partial file"
    )
    func outputCreatedMeanwhileIsKept() async throws {
        let (folder, output) = try outputPath()
        defer { try? FileManager.default.removeItem(at: folder) }
        let error = await #expect(throws: ValidationError.self) {
            try await PositionLogits.writeFile(at: output) { partial in
                try Data([1, 2]).write(to: partial)
                try Data([8]).write(to: URL(filePath: output))
            }
        }
        #expect(error?.message == "The output file already exists: \(output)")
        #expect(try Data(contentsOf: URL(filePath: output)) == Data([8]))
        #expect(try names(folder) == ["logits.f16"])
    }

    @Test("A failing writer leaves neither the output nor its partial file")
    func failingWriterLeavesNothing() async throws {
        let (folder, output) = try outputPath()
        defer { try? FileManager.default.removeItem(at: folder) }
        await #expect(throws: CancellationError.self) {
            try await PositionLogits.writeFile(at: output) { partial in
                try Data([1]).write(to: partial)
                throw CancellationError()
            }
        }
        #expect(try names(folder).isEmpty)
    }

    @Test("A chunk length near the largest integer is cut to the remaining positions without overflow")
    func cutsALargeChunkToTheRemainingPositions() {
        #expect(PositionLogits.pieces(1..<10, size: .max) == [1..<10])
        #expect(PositionLogits.pieces(0..<10, size: .max) == [0..<10])
    }

    @Test("The record has the keys and the shape of a fixed example")
    func recordMatchesFixedExample() throws {
        let record = PositionLogitsRecord(
            tokenCount: 10, tokensSha256: "digest", start: 3, chunk: 4, shape: [7, 151_936],
            prefixFunction: "prefill", prefixChunk: 4096, prefixCalls: 1, mainCalls: 2, mainHash: "hash",
            requestedCompute: "default", expectFrequentReshapes: true, cached: true, cacheQueryError: nil,
            specializeS: 0, nonFiniteCount: 0, loadS: 1.5, prefixS: 0.25, scoringS: 0.5)
        let encoded = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder.records.encode(record)) as? NSDictionary)
        let expected: NSDictionary = [
            "record": "logits", "token_count": 10, "tokens_sha256": "digest", "start": 3, "chunk": 4,
            "shape": [7, 151_936], "dtype": "float16", "byte_order": "little", "prefix_function": "prefill",
            "prefix_chunk": 4096, "prefix_calls": 1, "main_calls": 2, "main_hash": "hash",
            "requested_compute": "default", "expect_frequent_reshapes": true, "cached": true, "specialize_s": 0,
            "non_finite_count": 0, "load_s": 1.5, "prefix_s": 0.25, "scoring_s": 0.5,
        ]
        #expect(encoded == expected)
    }
}
