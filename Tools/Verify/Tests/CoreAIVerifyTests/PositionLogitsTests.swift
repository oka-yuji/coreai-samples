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

    @Test("A range is split into calls of the chunk length, with a shorter last call")
    func splitsRangesIntoCalls() {
        #expect(PositionLogits.pieces(3..<10, size: 4) == [3..<7, 7..<10])
        #expect(PositionLogits.pieces(5..<8, size: 1) == [5..<6, 6..<7, 7..<8])
        #expect(PositionLogits.pieces(0..<3, size: 4096) == [0..<3])
        #expect(PositionLogits.pieces(0..<0, size: 4096).isEmpty)
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
