import Foundation
import Testing

@_spi(Verification) @testable import CoreAIBackend

extension ModelIntegrationTests {
    private func positionLogitsInput(_ bundle: ModelBundle) async throws -> (tokens: [Int32], start: Int) {
        let tokenizer = try await bundle.loadTokenizer()
        let prompt = try tokenizer.applyChatTemplate(messages: [
            ["role": "user", "content": "Explain in a short paragraph why the sky is blue."]
        ])
        let answer = tokenizer.encode(
            text: "The sky looks blue because sunlight is scattered by the molecules of the air. "
                + "Sunlight contains every color, but the short blue and violet wavelengths are scattered "
                + "far more strongly than the long red and orange ones. This scattered light reaches our eyes "
                + "from every part of the sky, so the whole sky appears blue during the day. Our eyes are more "
                + "sensitive to blue than to violet, and some violet light is absorbed high in the atmosphere, "
                + "which is why we do not see a violet sky. At sunrise and sunset the light crosses much more "
                + "air, most of the blue is scattered away before it reaches us, and the sky turns red and orange.",
            addSpecialTokens: false)
        return ((prompt + answer).map(Int32.init), prompt.count - 1)
    }

    private func readPrefix(_ engine: Engine, _ tokens: [Int32], start: Int) async throws {
        for lower in stride(from: 0, to: start, by: engine.prefillChunkSize) {
            let piece = Array(tokens[lower..<min(lower + engine.prefillChunkSize, start)])
            try await engine.readInOneCall(piece, usePrefillFunction: true)
        }
    }

    private func readPositionLogits(_ engine: Engine, _ tokens: [Int32], start: Int, chunk: Int) async throws
        -> [[Float16]]
    {
        engine.reset()
        try await readPrefix(engine, tokens, start: start)
        var rows: [[Float16]] = []
        for lower in stride(from: start, to: tokens.count, by: chunk) {
            let piece = Array(tokens[lower..<min(lower + chunk, tokens.count)])
            try await engine.readInOneCall(piece)
            for row in piece.indices { rows.append(try engine.logits(row: row)) }
        }
        return rows
    }

    private func bitPatterns(_ values: [Float16]) -> [UInt16] {
        values.map { $0.bitPattern }
    }

    private func topTwo(_ values: [Float16]) -> String {
        let (first, second) = StateCheck.top2(values)
        return "\(first):\(values[first]) \(second):\(values[second])"
    }

    @Test(
        "The last row of a multi-token main call matches logits(), and rows are unavailable after a prefill-only read, reset, restore or unload",
        .enabled(if: !bundlePaths.isEmpty), arguments: bundlePaths)
    func positionLogitsFollowTheLastMainCall(path: String) async throws {
        let bundle = try ModelBundle(url: URL(filePath: path))
        let (tokens, start) = try await positionLogitsInput(bundle)
        let engine = try await Engine(bundle: bundle)
        try await readPrefix(engine, tokens, start: start)
        if engine.hasPrefillFunction {
            #expect(throws: LogitRowError.noLogits) { try engine.logits(row: 0) }
        }
        var point: Engine.Checkpoint?
        var calls = 0
        for lower in stride(from: start, to: tokens.count, by: 8) {
            let piece = Array(tokens[lower..<min(lower + 8, tokens.count)])
            try await engine.readInOneCall(piece)
            calls += 1
            #expect(try bitPatterns(engine.logits(row: piece.count - 1)) == bitPatterns(engine.logits()))
            #expect(throws: LogitRowError.rowOutOfRange(row: piece.count, rows: piece.count)) {
                try engine.logits(row: piece.count)
            }
            #expect(throws: LogitRowError.rowOutOfRange(row: -1, rows: piece.count)) { try engine.logits(row: -1) }
            if point == nil { point = engine.checkpoint() }
        }
        #expect(calls > 2)
        let saved = try #require(point)
        try engine.restore(saved)
        #expect(throws: LogitRowError.noLogits) { try engine.logits(row: 0) }
        #expect(bitPatterns(engine.logits()) == bitPatterns(saved.logits))
        try await engine.readInOneCall(Array(tokens[saved.tokens.count..<saved.tokens.count + 4]))
        #expect(try engine.logits(row: 3).count == engine.vocabSize)
        engine.reset()
        #expect(throws: LogitRowError.noLogits) { try engine.logits(row: 0) }
        try await engine.readInOneCall(Array(tokens.prefix(4)))
        #expect(try engine.logits(row: 3).count == engine.vocabSize)
        if engine.hasPrefillFunction {
            try await engine.readInOneCall(Array(tokens[4..<8]), usePrefillFunction: true)
            #expect(throws: LogitRowError.noLogits) { try engine.logits(row: 0) }
        }
        engine.unload()
        #expect(throws: LogitRowError.noLogits) { try engine.logits(row: 0) }
    }

    @Test(
        "Single-token chunks match stepping bit for bit, and chunks of 2, 4 and 8 stay finite and within the margin rule",
        .enabled(if: !bundlePaths.isEmpty), arguments: bundlePaths)
    func positionLogitsAcrossChunkLengths(path: String) async throws {
        let bundle = try ModelBundle(url: URL(filePath: path))
        let (tokens, start) = try await positionLogitsInput(bundle)
        let engine = try await Engine(bundle: bundle)
        let single = try await readPositionLogits(engine, tokens, start: start, chunk: 1)
        engine.reset()
        try await readPrefix(engine, tokens, start: start)
        var stepped: [[Float16]] = []
        for token in tokens[start...] {
            try await engine.step(token)
            stepped.append(engine.logits())
        }
        #expect(single.count == tokens.count - start)
        #expect(single.map(bitPatterns) == stepped.map(bitPatterns))
        let singleNonFinite = single.reduce(0) { $0 + $1.filter { !$0.isFinite }.count }
        #expect(singleNonFinite == 0)
        print(
            "position-logits chunk=1 positions=\(single.count) start=\(start) tokens=\(tokens.count)"
                + " identical_to_step=\(zip(single, stepped).filter { bitPatterns($0) == bitPatterns($1) }.count)"
                + " non_finite=\(singleNonFinite)")
        for chunk in [2, 4, 8] {
            let rows = try await readPositionLogits(engine, tokens, start: start, chunk: chunk)
            #expect(rows.count == single.count)
            var identical = 0
            var mismatches = 0
            var nearTies = 0
            var nonFinite = 0
            var maxAbsDiff: Float = 0
            for (offset, (reference, row)) in zip(single, rows).enumerated() {
                nonFinite += row.filter { !$0.isFinite }.count
                identical += bitPatterns(reference) == bitPatterns(row) ? 1 : 0
                let check = StateCheck(current: reference, reread: row)
                maxAbsDiff = max(maxAbsDiff, check.maxAbsDiff)
                if check.currentTop != check.rereadTop {
                    mismatches += 1
                    nearTies += check.passed ? 1 : 0
                }
                #expect(
                    check.passed,
                    "chunk \(chunk) position \(start + offset): chunk 1 \(topTwo(reference)), chunk \(chunk) \(topTwo(row)), \(check.marginTicks) ticks"
                )
            }
            #expect(nonFinite == 0)
            print(
                "position-logits chunk=\(chunk) positions=\(rows.count) identical=\(identical)"
                    + " top1_mismatches=\(mismatches) near_ties=\(nearTies) max_abs_diff=\(maxAbsDiff)"
                    + " non_finite=\(nonFinite)")
        }
    }
}
