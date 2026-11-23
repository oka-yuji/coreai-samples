import Foundation
import Testing

@_spi(Verification) @testable import CoreAIBackend

let outputRowsBundlePath = ProcessInfo.processInfo.environment["COREAI_OUTPUT_ROWS_BUNDLE"] ?? ""

extension ModelIntegrationTests {
    private func bitPatterns(_ values: [Float16]) -> [UInt16] {
        values.map(\.bitPattern)
    }

    @Test(
        "The outputs that a call reads can be read by row only until the next read, reset, restore or unload, and a name that the function does not have is refused before the call",
        .enabled(if: !outputRowsBundlePath.isEmpty))
    func outputRowsFollowTheLastCall() async throws {
        let engine = try await Engine(bundle: try ModelBundle(url: URL(filePath: outputRowsBundlePath)))
        let tokens: [Int32] = [2] + (1...11).map { Int32(1_000 + $0) }
        let both: Set<String> = ["hidden_states", "tap_hidden_states"]
        try await engine.readInOneCall(Array(tokens[0..<4]), outputs: both)
        let hidden = try engine.output("hidden_states", row: 3)
        let taps = try engine.output("tap_hidden_states", row: 0)
        #expect(!hidden.isEmpty && taps.count > hidden.count && taps.count % hidden.count == 0)
        #expect(throws: OutputRowError.rowOutOfRange(row: 4, rows: 4)) { try engine.output("hidden_states", row: 4) }
        #expect(try bitPatterns(engine.logits(row: 3)) == bitPatterns(engine.logits()))

        try await engine.readInOneCall([tokens[4]])
        #expect(throws: OutputRowError.notRead("hidden_states")) { try engine.output("hidden_states", row: 0) }
        try await engine.readInOneCall([tokens[5]], outputs: ["hidden_states"])
        #expect(try engine.output("hidden_states", row: 0).count == hidden.count)
        #expect(throws: OutputRowError.notRead("tap_hidden_states")) { try engine.output("tap_hidden_states", row: 0) }

        let read = engine.tokens.count
        for (names, refused) in [(["logits"], "logits"), (["hidden_states", "missing"], "missing")]
            as [(Set<String>, String)]
        {
            await #expect(throws: OutputRowError.unreadableOutput(refused)) {
                try await engine.readInOneCall([tokens[6]], outputs: names)
            }
        }
        #expect(engine.tokens.count == read)
        #expect(try engine.output("hidden_states", row: 0).count == hidden.count)

        let point = engine.checkpoint()
        try await engine.readInOneCall([tokens[6]], outputs: both)
        try engine.restore(point)
        #expect(throws: OutputRowError.notRead("hidden_states")) { try engine.output("hidden_states", row: 0) }
        try await engine.readInOneCall([tokens[6]], outputs: both)
        engine.reset()
        #expect(throws: OutputRowError.notRead("tap_hidden_states")) { try engine.output("tap_hidden_states", row: 0) }

        if engine.hasPrefillFunction {
            await #expect(throws: OutputRowError.unreadableOutput("hidden_states")) {
                try await engine.readInOneCall(
                    Array(tokens[0..<3]), usePrefillFunction: true, outputs: ["hidden_states"])
            }
            try await engine.readInOneCall(
                Array(tokens[0..<3]), usePrefillFunction: true, outputs: ["tap_hidden_states"])
            #expect(try engine.output("tap_hidden_states", row: 2).count == taps.count)
            #expect(throws: LogitRowError.noLogits) { try engine.logits(row: 0) }
            try await engine.readInOneCall([tokens[3]])
            #expect(throws: OutputRowError.notRead("tap_hidden_states")) {
                try engine.output("tap_hidden_states", row: 0)
            }
        }
        try await engine.readInOneCall([tokens[engine.tokens.count]], outputs: both)
        engine.unload()
        #expect(throws: OutputRowError.notRead("hidden_states")) { try engine.output("hidden_states", row: 0) }
    }
}
