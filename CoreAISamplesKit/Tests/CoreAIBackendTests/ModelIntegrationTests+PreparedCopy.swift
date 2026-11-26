import Foundation
import Testing

@testable import CoreAIBackend

extension ModelIntegrationTests {
    @Test(
        "Warming the prepared copy preserves empty state, statistics, greedy tokens and exact logits",
        .enabled(if: !bundlePaths.isEmpty), arguments: bundlePaths)
    func preparedCopyWarmUpPreservesGeneration(path: String) async throws {
        let bundle = try ModelBundle(url: URL(filePath: path))
        let baseline = try await preparedCopyOutput(bundle, warmUp: false)
        let warmed = try await preparedCopyOutput(bundle, warmUp: true)
        #expect(warmed.tokens == baseline.tokens)
        #expect(warmed.logits == baseline.logits)
    }

    private func preparedCopyOutput(_ bundle: ModelBundle, warmUp: Bool) async throws
        -> (tokens: [Int32], logits: [[UInt16]])
    {
        let engine = try await Engine(bundle: bundle)
        engine.measuresPreparation = true
        engine.stats.prefillTokens = 1
        engine.stats.prefillSeconds = 2
        engine.stats.decodeTokens = 3
        engine.stats.decodeSeconds = 4
        engine.stats.prepareSeconds = 5
        if warmUp { try await engine.warmUpPreparedCopy() }
        #expect(engine.tokens.isEmpty)
        #expect(!engine.hasLogits)
        #expect(engine.measuresPreparation)
        #expect(engine.stats.prefillTokens == 1)
        #expect(engine.stats.prefillSeconds == 2)
        #expect(engine.stats.decodeTokens == 3)
        #expect(engine.stats.decodeSeconds == 4)
        #expect(engine.stats.prepareSeconds == 5)
        try await engine.prefill([7, 11, 13])
        var logits: [[UInt16]] = []
        var tokens: [Int32] = []
        for _ in 0..<8 {
            logits.append(engine.logits().map(\.bitPattern))
            let next = engine.greedyToken()
            tokens.append(next)
            try await engine.step(next)
        }
        return (tokens, logits)
    }
}
