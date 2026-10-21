import Foundation
import Testing

@testable import CoreAIBackend

@Suite struct SamplerGoldenTests {
    struct Golden: Decodable {
        struct Run: Decodable, CustomTestStringConvertible {
            let logits: [String]
            let temperature: Float
            let k: Int?
            let p: Double?
            let seed: String
            let tokens: [Int32]
            let state: String

            var testDescription: String { "\(logits) t=\(temperature) k=\(k ?? 0) p=\(p ?? 0) seed=\(seed)" }
        }

        struct Pool: Decodable {
            let logits: String
            let temperature: Float
            let k: Int?
            let p: Double?
            let indices: [Int]
            let weight: String
        }

        let logits: [String: [String]]
        let pools: [Pool]
        let runs: [Run]

        private static let shared = Result {
            guard
                let url = Bundle.module.url(
                    forResource: "sampler-golden", withExtension: "json", subdirectory: "Resources")
            else { throw CocoaError(.fileReadNoSuchFile) }
            return try JSONDecoder().decode(Golden.self, from: Data(contentsOf: url))
        }

        static func load() throws -> Golden {
            try shared.get()
        }

        func values(_ name: String) throws -> [Float16] {
            try #require(logits[name]).map { Float16(bitPattern: try #require(UInt16($0, radix: 16))) }
        }

        func pool(_ name: String, temperature: Float, k: Int?, p: Double?) throws -> Pool {
            try #require(pools.first { $0.logits == name && $0.temperature == temperature && $0.k == k && $0.p == p })
        }
    }

    private static func mode(k: Int?, p: Double?) -> Sampler.Mode {
        k.map { .topK($0) } ?? p.map { .topP($0) } ?? .full
    }

    private func replay(_ run: Golden.Run, in golden: Golden) throws {
        let label = run.testDescription
        var sampler = Sampler(
            mode: Self.mode(k: run.k, p: run.p), temperature: run.temperature,
            seed: try #require(UInt64(run.seed, radix: 16)))
        var tokens: [Int32] = []
        let calls = run.tokens.count / run.logits.count
        for name in run.logits {
            let logits = try golden.values(name)
            for _ in 0..<calls { tokens.append(logits.withUnsafeBufferPointer { sampler.next($0) }) }
            let expected = try golden.pool(name, temperature: run.temperature, k: run.k, p: run.p)
            #expect(sampler.pool == expected.indices, "\(label) after \(name)")
        }
        #expect(tokens == run.tokens, "\(label)")
        #expect(sampler.random.state == UInt64(run.state, radix: 16), "\(label)")
    }

    @Test(
        "Sequences and random consumption from fixed logits and seeds in full, top-k, and top-p match the previous implementation",
        arguments: try Golden.load().runs.filter { $0.logits.count == 1 })
    func sequencesMatchThePreviousImplementation(run: Golden.Run) throws {
        try replay(run, in: Golden.load())
    }

    @Test(
        "Candidate order and selection probability computation match the previous implementation at tie and candidate-count boundaries"
    )
    func candidatesMatchThePreviousImplementation() throws {
        let golden = try Golden.load()
        #expect(golden.pools.count == 126)
        var fewerThanK = 0
        var tieAcrossK = 0
        var firstReachesP = 0
        for expected in golden.pools {
            let label = "\(expected.logits) t=\(expected.temperature) k=\(expected.k ?? 0) p=\(expected.p ?? 0)"
            let logits = try golden.values(expected.logits)
            var sampler = Sampler(
                mode: Self.mode(k: expected.k, p: expected.p), temperature: expected.temperature, seed: 0)
            _ = logits.withUnsafeBufferPointer { sampler.next($0) }
            let pool = sampler.pool
            let weight = pool.reduce(0.0) { $0 + Double(sampler.scratch[$1]) }
            #expect(pool == expected.indices, "\(label)")
            #expect(weight.bitPattern == UInt64(expected.weight, radix: 16), "\(label)")
            if let k = expected.k, let last = pool.last {
                fewerThanK += pool.count < k ? 1 : 0
                let tied = logits.indices.contains {
                    !pool.contains($0) && sampler.scratch[$0] == sampler.scratch[last]
                }
                tieAcrossK += tied ? 1 : 0
            }
            firstReachesP += expected.p != nil && pool.count == 1 ? 1 : 0
        }
        #expect(fewerThanK > 0)
        #expect(tieAcrossK > 0)
        #expect(firstReachesP > 0)
    }

    @Test("Reusing one sampler across vocabulary sizes 64, 8, and 64 matches the previous implementation")
    func reuseAcrossVocabularySizesMatchesThePreviousImplementation() throws {
        let golden = try Golden.load()
        let runs = golden.runs.filter { $0.logits.count > 1 }
        #expect(runs.count == 21)
        for run in runs {
            #expect(try run.logits.map { try golden.values($0).count } == [64, 8, 64])
            try replay(run, in: golden)
        }
    }
}
