import CryptoKit
import Foundation
import Testing

@testable import CoreAIBackend

let typedDecisionBundle = ProcessInfo.processInfo.environment["COREAI_TYPED_DECISION_BUNDLE"] ?? ""
private let typedDecisionModel = Task {
    try await TypedDecisionModel.load(bundle: URL(filePath: typedDecisionBundle))
}

struct TypedDecisionDetachedBackbone: TypedDecisionBackbone {
    let runtime: TypedDecisionRuntime
    let entered: AsyncStream<Void>.Continuation

    func forward(_ ids: [Int], capacity: Int) async throws -> [Float] {
        entered.yield()
        return try await Task { try await runtime.forward(ids, capacity: capacity) }.value
    }
}

struct TypedDecisionInferenceRow: Decodable, CustomTestStringConvertible {
    private struct Document: Decodable { let rows: [TypedDecisionInferenceRow] }

    enum CodingKeys: String, CodingKey {
        case id, kind, state, question, options, band, p2, p1m
        case tokenCount = "token_count"
        case tokensSHA256 = "tokens_sha256"
        case p2T = "p2_T"
    }

    let id: String
    let kind: TypedDecisionKind
    let state: String
    let question: String
    let options: [String]
    let tokenCount: Int
    let tokensSHA256: String
    let band: String
    let p2: [Double]
    let p2T: [Double]
    let p1m: [Double]

    var request: TypedDecisionRequest {
        TypedDecisionRequest(state: state, question: question, kind: kind, options: options)
    }

    var testDescription: String { id }

    static func rows() throws -> [TypedDecisionInferenceRow] {
        try TypedDecisionFixture.load("inference", as: Document.self).rows
    }
}

@Suite(.serialized, .enabled(if: !typedDecisionBundle.isEmpty))
struct TypedDecisionInferenceTests {
    static let bands = ["<=512", "513-2048", "2049-4096", "4097-8192"]

    static func digest(_ ids: [Int]) -> String {
        SHA256.hash(data: Data(ids.map(String.init).joined(separator: ",").utf8))
            .map { String(format: "%02x", $0) }.joined()
    }

    static func largest(_ actual: [Float], _ expected: [Double]) -> Double {
        guard actual.count == expected.count else { return .infinity }
        return zip(actual, expected).map { abs(Double($0) - $1) }.max() ?? 0
    }

    static func isDistribution(_ values: [Float]) -> Bool {
        values.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 } && abs(values.reduce(0, +) - 1) <= 1e-3
    }

    static func check(_ answer: TypedDecisionAnswer, _ ids: [Int], _ row: TypedDecisionInferenceRow) -> Double {
        #expect(ids.count == row.tokenCount && answer.tokenCount == row.tokenCount, "\(row.id)")
        #expect(digest(ids) == row.tokensSHA256, "\(row.id)")
        #expect(isDistribution(answer.probabilities) && isDistribution(answer.probabilitiesT1), "\(row.id)")
        #expect(answer.logits.count == 24 && answer.logits.allSatisfy(\.isFinite), "\(row.id)")
        return largest(answer.probabilitiesT1, row.p2)
    }

    @Test("Ten rows of F match stage 2 at temperature 1 within the band limits with the reference token IDs")
    func tenRowsMatchStageTwo() async throws {
        let model = try await typedDecisionModel.value
        print("typed-load \(model.loadStats)")
        let rows = try TypedDecisionInferenceRow.rows()
        #expect(rows.count == 10)
        var bands: [String: [Double]] = [:]
        var metrics: [[String: Any]] = []
        for row in rows {
            let ids = try await model.tokenIDs(for: row.request)
            let clock = ContinuousClock()
            let start = clock.now
            let answer = try await model.decide(row.request)
            let seconds = (clock.now - start) / .seconds(1)
            let error = Self.check(answer, ids, row)
            bands[row.band, default: []].append(error)
            let shipping = Self.largest(answer.probabilities, row.p2T)
            let merged = Self.largest(answer.probabilitiesT1, row.p1m)
            print("typed-inference \(row.id) tokens \(ids.count) e \(error) eT \(shipping) e1m \(merged) s \(seconds)")
            metrics.append([
                "id": row.id, "band": row.band, "tokens": ids.count, "seconds": seconds, "e": error,
                "eShipping": shipping, "eStage1m": merged, "footprint": try DecisionInferenceTests.footprint(),
            ])
        }
        for (band, errors) in bands {
            let mean = errors.reduce(0, +) / Double(errors.count)
            #expect((errors.max() ?? 1) <= 0.03, "Band \(band)")
            #expect(mean <= 0.01, "Band \(band)")
            print("typed-band \(band) rows \(errors.count) max \(errors.max() ?? 0) mean \(mean)")
        }
        let unevaluated = Self.bands.filter { bands[$0] == nil }
        print("typed-band unevaluated \(unevaluated)")
        try DecisionInferenceTests.writeMetrics(
            [
                "rows": metrics, "unevaluated": unevaluated,
                "load": [
                    "bundle": model.loadStats.bundleSeconds, "tokenizer": model.loadStats.tokenizerSeconds,
                    "head": model.loadStats.headSeconds, "backbone": model.loadStats.backboneSeconds,
                    "cached": model.loadStats.cachedSpecialization,
                ],
            ], name: "typed-parity")
    }

    @Test("Running each row twice in a row gives bit-identical logits and probabilities")
    func repeatedRowsAreBitIdentical() async throws {
        let model = try await typedDecisionModel.value
        for row in try TypedDecisionInferenceRow.rows() {
            let first = try await model.decide(row.request)
            let second = try await model.decide(row.request)
            #expect(first.logits.map(\.bitPattern) == second.logits.map(\.bitPattern), "\(row.id)")
            #expect(first.probabilities.map(\.bitPattern) == second.probabilities.map(\.bitPattern), "\(row.id)")
            #expect(first.probabilitiesT1.map(\.bitPattern) == second.probabilitiesT1.map(\.bitPattern), "\(row.id)")
        }
    }

    static func longRequest(_ row: TypedDecisionInferenceRow, repeats: Int = 120) -> TypedDecisionRequest {
        TypedDecisionRequest(
            state: Array(repeating: row.state, count: repeats).joined(separator: " "), question: row.question,
            kind: row.kind, options: row.options)
    }

    @Test("A request cancelled at the forward checkpoint ends with CancellationError and the next row still matches")
    func cancellationLeavesTheNextRowIntact() async throws {
        let model = try await typedDecisionModel.value
        let row = try #require(try TypedDecisionInferenceRow.rows().first)
        let before = try await model.decide(row.request)
        let long = Self.longRequest(row)
        let longTokens = try await model.tokenCount(for: long)
        let (started, continuation) = AsyncStream.makeStream(of: Void.self)
        await model.setForwardCheckpoint { continuation.yield() }
        let clock = ContinuousClock()
        let start = clock.now
        let task = Task { try await model.decide(long) }
        var iterator = started.makeAsyncIterator()
        #expect(await iterator.next() != nil)
        task.cancel()
        let result = await task.result
        let seconds = (clock.now - start) / .seconds(1)
        await model.setForwardCheckpoint(nil)
        switch result {
        case .success:
            Issue.record("Unverified: the \(longTokens)-token request finished before the cancellation took effect.")
        case .failure(let error):
            #expect(error is CancellationError, "\(error)")
        }
        print("typed-cancel tokens \(longTokens) outcome \(result) seconds \(seconds)")
        let ids = try await model.tokenIDs(for: row.request)
        let after = try await model.decide(row.request)
        let error = Self.check(after, ids, row)
        #expect(error <= 0.03, "\(row.id) e \(error)")
        #expect(after.logits.map(\.bitPattern) == before.logits.map(\.bitPattern))
        print("typed-cancel next \(row.id) e \(error)")
        let uncancelledStart = clock.now
        let uncancelled = try await model.decide(long)
        let uncancelledSeconds = (clock.now - uncancelledStart) / .seconds(1)
        #expect(Self.isDistribution(uncancelled.probabilitiesT1))
        let phase = seconds < uncancelledSeconds / 2 ? "before the run" : "after the run"
        print("typed-cancel uncancelled tokens \(longTokens) seconds \(uncancelledSeconds) caught \(phase)")
        try DecisionInferenceTests.writeMetrics(
            [
                "id": row.id, "tokens": longTokens, "cancelledSeconds": seconds,
                "uncancelledSeconds": uncancelledSeconds, "nextError": error, "caught": phase,
            ], name: "typed-cancel")
    }

    @Test("A request cancelled while its forward runs keeps the model busy until the run returns, then ends cancelled")
    func cancellationDuringTheRunWaitsForTheRun() async throws {
        let model = try await typedDecisionModel.value
        let bundle = model.bundle
        let (runtime, _) = try await TypedDecisionRuntime.load(bundle: bundle)
        let tokenizer = try await DecisionModel.loadTokenizer(bundle.tokenizerURL)
        let (entered, continuation) = AsyncStream.makeStream(of: Void.self)
        let detached = TypedDecisionModel(
            bundle: bundle, encode: { tokenizer.encode(text: $0, addSpecialTokens: false) },
            head: try TypedDecisionHead(url: bundle.headURL, width: bundle.hiddenSize),
            backbone: TypedDecisionDetachedBackbone(runtime: runtime, entered: continuation))
        let row = try #require(try TypedDecisionInferenceRow.rows().first)
        var iterator = entered.makeAsyncIterator()
        let before = try await detached.decide(row.request)
        #expect(await iterator.next() != nil)
        let clock = ContinuousClock()
        let start = clock.now
        let task = Task { try await detached.decide(Self.longRequest(row)) }
        #expect(await iterator.next() != nil)
        task.cancel()
        await #expect(throws: TypedDecisionError.busy) { try await detached.decide(row.request) }
        let result = await task.result
        let seconds = (clock.now - start) / .seconds(1)
        guard case .failure(let error) = result, error is CancellationError else {
            Issue.record("Expected CancellationError after the run, got \(result)")
            return
        }
        let after = try await detached.decide(row.request)
        #expect(await iterator.next() != nil)
        #expect(after.logits.map(\.bitPattern) == before.logits.map(\.bitPattern))
        print("typed-cancel-run seconds \(seconds) next \(row.id) bit-identical")
        let request = Self.longRequest(row, repeats: 164)
        let tokens = try await detached.tokenCount(for: request)
        let clockStart = clock.now
        _ = try await detached.decide(request)
        #expect(await iterator.next() != nil)
        let uncancelled = (clock.now - clockStart) / .seconds(1)
        var cancellations: [[String: Any]] = []
        for attempt in 1...3 {
            let timing = await cancelWhileRunning(entered: &iterator) {
                try await detached.decide(request)
            }
            #expect(timing.entered && timing.cancelled, "Attempt \(attempt): \(timing)")
            #expect(timing.requestToEndSeconds > uncancelled / 2, "Attempt \(attempt): \(timing)")
            cancellations.append(timing.record.merging(["attempt": attempt, "tokens": tokens]) { first, _ in first })
            print("typed-cancel-timing attempt \(attempt) tokens \(tokens) \(timing) uncancelled \(uncancelled)")
        }
        try DecisionInferenceTests.writeMetrics(
            [
                "id": row.id, "cancelledSeconds": seconds, "tokens": tokens, "uncancelledSeconds": uncancelled,
                "cancellations": cancellations,
            ], name: "typed-cancel-run")
    }

    @Test("A row run with the rounded capacity and with 8,192 positions records the probability difference")
    func roundedAndFullCapacityAgree() async throws {
        let model = try await typedDecisionModel.value
        let row = try #require(try TypedDecisionInferenceRow.rows().first)
        let ids = try await model.tokenIDs(for: row.request)
        let rounded = try await model.decide(row.request)
        let full = try await model.decide(row.request, capacity: 8_192)
        _ = Self.check(full, ids, row)
        let difference = Self.largest(rounded.probabilitiesT1, full.probabilitiesT1.map(Double.init))
        let shipping = Self.largest(rounded.probabilities, full.probabilities.map(Double.init))
        print(
            "typed-capacity \(row.id) capacity \(model.capacity(forTokens: ids.count)) and 8192 difference "
                + "\(difference) shipping \(shipping) above 1e-3 \(max(difference, shipping) > 1e-3)")
        #expect(Self.largest(full.probabilitiesT1, row.p2) <= 0.03)
        try DecisionInferenceTests.writeMetrics(
            [
                "id": row.id, "capacity": model.capacity(forTokens: ids.count), "difference": difference,
                "shippingDifference": shipping,
            ], name: "typed-capacity")
    }
}
