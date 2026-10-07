import Foundation
import Testing

@testable import CoreAIBackend

struct DecisionReuseFixture: Decodable {
    let id: String
    let prefixTokens: Int
    let requestA: String
    let requestB: String
    let referenceA: [[Float]]
    let referenceB: [[Float]]

    static func all() throws -> [Self] {
        let folder = try #require(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        return try JSONDecoder().decode(
            [Self].self, from: Data(contentsOf: folder.appendingPathComponent("DecisionInference/reuse.json")))
    }
}

struct DecisionContextPolicyTests {
    @Test("Review marks use the larger noul probability and mark unknown choices")
    func reviewPolicy() throws {
        let policy = try DecisionReviewPolicy.bundled()
        let no = DecisionAnswer.noul(
            id: "q",
            distribution: [
                .init(option: "true", probability: 0.1, logit: 0),
                .init(option: "false", probability: 0.9, logit: 1),
            ], probabilityTrue: 0.1)
        #expect(!policy.requestsReview(no))
        #expect(policy.requestsReview(answer("q", top: "\u{4e0d}\u{660e}", probability: 0.95)))
        #expect(policy.requestsReview(answer("q", top: "a", probability: 0.59)))
        #expect(!policy.requestsReview(answer("q", top: "a", probability: 0.6)))
    }

    @Test("Changed questions include added, removed, changed candidates and changed review marks")
    func changes() {
        let policy = DecisionReviewPolicy()
        let before = DecisionResult(
            answers: [
                answer("kept", top: "a", probability: 0.7),
                answer("marked", top: "a", probability: 0.7),
                answer("changed", top: "a", probability: 0.7),
                answer("removed", top: "a", probability: 0.7),
            ],
            model: "m", modelVersion: "v", inputTokens: 1, elapsedSeconds: 0)
        let after = [
            answer("changed", top: "b", probability: 0.7),
            answer("kept", top: "a", probability: 0.8),
            answer("marked", top: "a", probability: 0.55),
            answer("added", top: "a", probability: 0.7),
        ]
        #expect(
            DecisionResult.changedQuestions(previous: before, answers: after, policy: policy) == [
                "changed", "marked", "added", "removed",
            ])
        #expect(DecisionResult.changedQuestions(previous: nil, answers: after, policy: policy).isEmpty)
    }

    private func answer(_ id: String, top: String, probability: Float) -> DecisionAnswer {
        .choice(
            id: id,
            distribution: [
                .init(option: top, probability: probability, logit: 0),
                .init(option: "other", probability: 1 - probability, logit: -1),
            ], top: top)
    }
}

extension DecisionInferenceTests {
    @Test("Document reuse matches fresh inference and reference probabilities at 2K and 8K")
    func contextParityAndLatency() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["COREAI_DECISION_BUNDLE"])
        let model = try await DecisionModel.load(bundle: URL(filePath: path))
        var metrics: [[String: Any]] = []
        for fixture in try DecisionReuseFixture.all() {
            let requestA = try DecisionRequest(json: Data(fixture.requestA.utf8))
            let requestB = try DecisionRequest(json: Data(fixture.requestB.utf8))
            let fullA = try await model.decide(requestA)
            let fullB = try await model.decide(requestB)
            let context = try await model.context(state: requestA.state)
            #expect(context.prefixTokenCount == fixture.prefixTokens)
            let reusedA = try await context.decide(requestA.questions)
            let reusedB = try await context.decide(requestB.questions)
            #expect(reusedA.recomputedAll && reusedB.recomputedAll)
            #expect(reusedB.reusedTokens == fixture.prefixTokens)
            #expect(!reusedB.changedQuestions.isEmpty)
            for (result, full, reference) in [
                (reusedA, fullA, fixture.referenceA), (reusedB, fullB, fixture.referenceB),
            ] {
                let fullErrors = try Self.errors(
                    result, comparedWith: full.answers.map { $0.distribution.map(\.probability) })
                let referenceErrors = try Self.errors(result, comparedWith: reference)
                #expect((fullErrors.max() ?? 1) <= 0.011)
                #expect((referenceErrors.max() ?? 1) <= 0.03)
                #expect(referenceErrors.reduce(0, +) / Float(referenceErrors.count) <= 0.01)
                metrics.append([
                    "id": fixture.id, "tokens": result.inputTokens, "reusedTokens": result.reusedTokens,
                    "seconds": result.elapsedSeconds, "freshErrors": fullErrors, "referenceErrors": referenceErrors,
                ])
            }
            if fixture.id.contains("8k") {
                var times: [Double] = []
                for index in 0..<20 {
                    let result = try await context.decide(
                        index.isMultiple(of: 2) ? requestA.questions : requestB.questions)
                    times.append(result.elapsedSeconds)
                }
                let p95 = times.sorted()[Int(ceil(Double(times.count) * 0.95)) - 1]
                metrics.append(["id": fixture.id, "reuseSeconds": times, "p95Seconds": p95])
                #expect(p95 <= 1)
            }
        }
        try Self.writeMetrics(["contexts": metrics], name: "context")
    }

    @Test(
        "Cancellation after inference and rejected requests preserve the current document",
        .timeLimit(.minutes(3)))
    func contextCancellationAndLimits() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["COREAI_DECISION_BUNDLE"])
        let model = try await DecisionModel.load(bundle: URL(filePath: path))
        let fixture = try #require(try DecisionReuseFixture.all().first)
        let request = try DecisionRequest(json: Data(fixture.requestA.utf8))
        let context = try await model.context(state: request.state)
        let baseline = try await context.decide(request.questions)
        let gate = DecisionCheckpointGate()
        await model.setDecisionCheckpoint { await gate.hold() }
        let running = Task {
            do {
                let result = try await context.decide(request.questions)
                await gate.finished()
                return result
            } catch {
                await gate.finished()
                throw error
            }
        }
        await gate.waitForArrival()
        let (queuedEvents, queuedSignal) = AsyncStream<Void>.makeStream()
        await model.setQueuedCheckpoint { queuedSignal.yield(()) }
        let queued = Task { try await context.decide(request.questions) }
        var queuedIterator = queuedEvents.makeAsyncIterator()
        _ = await queuedIterator.next()
        queued.cancel()
        running.cancel()
        await gate.release()
        await #expect(throws: CancellationError.self) { try await running.value }
        await #expect(throws: CancellationError.self) { try await queued.value }
        await model.setDecisionCheckpoint(nil)
        await model.setQueuedCheckpoint(nil)
        queuedSignal.finish()
        let reloadCount = DecisionReloadCounter()
        await model.setReloadCheckpoint { await reloadCount.record() }
        let next = try await context.decide(request.questions)
        let reloads = await reloadCount.value
        #expect(reloads == 1)
        await model.setReloadCheckpoint(nil)
        #expect(
            (try Self.errors(next, comparedWith: baseline.answers.map { $0.distribution.map(\.probability) })).max()
                ?? 1 <= 1e-3)
        var oversized = request.questions
        oversized[0].instructions = String(repeating: " long", count: 1_200)
        await #expect(throws: DecisionError.self) { try await context.decide(oversized) }
        let afterRejection = try await context.decide(request.questions)
        #expect(
            (try Self.errors(afterRejection, comparedWith: baseline.answers.map { $0.distribution.map(\.probability) }))
                .max() ?? 1 <= 1e-3)
        let reordered = try await context.decide(Array(request.questions.reversed()))
        #expect(reordered.recomputedAll)
        let reduced = try await context.decide(Array(request.questions.dropLast()))
        #expect(reduced.recomputedAll)
        let expanded = try await context.decide(request.questions + [DecisionQuestion(id: "extra", criteria: .noul())])
        #expect(expanded.recomputedAll)
        let replacement = try await model.context(state: "A different document.")
        await #expect(throws: DecisionError.contextInvalidated) { try await context.decide(request.questions) }
        _ = try await replacement.decide([DecisionQuestion(id: "exists", criteria: .noul())])
    }

    @Test(
        "Alternating small and large question sets keeps the document memory bounded",
        .enabled(if: ProcessInfo.processInfo.environment["DECIDE_SHAPE_MEMORY"] == "1"))
    func questionSizeMemory() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["COREAI_DECISION_BUNDLE"])
        let model = try await DecisionModel.load(bundle: URL(filePath: path))
        let fixture = try #require(try DecisionReuseFixture.all().last)
        let request = try DecisionRequest(json: Data(fixture.requestA.utf8))
        let context = try await model.context(state: request.state)
        let small = [DecisionQuestion(id: "q0", criteria: .noul())]
        let large = (0..<12).map { DecisionQuestion(id: "q\($0)", criteria: .noul()) }
        _ = try await context.decide(small)
        let initial = try await context.decide(large)
        let baseline = try Self.footprint()
        var rows: [[String: Any]] = []
        var last = baseline
        for index in 0..<20 {
            let questions = index.isMultiple(of: 2) ? small : large
            let result = try await context.decide(questions)
            last = try Self.footprint()
            rows.append([
                "index": index, "questions": questions.count, "tokens": result.inputTokens,
                "reusedTokens": result.reusedTokens, "seconds": result.elapsedSeconds, "footprint": last,
            ])
            if !index.isMultiple(of: 2) {
                let errors = try Self.errors(
                    result, comparedWith: initial.answers.map { $0.distribution.map(\.probability) })
                #expect((errors.max() ?? 1) <= 1e-3)
            }
        }
        let growth = (Double(last) - Double(baseline)) / 20
        try Self.writeMetrics(["baseline": baseline, "queries": rows, "bytesPerQuery": growth], name: "query-memory")
        #expect(growth < 100_000_000)
    }

    @Test("Reload failure retains the document and concurrent context creation leaves exactly one current handle")
    func contextReloadFailureAndConcurrentCreation() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["COREAI_DECISION_BUNDLE"])
        let model = try await DecisionModel.load(bundle: URL(filePath: path))
        let fixture = try #require(try DecisionReuseFixture.all().first)
        let request = try DecisionRequest(json: Data(fixture.requestA.utf8))
        let context = try await model.context(state: request.state)
        let small = [DecisionQuestion(id: "q0", criteria: .noul())]
        let large = (0..<12).map { DecisionQuestion(id: "q\($0)", criteria: .noul()) }
        let baseline = try await context.decide(large)
        _ = try await context.decide(small)
        await model.setReloadCheckpoint { throw DecisionError.invalidBundle("Injected reload failure.") }
        await #expect(throws: DecisionError.invalidBundle("Injected reload failure.")) {
            try await context.decide(large)
        }
        await model.setReloadCheckpoint(nil)
        let recovered = try await context.decide(large)
        #expect(recovered.reusedTokens == context.prefixTokenCount)
        #expect(
            (try Self.errors(recovered, comparedWith: baseline.answers.map { $0.distribution.map(\.probability) }))
                .max() ?? 1 <= 1e-3)
        async let first = model.context(state: "The first document.")
        async let second = model.context(state: "A different second document.")
        let handles = try await [first, second]
        #expect(handles[0].key != handles[1].key)
        var accepted = 0
        var rejected = 0
        for handle in handles {
            do {
                let result = try await handle.decide(small)
                #expect(result.reusedTokens == handle.prefixTokenCount)
                accepted += 1
            } catch DecisionError.contextInvalidated {
                rejected += 1
            }
        }
        #expect(accepted == 1 && rejected == 1)
    }

    static func errors(_ result: DecisionResult, comparedWith expected: [[Float]]) throws -> [Float] {
        try #require(result.answers.count == expected.count)
        return try zip(result.answers, expected).map { answer, values in
            try #require(answer.distribution.count == values.count)
            return zip(answer.distribution.map(\.probability), values).map { abs($0 - $1) }.max() ?? 0
        }
    }
}

private actor DecisionCheckpointGate {
    private var arrived = false
    private var opened = false
    private var arrival: CheckedContinuation<Void, Never>?
    private var openings: [CheckedContinuation<Void, Never>] = []

    func hold() async {
        arrived = true
        arrival?.resume()
        arrival = nil
        if !opened { await withCheckedContinuation { openings.append($0) } }
    }

    func finished() {
        if !arrived {
            arrived = true
            arrival?.resume()
            arrival = nil
        }
    }

    func waitForArrival() async {
        if !arrived { await withCheckedContinuation { arrival = $0 } }
    }

    func release() {
        opened = true
        for continuation in openings { continuation.resume() }
        openings = []
    }
}

private actor DecisionReloadCounter {
    private(set) var value = 0

    func record() {
        value += 1
    }
}
