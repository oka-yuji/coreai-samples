import Foundation
import Testing

@testable import CoreAIBackend

actor TypedDecisionGate {
    private var opened = false
    private var waiting: [Int: CheckedContinuation<Void, Never>] = [:]
    private var arrivals = 0

    func hold() async {
        arrivals += 1
        let id = arrivals
        guard !opened else { return }
        await withCheckedContinuation { waiting[id] = $0 }
    }

    func pass() async {
        arrivals += 1
        let id = arrivals
        await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                if opened || Task.isCancelled { continuation.resume() } else { waiting[id] = continuation }
            }
        } onCancel: {
            Task { await self.release(id) }
        }
    }

    func open() {
        opened = true
        for continuation in waiting.values { continuation.resume() }
        waiting.removeAll()
    }

    private func release(_ id: Int) {
        waiting.removeValue(forKey: id)?.resume()
    }
}

actor TypedDecisionFakeBackbone: TypedDecisionBackbone {
    private(set) var counts: [Int] = []
    private(set) var capacities: [Int] = []
    private let hidden: [Float]
    private let gate: TypedDecisionGate?
    private let entered: AsyncStream<Int>.Continuation?

    init(gate: TypedDecisionGate? = nil, entered: AsyncStream<Int>.Continuation? = nil) {
        hidden = [1, 0, 0, 0]
        self.gate = gate
        self.entered = entered
    }

    func forward(_ ids: [Int], capacity: Int) async throws -> [Float] {
        try Task.checkCancellation()
        counts.append(ids.count)
        capacities.append(capacity)
        entered?.yield(counts.count)
        if counts.count == 1 { await gate?.hold() }
        return hidden
    }
}

private struct TypedDecisionFakeModel {
    let model: TypedDecisionModel
    let backbone: TypedDecisionFakeBackbone
    let root: URL

    static let encode: @Sendable (String) -> [Int] = { $0.unicodeScalars.map { Int($0.value) } }

    init(backbone: TypedDecisionFakeBackbone = TypedDecisionFakeBackbone(), head: Data? = nil) throws {
        let root = try TypedDecisionFakeBundle.create()
        try TypedDecisionFakeBundle.write(TypedDecisionFakeBundle.document(), to: root)
        let bundle = try TypedDecisionBundle(url: root)
        var headURL = bundle.headURL
        if let head {
            headURL = root.deletingLastPathComponent().appending(path: "custom-head.safetensors")
            try head.write(to: headURL)
        }
        self.root = root
        self.backbone = backbone
        model = TypedDecisionModel(
            bundle: bundle, encode: Self.encode, head: try TypedDecisionHead(url: headURL, width: 4),
            backbone: backbone)
    }

    func remove() {
        try? FileManager.default.removeItem(at: root.deletingLastPathComponent())
    }
}

private let noul = TypedDecisionRequest(state: "state", question: "question", kind: .noul, options: ["false", "true"])

@Suite(.timeLimit(.minutes(1)))
struct TypedDecisionModelTests {
    @Test("Cancelling a request during the backbone forward ends it with CancellationError and the next request runs")
    func cancellationDuringForward() async throws {
        let gate = TypedDecisionGate()
        let (entered, enteredContinuation) = AsyncStream.makeStream(of: Int.self)
        let (started, startedContinuation) = AsyncStream.makeStream(of: Void.self)
        let fake = try TypedDecisionFakeModel(
            backbone: TypedDecisionFakeBackbone(gate: gate, entered: enteredContinuation))
        defer { fake.remove() }
        await fake.model.setForwardCheckpoint { startedContinuation.yield() }
        let task = Task { try await fake.model.decide(noul) }
        var startedIterator = started.makeAsyncIterator()
        var enteredIterator = entered.makeAsyncIterator()
        #expect(await startedIterator.next() != nil)
        #expect(await enteredIterator.next() == 1)
        task.cancel()
        await gate.open()
        await #expect(throws: CancellationError.self) { try await task.value }
        let answer = try await fake.model.decide(noul)
        #expect(answer.probabilities.count == 2)
        #expect(await fake.backbone.counts.count == 2)
        #expect(await enteredIterator.next() == 2)
    }

    @Test("A second request while one is running fails with busy until the backbone that ignores cancellation returns")
    func busyUntilTheBackboneReturns() async throws {
        let checkpointGate = TypedDecisionGate()
        let backboneGate = TypedDecisionGate()
        let (entered, enteredContinuation) = AsyncStream.makeStream(of: Int.self)
        let (started, startedContinuation) = AsyncStream.makeStream(of: Void.self)
        let fake = try TypedDecisionFakeModel(
            backbone: TypedDecisionFakeBackbone(gate: backboneGate, entered: enteredContinuation))
        defer { fake.remove() }
        await fake.model.setForwardCheckpoint {
            startedContinuation.yield()
            await checkpointGate.pass()
        }
        let task = Task { try await fake.model.decide(noul) }
        var startedIterator = started.makeAsyncIterator()
        var enteredIterator = entered.makeAsyncIterator()
        #expect(await startedIterator.next() != nil)
        await #expect(throws: TypedDecisionError.busy) { try await fake.model.decide(noul) }
        #expect(await fake.backbone.counts.isEmpty)
        await checkpointGate.open()
        #expect(await enteredIterator.next() == 1)
        task.cancel()
        await #expect(throws: TypedDecisionError.busy) { try await fake.model.decide(noul) }
        #expect(await fake.backbone.counts.count == 1)
        await backboneGate.open()
        await #expect(throws: CancellationError.self) { try await task.value }
        let answer = try await fake.model.decide(noul)
        #expect(answer.tokenCount == TypedDecisionFakeModel.encode(try TypedDecisionRenderer.render(noul)).count)
        #expect(await fake.backbone.counts.count == 2)
    }

    @Test("Requests that the renderer refuses never reach the backbone")
    func refusedRequestsSkipTheBackbone() async throws {
        let fake = try TypedDecisionFakeModel()
        defer { fake.remove() }
        let options = (1...17).map { "option \($0)" }
        let wide = TypedDecisionRequest(state: "s", question: "q", kind: .choice, options: options)
        let error = await #expect(throws: TypedDecisionError.self) { try await fake.model.decide(wide) }
        guard case .invalidOptions? = error else {
            Issue.record("Expected invalidOptions, got \(String(describing: error))")
            return
        }
        let long = TypedDecisionRequest(
            state: String(repeating: "x", count: 8_192), question: "q", kind: .noul, options: ["false", "true"])
        let count = TypedDecisionFakeModel.encode(try TypedDecisionRenderer.render(long)).count
        await #expect(throws: TypedDecisionError.tooManyTokens(count: count, limit: 8_192)) {
            try await fake.model.decide(long)
        }
        #expect(await fake.backbone.counts.isEmpty)
    }

    @Test("Loading a bundle whose asset fails its SHA-256 throws invalidBundle before the tokenizer and the asset load")
    func brokenBundleFailsBeforeTheAsset() async throws {
        let root = try TypedDecisionFakeBundle.create()
        defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent()) }
        try TypedDecisionFakeBundle.write(TypedDecisionFakeBundle.document(), to: root)
        try Data("MAIN".utf8).write(to: root.appending(path: "fixture.aimodel/main.mlirb"))
        await #expect(throws: TypedDecisionError.invalidBundle("SHA-256 mismatch: fixture.aimodel/main.mlirb.")) {
            _ = try await TypedDecisionModel.load(bundle: root)
        }
    }

    @Test("The answer carries the head logits and probabilities and a tie selects the first option")
    func answerMatchesTheHead() async throws {
        var weight = [[Float]](repeating: [0, 0, 0, 0], count: 24)
        weight[8][0] = 2
        weight[9][0] = 2
        weight[10][0] = 1
        let head = TypedDecisionHeadFile.data(weight: weight, bias: [Float](repeating: 0, count: 24))
        let fake = try TypedDecisionFakeModel(head: head)
        defer { fake.remove() }
        let request = TypedDecisionRequest(state: "s", question: "q", kind: .choice, options: ["a", "b", "c"])
        let answer = try await fake.model.decide(request)
        let reference = try TypedDecisionHead(
            url: fake.root.deletingLastPathComponent().appending(path: "custom-head.safetensors"), width: 4)
        let logits = try reference.logits(hidden: [1, 0, 0, 0])
        let expected = try reference.probabilities(logits: logits, kind: .choice, count: 3, temperature: 0.75)
        #expect(answer.logits == logits)
        #expect(answer.probabilities == expected.shipping)
        #expect(answer.probabilitiesT1 == expected.plain)
        #expect(answer.probabilities != answer.probabilitiesT1)
        #expect(answer.probabilities[0] == answer.probabilities[1] && answer.probabilities[0] > answer.probabilities[2])
        #expect(answer.top == 0)
        #expect(answer.kind == .choice && answer.options == ["a", "b", "c"])
    }

    @Test("The backbone capacity is the token count rounded up to 256 and never exceeds 8,192")
    func capacityRoundsUpWithinTheLimit() async throws {
        let fake = try TypedDecisionFakeModel()
        defer { fake.remove() }
        let base = TypedDecisionFakeModel.encode(try TypedDecisionRenderer.render(noul)).count - "state".count
        let counts = [base, 255, 256, 257, 512, 513, 8_000, 8_191, 8_192]
        for count in counts {
            let request = TypedDecisionRequest(
                state: String(repeating: "s", count: count - base), question: "question", kind: .noul,
                options: ["false", "true"])
            #expect(try await fake.model.decide(request).tokenCount == count)
        }
        let expected = counts.map { min(($0 + 255) / 256 * 256, 8_192) }
        #expect(await fake.backbone.counts == counts)
        #expect(await fake.backbone.capacities == expected)
        #expect(expected.allSatisfy { $0 <= 8_192 && $0.isMultiple(of: 256) })
        _ = try await fake.model.decide(noul, capacity: 8_192)
        #expect(await fake.backbone.capacities.last == 8_192)
        #expect(fake.model.capacity(forTokens: 8_192) == 8_192 && fake.model.capacity(forTokens: 1) == 256)
    }
}
