import Foundation
import IrodoriInputLogic
import SpeechCore
import Synchronization

@testable import SpeechRuntime

struct TestSignal: Sendable {
    let stream: AsyncStream<Void>
    let continuation: AsyncStream<Void>.Continuation
    init() { (stream, continuation) = AsyncStream.makeStream(bufferingPolicy: .bufferingNewest(1)) }
    func fire() {
        continuation.yield(())
        continuation.finish()
    }
    func wait() async throws {
        var iterator = stream.makeAsyncIterator()
        _ = await iterator.next()
        try Task.checkCancellation()
    }
}

final class TestGate: Sendable {
    struct State {
        var open = false
        var waiters: [CheckedContinuation<Void, Never>] = []
    }
    let state = Mutex(State())
    func wait() async {
        await withCheckedContinuation { continuation in
            let immediate = state.withLock { value in
                if value.open { return true }
                value.waiters.append(continuation)
                return false
            }
            if immediate { continuation.resume() }
        }
    }
    func open() {
        let waiters = state.withLock { value in
            value.open = true
            let pending = value.waiters
            value.waiters.removeAll()
            return pending
        }
        for waiter in waiters { waiter.resume() }
    }
}

final class LifetimeLog: Sendable {
    let values = Mutex<[String]>([])
    func add(_ event: String) { values.withLock { $0.append(event) } }
    func snapshot() -> [String] { values.withLock { $0 } }
}

final class TestContext: SpeechContext {
    let frames: Int
    let log: LifetimeLog
    init(frames: Int, log: LifetimeLog) {
        self.frames = frames
        self.log = log
    }
    deinit { log.add("context-ended") }
}

final class PipelineLifetime: Sendable {
    let log: LifetimeLog
    init(log: LifetimeLog) { self.log = log }
    deinit { log.add("pipeline-ended") }
}

final class TestTokenizer: SpeechTokenizer {
    func prepare(_ text: String) throws -> IrodoriTextInput {
        if text == "empty" { throw IrodoriInputError.emptyText }
        return try IrodoriTextInput(
            normalizedText: text, bodyTokenIDs: [UInt32(text) ?? 13],
            originalTokenCount: text == "long" ? 256 : 1)
    }
}

struct TestPipeline: SpeechPipeline {
    let lifetime: PipelineLifetime
    let driver: TestDriver
    @concurrent
    func prepare(_ input: IrodoriTextInput) async throws -> any SpeechContext {
        lifetime.log.add("prepare")
        return TestContext(frames: Int(input.tokenIDs[1]), log: lifetime.log)
    }
    @concurrent
    func generate(_ context: any SpeechContext, seed: UInt64) async throws -> [Float] {
        try await driver.generate(frames: context.frames, seed: seed)
    }
}

actor TestDriver {
    let started = TestSignal()
    let gate = TestGate()
    let log = LifetimeLog()
    var block = false
    var fail = false
    var factoryCount = 0
    var seeds: [UInt64] = []
    func configure(block: Bool = false, fail: Bool = false) {
        self.block = block
        self.fail = fail
    }
    func makePipeline() -> any SpeechPipeline {
        factoryCount += 1
        log.add("factory-\(factoryCount)")
        return TestPipeline(lifetime: PipelineLifetime(log: log), driver: self)
    }
    func generate(frames: Int, seed: UInt64) async throws -> [Float] {
        seeds.append(seed)
        started.fire()
        if block { await gate.wait() }
        if fail { throw SpeechError.invalidSamples }
        return [Float](repeating: 0, count: frames * 1920)
    }
}

func testDependencies(_ driver: TestDriver, releasing: TestSignal = TestSignal()) -> SpeechDependencies {
    SpeechDependencies(
        validate: {
            ValidatedSpeechBundle(directory: $0, assets: [], tokenizer: $0.appendingPathComponent("tokenizer.json"))
        },
        tokenizer: { _ in TestTokenizer() },
        pipeline: { _ in await driver.makePipeline() },
        didRequestRelease: { releasing.fire() })
}
