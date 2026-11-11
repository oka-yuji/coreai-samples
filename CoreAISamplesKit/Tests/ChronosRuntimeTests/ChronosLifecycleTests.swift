import ChronosCore
import Foundation
import Testing

@testable import ChronosRuntime

@Suite(.timeLimit(.minutes(1)))
struct ChronosLifecycleTests {
    @Test("Repeated predictions reuse the model and release forces a fresh load")
    func reuse() async throws {
        let events = AsyncStream<String>.makeStream()
        defer { events.continuation.finish() }
        let factory = ControlledFactory(events: events.continuation)
        let runtime = ChronosRuntime(factory: { try await factory.make() })
        let context = try ChronosContext([1])
        let first = try await runtime.predict(context)
        let second = try await runtime.predict(context)
        #expect(first.loaded)
        #expect(!second.loaded)
        #expect(first.prediction == second.prediction)
        await runtime.release()
        let third = try await runtime.predict(context)
        #expect(third.loaded)
        #expect(await factory.loads == 2)
        await runtime.release()
    }

    @Test("Release joins noncooperative loading and inference for every waiter", arguments: [false, true])
    func releaseJoin(inference: Bool) async throws {
        let events = AsyncStream<String>.makeStream()
        defer { events.continuation.finish() }
        var iterator = events.stream.makeAsyncIterator()
        let gate = ChronosGate()
        defer { gate.finish(.failure(CancellationError())) }
        let factory = ControlledFactory(events: events.continuation, gate: gate, pauseInference: inference)
        let runtime = ChronosRuntime(
            factory: { try await factory.make() },
            observe: { events.continuation.yield($0 == .releaseWaiting ? "waiting" : "released") })
        let context = try ChronosContext([1])
        let operation = Task { try await runtime.predict(context) }
        #expect(await iterator.next() == "load-1")
        if inference { #expect(await iterator.next() == "predict-1") }
        await #expect(throws: ChronosError.busy) { try await runtime.predict(context) }
        let firstRelease = Task { await runtime.release() }
        #expect(await iterator.next() == "waiting")
        let secondRelease = Task { await runtime.release() }
        #expect(await iterator.next() == "waiting")
        operation.cancel()
        firstRelease.cancel()
        await #expect(throws: ChronosError.busy) { try await runtime.predict(context) }
        gate.finish()
        #expect(await iterator.next() == "drop-1")
        #expect(await iterator.next() == "released")
        let next = Task { try await runtime.predict(context) }
        await firstRelease.value
        await secondRelease.value
        switch await operation.result {
        case .success: Issue.record("A closing prediction must not publish a late result")
        case .failure(let error): #expect(error is CancellationError)
        }
        let newResult = try await next.value
        #expect(newResult.loaded)
        let reused = try await runtime.predict(context)
        #expect(!reused.loaded)
        #expect(await factory.loads == 2)
        await runtime.release()
    }

    @Test("Cancellation without release discards late work and reloads cleanly", arguments: [false, true])
    func cancellation(inference: Bool) async throws {
        let events = AsyncStream<String>.makeStream()
        defer { events.continuation.finish() }
        var iterator = events.stream.makeAsyncIterator()
        let gate = ChronosGate()
        defer { gate.finish(.failure(CancellationError())) }
        let factory = ControlledFactory(events: events.continuation, gate: gate, pauseInference: inference)
        let runtime = ChronosRuntime(factory: { try await factory.make() })
        let context = try ChronosContext([1])
        let operation = Task { try await runtime.predict(context) }
        #expect(await iterator.next() == "load-1")
        if inference { #expect(await iterator.next() == "predict-1") }
        operation.cancel()
        await #expect(throws: ChronosError.busy) { try await runtime.predict(context) }
        gate.finish()
        switch await operation.result {
        case .success: Issue.record("Cancellation must reject a noncooperative result")
        case .failure(let error): #expect(error is CancellationError)
        }
        #expect(await iterator.next() == "drop-1")
        #expect(try await runtime.predict(context).loaded)
        #expect(await factory.loads == 2)
        await runtime.release()
    }

    @Test("Load and inference errors unblock release and allow retry", arguments: [false, true])
    func failures(inference: Bool) async throws {
        let events = AsyncStream<String>.makeStream()
        defer { events.continuation.finish() }
        var iterator = events.stream.makeAsyncIterator()
        let gate = ChronosGate()
        defer { gate.finish(.failure(CancellationError())) }
        let factory = ControlledFactory(events: events.continuation, gate: gate, pauseInference: inference)
        let runtime = ChronosRuntime(
            factory: { try await factory.make() },
            observe: { events.continuation.yield($0 == .releaseWaiting ? "waiting" : "released") })
        let context = try ChronosContext([1])
        let operation = Task { try await runtime.predict(context) }
        #expect(await iterator.next() == "load-1")
        if inference { #expect(await iterator.next() == "predict-1") }
        let release = Task { await runtime.release() }
        #expect(await iterator.next() == "waiting")
        gate.finish(.failure(ChronosError.invalidModel))
        switch await operation.result {
        case .success: Issue.record("The fake failure must reach the caller")
        case .failure(let error): #expect(error as? ChronosError == .invalidModel)
        }
        await release.value
        #expect(try await runtime.predict(context).loaded)
        #expect(await factory.loads == 2)
        await runtime.release()
    }

    @Test("An already cancelled request never calls the factory")
    func cancelledBeforeStart() async throws {
        let gate = ChronosGate()
        let events = AsyncStream<String>.makeStream()
        defer { events.continuation.finish() }
        let factory = ControlledFactory(events: events.continuation)
        let runtime = ChronosRuntime(factory: { try await factory.make() })
        let context = try ChronosContext([1])
        let operation = Task {
            try await gate.wait()
            return try await runtime.predict(context)
        }
        operation.cancel()
        gate.finish()
        switch await operation.result {
        case .success: Issue.record("A cancelled request must fail")
        case .failure(let error): #expect(error is CancellationError)
        }
        #expect(await factory.loads == 0)
        await runtime.release()
    }
}

private actor ControlledFactory {
    let events: AsyncStream<String>.Continuation
    let gate: ChronosGate?
    let pauseInference: Bool
    private(set) var loads = 0

    init(events: AsyncStream<String>.Continuation, gate: ChronosGate? = nil, pauseInference: Bool = false) {
        self.events = events
        self.gate = gate
        self.pauseInference = pauseInference
    }

    func make() async throws -> any ChronosPredictor {
        loads += 1
        let number = loads
        events.yield("load-\(number)")
        let predictor = LifetimePredictor(
            events: events, number: number, gate: number == 1 && pauseInference ? gate : nil)
        if number == 1, !pauseInference { try await gate?.wait() }
        return predictor
    }
}

private final class LifetimePredictor: ChronosPredictor {
    let identity = ChronosTestFolder.identity
    let events: AsyncStream<String>.Continuation
    let number: Int
    let gate: ChronosGate?

    init(events: AsyncStream<String>.Continuation, number: Int, gate: ChronosGate?) {
        self.events = events
        self.number = number
        self.gate = gate
    }

    deinit { events.yield("drop-\(number)") }

    func predict(_ context: ChronosContext) async throws -> ChronosPrediction {
        events.yield("predict-\(number)")
        try await gate?.wait()
        return try ChronosPrediction(quantiles: ChronosTestFolder.rows())
    }
}
