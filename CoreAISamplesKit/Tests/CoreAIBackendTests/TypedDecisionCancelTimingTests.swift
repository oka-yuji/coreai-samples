import Foundation
import Synchronization
import Testing

struct TypedDecisionCancelTiming: Equatable {
    let entered: Bool
    let cancelled: Bool
    let requestToEndSeconds: Double
    let startToEndSeconds: Double

    var record: [String: Any] {
        [
            "entered": entered, "cancelled": cancelled, "request_to_error_seconds": requestToEndSeconds,
            "start_to_error_seconds": startToEndSeconds,
        ]
    }
}

func cancelWhileRunning<Value: Sendable>(
    entered: inout AsyncStream<Void>.Iterator,
    now: @escaping @Sendable () -> ContinuousClock.Instant = { ContinuousClock.now },
    _ operation: @escaping @Sendable () async throws -> Value
) async -> TypedDecisionCancelTiming {
    let start = now()
    let task = Task { try await operation() }
    let observed = await entered.next() != nil
    let requested = now()
    task.cancel()
    let result = await task.result
    let end = now()
    var cancelled = false
    if case .failure(let error) = result { cancelled = error is CancellationError }
    return TypedDecisionCancelTiming(
        entered: observed, cancelled: cancelled, requestToEndSeconds: (end - requested) / .seconds(1),
        startToEndSeconds: (end - start) / .seconds(1))
}

struct TypedDecisionCancelTimingTests {
    static func clock(_ offsets: [Duration]) -> @Sendable () -> ContinuousClock.Instant {
        let origin = ContinuousClock.now
        let queue = Mutex(offsets)
        return { origin + queue.withLock { $0.removeFirst() } }
    }

    @Test("The cancellation time runs from the request to the error of a run that ignores the cancellation")
    func requestToError() async {
        let (entered, continuation) = AsyncStream.makeStream(of: Void.self)
        var iterator = entered.makeAsyncIterator()
        let elapsed = Mutex(Duration.zero)
        let origin = ContinuousClock.now
        let now: @Sendable () -> ContinuousClock.Instant = { origin + elapsed.withLock { $0 } }
        let timing = await cancelWhileRunning(entered: &iterator, now: now) {
            for _ in 0..<200 { await Task.yield() }
            elapsed.withLock { $0 = .milliseconds(1_500) }
            continuation.yield()
            while !Task.isCancelled { await Task.yield() }
            for _ in 0..<200 { await Task.yield() }
            elapsed.withLock { $0 = .seconds(4) }
            try Task.checkCancellation()
        }
        #expect(
            timing
                == TypedDecisionCancelTiming(
                    entered: true, cancelled: true, requestToEndSeconds: 2.5, startToEndSeconds: 4))
    }

    @Test("A run that finishes before the cancellation is not reported as cancelled")
    func finishedFirst() async {
        let (entered, continuation) = AsyncStream.makeStream(of: Void.self)
        var iterator = entered.makeAsyncIterator()
        let timing = await cancelWhileRunning(
            entered: &iterator, now: Self.clock([.zero, .seconds(1), .seconds(1)])
        ) {
            continuation.yield()
            return 1
        }
        #expect(timing.entered && !timing.cancelled && timing.requestToEndSeconds == 0)
    }
}
