import Foundation
import SpokenSearchFeature

@MainActor
public final class SpokenFileTranscriber {
    private var stopRequest: AsyncStream<Never>.Continuation?
    private var stopWaiters: [CheckedContinuation<Void, Never>] = []

    public init() {}

    public func transcribe(_ url: URL) async throws -> SpokenTranscript {
        let job = SpokenFileTranscriberJob()
        return try await run(recognize: { try await job.recognize(url) }, finish: { await job.finish() })
    }

    func run(
        recognize: @escaping @Sendable () async throws -> SpokenTranscript,
        finish: @escaping @Sendable () async -> Void
    ) async throws -> SpokenTranscript {
        try Task.checkCancellation()
        guard stopRequest == nil else { throw SpokenRecognitionError.busy }
        let (stopped, request) = AsyncStream<Never>.makeStream()
        stopRequest = request
        defer {
            stopRequest = nil
            for waiter in stopWaiters { waiter.resume() }
            stopWaiters.removeAll()
        }

        let result = await withTaskGroup(of: Result<SpokenTranscript, any Error>.self) { group in
            group.addTask {
                do {
                    return .success(try await recognize())
                } catch {
                    return .failure(error)
                }
            }
            group.addTask {
                var iterator = stopped.makeAsyncIterator()
                _ = await iterator.next()
                return .failure(CancellationError())
            }
            let first = await group.next()
            group.cancelAll()
            request.finish()
            await finish()
            return first
        }
        guard let result, stopWaiters.isEmpty, !Task.isCancelled else { throw CancellationError() }
        return try result.get()
    }

    public func cancel() async {
        guard let stopRequest else { return }
        await withCheckedContinuation { waiter in
            stopWaiters.append(waiter)
            stopRequest.finish()
        }
    }
}
