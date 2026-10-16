import Foundation
import Synchronization

public struct PreparationProgress: Equatable, Sendable {
    public let requestID: UUID
    public let transcriptID: String
    public let processedTokens: Int
    public let totalTokens: Int
}

final class PreparationProgressHub: Sendable {
    private let subscribers = Mutex<[UUID: AsyncStream<PreparationProgress>.Continuation]>([:])

    func stream() -> AsyncStream<PreparationProgress> {
        let id = UUID()
        let (stream, continuation) = AsyncStream.makeStream(
            of: PreparationProgress.self, bufferingPolicy: .bufferingNewest(1))
        subscribers.withLock { $0[id] = continuation }
        continuation.onTermination = { [weak self] _ in
            self?.subscribers.withLock { _ = $0.removeValue(forKey: id) }
        }
        return stream
    }

    func publish(_ progress: PreparationProgress) {
        let current = subscribers.withLock { Array($0.values) }
        for subscriber in current { subscriber.yield(progress) }
    }
}
