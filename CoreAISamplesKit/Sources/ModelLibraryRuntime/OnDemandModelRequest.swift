import Foundation

@MainActor
final class OnDemandModelRequest {
    let owner: UUID
    let generation: Int
    let isCurrent: @MainActor () -> Bool
    private var acceptsWaiters = true
    private var waiters: [UUID: AsyncStream<Never>.Continuation] = [:]

    init(owner: UUID, generation: Int, isCurrent: @escaping @MainActor () -> Bool) {
        self.owner = owner
        self.generation = generation
        self.isCurrent = isCurrent
    }

    func wait() async {
        guard acceptsWaiters, !Task.isCancelled else { return }
        let id = UUID()
        let (stream, continuation) = AsyncStream<Never>.makeStream()
        waiters[id] = continuation
        defer { waiters[id] = nil }
        var iterator = stream.makeAsyncIterator()
        _ = await iterator.next()
    }

    func endWaiting() {
        acceptsWaiters = false
        for waiter in waiters.values { waiter.finish() }
        waiters.removeAll()
    }
}
