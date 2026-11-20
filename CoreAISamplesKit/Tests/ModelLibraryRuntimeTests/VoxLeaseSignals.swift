import Foundation
import Synchronization

struct VoxLeaseProbeError: Error, CustomStringConvertible {
    let description: String
}

func requireLease(_ ok: Bool, _ message: String) throws {
    guard ok else { throw VoxLeaseProbeError(description: message) }
}

final class VoxLeaseSignals: Sendable {
    private struct Waiter {
        let label: String
        let count: Int
        let continuation: AsyncStream<Void>.Continuation
    }
    private struct State {
        var log: [String] = []
        var waiters: [UUID: Waiter] = [:]
    }
    private let state = Mutex(State())
    var log: [String] { state.withLock { $0.log } }

    func record(_ label: String) {
        let ready = state.withLock { s in
            s.log.append(label)
            let count = s.log.count { $0 == label }
            let matches = s.waiters.filter { $0.value.label == label && $0.value.count <= count }
            for id in matches.keys { s.waiters.removeValue(forKey: id) }
            return matches.map(\.value.continuation)
        }
        for item in ready {
            item.yield(())
            item.finish()
        }
    }

    func wait(_ label: String, _ count: Int = 1) async throws {
        try Task.checkCancellation()
        let id = UUID()
        let (stream, continuation) = AsyncStream<Void>.makeStream()
        let arrived = state.withLock { s in
            if s.log.count(where: { $0 == label }) >= count { return true }
            s.waiters[id] = Waiter(label: label, count: count, continuation: continuation)
            return false
        }
        defer {
            _ = state.withLock { $0.waiters.removeValue(forKey: id) }
            continuation.finish()
        }
        if !arrived {
            var iterator = stream.makeAsyncIterator()
            guard await iterator.next() != nil else { throw CancellationError() }
        }
        try Task.checkCancellation()
    }
}

final class VoxLeaseGate: Sendable {
    private struct State {
        var closed: Bool
        var waiters: [CheckedContinuation<Void, Never>] = []
    }
    private let state: Mutex<State>
    init(closed: Bool) { state = Mutex(State(closed: closed)) }
    func waitIgnoringCancellation() async {
        await withCheckedContinuation { waiter in
            let open = state.withLock { state in
                guard state.closed else { return true }
                state.waiters.append(waiter)
                return false
            }
            if open { waiter.resume() }
        }
    }
    func open() {
        let pending = state.withLock { state in
            state.closed = false
            defer { state.waiters.removeAll() }
            return state.waiters
        }
        for item in pending { item.resume() }
    }
}
