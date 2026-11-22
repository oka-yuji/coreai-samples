import Synchronization

@testable import coreai_verify

final class FakeEvaluationLogitsSource: LogitsSource, Sendable {
    struct Failure: Error {}

    struct Item: Sendable {
        var rows: [[Float16]]
        var failsAtPrefill = false
        var failsAtStep: Int?
        var cancelsAtPrefill = false
    }

    private struct State {
        var items: [Item]
        var current = Item(rows: [])
        var position = 0
        var calls: [String] = []
    }

    private let state: Mutex<State>

    init(_ items: [Item]) {
        state = Mutex(State(items: items))
    }

    var calls: [String] { state.withLock { $0.calls } }

    func reset() {
        state.withLock { $0.calls.append("reset") }
    }

    func prefill(_ input: [Int32], usePrefillFunction: Bool) async throws {
        try state.withLock { state in
            state.calls.append("prefill \(input.count) \(usePrefillFunction)")
            state.current = state.items.isEmpty ? Item(rows: [[0, 0, 0, 1]]) : state.items.removeFirst()
            state.position = 0
            if state.current.cancelsAtPrefill { throw CancellationError() }
            if state.current.failsAtPrefill { throw Failure() }
        }
    }

    func step(_ token: Int32) async throws {
        try state.withLock { state in
            state.calls.append("step \(token)")
            state.position += 1
            if state.position == state.current.failsAtStep { throw Failure() }
        }
    }

    func logits() -> [Float16] {
        state.withLock { $0.current.rows[$0.position] }
    }
}
