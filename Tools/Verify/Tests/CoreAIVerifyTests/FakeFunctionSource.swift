import Synchronization

@testable import coreai_verify

final class FakeFunctionSource: FunctionSource, Sendable {
    struct Failure: Error, Equatable {}

    struct Script: Sendable {
        var mainHash: String? = "0123abcd"
        var functions = [FunctionFixture.signature]
        var load = FunctionLoad(
            requestedCompute: "default", expectFrequentReshapes: true, cached: false, cacheQueryError: nil,
            specializeS: 0.5, loadS: 0.75)
        var failsAtLoad = false
        var cancelsAtLoad = false
        var cancelsTaskAtLoad = false
        var outputs = [FunctionFixture.outputs()]
        var failsAtCall: Int?
        var cancelsAtCall: Int?
        var cancelsTaskAtCall: Int?
        var beforeCall: (@Sendable (Int) -> Void)?
    }

    private struct State {
        var calls: [String] = []
        var received: [[FunctionTensor]] = []
    }

    private let script: Script
    private let state = Mutex(State())

    init(_ script: Script = Script()) {
        self.script = script
    }

    var calls: [String] { state.withLock { $0.calls } }
    var received: [[FunctionTensor]] { state.withLock { $0.received } }

    func describe() throws -> (mainHash: String?, functions: [FunctionSignature]) {
        state.withLock { $0.calls.append("describe") }
        return (script.mainHash, script.functions)
    }

    func load(_ function: String) async throws -> FunctionLoad {
        state.withLock { $0.calls.append("load \(function)") }
        if script.cancelsAtLoad { throw CancellationError() }
        if script.failsAtLoad { throw Failure() }
        if script.cancelsTaskAtLoad { Self.cancelTheTask() }
        return script.load
    }

    func call(_ inputs: [FunctionTensor]) async throws -> [FunctionTensor] {
        let number = state.withLock { state in
            state.calls.append("call")
            state.received.append(inputs)
            return state.received.count
        }
        script.beforeCall?(number)
        if script.cancelsAtCall == number { throw CancellationError() }
        if script.failsAtCall == number { throw Failure() }
        if script.cancelsTaskAtCall == number { Self.cancelTheTask() }
        return script.outputs[min(number, script.outputs.count) - 1]
    }

    private static func cancelTheTask() {
        _ = withUnsafeCurrentTask { $0?.cancel() }
    }
}
