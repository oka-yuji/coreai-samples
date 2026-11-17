import Synchronization

@testable import SpeechRuntime

final class FakeVoxCPMLog: Sendable {
    struct Call: Sendable {
        let generation: Int
        let function: VoxCPMFunction
        let inputs: [String: VoxCPMInput]
        let outputs: [String: [Float]]
        let stateSum: Float?
    }

    enum Event: Equatable, Sendable {
        case created(Int)
        case called(Int)
        case released(Int)
    }

    private let state = Mutex<(calls: [Call], events: [Event], created: Int)>(([], [], 0))

    var calls: [Call] { state.withLock { $0.calls } }
    var events: [Event] { state.withLock { $0.events } }

    func create() -> Int {
        state.withLock {
            $0.created += 1
            $0.events.append(.created($0.created))
            return $0.created
        }
    }

    func record(_ call: Call) {
        state.withLock {
            $0.calls.append(call)
            $0.events.append(.called(call.generation))
        }
    }

    func release(_ generation: Int) {
        state.withLock { $0.events.append(.released(generation)) }
    }

    func calls(in generation: Int) -> [Call] {
        calls.filter { $0.generation == generation }
    }
}
