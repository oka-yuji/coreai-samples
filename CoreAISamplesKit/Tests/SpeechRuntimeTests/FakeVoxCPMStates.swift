import Synchronization

@testable import SpeechRuntime

final class FakeVoxCPMStates: Sendable {
    let generation: Int
    private let log: FakeVoxCPMLog
    private let storage = Mutex<(sums: [String: Float], counts: [VoxCPMFunction: Int], calls: Int)>(([:], [:], 0))

    init(log: FakeVoxCPMLog) {
        self.log = log
        generation = log.create()
    }

    deinit { log.release(generation) }

    func next(_ function: VoxCPMFunction) -> (call: Int, occurrence: Int) {
        storage.withLock {
            let occurrence = $0.counts[function, default: 0]
            $0.counts[function] = occurrence + 1
            $0.calls += 1
            return ($0.calls - 1, occurrence)
        }
    }

    func sum(_ group: String) -> Float {
        storage.withLock { $0.sums[group, default: 0] }
    }

    func write(_ group: String, _ value: Float) {
        storage.withLock { $0.sums[group, default: 0] += value }
    }
}
