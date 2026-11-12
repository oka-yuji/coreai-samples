import Dispatch
import Foundation
import Observation
import Synchronization

@testable import ChronosDemoRuntime

final class ChronosPublicationCheckpoint: Sendable {
    private struct Reference: Sendable { weak var permit: ChronosPublicationPermit? }
    private struct State {
        var references: [Reference] = []
        var writes: [Data] = []
    }
    private let state = Mutex(State())
    private let release = DispatchSemaphore(value: 0)
    let paused = AsyncStream<Void>.makeStream()

    var attempts: Int { state.withLock { $0.references.count } }
    var retainedPermits: Int { state.withLock { $0.references.filter { $0.permit != nil }.count } }
    var writes: [Data] { state.withLock { $0.writes } }

    func write(_ data: Data, _ file: FileHandle) throws {
        state.withLock { $0.writes.append(data) }
        try file.write(contentsOf: data)
    }

    func beforePublish(_ permit: ChronosPublicationPermit) {
        let count = state.withLock {
            $0.references.append(Reference(permit: permit))
            return $0.references.count
        }
        if count == 1 {
            paused.continuation.yield(())
            release.wait()
        }
    }

    func resume() { release.signal() }

    func finish() {
        resume()
        paused.continuation.finish()
    }
}

@MainActor
@Observable
final class ChronosPublicationOwner {
    var revision = 0
}
