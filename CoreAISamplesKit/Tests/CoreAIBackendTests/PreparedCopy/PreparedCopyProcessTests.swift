import Foundation
import Testing

@testable import CoreAIBackend

@Suite(.timeLimit(.minutes(1))) struct PreparedCopyProcessTests {
    private static let mark = PreparedCopyMark(
        attempt: "probe", pid: 201, start: 3_000, kind: .normal, stage: .load, at: 0)

    @Test(
        "AC05 A mark is dead when its PID has another start time, has no process or is a zombie, and unknown when the probe fails",
        arguments: zip(
            [
                PreparedCopyProcess.Probe.present(start: 3_001, zombie: false), .absent,
                .present(start: 3_000, zombie: true), .unavailable,
            ],
            [PreparedCopyLiveness.dead, .dead, .dead, .unknown]))
    func probedLiveness(probe: PreparedCopyProcess.Probe, liveness: PreparedCopyLiveness) {
        #expect(PreparedCopyProcess.liveness(of: Self.mark, probe: probe) == liveness)
    }

    @Test(
        "AC05 Under the real probe the current process is alive, its PID with another start time is dead and an unused PID is absent"
    )
    func realProbe() throws {
        let current = try #require(PreparedCopyProcess.current())
        #expect(current.pid == getpid())
        var mark = PreparedCopyMark(
            attempt: "current", pid: current.pid, start: current.start, kind: .normal, stage: .load, at: 0)
        #expect(PreparedCopyProcess.liveness(of: mark) == .alive)
        mark.start += 1
        #expect(PreparedCopyProcess.liveness(of: mark) == .dead)
        #expect(PreparedCopyProcess.probe(Int32.max) == .absent)
    }
}
