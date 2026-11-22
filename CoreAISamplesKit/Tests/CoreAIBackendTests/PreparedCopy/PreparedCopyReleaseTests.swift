import Foundation
import Testing

@testable import CoreAIBackend

@Suite(.timeLimit(.minutes(1))) struct PreparedCopyReleaseTests {
    private typealias Fixture = PreparedCopyFixture

    @Test(
        "After a recovery whose deletion threw, a manual clear succeeds and removes the ledger, and the next load is normal"
    )
    func clearAfterFailedDeletion() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(Fixture.record([Fixture.dead]))
        let recovery = fixture.recovery()
        await #expect(throws: POSIXError(.EIO)) {
            _ = try await recovery.load(delete: { throw POSIXError(.EIO) }, load: { 1 }, warmUp: { _ in })
        }
        #expect(recovery.record().failed)
        let recorder = PreparedCopyRecorder()
        try recovery.clear(delete: { recorder.note("clear") })
        #expect(!Fixture.exists(recovery.file))
        let result = try await recorder.load(recovery)
        #expect(!result.recovered)
        #expect(recorder.events == ["clear", "load", "warmUp"])
    }

    @Test(
        "When the warm-up of a recovery throws another error, the ledger becomes terminal with the own mark at the warm-up stage and a manual clear succeeds"
    )
    func failedRecoveryWarmUpIsTerminal() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(Fixture.record([Fixture.dead]))
        let recovery = fixture.recovery()
        await #expect(throws: POSIXError(.EIO)) {
            _ = try await recovery.load(delete: {}, load: { 1 }, warmUp: { _ in throw POSIXError(.EIO) })
        }
        let record = recovery.record()
        let mark = try #require(record.marks.first)
        #expect(
            record
                == Fixture.record(
                    [Fixture.own(mark.attempt, .recovery, .warmUp)], recoveries: 1, suspects: [Fixture.dead],
                    failed: true))
        let recorder = PreparedCopyRecorder()
        try recovery.clear(delete: { recorder.note("clear") })
        #expect(recorder.events == ["clear"])
    }

    @Test(
        "When the warm-up of a normal load throws, only its own mark is removed, other living marks stay and a manual clear succeeds"
    )
    func failedNormalWarmUpRemovesOwnMark() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(Fixture.record([Fixture.alive]))
        let recovery = fixture.recovery()
        let recorder = PreparedCopyRecorder()
        await #expect(throws: POSIXError(.EIO)) {
            _ = try await recovery.load(
                delete: { recorder.note("delete") }, load: { 1 }, warmUp: { _ in throw POSIXError(.EIO) })
        }
        #expect(recovery.record() == Fixture.record([Fixture.alive]))
        try recovery.clear(delete: { recorder.note("clear") })
        #expect(recorder.events == ["clear"])
    }

    @Test("During a recovery the lock is free inside the notification, deletion, load and warm-up closures")
    func recoveryClosuresRunWithoutLock() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(Fixture.record([Fixture.dead]))
        let recovery = fixture.recovery()
        var free: [String: Bool] = [:]
        let result = try await recovery.load(
            onRecovery: { free["onRecovery"] = fixture.lockIsFree() },
            delete: { free["delete"] = fixture.lockIsFree() },
            load: {
                free["load"] = fixture.lockIsFree()
                return 1
            }, warmUp: { _ in free["warmUp"] = fixture.lockIsFree() })
        #expect(result.recovered)
        #expect(free == ["onRecovery": true, "delete": true, "load": true, "warmUp": true])
    }

    @Test(
        "A living recovery mark whose use is free is a recovery that could not record its end, so a load fails as recovery failed without changing the ledger"
    )
    func stalledRecoveryMarkFails() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(Fixture.record([Fixture.aliveRecovery], recoveries: 1, suspects: [Fixture.dead]))
        let before = try fixture.bytes()
        let recorder = PreparedCopyRecorder()
        await #expect(throws: PreparedCopyError.recoveryFailed) {
            _ = try await recorder.load(fixture.recovery())
        }
        let after = try fixture.bytes()
        #expect(after == before)
        #expect(recorder.events.isEmpty)
    }

    @Test(
        "When a failed deletion cannot record the terminal state, a repeated request from the same process still returns recovery failed rather than preparing"
    )
    func unrecordedDeletionFailureIsTerminal() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(Fixture.record([Fixture.dead]))
        let path = fixture.directory.path(percentEncoded: false)
        defer { chmod(path, 0o700) }
        let recovery = fixture.recovery()
        let recorder = PreparedCopyRecorder()
        await #expect(throws: POSIXError(.EIO)) {
            _ = try await recovery.load(
                delete: {
                    recorder.note("delete")
                    #expect(chmod(path, 0o500) == 0)
                    throw POSIXError(.EIO)
                }, load: { 1 }, warmUp: { _ in })
        }
        let record = recovery.record()
        let mark = try #require(record.marks.first)
        #expect(
            record
                == Fixture.record(
                    [Fixture.own(mark.attempt, .recovery, .load)], recoveries: 1, suspects: [Fixture.dead]))
        await #expect(throws: PreparedCopyError.recoveryFailed) {
            _ = try await recorder.load(recovery)
        }
        #expect(recorder.events == ["delete"])
    }

    @Test(
        "A living recovery mark that could not record its end still fails as recovery failed while another load holds the use, and the ledger stays unchanged"
    )
    func stalledRecoveryMarkFailsWhileShared() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let recorder = PreparedCopyRecorder()
        do {
            let held = try await recorder.load(fixture.recovery(Fixture.second), label: "second.")
            try fixture.write(
                Fixture.record(
                    [Fixture.own("stalled", .recovery, .load)], recoveries: 1, suspects: [Fixture.dead]))
            let before = try fixture.bytes()
            await #expect(throws: PreparedCopyError.recoveryFailed) {
                _ = try await recorder.load(fixture.recovery(Fixture.first), label: "first.")
            }
            let after = try fixture.bytes()
            #expect(after == before)
            withExtendedLifetime(held) {}
        }
        #expect(recorder.events == ["second.load", "second.warmUp"])
    }

    @Test(
        "When a cancelled recovery cannot remove its mark, the next request fails as recovery failed and a manual clear makes the asset loadable again"
    )
    func unrecordedCancellationNeedsClear() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(Fixture.record([Fixture.dead]))
        let path = fixture.directory.path(percentEncoded: false)
        defer { chmod(path, 0o700) }
        let recovery = fixture.recovery()
        await #expect(throws: CancellationError.self) {
            _ = try await recovery.load(
                delete: {},
                load: {
                    #expect(chmod(path, 0o500) == 0)
                    throw CancellationError()
                }, warmUp: { (_: Int) in })
        }
        #expect(chmod(path, 0o700) == 0)
        let recorder = PreparedCopyRecorder()
        await #expect(throws: PreparedCopyError.recoveryFailed) {
            _ = try await recorder.load(recovery)
        }
        try recovery.clear(delete: { recorder.note("clear") })
        let result = try await recorder.load(recovery)
        #expect(!result.recovered)
        #expect(recorder.events == ["clear", "load", "warmUp"])
    }
}
