import Foundation
import Testing

@testable import CoreAIBackend

@Suite(.timeLimit(.minutes(1))) struct PreparedCopyRecoveryTests {
    private typealias Fixture = PreparedCopyFixture

    @Test(
        "AC01 A ledger left with only a dead process's mark is judged for recovery before the query and counts that mark once"
    )
    func recoversFromDeadMark() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(Fixture.record([Fixture.dead]))
        let recovery = fixture.recovery()
        let recorder = PreparedCopyRecorder()
        var saved: PreparedCopyRecord?
        let result = try await recovery.load(
            onRecovery: { saved = recovery.record() }, delete: { recorder.note("delete") }, load: { 7 },
            warmUp: { _ in })
        #expect(result.recovered)
        #expect(result.value == 7)
        let record = try #require(saved)
        let mark = try #require(record.marks.first)
        #expect(UUID(uuidString: mark.attempt) != nil)
        #expect(
            record
                == Fixture.record(
                    [Fixture.own(mark.attempt, .recovery, .load)], recoveries: 1, suspects: [Fixture.dead]))
        #expect(recorder.events == ["delete"])
    }

    @Test("AC02 A ledger left with a dead recovery mark returns a failure without deleting or specializing")
    func deadRecoveryMarkFails() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(Fixture.record([Fixture.deadRecovery], recoveries: 1, suspects: [Fixture.dead]))
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
        "AC03 After the deletion throws, a repeated request from the same process returns recovery failed rather than preparing"
    )
    func failedDeletionIsTerminal() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(Fixture.record([Fixture.dead]))
        let recovery = fixture.recovery()
        let recorder = PreparedCopyRecorder()
        await #expect(throws: POSIXError(.EIO)) {
            _ = try await recovery.load(
                delete: {
                    recorder.note("delete")
                    throw POSIXError(.EIO)
                },
                load: {
                    recorder.note("load")
                    return 1
                }, warmUp: { _ in })
        }
        await #expect(throws: PreparedCopyError.recoveryFailed) {
            _ = try await recorder.load(recovery)
        }
        #expect(recorder.events == ["delete"])
    }

    @Test(
        "AC07 After the warm-up result is read, the own mark and the dead marks are removed and the recovery count returns to zero"
    )
    func completedRecoveryClearsMarks() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(Fixture.record([Fixture.dead]))
        let recovery = fixture.recovery()
        let result = try await recovery.load(
            delete: {}, load: { 1 },
            warmUp: { _ in
                var record = recovery.record()
                record.marks.append(Fixture.lateDead)
                try fixture.write(record)
            })
        #expect(result.recovered)
        #expect(recovery.record() == Fixture.empty)
    }

    @Test("AC08 When the deletion throws, loading stops, the error is returned and the mark is kept")
    func failedDeletionKeepsMark() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(Fixture.record([Fixture.dead]))
        let recovery = fixture.recovery()
        let recorder = PreparedCopyRecorder()
        await #expect(throws: POSIXError(.EIO)) {
            _ = try await recovery.load(
                delete: { throw POSIXError(.EIO) },
                load: {
                    recorder.note("load")
                    return 1
                }, warmUp: { _ in recorder.note("warmUp") })
        }
        #expect(recorder.events.isEmpty)
        let record = recovery.record()
        let mark = try #require(record.marks.first)
        #expect(
            record
                == Fixture.record(
                    [Fixture.own(mark.attempt, .recovery, .load)], recoveries: 1, suspects: [Fixture.dead], failed: true
                ))
    }

    @Test(
        "AC19 A reload does not delete the entry even with a dead mark and fails when the ledger is terminal",
        arguments: zip(
            [
                PreparedCopyFixture.record([PreparedCopyFixture.dead]),
                PreparedCopyFixture.record([PreparedCopyFixture.dead], failed: true),
                PreparedCopyFixture.record([PreparedCopyFixture.dead], recoveries: 1),
            ],
            [PreparedCopyError.reloadRequired, .recoveryFailed, .recoveryFailed]))
    func reloadKeepsEntry(record: PreparedCopyRecord, error: PreparedCopyError) async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(record)
        let before = try fixture.bytes()
        let recorder = PreparedCopyRecorder()
        await #expect(throws: error) {
            _ = try await recorder.load(fixture.recovery(), reload: true)
        }
        let after = try fixture.bytes()
        #expect(after == before)
        #expect(recorder.events.isEmpty)
    }

    @Test(
        "AC20 Right after the recovery slot is saved and before the entry is deleted, the notification arrives exactly once"
    )
    func notifiesOnceBeforeDeletion() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(Fixture.record([Fixture.dead]))
        let recovery = fixture.recovery()
        let recorder = PreparedCopyRecorder()
        _ = try await recorder.load(recovery)
        #expect(recorder.events == ["onRecovery", "delete", "load", "warmUp"])
        _ = try await recorder.load(recovery, label: "normal.")
        #expect(recorder.events == ["onRecovery", "delete", "load", "warmUp", "normal.load", "normal.warmUp"])
    }

    @Test(
        "A cancelled load during a recovery removes only the own mark without a terminal state and the next load is normal"
    )
    func cancelledRecoveryIsNotTerminal() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(Fixture.record([Fixture.dead]))
        let recovery = fixture.recovery()
        await #expect(throws: CancellationError.self) {
            _ = try await recovery.load(delete: {}, load: { throw CancellationError() }, warmUp: { (_: Int) in })
        }
        #expect(recovery.record() == Fixture.record([], recoveries: 1, suspects: [Fixture.dead]))
        let recorder = PreparedCopyRecorder()
        let result = try await recorder.load(recovery)
        #expect(!result.recovered)
        #expect(recovery.record() == Fixture.empty)
        #expect(recorder.events == ["load", "warmUp"])
    }

    @Test("Another error from the load during a recovery makes the ledger terminal and the next load fails")
    func failedRecoveryIsTerminal() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(Fixture.record([Fixture.dead]))
        let recovery = fixture.recovery()
        await #expect(throws: POSIXError(.EIO)) {
            _ = try await recovery.load(delete: {}, load: { throw POSIXError(.EIO) }, warmUp: { (_: Int) in })
        }
        let record = recovery.record()
        let mark = try #require(record.marks.first)
        #expect(
            record
                == Fixture.record(
                    [Fixture.own(mark.attempt, .recovery, .load)], recoveries: 1, suspects: [Fixture.dead], failed: true
                ))
        let recorder = PreparedCopyRecorder()
        await #expect(throws: PreparedCopyError.recoveryFailed) {
            _ = try await recorder.load(recovery)
        }
        #expect(recorder.events.isEmpty)
    }
}
