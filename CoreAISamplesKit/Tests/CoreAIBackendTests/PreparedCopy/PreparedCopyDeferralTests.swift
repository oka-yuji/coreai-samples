import Foundation
import Testing

@testable import CoreAIBackend

@Suite(.timeLimit(.minutes(1))) struct PreparedCopyDeferralTests {
    private typealias Fixture = PreparedCopyFixture

    @Test("AC04 A mark of another living process is not counted as a leftover even while that process is stopped")
    func livingMarkIsNotLeftover() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(Fixture.record([Fixture.alive]))
        let recovery = fixture.recovery()
        let recorder = PreparedCopyRecorder()
        let result = try await recorder.load(recovery)
        #expect(!result.recovered)
        #expect(recorder.events == ["load", "warmUp"])
        #expect(recovery.record() == Fixture.record([Fixture.alive]))
        #expect(
            PreparedCopyProcess.liveness(of: Fixture.alive, probe: .present(start: Fixture.alive.start, zombie: false))
                == .alive)
    }

    @Test("AC06 When two processes see the same leftover mark at the same time, only one of them starts a recovery")
    func onlyOneRecoveryStarts() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(Fixture.record([Fixture.dead]))
        let first = fixture.recovery(Fixture.first)
        let second = fixture.recovery(Fixture.second)
        let recorder = PreparedCopyRecorder()
        let result = try await first.load(
            delete: { recorder.note("first.delete") },
            load: {
                try #require(fixture.lockIsFree())
                await #expect(throws: PreparedCopyError.preparing) {
                    _ = try await recorder.load(second, label: "second.")
                }
                return 1
            }, warmUp: { _ in })
        #expect(result.recovered)
        #expect(recorder.events == ["first.delete"])
    }

    @Test("AC17 An asset with a load in use is not deleted and its recovery count does not increase")
    func inUseAssetIsKept() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let second = fixture.recovery(Fixture.second)
        let recorder = PreparedCopyRecorder()
        do {
            let held = try await recorder.load(fixture.recovery(Fixture.first), label: "first.")
            try fixture.write(Fixture.record([Fixture.dead]))
            let before = try fixture.bytes()
            await #expect(throws: PreparedCopyError.preparing) {
                _ = try await recorder.load(second, label: "second.")
            }
            let after = try fixture.bytes()
            #expect(after == before)
            #expect(!held.recovered)
            withExtendedLifetime(held) {}
        }
        let result = try await recorder.load(second, label: "second.")
        #expect(result.recovered)
        #expect(
            recorder.events == [
                "first.load", "first.warmUp", "second.onRecovery", "second.delete", "second.load", "second.warmUp",
            ])
    }

    @Test("AC18 Nothing is deleted while a mark whose liveness cannot be confirmed is present")
    func unknownMarkPreventsDeletion() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let recovery = fixture.recovery()
        let recorder = PreparedCopyRecorder()
        try fixture.write(Fixture.record([Fixture.dead, Fixture.unknown]))
        let before = try fixture.bytes()
        await #expect(throws: PreparedCopyError.preparing) {
            _ = try await recorder.load(recovery)
        }
        let after = try fixture.bytes()
        #expect(after == before)
        try fixture.write(Fixture.record([Fixture.unknown]))
        let result = try await recorder.load(recovery)
        #expect(!result.recovered)
        #expect(recovery.record() == Fixture.record([Fixture.unknown]))
        #expect(recorder.events == ["load", "warmUp"])
    }

    @Test("AC21 A manual clear is refused while a load is in use and otherwise deletes the entry and the ledger")
    func clearRespectsUse() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let recovery = fixture.recovery()
        let base = fixture.directory.appending(path: Fixture.base(Fixture.key))
        #expect(recovery.file.path(percentEncoded: false) == base.path(percentEncoded: false) + ".json")
        let recorder = PreparedCopyRecorder()
        do {
            let held = try await recorder.load(recovery)
            #expect(throws: PreparedCopyError.inUse) {
                try recovery.clear(delete: { recorder.note("clear") })
            }
            #expect(Fixture.exists(recovery.file))
            withExtendedLifetime(held) {}
        }
        try recovery.clear(delete: { recorder.note("clear") })
        #expect(recorder.events == ["load", "warmUp", "clear"])
        #expect(!Fixture.exists(recovery.file))
        #expect(Fixture.exists(base.appendingPathExtension("lock")))
        #expect(Fixture.exists(base.appendingPathExtension("use")))
    }

    @Test("While the use of a completed recovery is held, another normal load succeeds and a manual clear is refused")
    func recoveredUseIsShared() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(Fixture.record([Fixture.dead]))
        let second = fixture.recovery(Fixture.second)
        let recorder = PreparedCopyRecorder()
        let held = try await recorder.load(fixture.recovery(Fixture.first), label: "first.")
        #expect(held.recovered)
        do {
            let other = try await recorder.load(second, label: "second.")
            #expect(!other.recovered)
        }
        #expect(throws: PreparedCopyError.inUse) {
            try second.clear(delete: { recorder.note("clear") })
        }
        #expect(
            recorder.events == [
                "first.onRecovery", "first.delete", "first.load", "first.warmUp", "second.load", "second.warmUp",
            ])
        withExtendedLifetime(held) {}
    }
}
