import Foundation
import Testing

@testable import CoreAIBackend

@Suite(.timeLimit(.minutes(1))) struct PreparedCopyLedgerTests {
    private typealias Fixture = PreparedCopyFixture

    @Test("A normal load leaves no mark in the record and does not delete")
    func normalLoadLeavesNoMark() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let recovery = fixture.recovery()
        let recorder = PreparedCopyRecorder()
        let result = try await recorder.load(recovery)
        #expect(!result.recovered)
        #expect(result.value == 1)
        #expect(recovery.record() == Fixture.empty)
        #expect(recorder.events == ["load", "warmUp"])
    }

    @Test("A normal load whose load throws removes only its own mark, keeps other living marks and releases its use")
    func failedNormalLoadRemovesOwnMark() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(Fixture.record([Fixture.alive]))
        let recovery = fixture.recovery()
        let recorder = PreparedCopyRecorder()
        await #expect(throws: POSIXError(.EIO)) {
            _ = try await recovery.load(
                delete: { recorder.note("delete") }, load: { throw POSIXError(.EIO) }, warmUp: { (_: Int) in })
        }
        #expect(recovery.record() == Fixture.record([Fixture.alive]))
        try recovery.clear(delete: { recorder.note("clear") })
        #expect(recorder.events == ["clear"])
    }

    @Test("A ledger that is not valid JSON is read as empty, the load is normal and the ledger is rewritten readably")
    func unreadableLedgerIsRewritten() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(raw: Data("not json".utf8))
        let recovery = fixture.recovery()
        let recorder = PreparedCopyRecorder()
        let result = try await recorder.load(recovery)
        #expect(!result.recovered)
        #expect(recorder.events == ["load", "warmUp"])
        let ledger = try JSONDecoder().decode(PreparedCopyLedger.self, from: fixture.bytes())
        #expect(ledger.namespace == Fixture.key.namespace)
        #expect(ledger.osBuild == Fixture.key.osBuild)
        #expect(ledger.assetHash == Fixture.key.assetHash)
        #expect(recovery.record() == Fixture.empty)
    }

    @Test(
        "A ledger whose namespace, OS build or asset hash differs from the key is read as empty and rewritten with the key's values",
        arguments: [
            PreparedCopyFixture.ledger(
                PreparedCopyFixture.record([PreparedCopyFixture.dead]), namespace: "com.example.other"),
            PreparedCopyFixture.ledger(PreparedCopyFixture.record([PreparedCopyFixture.dead]), osBuild: "27A999"),
            PreparedCopyFixture.ledger(
                PreparedCopyFixture.record([PreparedCopyFixture.dead]), assetHash: String(repeating: "a5", count: 32)),
        ])
    func mismatchedLedgerIsEmpty(ledger: PreparedCopyLedger) async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(ledger: ledger)
        let recovery = fixture.recovery()
        let recorder = PreparedCopyRecorder()
        let result = try await recorder.load(recovery)
        #expect(!result.recovered)
        #expect(recorder.events == ["load", "warmUp"])
        let rewritten = try JSONDecoder().decode(PreparedCopyLedger.self, from: fixture.bytes())
        #expect(rewritten.namespace == Fixture.key.namespace)
        #expect(rewritten.osBuild == Fixture.key.osBuild)
        #expect(rewritten.assetHash == Fixture.key.assetHash)
        #expect(recovery.record() == Fixture.empty)
    }

    @Test(
        "Two keys of one asset with different options use separate ledger files and one's dead mark does not affect the other"
    )
    func optionsKeepSeparateLedgers() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(Fixture.record([Fixture.dead]))
        let other = fixture.recovery(key: Fixture.otherKey)
        let recorder = PreparedCopyRecorder()
        let result = try await recorder.load(other, label: "other.")
        #expect(!result.recovered)
        #expect(other.record() == Fixture.empty)
        #expect(fixture.recovery().record() == Fixture.record([Fixture.dead]))
        #expect(other.file != fixture.recovery().file)
        let recovered = try await recorder.load(fixture.recovery())
        #expect(recovered.recovered)
        #expect(other.record() == Fixture.empty)
        #expect(
            recorder.events == ["other.load", "other.warmUp", "onRecovery", "delete", "load", "warmUp"])
        let names = try FileManager.default.contentsOfDirectory(atPath: fixture.directory.path(percentEncoded: false))
        #expect(names.filter { $0.hasSuffix(".json") }.count == 2)
    }

    @Test("When the warm-up is called, the own mark in the record is at the warm-up stage")
    func warmUpStageIsRecorded() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let recovery = fixture.recovery()
        var atLoad: PreparedCopyRecord?
        var atWarmUp: PreparedCopyRecord?
        _ = try await recovery.load(
            delete: {},
            load: {
                atLoad = recovery.record()
                return 1
            }, warmUp: { _ in atWarmUp = recovery.record() })
        let loading = try #require(atLoad)
        let warming = try #require(atWarmUp)
        let mark = try #require(loading.marks.first)
        #expect(loading == Fixture.record([Fixture.own(mark.attempt, .normal, .load)]))
        #expect(warming == Fixture.record([Fixture.own(mark.attempt, .normal, .warmUp)]))
    }
}
