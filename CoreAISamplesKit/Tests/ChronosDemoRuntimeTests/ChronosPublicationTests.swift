import ChronosFeature
import FeatureStore
import Foundation
import Synchronization
import Testing

@testable import ChronosDemoRuntime

@MainActor
@Suite(.serialized, .timeLimit(.minutes(1)))
struct ChronosPublicationTests {
    @Test(
        "J2 A prepared check retries identical bytes once without another request and ends every observation",
        arguments: ["unrelated", "journal", "noop", "disk", "cancel", "close", "back", "select", "suspend", "ended"])
    func checks(_ change: String) async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let profile = try ChronosAPIFixture.syntheticProfile(areas: NYCAPIContract.zoneIDs)
        let directory = root.appending(path: "journal")
        let originalFiles = ChronosJournalFiles(directory: directory, verification: profile)
        let record = try await ChronosJournalFixture.record()
        try await originalFiles.create(record, permit: ChronosPublicationPermit())
        let actuals = try await ChronosJournalFixture.actuals(origin: record.input.snapshot.origin, profile: profile)
        let queuedBatch = try await ChronosJournalFixture.batch()
        let checkpoint = ChronosPublicationCheckpoint()
        defer { checkpoint.finish() }
        let files = ChronosJournalFiles(
            directory: directory, verification: profile, write: checkpoint.write,
            beforePublish: {
                checkpoint.beforePublish($0)
                if change == "disk" { throw POSIXError(.ENOSPC) }
            })
        let owner = ChronosPublicationOwner()
        let requests = Mutex(0)
        let id = UUID()
        var identifiers = 0
        var dates = 0
        let runner = ChronosJournalRunner(
            files: files,
            actuals: { _, _ in
                requests.withLock { $0 += 1 }
                return actuals
            },
            permits: {
                _ = owner.revision
                return true
            },
            now: {
                dates += 1
                return ChronosAPIFixture.now
            },
            makeID: {
                identifiers += 1
                return id
            })
        var state = ChronosJournalFeature.State()
        state.presented = true
        state.selectedID = record.id
        state.detail = ChronosJournalDetail(record: record, checks: [])
        _ = ChronosJournalFeature.reduce(&state, .check)
        let journal = Store<ChronosJournalFeature>(initialState: state) { _, _ in }
        runner.attach(to: journal)
        var delivered = 0
        let operation = Task {
            await runner.run(.check(state.ticket, record.id)) {
                delivered += 1
                journal.send($0)
            }
        }
        var paused = checkpoint.paused.stream.makeAsyncIterator()
        _ = await paused.next()
        switch change {
        case "unrelated": owner.revision += 1
        case "journal": journal.send(.capture(queuedBatch))
        case "noop", "disk": journal.send(.appear)
        case "cancel":
            owner.revision += 1
            operation.cancel()
        case "close": journal.send(.close)
        case "back": journal.send(.back)
        case "select": journal.send(.select(UUID()))
        case "suspend": journal.send(.suspend)
        default: journal.send(.ended)
        }
        checkpoint.resume()
        await operation.value
        let detail = try await files.detail(record.id)
        let retries = change == "unrelated" || change == "journal"
        let succeeds = retries || change == "noop"
        #expect(detail.record == record && detail.checks.count == (succeeds ? 1 : 0))
        #expect(delivered == (succeeds || change == "disk" ? 1 : 0))
        if change == "disk" { #expect(journal.state.error == .persistence) }
        #expect(identifiers == 1 && dates == 1 && requests.withLock { $0 } == 1)
        #expect(checkpoint.attempts == (retries ? 2 : 1))
        #expect(checkpoint.retainedPermits == 0)
        if succeeds {
            #expect(detail.checks.first?.id == id && detail.checks.first?.checkedAt == ChronosAPIFixture.now)
            #expect(checkpoint.writes.allSatisfy { $0 == checkpoint.writes.first })
        }
        journal.send(.ended)
    }

    @Test("J2 A newer capture cannot invalidate the first pending save or replace its UUID and bytes")
    func queuedSave() async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let checkpoint = ChronosPublicationCheckpoint()
        defer { checkpoint.finish() }
        let files = ChronosJournalFiles(
            directory: root.appending(path: "journal"), write: checkpoint.write, beforePublish: checkpoint.beforePublish
        )
        let ids = [UUID(), UUID()]
        var identifiers = 0
        let runner = ChronosJournalRunner(
            files: files, actuals: { _, _ in throw ChronosDemoFailure.unavailable },
            now: { ChronosAPIFixture.now },
            makeID: {
                defer { identifiers += 1 }
                return ids[identifiers]
            },
            device: { ChronosJournalFixture.device })
        let completions = AsyncStream<Void>.makeStream()
        defer { completions.continuation.finish() }
        let journal = Store<ChronosJournalFeature>(initialState: ChronosJournalFeature.State()) { effect, send in
            await runner.run(effect, send: send)
            completions.continuation.yield(())
        }
        runner.attach(to: journal)
        let first = try await ChronosJournalFixture.batch()
        let second = try await ChronosJournalFixture.batch(step: 1)
        journal.send(.capture(first))
        var paused = checkpoint.paused.stream.makeAsyncIterator()
        _ = await paused.next()
        journal.send(.capture(second))
        checkpoint.resume()
        var completed = completions.stream.makeAsyncIterator()
        _ = await completed.next()
        _ = await completed.next()
        #expect(identifiers == 2 && checkpoint.attempts == 3)
        #expect(checkpoint.writes[0] == checkpoint.writes[1])
        #expect(try await files.record(ids[0]).input == first.capturedInput)
        #expect(try await files.record(ids[1]).input == second.capturedInput)
        #expect(try await files.list().count == 2)
        #expect(journal.state.saves.isEmpty && journal.state.saveError == nil)
        #expect(checkpoint.retainedPermits == 0)
        journal.send(.ended)
    }

    @Test("J2 A prepared check refuses a different original record at the same UUID")
    func replacedOriginal() async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let profile = try ChronosAPIFixture.syntheticProfile(areas: NYCAPIContract.zoneIDs)
        let directory = root.appending(path: "journal")
        let files = ChronosJournalFiles(directory: directory, verification: profile)
        let record = try await ChronosJournalFixture.record()
        try await files.create(record, permit: ChronosPublicationPermit())
        let prepared = try await files.prepare(
            id: UUID(), record: record, checkedAt: ChronosAPIFixture.now,
            actuals: ChronosJournalFixture.actuals(origin: record.input.snapshot.origin, profile: profile))
        let path = directory.appending(path: "\(record.id.uuidString)/prediction.json")
        try ChronosJournalFixture.rewrite(path) { value in
            value["savedAt"] = record.savedAt.addingTimeInterval(1).timeIntervalSinceReferenceDate
        }
        #expect(try await files.record(record.id) != record)
        await #expect(throws: ChronosDemoFailure.invalidData) {
            try await files.publish(prepared, permit: ChronosPublicationPermit())
        }
        #expect(try await files.detail(record.id).checks.isEmpty)
    }
}
