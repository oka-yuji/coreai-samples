import ChronosFeature
import FeatureStore
import Foundation
import ModelLibraryFeature
import Observation
import Testing

@testable import ChronosDemoRuntime
@testable import ModelLibraryRuntime

@MainActor
@Suite(.serialized, .timeLimit(.minutes(1)))
struct ChronosModelsPublicationTests {
    @Test(
        "J2 Models revoke record and check publication before delivering journal cancellation",
        arguments: ["close", "suspend", "direct", "abandon", "release"], [false, true])
    func withdrawal(_ transition: String, _ checking: Bool) async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let profile = try ChronosAPIFixture.syntheticProfile(areas: NYCAPIContract.zoneIDs)
        let directory = root.appending(path: "journal")
        let original = ChronosJournalFiles(directory: directory, verification: profile)
        let record = try await ChronosJournalFixture.record()
        if checking { try await original.create(record, permit: ChronosPublicationPermit()) }
        let checkpoint = ChronosPublicationCheckpoint()
        defer { checkpoint.finish() }
        let files = ChronosJournalFiles(
            directory: directory, verification: profile, beforePublish: checkpoint.beforePublish)
        let fake = try ChronosAPITransportFake()
        let api = NYCAPIClient(
            transport: { try await fake.receive($0) }, now: { ChronosAPIFixture.now }, verification: profile)
        let harness = try await ChronosPublicationModelsFixture.make(
            journal: ChronosJournalAssembly(files: files, api: api))
        defer {
            harness.fixture.remove()
            harness.adapter.finish()
        }
        if checking {
            harness.journal.send(.open)
            harness.journal.send(.select(record.id))
            for await state in Observations({ harness.journal.state }) { if state.detail != nil { break } }
            harness.journal.send(.check)
        } else {
            harness.journal.send(.capture(try await ChronosJournalFixture.batch()))
        }
        var paused = checkpoint.paused.stream.makeAsyncIterator()
        _ = await paused.next()
        let unchanged = harness.library.state
        var direct: Task<Void, Never>?
        switch transition {
        case "close": harness.library.send(.closeDemo)
        case "suspend": harness.library.send(.suspendDemo)
        case "direct":
            direct = Task { await harness.assembly.runner.run(.stopDemo(harness.session, harness.session.id)) { _ in } }
        case "release":
            direct = Task { await harness.assembly.runner.release(owner: harness.session.id.owner) }
        default: harness.assembly.runner.abandon(owner: harness.session.id.owner)
        }
        if transition != "direct" && transition != "release" { #expect(!harness.gate.isPermitted()) }
        var stopping = harness.adapter.stopping.stream.makeAsyncIterator()
        _ = await stopping.next()
        #expect(!harness.gate.isPermitted())
        #expect(harness.journal.state.active && !harness.journal.state.ended)
        #expect(checking ? harness.journal.state.checking : harness.journal.state.saving)
        if ["direct", "abandon", "release"].contains(transition) { #expect(harness.library.state == unchanged) }
        checkpoint.resume()
        if checking {
            #expect(try await files.detail(record.id).checks.isEmpty)
        } else {
            #expect(try await files.list().isEmpty)
        }
        #expect(checkpoint.attempts == 1)
        await harness.stop()
        await direct?.value
        #expect(await harness.storage.activeLeases(identity: harness.session.identity) == 0)
    }

    @Test(
        "J2 Unrelated Models and competing owner changes retry the same save without observation cycles",
        arguments: [false, true])
    func unrelated(_ competing: Bool) async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let checkpoint = ChronosPublicationCheckpoint()
        defer { checkpoint.finish() }
        let files = ChronosJournalFiles(
            directory: root.appending(path: "journal"), write: checkpoint.write, beforePublish: checkpoint.beforePublish
        )
        let harness = try await ChronosPublicationModelsFixture.make(journal: ChronosJournalAssembly(files: files))
        defer {
            harness.fixture.remove()
            harness.adapter.finish()
        }
        var competitor: (Store<ModelLibraryFeature>, ModelDemoSession)?
        if competing {
            let second = harness.assembly.makeLibrary(catalog: harness.catalog, capabilities: [:], owner: UUID())
            second.store.send(.refresh)
            for await state in Observations({ second.store.state }) {
                if state.installations[harness.fixture.artifact.id]?.phase == .ready { break }
            }
            second.store.send(.openDemo(ChronosDemoCatalog.selection))
            let session = try #require(second.store.state.session)
            let gate = harness.assembly.runner.gate(for: session, in: second.store)
            await harness.assembly.runner.prepare(session)
            #expect(!gate.isPermitted() && gate.retainPermission()?.isPermitted() == false)
            #expect(harness.adapter.sessions.bind(session, gate: gate) == nil)
            competitor = (second.store, session)
        }
        harness.journal.send(.capture(try await ChronosJournalFixture.batch()))
        var paused = checkpoint.paused.stream.makeAsyncIterator()
        _ = await paused.next()
        if let (_, session) = competitor {
            await harness.assembly.runner.run(.stopDemo(session, session.id)) { _ in }
        } else {
            harness.library.send(.capabilityChanged(.camera, .available))
        }
        #expect(harness.gate.isPermitted())
        checkpoint.resume()
        var actions = harness.adapter.actions.stream.makeAsyncIterator()
        while let action = await actions.next() { if case .saved = action { break } }
        let records = try await files.list()
        #expect(records.count == 1 && harness.journal.state.savedID == records.first?.id)
        #expect(harness.journal.state.saveError == nil && checkpoint.attempts == 2)
        #expect(checkpoint.writes[0] == checkpoint.writes[1])
        #expect(checkpoint.retainedPermits == 0)
        withExtendedLifetime(competitor) {}
        await harness.stop()
    }
}
