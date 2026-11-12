import ChronosFeature
import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryRuntime
import Testing

@testable import ChronosDemoRuntime

@MainActor
@Suite(.serialized, .timeLimit(.minutes(1)))
struct ChronosSessionPublicationTests {
    @Test("J2 An in-flight publication retains its session only until the attempt ends")
    func ownerLifetime() async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let checkpoint = ChronosPublicationCheckpoint()
        defer { checkpoint.finish() }
        let files = ChronosJournalFiles(
            directory: root.appending(path: "journal"), beforePublish: checkpoint.beforePublish)
        let released = AsyncStream<Void>.makeStream()
        defer { released.continuation.finish() }
        let catalog = try await ChronosRepository.bundled().load()
        var lifetime: SessionLifetime? = SessionLifetime(released: released.continuation)
        var sessions: ChronosDemoSessions? = ChronosDemoSessions(
            makeService: { [lifetime] _, _ in
                withExtendedLifetime(lifetime) { SessionPublicationService(catalog: catalog) }
            }, journal: ChronosJournalAssembly(files: files))
        lifetime = nil
        weak let retained = sessions
        let session = session()
        let gate = DemoExecutionGate(session: session, isPermitted: { true })
        _ = sessions?.bind(session, gate: gate)
        let journal = try #require(sessions?.journal(for: session))
        journal.send(.capture(try await ChronosJournalFixture.batch()))
        var paused = checkpoint.paused.stream.makeAsyncIterator()
        _ = await paused.next()
        sessions = nil
        #expect(retained != nil)
        checkpoint.resume()
        var ended = released.stream.makeAsyncIterator()
        _ = await ended.next()
        #expect(retained == nil)
        #expect(try await files.list().count == 1)
        #expect(checkpoint.retainedPermits == 0)
        journal.send(.ended)
    }

    @Test(
        "J2 A direct session stop withdraws permission before returning and an old stop cannot revoke its replacement")
    func sessionStop() async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let checkpoint = ChronosPublicationCheckpoint()
        defer { checkpoint.finish() }
        let files = ChronosJournalFiles(
            directory: root.appending(path: "journal"), beforePublish: checkpoint.beforePublish)
        let catalog = try await ChronosRepository.bundled().load()
        let sessions = ChronosDemoSessions(
            makeService: { _, _ in SessionPublicationService(catalog: catalog) },
            journal: ChronosJournalAssembly(files: files))
        let first = session()
        _ = sessions.bind(first, gate: DemoExecutionGate(session: first, isPermitted: { true }))
        let journal = try #require(sessions.journal(for: first))
        journal.send(.capture(try await ChronosJournalFixture.batch()))
        var paused = checkpoint.paused.stream.makeAsyncIterator()
        _ = await paused.next()
        await sessions.stop(first)
        checkpoint.resume()
        #expect(try await files.list().isEmpty)
        let second = ModelDemoSession(
            id: LibraryTicket(owner: first.id.owner, number: first.id.number + 1), selection: first.selection,
            resource: first.resource)
        _ = sessions.bind(second, gate: DemoExecutionGate(session: second, isPermitted: { true }))
        let current = try #require(sessions.journal(for: second))
        await sessions.stop(first)
        #expect(current.state.active && !current.state.ended)
        await sessions.stop(second)
    }

    private func session() -> ModelDemoSession {
        let resource = ValidatedModelResource(
            modelID: ChronosDemoCatalog.selection.modelID, artifactID: "chronos-fixture", contract: "chronos2-fp32",
            revision: String(repeating: "a", count: 64), upstreamRevision: nil,
            directory: URL(filePath: "/unused-chronos"),
            sizeBytes: 1, sourceKind: .localManifest)
        return ModelDemoSession(
            id: LibraryTicket(owner: UUID(), number: 1), selection: ChronosDemoCatalog.selection,
            resource: resource)
    }
}

private final class SessionLifetime: Sendable {
    let released: AsyncStream<Void>.Continuation
    init(released: AsyncStream<Void>.Continuation) { self.released = released }
    deinit { released.yield(()) }
}

@MainActor
private final class SessionPublicationService: ChronosService {
    let catalog: ChronosCatalog
    init(catalog: ChronosCatalog) { self.catalog = catalog }
    func load() async throws -> ChronosLoadedDemo { ChronosLoadedDemo(catalog: catalog, liveAvailable: true) }
    func predict(_ request: ChronosForecastRequest) async throws -> ChronosForecastBatch {
        throw ChronosDemoFailure.unavailable
    }
    func release() async {}
}
