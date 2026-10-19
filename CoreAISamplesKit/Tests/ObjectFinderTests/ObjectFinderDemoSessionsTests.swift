import FeatureStore
import Foundation
import ObjectFinderCore
import ObjectFinderFeature
import Testing

@testable import ModelLibraryFeature
@testable import ObjectFinderRuntime

private final class PreviewProbe {}

@MainActor
private final class RecordingFinderService: ObjectFinderService {
    let searching = AsyncStream<Void>.makeStream()
    let stopping = AsyncStream<Void>.makeStream()
    private(set) var searches: [URL] = []
    private(set) var stops: [Int] = []
    var holdsStop = false
    var stopFailure: ObjectFinderFailure?
    private var search: CheckedContinuation<Void, Never>?
    private var heldStops: [CheckedContinuation<Void, Never>] = []

    func search(
        folder: URL, query: String, minimumSimilarity: Double, rotation: ObjectCameraRotation, started: () -> Void,
        receive: (ObjectFinderFrame) -> Void
    ) async throws {
        searches.append(folder)
        started()
        searching.continuation.yield(())
        await withCheckedContinuation { search = $0 }
        throw CancellationError()
    }

    func stop(session: Int, retainingModel: Bool) async throws {
        stops.append(session)
        search?.resume()
        search = nil
        if holdsStop {
            stopping.continuation.yield(())
            await withCheckedContinuation { heldStops.append($0) }
        }
        if let stopFailure { throw stopFailure }
    }

    func notify() {}
    func setRotation(_ rotation: ObjectCameraRotation) {}

    func releaseStops() {
        holdsStop = false
        let held = heldStops
        heldStops = []
        for continuation in held { continuation.resume() }
    }
}

@MainActor
private final class RuntimeFactory {
    private(set) var services: [RecordingFinderService] = []

    func make() -> (preview: PreviewProbe, service: any ObjectFinderService) {
        let service = RecordingFinderService()
        services.append(service)
        return (PreviewProbe(), service)
    }
}

@MainActor
private final class LibraryGate {
    var ticket: LibraryTicket?
    var installed: ValidatedModelResource?
    let checked = AsyncStream<Void>.makeStream()

    func permits(_ session: ModelDemoSession) -> @MainActor () -> Bool {
        { [weak self] in
            guard let self else { return false }
            checked.continuation.yield(())
            return ticket == session.id && installed == session.resource
        }
    }
}

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ObjectFinderDemoSessionsTests {
    private let owner = UUID()
    private let selection = DemoSelection(modelID: "embeddinggemma2", demoID: "object-finder")
    private let selected = URL(filePath: "/models/embeddinggemma2/selected")
    private let replacement = URL(filePath: "/models/embeddinggemma2/replacement")

    private func resource(
        _ directory: URL, revision: String = String(repeating: "a", count: 40)
    ) -> ValidatedModelResource {
        ValidatedModelResource(
            modelID: "embeddinggemma2", artifactID: directory.lastPathComponent, contract: "embeddinggemma2-fixture",
            revision: revision, upstreamRevision: nil, directory: directory, sizeBytes: 1)
    }

    private func session(_ number: Int, _ resource: ValidatedModelResource) -> ModelDemoSession {
        ModelDemoSession(id: LibraryTicket(owner: owner, number: number), selection: selection, resource: resource)
    }

    @Test(
        "CAMERA-01 A closed gate for a missing artifact, another revision or an old session ticket starts no search and never calls the service"
    )
    func closedGates() async throws {
        let factory = RuntimeFactory()
        let sessions = ObjectFinderDemoSessions(makeRuntime: factory.make)
        let library = LibraryGate()
        var checks = library.checked.stream.makeAsyncIterator()
        let first = session(1, resource(selected))
        let runtime = sessions.bind(first, permits: library.permits(first))
        runtime.store.send(.queryChanged("glasses"))
        let closed: [(LibraryTicket?, ValidatedModelResource?)] = [
            (first.id, nil), (first.id, resource(selected, revision: String(repeating: "f", count: 40))),
            (LibraryTicket(owner: owner, number: 2), first.resource),
        ]
        for (ticket, installed) in closed {
            library.ticket = ticket
            library.installed = installed
            runtime.store.send(.start)
            _ = await checks.next()
            #expect(runtime.store.state.phase == .idle && runtime.store.state.failure == .notPermitted)
        }
        let service = try #require(factory.services.first)
        #expect(factory.services.count == 1 && service.searches.isEmpty && service.stops.isEmpty)

        library.ticket = first.id
        library.installed = first.resource
        var searching = service.searching.stream.makeAsyncIterator()
        runtime.store.send(.start)
        _ = await searching.next()
        #expect(service.searches == [selected] && runtime.store.state.phase == .searching)
        try await sessions.stop(first)
        #expect(runtime.store.state.phase == .idle)
    }

    @Test(
        "CAMERA-01 Rebinding the same resource reuses its Store and preview and follows the newest gate, so an old gate neither blocks nor starts a search"
    )
    func newestGate() async throws {
        let factory = RuntimeFactory()
        let sessions = ObjectFinderDemoSessions(makeRuntime: factory.make)
        let library = LibraryGate()
        var checks = library.checked.stream.makeAsyncIterator()
        let installed = resource(selected)
        let first = session(1, installed)
        let second = session(2, installed)
        library.installed = installed
        let shown = sessions.bind(first, permits: library.permits(first))
        library.ticket = second.id
        let rebound = sessions.bind(second, permits: library.permits(second))
        #expect(rebound.store === shown.store && rebound.preview === shown.preview && factory.services.count == 1)
        let service = try #require(factory.services.first)
        var searching = service.searching.stream.makeAsyncIterator()
        rebound.store.send(.queryChanged("keys"))
        rebound.store.send(.start)
        _ = await searching.next()
        _ = await checks.next()
        #expect(service.searches == [selected])
        try await sessions.stop(second)

        library.ticket = first.id
        rebound.store.send(.start)
        _ = await checks.next()
        #expect(rebound.store.state.failure == .notPermitted && service.searches == [selected])
    }

    @Test(
        "CAMERA-04 BOX-03 A different validated resource gets a new Store, preview and service that keep the target, threshold and box limit, and the old runtime cannot run again"
    )
    func replacedResource() async throws {
        let factory = RuntimeFactory()
        let sessions = ObjectFinderDemoSessions(makeRuntime: factory.make)
        let library = LibraryGate()
        let first = session(1, resource(selected))
        library.ticket = first.id
        library.installed = first.resource
        let old = sessions.bind(first, permits: library.permits(first))
        let oldService = try #require(factory.services.first)
        var oldSearching = oldService.searching.stream.makeAsyncIterator()
        old.store.send(.queryChanged("glasses"))
        old.store.send(.thresholdChanged(0.72))
        old.store.send(.maximumBoxesChanged(1))
        old.store.send(.start)
        _ = await oldSearching.next()
        try await sessions.stop(first)

        let second = session(2, resource(replacement, revision: String(repeating: "e", count: 40)))
        library.ticket = second.id
        library.installed = second.resource
        let replaced = sessions.bind(second, permits: library.permits(second))
        let replacedService = try #require(factory.services.last)
        #expect(replaced.store !== old.store && replaced.preview !== old.preview && factory.services.count == 2)
        var expected = ObjectFinderState(folder: replacement, execution: .managed)
        expected.query = "glasses"
        expected.submittedQuery = "glasses"
        expected.minimumSimilarity = 0.72
        expected.maximumBoxes = 1
        expected.session = 1
        #expect(replaced.store.state == expected && replacedService.searches.isEmpty)
        #expect(sessions.runtime(for: selection)?.store === replaced.store)

        var rejected: ObjectFinderFailure?
        await old.runner.run(.search(9, selected, "glasses", 0.72, .init(session: 9, revision: 0, angle: 90))) {
            if case .failed(_, let failure) = $0 { rejected = failure }
        }
        #expect(rejected == .notPermitted && oldService.searches == [selected])

        var replacedSearching = replacedService.searching.stream.makeAsyncIterator()
        replaced.store.send(.start)
        _ = await replacedSearching.next()
        #expect(replacedService.searches == [replacement] && oldService.searches == [selected])
        try await sessions.stop(second)
    }

    @Test(
        "CAMERA-03 Stopping awaits the release and throws its failure until a retry succeeds, a stop for an older session leaves the shown session running, and a session that never started stops successfully"
    )
    func stopping() async throws {
        let factory = RuntimeFactory()
        let sessions = ObjectFinderDemoSessions(makeRuntime: factory.make)
        let library = LibraryGate()
        let installed = resource(selected)
        let first = session(1, installed)
        try await sessions.stop(first)
        #expect(factory.services.isEmpty)

        library.ticket = first.id
        library.installed = installed
        let runtime = sessions.bind(first, permits: library.permits(first))
        try await sessions.stop(first)
        let service = try #require(factory.services.first)
        #expect(runtime.store.state.phase == .idle && service.searches.isEmpty)

        var searching = service.searching.stream.makeAsyncIterator()
        var stopping = service.stopping.stream.makeAsyncIterator()
        runtime.store.send(.queryChanged("glasses"))
        runtime.store.send(.start)
        _ = await searching.next()
        service.holdsStop = true
        let held = Task { try await sessions.stop(first) }
        _ = await stopping.next()
        #expect(runtime.store.state.phase == .stopping)
        service.releaseStops()
        try await held.value
        #expect(runtime.store.state.phase == .idle && !runtime.store.state.releasePending)

        runtime.store.send(.start)
        _ = await searching.next()
        service.stopFailure = .operation("release failed")
        await #expect(throws: ObjectFinderReleaseFailure(failure: .operation("release failed"))) {
            try await sessions.stop(first)
        }
        #expect(runtime.store.state.releasePending && !runtime.store.state.canStart)
        service.stopFailure = nil
        try await sessions.stop(first)
        #expect(runtime.store.state.phase == .idle && !runtime.store.state.releasePending)

        let second = session(2, installed)
        library.ticket = second.id
        _ = sessions.bind(second, permits: library.permits(second))
        runtime.store.send(.start)
        _ = await searching.next()
        let running = runtime.store.state.session
        try await sessions.stop(first)
        #expect(!service.stops.contains(running) && runtime.store.state.phase == .searching)
        try await sessions.stop(second)
        #expect(runtime.store.state.phase == .idle && service.stops.contains(running))
        #expect(service.searches.count == 3)
    }

    @Test(
        "CAMERA-04 After a stop the shared Store keeps the target and settings for the next session and its preview, while frames and boxes are gone and nothing restarts"
    )
    func keptInputs() async throws {
        let factory = RuntimeFactory()
        let sessions = ObjectFinderDemoSessions(makeRuntime: factory.make)
        let library = LibraryGate()
        let installed = resource(selected)
        let first = session(1, installed)
        library.ticket = first.id
        library.installed = installed
        let runtime = sessions.bind(first, permits: library.permits(first))
        let service = try #require(factory.services.first)
        var searching = service.searching.stream.makeAsyncIterator()
        runtime.store.send(.queryChanged("glasses"))
        runtime.store.send(.thresholdChanged(0.72))
        runtime.store.send(.start)
        _ = await searching.next()
        let candidate = ObjectCandidate(
            id: 0, region: .init(x: 0.2, y: 0.2, width: 0.3, height: 0.3), similarity: 0.8)
        runtime.store.send(.analyzed(1, ObjectFinderFrame(candidates: [candidate], seconds: 0.1)))
        #expect(runtime.store.state.matches == [candidate])
        runtime.store.send(.queryChanged("keys"))
        try await sessions.stop(first)

        let second = session(2, installed)
        library.ticket = second.id
        let rebound = sessions.bind(second, permits: library.permits(second))
        let kept = try #require(sessions.runtime(for: selection)?.store.state)
        #expect(rebound.store === runtime.store && kept.query == "keys" && kept.submittedQuery == "glasses")
        #expect(kept.minimumSimilarity == 0.72 && kept.phase == .idle && kept.frame == nil && kept.matches.isEmpty)
        #expect(service.searches == [selected] && factory.services.count == 1)
    }
}
