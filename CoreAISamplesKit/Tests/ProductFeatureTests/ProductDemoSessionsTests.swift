import FeatureStore
import Foundation
import Observation
import ProductCore
import ProductFeature
import Testing

@testable import ModelLibraryFeature
@testable import ProductRuntime

@MainActor
struct ProductDemoSessionsTests {
    private let stamp = Date(timeIntervalSince1970: 0)

    private var draft: ProductDraft {
        var draft = ProductDraft(mode: .text)
        draft.text = "Memory: 16 GB"
        return draft
    }

    private var closed: String { ProductExecutionClosed().errorDescription ?? "" }

    @Test("Returning to the first owner does not permit that owner's retired product runtime", .timeLimit(.minutes(1)))
    func retiredOwnerCannotUseLatestGate() async throws {
        let library = try ProductLibrary()
        try library.install("fp32")
        let first = try library.open()
        let services = ProductServices()
        let sessions = ProductDemoSessions(makeService: services.make)
        let retired = sessions.bind(first, permits: { true })
        retired.send(.draftChanged(draft))
        try await sessions.stop(first)
        for await state in Observations({ retired.state }) where !state.busy { break }
        let middle = ModelDemoSession(
            id: LibraryTicket(owner: UUID(), number: 1), selection: first.selection, resource: first.resource)
        _ = sessions.bind(middle, permits: { true })
        try await sessions.stop(middle)
        let last = ModelDemoSession(
            id: LibraryTicket(owner: first.id.owner, number: first.id.number + 1),
            selection: first.selection, resource: first.resource)
        let current = sessions.bind(last, permits: { true })
        #expect(retired !== current && services.created.count == 3)
        retired.send(.appeared)
        retired.send(.submit(stamp))
        for await state in Observations({ retired.state }) where !state.busy { break }
        #expect(retired.state.phase == .failed(closed))
        #expect(services.created[0].folders.isEmpty && services.created[2].log.isEmpty)
        try await sessions.stop(first)
        #expect(current.state.active && services.created[2].log.isEmpty)
    }

    private func settle(_ store: Store<ProductFeature>) async {
        for _ in 0..<10 where store.state.busy { await Task.yield() }
    }

    @Test(
        "Imports and extracts only with the selected context and never calls the service for a missing, other-revision or old-session request",
        .timeLimit(.minutes(1)), arguments: ProductRequest.allCases)
    func deniedRequests(request: ProductRequest) async throws {
        let library = try ProductLibrary()
        let session: ModelDemoSession
        switch request {
        case .missing:
            session = ModelDemoSession(
                id: LibraryTicket(owner: library.state.owner, number: 1), selection: library.selection,
                resource: library.resource("fp32"))
        case .otherRevision:
            try library.install("fp32")
            let current = try library.open()
            session = ModelDemoSession(
                id: LibraryTicket(owner: library.state.owner, number: current.id.number + 1),
                selection: library.selection,
                resource: library.resource("fp32", revision: ProductLibrary.otherRevision))
        case .oldSession:
            try library.install("fp32")
            session = try library.open()
            library.send(.suspendDemo)
            let next = try library.finishStop()
            #expect(next?.identity == session.identity && next?.id != session.id)
        }
        let services = ProductServices()
        let sessions = ProductDemoSessions(makeService: services.make)
        let store = sessions.bind(session, permits: library.gate(session))
        store.send(.draftChanged(draft))
        store.send(.submit(stamp))
        await settle(store)
        #expect(store.state.phase == .failed(closed) && store.state.cards.isEmpty)
        #expect(services.created.count == 1 && services.created.allSatisfy { $0.log.isEmpty })
    }

    @Test(
        "Re-entering with the same artifact reuses the Store and runs with the newest gate after the previous session's gate closes",
        .timeLimit(.minutes(1)))
    func sameIdentity() async throws {
        let library = try ProductLibrary()
        try library.install("fp32")
        let first = try library.open()
        let services = ProductServices()
        let sessions = ProductDemoSessions(makeService: services.make)
        let store = sessions.bind(first, permits: library.gate(first))
        let service = try #require(services.created.first)
        var events = service.events.makeAsyncIterator()
        store.send(.draftChanged(draft))
        library.send(.suspendDemo)
        try await sessions.stop(first)
        let stopped = try library.finishStop()
        let second = try #require(stopped)
        try await receive(&events, "release", "release")
        #expect(second.id != first.id && second.identity == first.identity && !library.gate(first)())
        store.send(.appeared)
        store.send(.submit(stamp))
        await settle(store)
        #expect(store.state.phase == .failed(closed) && service.log == ["release", "release"])
        #expect(sessions.bind(second, permits: library.gate(second)) === store)
        store.send(.submit(stamp))
        try await receive(&events, "extract finished")
        #expect(store.state.cards.map(\.resourceIdentity) == [ProductLibrary.identity("fp32")])
        #expect(services.created.count == 1)
        #expect(service.log == ["release", "release", "source", "source finished", "extract", "extract finished"])
    }

    @Test(
        "A different artifact identity creates a new Store and service that keep the draft, cards and card numbering",
        .timeLimit(.minutes(1)))
    func otherIdentity() async throws {
        let library = try ProductLibrary()
        try library.install("fp32")
        try library.install("int8")
        let first = try library.open()
        let services = ProductServices()
        let sessions = ProductDemoSessions(makeService: services.make)
        let store = sessions.bind(first, permits: library.gate(first))
        let service = try #require(services.created.first)
        var events = service.events.makeAsyncIterator()
        store.send(.draftChanged(draft))
        store.send(.submit(stamp))
        try await receive(&events, "extract finished")
        var kept = draft
        kept.text = "Storage: 1 TB"
        store.send(.draftChanged(kept))
        library.send(.selectArtifact(modelID: "gliner", artifactID: "int8"))
        try await sessions.stop(first)
        let stopped = try library.finishStop()
        let next = try #require(stopped)
        let replacement = sessions.bind(next, permits: library.gate(next))
        let fresh = try #require(services.created.last)
        #expect(next.resource.artifactID == "int8" && replacement !== store)
        #expect(services.created.count == 2 && fresh !== service)
        #expect(replacement.state.draft == kept && replacement.state.cards == store.state.cards)
        #expect(replacement.state.generation == store.state.generation)
        var freshEvents = fresh.events.makeAsyncIterator()
        replacement.send(.submit(stamp))
        try await receive(&freshEvents, "extract finished")
        #expect(fresh.folders == [URL(filePath: "/models/int8")])
        #expect(service.folders == [URL(filePath: "/models/fp32")])
        #expect(
            replacement.state.cards.map(\.resourceIdentity) == [
                ProductLibrary.identity("fp32"), ProductLibrary.identity("int8"),
            ])
        #expect(Set(replacement.state.cards.map(\.id)).count == 2)
        #expect(sessions.store(for: library.selection) === replacement)
    }

    @Test(
        "Stopping closes the Store before joining the release, reports a release failure, and ignores a session with another identity",
        .timeLimit(.minutes(1)))
    func stopOrdering() async throws {
        let library = try ProductLibrary()
        try library.install("fp32")
        let session = try library.open()
        let services = ProductServices()
        let sessions = ProductDemoSessions(makeService: services.make)
        try await sessions.stop(session)
        let store = sessions.bind(session, permits: library.gate(session))
        let service = try #require(services.created.first)
        var events = service.events.makeAsyncIterator()
        let stale = ModelDemoSession(
            id: session.id, selection: session.selection,
            resource: library.resource("fp32", revision: ProductLibrary.otherRevision))
        try await sessions.stop(stale)
        #expect(service.log.isEmpty && store.state.active && store.state.generation == 0)
        service.holdsRelease = true
        let progress = StopProgress()
        let stopping = Task {
            try await sessions.stop(session)
            progress.finished = true
        }
        try await receive(&events, "release")
        #expect(!store.state.active && store.state.phase == .stopping && !progress.finished)
        service.holdsRelease = false
        service.resumeReleases()
        try await stopping.value
        #expect(progress.finished)
        service.releaseFailure = .busy
        await #expect(throws: ProductError.busy) { try await sessions.stop(session) }
    }

    @Test(
        "The draft, cards and sources stay available for the preview after the session stops and the model is deleted",
        .timeLimit(.minutes(1)))
    func keepsDataAfterStop() async throws {
        let library = try ProductLibrary()
        try library.install("fp32")
        let session = try library.open()
        let services = ProductServices()
        let sessions = ProductDemoSessions(makeService: services.make)
        let store = sessions.bind(session, permits: library.gate(session))
        let service = try #require(services.created.first)
        var events = service.events.makeAsyncIterator()
        store.send(.draftChanged(draft))
        store.send(.submit(stamp))
        try await receive(&events, "extract finished")
        var kept = draft
        kept.text = "Storage: 1 TB"
        store.send(.draftChanged(kept))
        let cards = store.state.cards
        library.send(.closeDemo)
        try await sessions.stop(session)
        let remaining = try library.finishStop()
        #expect(remaining == nil)
        library.send(.deleteRequested("fp32"))
        let request = try #require(library.state.installations["fp32"]?.requestID)
        library.send(.deleted("fp32", request))
        #expect(library.state.installations["fp32"]?.phase == .missing)
        let preserved = try #require(sessions.store(for: library.selection))
        #expect(preserved === store && preserved.state.draft == kept && preserved.state.cards == cards)
        #expect(cards.count == 1 && cards.first?.memo.source.text == "Memory: 16 GB")
    }
}

enum ProductRequest: CaseIterable, Sendable {
    case missing, otherRevision, oldSession
}

@MainActor
final class StopProgress {
    var finished = false
}

@MainActor
final class ProductServices {
    private(set) var created: [ProductServiceFake] = []

    func make() -> any ProductService {
        let service = ProductServiceFake()
        created.append(service)
        return service
    }
}

@MainActor
final class ProductLibrary {
    static let revision = "0123456789abcdef0123456789abcdef01234567"
    static let otherRevision = "fedcba9876543210fedcba9876543210fedcba98"
    let selection = DemoSelection(modelID: "gliner", demoID: "comparison")
    private(set) var state: ModelLibraryState

    init() throws {
        let artifacts = ["fp32", "int8"].map { id in
            ModelArtifact(
                id: id, title: id, contract: "gliner-gl4-" + id, platforms: [.iOS], sizeBytes: 4,
                download: ModelDownload(repository: "owner/fixture", revision: ProductLibrary.revision),
                upstream: ModelUpstream(repository: "upstream/fixture", revision: ProductLibrary.revision),
                files: [ModelArtifactFile(path: "model.bin", bytes: 4, sha256: String(repeating: "c", count: 64))])
        }
        let model = LibraryModel(
            id: "gliner", name: "GLiNER", category: "Fixture", artifacts: artifacts, defaultArtifactID: "fp32",
            platforms: [.iOS], demos: [ModelDemoLink(demoID: "comparison", platforms: [.iOS])])
        let demo = DemoDescriptor(
            id: "comparison", title: "Comparison", systemImage: "doc.text", summary: "Fixture", platforms: [.iOS])
        state = ModelLibraryState(
            catalog: try ModelCatalog(models: [model], demos: [demo]),
            environment: LibraryEnvironment(platform: .iOS, connectedDemos: ["comparison"]), owner: UUID())
    }

    static func identity(_ artifactID: String) -> String { "gliner/\(artifactID)@\(revision)" }

    func resource(_ artifactID: String, revision: String = ProductLibrary.revision) -> ValidatedModelResource {
        ValidatedModelResource(
            modelID: "gliner", artifactID: artifactID, contract: "gliner-gl4-" + artifactID, revision: revision,
            upstreamRevision: ProductLibrary.revision, directory: URL(filePath: "/models/" + artifactID), sizeBytes: 4)
    }

    func install(_ artifactID: String) throws {
        send(.downloadRequested(artifactID))
        let request = try #require(state.installations[artifactID]?.requestID)
        send(.downloaded(artifactID, request, URL(filePath: "/models/" + artifactID)))
        send(.validated(artifactID, request, resource(artifactID)))
        #expect(state.installations[artifactID]?.phase == .ready)
    }

    func open() throws -> ModelDemoSession {
        send(.openDemo(selection))
        return try #require(state.session)
    }

    func finishStop() throws -> ModelDemoSession? {
        let session = try #require(state.session)
        let request = try #require(state.stopRequestID)
        send(.demoStopped(session.id, request))
        return state.session
    }

    func gate(_ session: ModelDemoSession) -> @MainActor () -> Bool {
        { [weak self] in self?.state.permits(session.id, selection: session.selection) == true }
    }

    func send(_ action: ModelLibraryAction) {
        _ = ModelLibraryFeature.reduce(&state, action)
    }
}
