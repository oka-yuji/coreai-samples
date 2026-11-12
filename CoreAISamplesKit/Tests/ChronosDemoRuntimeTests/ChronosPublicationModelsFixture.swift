import ChronosFeature
import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTestSupport
import ModelLibraryTransfer
import Observation
import SwiftUI
import Testing

@testable import ChronosDemoRuntime
@testable import ModelLibraryRuntime

@MainActor
struct ChronosPublicationModelsFixture {
    let fixture: ImageBundleFixture
    let storage: ManagedModelStore
    let assembly: ModelLibraryAssembly
    let catalog: ModelCatalog
    let adapter: ChronosPublicationAdapter
    let library: Store<ModelLibraryFeature>
    let session: ModelDemoSession
    let gate: DemoExecutionGate
    let journal: Store<ChronosJournalFeature>

    static func make(journal: ChronosJournalAssembly) async throws -> Self {
        let fixture = try ImageBundleFixture()
        let storage = fixture.store()
        let model = LibraryModel(
            id: ChronosDemoCatalog.selection.modelID, name: "Chronos CPU fixture", category: "Fixture",
            artifacts: [fixture.artifact], defaultArtifactID: fixture.artifact.id, platforms: [.iOS],
            demos: [ModelDemoLink(demoID: ChronosDemoCatalog.selection.demoID, platforms: [.iOS])])
        let catalog = try ModelCatalog(
            models: [model],
            demos: [
                DemoDescriptor(
                    id: ChronosDemoCatalog.selection.demoID, title: "Chronos fixture", systemImage: "chart.xyaxis.line",
                    summary: "CPU publication fixture", platforms: [.iOS])
            ])
        let staged = try await storage.copyForImport(
            from: fixture.source, model: model, artifact: fixture.artifact, progress: { _ in })
        _ = try await storage.promote(staging: staged, model: model, artifact: fixture.artifact)
        let adapter = try await ChronosPublicationAdapter(journal: journal)
        let assembly = ModelLibraryAssembly(adapters: [adapter], store: storage, transfer: ArtifactTransfer())
        let bound = assembly.makeLibrary(catalog: catalog, capabilities: [:], owner: UUID())
        let library = bound.store
        library.send(.refresh)
        for await state in Observations({ library.state }) {
            if state.installations[fixture.artifact.id]?.phase == .ready { break }
        }
        library.send(.openDemo(ChronosDemoCatalog.selection))
        let session = try #require(library.state.session)
        _ = bound.rendering.content(for: session)
        await assembly.runner.prepare(session)
        _ = bound.rendering.content(for: session)
        let gate = try #require(adapter.gate)
        let journalStore = try #require(adapter.sessions.journal(for: session))
        try #require(gate.isPermitted())
        return Self(
            fixture: fixture, storage: storage, assembly: assembly, catalog: catalog, adapter: adapter,
            library: library, session: session, gate: gate, journal: journalStore)
    }

    func stop() async {
        adapter.resumeStop()
        await assembly.runner.run(.stopDemo(session, session.id)) { _ in }
        library.send(.closeDemo)
        adapter.finish()
    }
}

@MainActor
final class ChronosPublicationAdapter: ModelDemoAdapter {
    let demoID = ChronosDemoCatalog.selection.demoID
    let sessions: ChronosDemoSessions
    let stopping = AsyncStream<Void>.makeStream()
    let actions = AsyncStream<ChronosJournalFeature.Action>.makeStream()
    private let stopRelease = AsyncStream<Void>.makeStream()
    var holdsStop = true
    var gate: DemoExecutionGate?

    init(journal: ChronosJournalAssembly) async throws {
        let catalog = try await ChronosRepository.bundled().load()
        let sink = actions.continuation
        sessions = ChronosDemoSessions(
            makeService: { _, _ in ChronosPublicationService(catalog: catalog) }, journal: journal,
            observeJournal: { _, action in sink.yield(action) })
    }

    func preview(for selection: DemoSelection) -> AnyView { AnyView(EmptyView()) }

    func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
        self.gate = gate
        _ = sessions.bind(session, gate: gate)
        return AnyView(EmptyView())
    }

    func stop(_ session: ModelDemoSession) async throws {
        stopping.continuation.yield(())
        if holdsStop {
            var release = stopRelease.stream.makeAsyncIterator()
            _ = await release.next()
        }
        await sessions.stop(session)
    }

    func resumeStop() {
        holdsStop = false
        stopRelease.continuation.yield(())
    }

    func finish() {
        resumeStop()
        stopping.continuation.finish()
        stopRelease.continuation.finish()
        actions.continuation.finish()
    }
}

@MainActor
private final class ChronosPublicationService: ChronosService {
    let catalog: ChronosCatalog
    init(catalog: ChronosCatalog) { self.catalog = catalog }
    func load() async throws -> ChronosLoadedDemo { ChronosLoadedDemo(catalog: catalog, liveAvailable: true) }
    func predict(_ request: ChronosForecastRequest) async throws -> ChronosForecastBatch {
        throw ChronosDemoFailure.unavailable
    }
    func release() async {}
}
