import CryptoKit
import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTransfer
import ReceiptCore
import ReceiptFeature
import SwiftUI
import Testing

@testable import ModelLibraryRuntime
@testable import ReceiptDemoRuntime

@MainActor
struct ReceiptModelsIntegrationTests {
    @Test(
        "Models verifies the public JEV bundle and the production receipt session exports the synthetic ledger",
        .enabled(if: ProcessInfo.processInfo.environment["RECEIPT_MODELS_BUNDLE"] != nil), .timeLimit(.minutes(10)))
    func nativeWorkflow() async throws {
        let environment = ProcessInfo.processInfo.environment
        let source = URL(filePath: try #require(environment["RECEIPT_MODELS_BUNDLE"]))
        let originalManifest = try Data(contentsOf: source.appending(path: "decision.json"))
        let root = FileManager.default.temporaryDirectory.appending(path: "receipt-models-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let storage = ManagedModelStore(root: root.appending(path: "models"))
        let artifact = try JEVModelArtifact.make()
        let catalog = try ReceiptDemoCatalog.make(artifact: artifact)
        let model = try #require(catalog.models.first)
        let migration = await storage.migrate(candidates: [source], model: model, artifact: artifact)
        guard case .migrated(let installed) = migration.outcomes[source] else {
            Issue.record("The pinned public JEV bundle was not installed")
            return
        }
        let events = AsyncStream<ReceiptAction>.makeStream()
        let adapter = ReceiptIntegrationAdapter(events: events.continuation)
        let assembly = ModelLibraryAssembly(adapters: [adapter], store: storage, transfer: ArtifactTransfer())
        let effects = AsyncStream<ModelLibraryEffect>.makeStream()
        let library = Store<ModelLibraryFeature>(
            initialState: ModelLibraryState(
                catalog: catalog, environment: LibraryEnvironment(platform: .macOS, connectedDemos: ["receipt-desk"]),
                owner: UUID())
        ) { effect, send in
            await assembly.runner.run(effect, send: send)
            effects.continuation.yield(effect)
        }
        let rendering = ModelLibraryRendering(registry: assembly.registry, runner: assembly.runner, library: library)
        var completed = effects.stream.makeAsyncIterator()
        var evidence: [String: Any] = [:]
        library.send(.refresh)
        while let effect = await completed.next() { if case .inspect = effect { break } }
        #expect(library.state.installations[artifact.id]?.phase == .ready)
        library.send(.openDemo(ReceiptDemoCatalog.selection))
        let session = try #require(library.state.session)
        do {
            _ = rendering.content(for: session)
            library.send(.requestDemoPreparation(session.id))
            while let effect = await completed.next() { if case .prepareDemo = effect { break } }
            #expect(library.state.demoPreparation == .available)
            _ = rendering.content(for: session)
            let desk = try #require(adapter.store)
            #expect(!desk.state.modelPrepared && desk.state.items.isEmpty)
            #expect(await storage.activeLeases(identity: installed.identity) == 1)
            desk.send(.samples)
            var actions = events.stream.makeAsyncIterator()
            while await actions.next() != nil { if desk.state.phase == .idle { break } }
            try #require(desk.state.items.count == 4)
            #expect(desk.state.message == nil && desk.state.items.allSatisfy { $0.failure == nil })
            #expect(desk.state.items.map { $0.record?.fields.category } == [.supplies, .meals, .travel, .review])
            #expect(desk.state.items.map { $0.record?.fields.amount } == ["1320", "1100", "900", "880"])
            #expect(desk.state.items.allSatisfy { $0.record?.fields.date == "2026-11-06" })
            desk.send(.confirmTogether)
            for item in desk.state.items where item.record?.canConfirm == true && !item.isConfirmed {
                desk.send(.confirm(item.id))
            }
            let ledger = try #require(desk.state.ledger)
            #expect(ledger.totalYen == 3320 && ledger.rows.count == 3)
            desk.send(.requestExport)
            let selection = try #require(desk.state.exportSelection)
            desk.send(.exportSelected(selection.id, root))
            while await actions.next() != nil { if desk.state.phase == .idle { break } }
            let saved = try #require(desk.state.saved)
            let csv = try Data(contentsOf: saved.directory.appending(path: "ledger.csv"))
            try #require(csv == ledger.csv)
            for row in ledger.rows {
                let bytes = try Data(contentsOf: saved.directory.appending(path: row.relativePath))
                let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
                try #require(digest == row.sourceSHA256)
            }
            evidence = [
                "format": "receipt-models-integration/1", "os": ProcessInfo.processInfo.operatingSystemVersionString,
                "revision": JEVModelArtifact.revision, "manifestSHA256": JEVModelArtifact.manifestSHA256,
                "categories": desk.state.items.compactMap { $0.record?.fields.category.rawValue },
                "confirmedCount": ledger.rows.count, "totalYen": String(ledger.totalYen),
                "originalHashesMatched": true, "csvMatched": true,
            ]
        } catch {
            await withTaskCancellationShield {
                await assembly.runner.run(.stopDemo(session, session.id), send: { _ in })
            }
            throw error
        }
        library.send(.closeDemo)
        while let effect = await completed.next() { if case .stopDemo = effect { break } }
        #expect(library.state.session == nil && library.state.pendingDestination == nil)
        let finalLeases = await storage.activeLeases(identity: installed.identity)
        try #require(finalLeases == 0)
        evidence["leasesAfterClose"] = finalLeases
        #expect(adapter.store?.state.ended == true)
        #expect(try await storage.delete(identity: installed.identity) == installed.sizeBytes)
        #expect(try Data(contentsOf: source.appending(path: "decision.json")) == originalManifest)
        if let reportPath = environment["RECEIPT_MODELS_REPORT"] {
            try JSONSerialization.data(withJSONObject: evidence, options: [.prettyPrinted, .sortedKeys])
                .write(to: URL(filePath: reportPath), options: .withoutOverwriting)
        }
        events.continuation.finish()
        effects.continuation.finish()
    }
}

@MainActor
private final class ReceiptIntegrationAdapter: ModelDemoAdapter {
    let demoID = "receipt-desk"
    private let sessions: ReceiptDemoSessions
    var store: Store<ReceiptFeature>?

    init(events: AsyncStream<ReceiptAction>.Continuation) {
        sessions = ReceiptDemoSessions(
            makeService: { LiveReceiptService() }, observe: { _, action in events.yield(action) })
    }
    func preview(for selection: DemoSelection) -> AnyView { AnyView(EmptyView()) }
    func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
        guard gate.session == session, gate.isPermitted() else { return preview(for: session.selection) }
        store = sessions.bind(session, permits: gate.isPermitted)
        return AnyView(EmptyView())
    }
    func stop(_ session: ModelDemoSession) async throws { await sessions.stop(session) }
}
