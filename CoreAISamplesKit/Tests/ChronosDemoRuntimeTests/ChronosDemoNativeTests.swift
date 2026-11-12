import ChronosCore
import ChronosFeature
import CryptoKit
import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTransfer
import SwiftUI
import Testing

@testable import ChronosDemoRuntime
@testable import ModelLibraryRuntime

@MainActor
struct ChronosDemoNativeTests {
    @Test(
        "D5 Models and production sessions match all sixty official fixed-scene outputs and return every lease",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_CHRONOS2_DEMO_BUNDLE"] != nil),
        .timeLimit(.minutes(30)))
    func fixedScenes() async throws {
        let environment = ProcessInfo.processInfo.environment
        let source = URL(filePath: try #require(environment["COREAI_CHRONOS2_DEMO_BUNDLE"]))
        let referenceURL = URL(filePath: try #require(environment["COREAI_CHRONOS2_DEMO_REFERENCE"]))
        let referenceData = try Data(contentsOf: referenceURL)
        let referenceHash = SHA256.hash(data: referenceData).map { String(format: "%02x", $0) }.joined()
        try #require(referenceHash == "8f894aaac769294a3e6843225e6f86519f85090d747b16d94a288fb987661470")
        let reference = try JSONDecoder().decode(ChronosScenesReference.self, from: referenceData)
        try #require(
            reference.format == "chronos2-nyc-reference/1" && reference.modelRevision == ChronosContract.revision)
        try #require(reference.atol == 0.0001 && reference.rtol == 0.0001 && reference.cases.count == 60)
        try #require(reference.sourceHashes == ChronosRepository.hashes.filter { $0.key != "forecasts.json" })
        let directory =
            try environment["COREAI_CHRONOS2_DEMO_DATA"].map { URL(filePath: $0) }
            ?? ChronosRepository.bundledDirectory()
        let catalog = try MacDemoCatalog.make()
        let model = try #require(catalog.model("chronos2"))
        let artifact = try ChronosModelArtifact.make()
        let root = FileManager.default.temporaryDirectory.appending(path: "chronos-demo-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let storage = ManagedModelStore(root: root.appending(path: "models"))
        let staging = try await storage.copyForImport(from: source, model: model, artifact: artifact) { _ in }
        let installed = try await storage.promote(staging: staging, model: model, artifact: artifact)
        let adapter = ChronosNativeAdapter(
            directory: directory,
            journal: ChronosJournalAssembly(files: ChronosJournalFiles(directory: root.appending(path: "journal"))))
        let assembly = ModelLibraryAssembly(adapters: [adapter], store: storage, transfer: ArtifactTransfer())
        let effects = AsyncStream<ModelLibraryEffect>.makeStream()
        let library = Store<ModelLibraryFeature>(
            initialState: ModelLibraryState(
                catalog: catalog, environment: LibraryEnvironment(platform: .macOS, connectedDemos: [adapter.demoID]),
                owner: UUID())
        ) { effect, send in
            await assembly.runner.run(effect, send: send)
            effects.continuation.yield(effect)
        }
        let rendering = ModelLibraryRendering(registry: assembly.registry, runner: assembly.runner, library: library)
        var completed = effects.stream.makeAsyncIterator()
        library.send(.refresh)
        while let effect = await completed.next() {
            if case .inspect = effect, library.state.installations[artifact.id]?.phase == .ready { break }
        }
        library.send(.openDemo(ChronosDemoCatalog.selection))
        let session = try #require(library.state.session)
        var elements = 0
        do {
            _ = rendering.content(for: session)
            library.send(.requestDemoPreparation(session.id))
            while let effect = await completed.next() { if case .prepareDemo = effect { break } }
            try #require(library.state.demoPreparation == .available)
            _ = rendering.content(for: session)
            let store = try #require(adapter.store)
            var actions = adapter.actions.stream.makeAsyncIterator()
            while await actions.next() != nil { if store.state.phase == .ready { break } }
            try #require(store.state.mode == .live && store.state.forecast == nil)
            try #require(await storage.activeLeases(identity: installed.identity) == 1)
            let demo = try #require(store.state.catalog)
            for scene in demo.scenarios {
                store.send(.selectScenario(scene.id))
                store.send(.predict)
                while await actions.next() != nil { if store.state.phase == .ready { break } }
                let forecast = try #require(store.state.forecast)
                try #require(store.state.error == nil && forecast.mode == .live && forecast.originStep == 0)
                try #require(forecast.identity?.manifest.sha256 == installed.revision)
                for area in demo.areas {
                    let expected = try #require(
                        reference.cases.first {
                            $0.scenarioID == scene.id && $0.areaID == area.id && $0.originStep == 0
                        })
                    let raw = try #require(forecast.area(area.id)?.raw)
                    try #require(expected.quantiles.count == 1 && expected.quantiles[0].count == 21)
                    for row in 0..<21 {
                        try #require(expected.quantiles[0][row].count == 16)
                        for step in 0..<16 {
                            let value = expected.quantiles[0][row][step]
                            try #require(
                                value.isFinite
                                    && abs(Double(raw.quantiles[row][step]) - value) <= 0.0001 + 0.0001 * abs(value))
                            elements += 1
                        }
                    }
                }
            }
        } catch {
            await withTaskCancellationShield {
                await assembly.runner.run(.stopDemo(session, session.id), send: { _ in })
            }
            throw error
        }
        library.send(.closeDemo)
        while let effect = await completed.next() { if case .stopDemo = effect { break } }
        try #require(elements == 20_160)
        try #require(await storage.activeLeases(identity: installed.identity) == 0)
        #expect(adapter.store?.state.ended == true && library.state.session == nil)
        if let output = environment["COREAI_CHRONOS2_DEMO_REPORT"] {
            try JSONSerialization.data(
                withJSONObject: [
                    "format": "chronos2-demo-native/1", "referenceSHA256": referenceHash,
                    "manifestSHA256": installed.revision, "cases": 60, "elements": elements,
                    "atol": 0.0001, "rtol": 0.0001, "leasesAfterClose": 0,
                    "scope":
                        "Production Models, Store and Service; fixed scenes only. GUI and device acceptance are separate.",
                ], options: [.prettyPrinted, .sortedKeys]
            ).write(to: URL(filePath: output), options: .withoutOverwriting)
        }
        adapter.actions.continuation.finish()
        adapter.journalActions.continuation.finish()
        effects.continuation.finish()
    }
}

struct ChronosScenesReference: Decodable {
    struct Entry: Decodable {
        let scenarioID: String
        let originStep: Int
        let areaID: String
        let quantiles: [[[Double]]]
    }
    let format: String
    let modelRevision: String
    let sourceHashes: [String: String]
    let atol: Double
    let rtol: Double
    let cases: [Entry]
}

@MainActor
final class ChronosNativeAdapter: ModelDemoAdapter {
    let demoID = ChronosDemoCatalog.selection.demoID
    let actions = AsyncStream<ChronosFeature.Action>.makeStream()
    let journalActions = AsyncStream<ChronosJournalFeature.Action>.makeStream()
    let sessions: ChronosDemoSessions
    var store: Store<ChronosFeature>?
    var journalStore: Store<ChronosJournalFeature>?

    init(directory: URL, api: NYCAPIClient? = nil, journal: ChronosJournalAssembly = .shared) {
        let sink = actions.continuation
        let journalSink = journalActions.continuation
        sessions = ChronosDemoSessions(
            makeService: { resource, permits in
                LiveChronosService(
                    resource: resource, repository: ChronosRepository(directory: directory), api: api, permits: permits)
            }, journal: journal, observe: { _, action in sink.yield(action) },
            observeJournal: { _, action in journalSink.yield(action) })
    }
    func preview(for selection: DemoSelection) -> AnyView { AnyView(EmptyView()) }
    func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
        guard gate.session == session, gate.isPermitted() else { return AnyView(EmptyView()) }
        store = sessions.bind(session, gate: gate)
        journalStore = sessions.journal(for: session)
        return AnyView(EmptyView())
    }
    func stop(_ session: ModelDemoSession) async throws { await sessions.stop(session) }
}
