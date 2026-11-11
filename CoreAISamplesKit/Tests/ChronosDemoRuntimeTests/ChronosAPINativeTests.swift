import ChronosCore
import ChronosFeature
import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTransfer
import Testing

@testable import ChronosDemoRuntime
@testable import ModelLibraryRuntime

@MainActor
struct ChronosAPINativeTests {
    @Test(
        "A5 Live NYC history and production Models sessions match 20,160 inputs and all 6,720 raw native outputs",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_CHRONOS2_API_BUNDLE"] != nil), .timeLimit(.minutes(30))
    )
    func liveHistoryAndNative() async throws {
        let environment = ProcessInfo.processInfo.environment
        let source = URL(filePath: try #require(environment["COREAI_CHRONOS2_API_BUNDLE"]))
        let referenceURL = URL(filePath: try #require(environment["COREAI_CHRONOS2_API_REFERENCE"]))
        let reportURL = URL(filePath: try #require(environment["COREAI_CHRONOS2_API_REPORT"]))
        let historyURL = reportURL.deletingPathExtension().appendingPathExtension("history.json")
        try #require(
            !FileManager.default.fileExists(atPath: reportURL.path)
                && !FileManager.default.fileExists(atPath: historyURL.path))
        let referenceData = try Data(contentsOf: referenceURL)
        try #require(
            NYCAPIContract.hash(referenceData) == "8f894aaac769294a3e6843225e6f86519f85090d747b16d94a288fb987661470")
        let reference = try JSONDecoder().decode(ChronosScenesReference.self, from: referenceData)
        try #require(
            reference.modelRevision == ChronosContract.revision && reference.atol == 0.0001 && reference.rtol == 0.0001)
        let catalog = try MacDemoCatalog.make()
        let model = try #require(catalog.model("chronos2"))
        let artifact = try ChronosModelArtifact.make()
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let storage = ManagedModelStore(root: root.appending(path: "models"))
        let staging = try await storage.copyForImport(from: source, model: model, artifact: artifact) { _ in }
        let installed = try await storage.promote(staging: staging, model: model, artifact: artifact)
        let adapter = try ChronosNativeAdapter(directory: ChronosRepository.bundledDirectory(), api: NYCAPIClient())
        let assembly = ModelLibraryAssembly(adapters: [adapter], store: storage, transfer: ArtifactTransfer())
        let effects = AsyncStream<ModelLibraryEffect>.makeStream()
        defer {
            effects.continuation.finish()
            adapter.actions.continuation.finish()
        }
        let library = Store<ModelLibraryFeature>(
            initialState: ModelLibraryState(
                catalog: catalog,
                environment: LibraryEnvironment(platform: .macOS, connectedDemos: [adapter.demoID]), owner: UUID())
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
        var batch: ChronosForecastBatch?
        do {
            _ = rendering.content(for: session)
            library.send(.requestDemoPreparation(session.id))
            while let effect = await completed.next() { if case .prepareDemo = effect { break } }
            try #require(library.state.demoPreparation == .available)
            _ = rendering.content(for: session)
            let store = try #require(adapter.store)
            var actions = adapter.actions.stream.makeAsyncIterator()
            while await actions.next() != nil { if store.state.phase == .ready { break } }
            let fixed = try #require(store.state.catalog?.scenario("friday-evening"))
            try #require(await storage.activeLeases(identity: installed.identity) == 1)
            store.send(.selectSource(.api))
            store.send(.refreshCoverage)
            while await actions.next() != nil { if store.state.phase == .ready { break } }
            try #require(store.state.error == nil)
            store.send(.selectDate(try ChronosAPIFixture.date("2025-01-24T22:00:00Z")))
            try #require(store.state.canFetchHistory)
            store.send(.fetchHistory)
            while await actions.next() != nil { if store.state.phase == .ready { break } }
            try #require(store.state.error == nil)
            let history = try #require(store.state.selectedScenario?.apiHistory)
            try NYCAPIContract.encode(history).write(to: historyURL, options: .withoutOverwriting)
            for series in history.series {
                try #require(series.values == fixed.history[series.areaID]?.map(Optional.some))
            }
            try #require(history.series.reduce(0) { $0 + $1.values.count } == 20_160)
            store.send(.predict)
            while await actions.next() != nil { if store.state.phase == .ready { break } }
            let forecast = try #require(store.state.forecast)
            try #require(
                store.state.error == nil && forecast.mode == .live && forecast.inputSnapshot?.history == history)
            var count = 0
            for area in forecast.areas {
                let expected = try #require(
                    reference.cases.first {
                        $0.scenarioID == "friday-evening" && $0.areaID == area.areaID && $0.originStep == 0
                    })
                let raw = try #require(area.raw)
                try #require(expected.quantiles.count == 1 && expected.quantiles[0].count == 21)
                for row in 0..<21 {
                    try #require(expected.quantiles[0][row].count == 16)
                    for step in 0..<16 {
                        let value = expected.quantiles[0][row][step]
                        try #require(
                            value.isFinite
                                && abs(Double(raw.quantiles[row][step]) - value) <= 0.0001 + 0.0001 * abs(value))
                        count += 1
                    }
                }
            }
            try #require(count == 6720)
            batch = forecast
        } catch {
            await withTaskCancellationShield {
                await assembly.runner.run(.stopDemo(session, session.id), send: { _ in })
            }
            throw error
        }
        library.send(.closeDemo)
        while let effect = await completed.next() { if case .stopDemo = effect { break } }
        try #require(await storage.activeLeases(identity: installed.identity) == 0)
        try NYCAPIContract.encode(Report(forecast: #require(batch))).write(to: reportURL, options: .withoutOverwriting)
    }

    private struct Report: Encodable {
        let forecast: ChronosForecastBatch
        let format = "chronos2-api-native/1"
        let inputElements = 20_160
        let rawOutputElements = 6720
        let atol = 0.0001
        let rtol = 0.0001
        let leasesAfterClose = 0
    }
}
