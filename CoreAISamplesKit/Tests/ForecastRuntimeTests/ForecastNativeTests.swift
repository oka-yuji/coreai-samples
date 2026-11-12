import ChronosCore
import CryptoKit
import FeatureStore
import ForecastFeature
import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTransfer
import SwiftUI
import Testing

@testable import ForecastRuntime
@testable import ModelLibraryRuntime

@MainActor
struct ForecastNativeTests {
    @Test(
        "S4 Production Models and sales sessions match every official scalar, close with zero leases and reopen",
        .enabled(
            if: ["COREAI_FORECAST_BUNDLE", "COREAI_SALES_CSV", "COREAI_FORECAST_REFERENCE", "COREAI_FORECAST_REPORT"]
                .allSatisfy { ProcessInfo.processInfo.environment[$0] != nil }),
        .timeLimit(.minutes(30)))
    func sales() async throws {
        let environment = ProcessInfo.processInfo.environment
        let bundle = URL(filePath: try #require(environment["COREAI_FORECAST_BUNDLE"]))
        let csv = URL(filePath: try #require(environment["COREAI_SALES_CSV"]))
        let referenceURL = URL(filePath: try #require(environment["COREAI_FORECAST_REFERENCE"]))
        let output = URL(filePath: try #require(environment["COREAI_FORECAST_REPORT"]))
        try #require(!FileManager.default.fileExists(atPath: output.path))
        let sourceBytes = try Data(contentsOf: csv)
        try SalesSource.validate(sourceBytes)
        let sourceHash = SHA256.hash(data: sourceBytes).map { String(format: "%02x", $0) }.joined()
        let referenceData = try Data(contentsOf: referenceURL)
        let reference = try JSONDecoder().decode(ForecastReference.self, from: referenceData)
        try #require(!reference.cases.isEmpty && Set(reference.cases.map(\.id)).count == reference.cases.count)
        let root = URL.temporaryDirectory.appending(path: "forecast-native-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let storage = ManagedModelStore(root: root.appending(path: "models"))
        let catalog = try MacDemoCatalog.make()
        let model = try #require(catalog.model("chronos2"))
        let artifact = try ChronosModelArtifact.make()
        var memory = [ForecastMemory.capture("beforeImport")]
        let staging = try await storage.copyForImport(from: bundle, model: model, artifact: artifact) { _ in }
        let installed = try await storage.promote(staging: staging, model: model, artifact: artifact)
        let adapter = ForecastNativeAdapter(source: sourceBytes, cache: root.appending(path: "cache"))
        let assembly = ModelLibraryAssembly(adapters: [adapter], store: storage, transfer: ArtifactTransfer())
        let effects = AsyncStream<ModelLibraryEffect>.makeStream()
        defer {
            effects.continuation.finish()
            adapter.actions.continuation.finish()
        }
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
        var records: [ForecastNativeRecord] = []
        var actions = adapter.actions.stream.makeAsyncIterator()
        for pass in 1...2 {
            library.send(.openDemo(ChronosDemoCatalog.salesSelection))
            let session = try #require(library.state.session)
            do {
                _ = rendering.content(for: session)
                library.send(.requestDemoPreparation(session.id))
                while let effect = await completed.next() { if case .prepareDemo = effect { break } }
                try #require(library.state.demoPreparation == .available)
                _ = rendering.content(for: session)
                let store = try #require(adapter.store)
                try #require(store.state.phase == .idle && store.state.dataset == nil && store.state.result == nil)
                store.send(.loadData)
                while await actions.next() != nil { if !store.state.isWorking { break } }
                let dataset = try #require(store.state.dataset)
                try #require(store.state.error == nil && store.state.result == nil)
                try #require(await storage.activeLeases(identity: installed.identity) == 1)
                for item in reference.cases {
                    let request = try dataset.request(
                        seriesID: item.seriesID, originIndex: item.originIndex, horizon: item.horizon)
                    try #require(request.context == item.context && request.context.count == 365)
                    let context = try ChronosContext(request.context.map { $0 ?? .nan })
                    try #require(context.rawCount == 365 && context.paddedValues.count == 368)
                    try #require(
                        zip(context.paddedValues.suffix(365), item.context).allSatisfy { raw, expected in
                            expected.map { raw == $0 } ?? raw.isNaN
                        })
                    store.send(.selectSeries(item.seriesID))
                    store.send(.selectHorizon(item.horizon))
                    store.send(.selectOrigin(item.originIndex))
                    try #require(store.state.canPredict && store.state.request == request && store.state.result == nil)
                    memory.append(ForecastMemory.capture("beforePrediction"))
                    store.send(.predict)
                    while await actions.next() != nil { if !store.state.isWorking { break } }
                    let result = try #require(store.state.result)
                    try #require(store.state.error == nil && result.request == request)
                    try #require(
                        result.identity.source == .expected && result.identity.manifest.sha256 == installed.revision)
                    for file in artifact.files where file.path != "chronos2.json" {
                        try #require(
                            result.identity.files[file.path]
                                == ChronosFileIdentity(bytes: file.bytes, sha256: file.sha256))
                    }
                    try #require(item.quantiles.count == 21)
                    var maximumAbsoluteError = 0.0
                    for row in 0..<21 {
                        try #require(item.quantiles[row].count == 16)
                        for step in 0..<16 {
                            let expected = item.quantiles[row][step]
                            let error = abs(Double(result.raw.quantiles[row][step]) - expected)
                            try #require(expected.isFinite && error <= 0.0001 + 0.0001 * abs(expected))
                            maximumAbsoluteError = max(maximumAbsoluteError, error)
                        }
                    }
                    if pass == 2 {
                        let first = try #require(records.first { $0.id == item.id && $0.pass == 1 })
                        try #require(first.result.request == result.request && first.result.identity == result.identity)
                    }
                    records.append(
                        ForecastNativeRecord(
                            id: item.id, pass: pass, result: result, metrics: result.metrics,
                            scalars: 336, maximumAbsoluteError: maximumAbsoluteError))
                    memory.append(ForecastMemory.capture("afterPrediction"))
                }
            } catch {
                await withTaskCancellationShield {
                    await assembly.runner.run(.stopDemo(session, session.id), send: { _ in })
                }
                throw error
            }
            library.send(.closeDemo)
            while let effect = await completed.next() { if case .stopDemo = effect { break } }
            try #require(await storage.activeLeases(identity: installed.identity) == 0)
            try #require(library.state.session == nil && adapter.store?.state.ended == true)
            memory.append(ForecastMemory.capture("afterClose"))
        }
        let report = ForecastNativeReport(
            sourceSHA256: sourceHash, sourceBytes: sourceBytes.count,
            referenceSHA256: SHA256.hash(data: referenceData).map { String(format: "%02x", $0) }.joined(),
            records: records, memory: memory)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(report).write(to: output, options: .withoutOverwriting)
        #expect(SHA256.hash(data: try Data(contentsOf: csv)) == SHA256.hash(data: sourceBytes))
    }
}

struct ForecastReference: Decodable {
    struct Entry: Decodable {
        let id: String
        let seriesID: String
        let originIndex: Int
        let horizon: Int
        let context: [Float?]
        let quantiles: [[Double]]

        enum CodingKeys: String, CodingKey {
            case id, horizon, context, quantiles
            case seriesID = "series_id"
            case originIndex = "origin_index"
        }
    }
    let cases: [Entry]
}

struct ForecastNativeRecord: Encodable {
    let id: String
    let pass: Int
    let result: ForecastResult
    let metrics: ForecastMetrics
    let scalars: Int
    let maximumAbsoluteError: Double
}

struct ForecastNativeReport: Encodable {
    let format = "chronos2-sales-native/1"
    let scope =
        "Production Models, Store, Service and Sessions; rendering adapter replaced. Not GUI or iPhone acceptance."
    let os = ProcessInfo.processInfo.operatingSystemVersionString
    let machine = ForecastMemory.machine()
    let precision = "FP32"
    let atol = 0.0001
    let rtol = 0.0001
    let leasesAfterEachClose = 0
    let sourceSHA256: String
    let sourceBytes: Int
    let referenceSHA256: String
    let records: [ForecastNativeRecord]
    let memory: [ForecastMemory]
}

@MainActor
final class ForecastNativeAdapter: ModelDemoAdapter {
    let demoID = ChronosDemoCatalog.salesSelection.demoID
    let actions = AsyncStream<ForecastFeature.Action>.makeStream()
    let sessions: ForecastDemoSessions
    var store: Store<ForecastFeature>?

    init(source: Data, cache: URL) {
        let sink = actions.continuation
        sessions = ForecastDemoSessions(
            makeService: { resource, permits in
                LiveForecastService(
                    resource: resource, repository: SalesDataRepository(cacheDirectory: cache, fetch: { source }),
                    permits: permits)
            }, observe: { _, action in sink.yield(action) })
    }

    func preview(for selection: DemoSelection) -> AnyView { AnyView(EmptyView()) }
    func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
        store = sessions.bind(session, gate: gate)
        return AnyView(EmptyView())
    }
    func stop(_ session: ModelDemoSession) async throws { await sessions.stop(session) }
}
