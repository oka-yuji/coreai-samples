import ChronosCore
import ChronosRuntime
import ForecastFeature
import Foundation
import ModelLibraryFeature
import Testing

@testable import ForecastRuntime

enum ForecastFixture {
    static func csv() -> Data {
        let items = [42, 45, 21, 10].enumerated().flatMap { brand, count in
            (1...count).map { "B\(brand + 1)_\($0)" }
        }
        let header = ["DATE"] + items.map { "QTY_" + $0 } + items.map { "PROMO_" + $0 }
        let rows =
            [header]
            + ["2014-01-02", "2015-01-02", "2018-12-31"].map {
                [$0] + Array(repeating: "2001", count: 118) + Array(repeating: "0", count: 118)
            }
        return Data(rows.map { $0.joined(separator: ",") }.joined(separator: "\n").utf8)
    }

    static func directory() throws -> URL {
        let root = URL.temporaryDirectory.appending(path: "forecast-synthetic-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    static func repository(_ root: URL) -> SalesDataRepository {
        SalesDataRepository(cacheDirectory: root, fetch: { csv() }, admit: SalesDataset.parse)
    }

    static func session(_ number: Int, kind: ForecastKind = .productSales) -> ModelDemoSession {
        let owner = UUID()
        return ModelDemoSession(
            id: LibraryTicket(owner: owner, number: number), selection: kind.selection,
            resource: ValidatedModelResource(
                modelID: "chronos2", artifactID: "chronos2-fp32", contract: "chronos2-fp32-v1",
                revision: String(repeating: "a", count: 64), upstreamRevision: ChronosContract.revision,
                directory: URL(filePath: "/unused-synthetic-forecast"), sizeBytes: 1, sourceKind: .localManifest))
    }

    static func identity(_ hash: String = String(repeating: "a", count: 64)) -> ChronosBundleIdentity {
        ChronosBundleIdentity(source: .expected, manifest: .init(bytes: 1, sha256: hash), files: [:])
    }

    static func result(_ request: ForecastRequest) throws -> ForecastResult {
        try ForecastResult(
            request: request,
            raw: ChronosPrediction(quantiles: Array(repeating: Array(repeating: -2, count: 16), count: 21)),
            identity: identity(), seconds: 0, loaded: true, timingBoundary: "Synthetic")
    }
}

@MainActor
final class ForecastServiceFake: ForecastService {
    let dataset: ForecastDataset
    let events = AsyncStream<String>.makeStream()
    var pause: String?
    var failing = false
    var continuation: CheckedContinuation<Void, Never>?
    var busy = false
    var releases = 0
    var loads = 0
    var predictions = 0
    var releasedWhileBusy = false

    init(dataset: ForecastDataset? = nil) throws {
        self.dataset = try dataset ?? SalesDataset.parse(ForecastFixture.csv())
    }

    func load(refresh: Bool) async throws -> ForecastDataset {
        loads += 1
        try await operation("load")
        return dataset
    }

    func predict(_ request: ForecastRequest) async throws -> ForecastResult {
        predictions += 1
        try await operation("predict")
        return try ForecastFixture.result(request)
    }

    func release() async {
        releasedWhileBusy = releasedWhileBusy || busy
        releases += 1
        events.continuation.yield("released")
    }

    func finish() {
        continuation?.resume()
        continuation = nil
    }

    private func operation(_ name: String) async throws {
        busy = true
        defer { busy = false }
        if pause == name {
            await withCheckedContinuation {
                continuation = $0
                events.continuation.yield(name)
            }
        } else {
            events.continuation.yield(name)
        }
        if failing { throw ForecastFailure.predictionFailed }
    }
}

actor ForecastPredictorFake: ChronosPredictor {
    nonisolated let identity: ChronosBundleIdentity
    var contexts: [[Float]] = []

    init(identity: ChronosBundleIdentity) { self.identity = identity }

    func predict(_ context: ChronosContext) async throws -> ChronosPrediction {
        contexts.append(context.paddedValues)
        return try ChronosPrediction(quantiles: Array(repeating: Array(repeating: -2, count: 16), count: 21))
    }
}
