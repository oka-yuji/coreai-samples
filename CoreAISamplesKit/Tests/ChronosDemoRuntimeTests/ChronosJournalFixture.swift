import ChronosCore
import ChronosFeature
import Foundation
import Testing

@testable import ChronosDemoRuntime

enum ChronosJournalFixture {
    static var identity: ChronosBundleIdentity {
        ChronosBundleIdentity(
            source: ChronosSource(modelId: ChronosContract.source, revision: String(repeating: "c", count: 40)),
            manifest: ChronosFileIdentity(bytes: 1824, sha256: String(repeating: "a", count: 64)),
            files: Dictionary(
                uniqueKeysWithValues: [
                    "LICENSE", "NOTICE", "SOURCE_README.md", "reference.json", "verification.json",
                    "Model.aimodel/main.hash", "Model.aimodel/main.mlirb", "Model.aimodel/metadata.json",
                ].map { ($0, ChronosFileIdentity(bytes: 123, sha256: String(repeating: "b", count: 64))) }))
    }

    static func batch(step: Int = 0, snapshot: ForecastInputSnapshot? = nil) async throws -> ChronosForecastBatch {
        let repository = try ChronosRepository.bundled()
        var catalog = try await repository.load()
        if let snapshot { catalog = try catalog.replacingWithAPI(snapshot.history, actuals: snapshot.actuals) }
        let request = try #require(catalog.request(scenarioID: catalog.scenarios[0].id, step: step, mode: .live))
        let captured = try await repository.capture(request)
        return try ChronosForecastBatch(
            scenarioID: request.scenarioID, originStep: step, mode: .live,
            areas: request.series.enumerated().map { index, series in
                try ChronosAreaForecast(
                    areaID: series.areaID,
                    raw: ChronosPrediction(
                        quantiles: (0..<21).map { quantile in
                            (0..<16).map { Float(index + quantile + $0) / 4 - 3 }
                        }), identity: identity)
            }, processingSeconds: 0.125, identity: identity, inputSnapshot: request.snapshot,
            inputSHA256: request.snapshot.map { try NYCAPIContract.hash(NYCAPIContract.encode($0)) },
            capturedInput: captured)
    }

    static func record(id: UUID = UUID(), step: Int = 0) async throws -> ChronosForecastRecord {
        try ChronosForecastRecord(
            id: id, savedAt: ChronosAPIFixture.now, batch: await batch(step: step), device: device)
    }

    static var device: ChronosRecordDevice {
        ChronosRecordDevice(hardwareModel: "Mac17,6", operatingSystem: "macOS", operatingSystemVersion: "27.2.1")
    }

    static func actuals(origin: Date, profile: NYCVerificationProfile, revision: String? = nil) async throws
        -> APIWindow
    {
        let dataset = APIDataset(
            year: 2025, id: "irx2-vgm4", revision: revision ?? ChronosAPIFixture.dataset().revision)
        let transport = try ChronosAPITransportFake(
            catalog: ChronosAPIFixture.catalog([ChronosAPIFixture.item(dataset)]))
        let api = NYCAPIClient(
            transport: { try await transport.receive($0) }, now: { ChronosAPIFixture.now }, retryDelay: {},
            verification: profile)
        return try await api.window(start: origin, count: 4, areaIDs: NYCAPIContract.zoneIDs, refresh: true)
    }

    static func rewrite(_ url: URL, transform: (inout [String: Any]) throws -> Void) throws {
        var envelope = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let encoded = try #require(envelope["payload"] as? String)
        let bytes = try #require(Data(base64Encoded: encoded))
        var value = try #require(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
        try transform(&value)
        let changed = try JSONSerialization.data(
            withJSONObject: value, options: [.sortedKeys, .withoutEscapingSlashes])
        envelope["payload"] = changed.base64EncodedString()
        envelope["sha256"] = NYCAPIContract.hash(changed)
        try JSONSerialization.data(withJSONObject: envelope, options: [.sortedKeys, .withoutEscapingSlashes]).write(
            to: url)
    }
}
