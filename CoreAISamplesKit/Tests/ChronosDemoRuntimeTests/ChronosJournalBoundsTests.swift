import ChronosFeature
import Foundation
import Testing

@testable import ChronosDemoRuntime

struct ChronosJournalBoundsTests {
    @Test("J1 Decoded window counts cannot overflow before validation", arguments: [Int.min, Int.max])
    func decodedCount(_ count: Int) throws {
        let window = try APIWindow(
            start: ChronosAPIFixture.date("2025-01-24T22:00:00Z"), count: 1,
            series: [CountSeries(areaID: "230", values: [nil])], availability: [.unpublished], datasets: [],
            fetchedAt: ChronosAPIFixture.now, catalog: ChronosAPIFixture.coverage().evidence,
            metadata: [], responses: [])
        var object = try #require(JSONSerialization.jsonObject(with: NYCAPIContract.encode(window)) as? [String: Any])
        object["count"] = count
        let decoded = try NYCAPIContract.decode(APIWindow.self, JSONSerialization.data(withJSONObject: object))
        #expect(decoded.end.timeIntervalSince1970.isFinite)
        #expect(throws: ChronosDemoFailure.invalidData) { try NYCWindowValidation.validate(decoded) }
        #expect(throws: ChronosDemoFailure.invalidData) { try decoded.requireVerifiedHistory() }
    }

    @Test("J1 Invalid acquisition dates are rejected before catalog calendar queries", arguments: [1e20, 1e100])
    func acquisitionDate(_ seconds: Double) throws {
        let source = try ChronosAPIFixture.coverage().evidence
        let evidence = APIEvidence(
            url: source.url, status: source.status, acquiredAt: Date(timeIntervalSince1970: seconds),
            data: source.data, sha256: source.sha256)
        #expect(throws: ChronosDemoFailure.invalidData) {
            _ = try NYCCatalogValidation.read(evidence, now: evidence.acquiredAt)
        }
    }

    @Test("J1 Rehashed API records reject extreme dates before chunking and integer conversion")
    func rehashedRecord() async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let profile = try ChronosAPIFixture.syntheticProfile(areas: NYCAPIContract.zoneIDs)
        let transport = try ChronosAPITransportFake()
        let api = NYCAPIClient(
            transport: { try await transport.receive($0) }, now: { ChronosAPIFixture.now }, retryDelay: {},
            verification: profile)
        let origin = try ChronosAPIFixture.date("2025-01-24T22:00:00Z")
        let history = try await api.window(
            start: origin.addingTimeInterval(-1008 * 1800), count: 1008, areaIDs: NYCAPIContract.zoneIDs)
        let snapshot = try ForecastInputSnapshot(history: history, actuals: nil, step: 0)
        let record = try ChronosForecastRecord(
            id: UUID(), savedAt: ChronosAPIFixture.now, batch: await ChronosJournalFixture.batch(snapshot: snapshot),
            device: ChronosJournalFixture.device)
        let directory = root.appending(path: "journal")
        let files = ChronosJournalFiles(directory: directory, verification: profile)
        try await files.create(record, permit: ChronosPublicationPermit())
        let path = directory.appending(path: "\(record.id.uuidString)/prediction.json")
        try ChronosJournalFixture.rewrite(path) { value in
            var input = try #require(value["input"] as? [String: Any])
            var capture = try #require(input["snapshot"] as? [String: Any])
            var api = try #require(capture["api"] as? [String: Any])
            var history = try #require(api["history"] as? [String: Any])
            let hostile = Date(timeIntervalSince1970: 1e100).timeIntervalSinceReferenceDate
            capture["start"] = hostile
            capture["origin"] = hostile
            api["start"] = hostile
            api["origin"] = hostile
            history["start"] = hostile
            history["datasets"] = []
            history["metadata"] = []
            history["responses"] = []
            history["availability"] = Array(repeating: "unpublished", count: 1008)
            history["series"] = NYCAPIContract.zoneIDs.map {
                ["areaID": $0, "values": Array(repeating: NSNull(), count: 1008)] as [String: Any]
            }
            api["history"] = history
            capture["api"] = api
            let typed = try ChronosJournalCodec.decode(
                ChronosInputCapture.self, JSONSerialization.data(withJSONObject: capture))
            let bytes = try NYCAPIContract.encode(typed)
            input["snapshot"] = capture
            input["bytes"] = bytes.base64EncodedString()
            input["sha256"] = NYCAPIContract.hash(bytes)
            value["input"] = input
            value["inputSHA256"] = try NYCAPIContract.hash(NYCAPIContract.encode(typed.api))
        }
        await #expect(throws: ChronosDemoFailure.invalidData) { try await files.record(record.id) }
    }

    @Test(
        "J1 Finite dates beyond the four-digit calendar domain cannot construct an API window",
        arguments: [1e20, 1e100])
    func windowDates(_ seconds: Double) throws {
        #expect(throws: ChronosDemoFailure.invalidData) {
            _ = try APIWindow(
                start: Date(timeIntervalSince1970: seconds), count: 1,
                series: [CountSeries(areaID: "230", values: [nil])], availability: [.unpublished], datasets: [],
                fetchedAt: ChronosAPIFixture.now, catalog: ChronosAPIFixture.coverage().evidence,
                metadata: [], responses: [])
        }
    }

    @Test("J1 A finite but unrepresentable saved time is rejected before publication", arguments: [1e20, 1e100])
    func savedTime(_ seconds: Double) async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let record = try ChronosForecastRecord(
            id: UUID(), savedAt: Date(timeIntervalSince1970: seconds), batch: await ChronosJournalFixture.batch(),
            device: ChronosJournalFixture.device)
        let files = ChronosJournalFiles(directory: root.appending(path: "journal"))
        await #expect(throws: ChronosDemoFailure.invalidData) {
            try await files.create(record, permit: ChronosPublicationPermit())
        }
    }
}
