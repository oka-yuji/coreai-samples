import ChronosFeature
import Foundation
import Testing

@testable import ChronosDemoRuntime

struct ChronosJournalDateTests {
    @Test("J1 Earlier millisecond journal envelopes are rejected before reading their dates")
    func earlierEnvelope() async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appending(path: "journal")
        let files = ChronosJournalFiles(directory: directory)
        let record = try await ChronosJournalFixture.record()
        try await files.create(record, permit: ChronosPublicationPermit())
        let path = directory.appending(path: "\(record.id.uuidString)/prediction.json")
        var envelope = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any])
        envelope["schema"] = "chronos-journal-envelope/1"
        try JSONSerialization.data(withJSONObject: envelope).write(to: path)
        await #expect(throws: ChronosDemoFailure.invalidData) { try await files.record(record.id) }
    }

    @Test(
        "J1 Native reference-clock timestamps survive record and check persistence exactly",
        arguments: [0.00000011920928955078125, 0.1234568, 0.5])
    func referenceClock(_ fraction: Double) async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let captured = Date(timeIntervalSinceReferenceDate: 815_000_000 + fraction)
        let profile = try ChronosAPIFixture.syntheticProfile(areas: NYCAPIContract.zoneIDs)
        let transport = try ChronosAPITransportFake()
        let client = NYCAPIClient(
            transport: { try await transport.receive($0) }, now: { captured }, retryDelay: {},
            verification: profile)
        let origin = try ChronosAPIFixture.date("2025-01-24T22:00:00Z")
        let history = try await client.window(
            start: origin.addingTimeInterval(-1008 * 1800), count: 1008, areaIDs: NYCAPIContract.zoneIDs)
        let snapshot = try ForecastInputSnapshot(history: history, actuals: nil, step: 0)
        let batch = try await ChronosJournalFixture.batch(snapshot: snapshot)
        let record = try ChronosForecastRecord(
            id: UUID(), savedAt: captured.addingTimeInterval(1), batch: batch, device: ChronosJournalFixture.device)
        let files = ChronosJournalFiles(directory: root.appending(path: "journal"), verification: profile)
        try await files.create(record, permit: ChronosPublicationPermit())
        let restored = try await files.record(record.id)
        #expect(restored.input.snapshot.api == snapshot)
        #expect(
            restored.savedAt.timeIntervalSinceReferenceDate.bitPattern
                == record.savedAt.timeIntervalSinceReferenceDate.bitPattern)
        #expect(restored == record)
        #expect(restored.input.bytes == record.input.bytes)
        #expect(restored.inputSHA256 == record.inputSHA256)
        let actuals = try await client.window(start: origin, count: 4, areaIDs: NYCAPIContract.zoneIDs)
        let check = try ChronosActualCheck(
            id: UUID(), record: record, checkedAt: captured.addingTimeInterval(2), actuals: actuals)
        try await files.append(check, permit: ChronosPublicationPermit())
        let detail = try await files.detail(record.id)
        #expect(detail.record == record)
        #expect(detail.checks == [check])
    }
}
