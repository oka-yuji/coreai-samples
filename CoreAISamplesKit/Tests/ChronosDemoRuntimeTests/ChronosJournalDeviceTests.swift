import ChronosFeature
import Foundation
import Testing

@testable import ChronosDemoRuntime

struct ChronosJournalDeviceTests {
    @Test(
        "J1 Saved records retain model identifiers from each Mac family and mobile devices",
        arguments: [
            "Mac17,6", "MacBookPro18,3", "MacBookAir10,1", "Macmini9,1", "iMac21,1", "iPhone18,1", "iPad16,3",
            "unknown",
        ])
    func models(_ model: String) async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let files = ChronosJournalFiles(directory: root.appending(path: "journal"))
        let device = ChronosRecordDevice(
            hardwareModel: model, operatingSystem: "macOS", operatingSystemVersion: "27.2.1")
        let record = try ChronosForecastRecord(
            id: UUID(), savedAt: ChronosAPIFixture.now, batch: await ChronosJournalFixture.batch(), device: device)
        try await files.create(record, permit: ChronosPublicationPermit())
        #expect(try await files.record(record.id).device == device)
    }

    @Test(
        "J1 Hardware identifiers reject names paths trailing data and oversized values",
        arguments: ["personal-name", "../Mac17,6", "Mac17,6\n", "Mac17,6 suffix", String(repeating: "Mac", count: 50)])
    func invalidModels(_ model: String) async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let files = ChronosJournalFiles(directory: root.appending(path: "journal"))
        let device = ChronosRecordDevice(
            hardwareModel: model, operatingSystem: "macOS", operatingSystemVersion: "27.2.1")
        let record = try ChronosForecastRecord(
            id: UUID(), savedAt: ChronosAPIFixture.now, batch: await ChronosJournalFixture.batch(), device: device)
        await #expect(throws: ChronosDemoFailure.invalidData) {
            try await files.create(record, permit: ChronosPublicationPermit())
        }
    }
}
