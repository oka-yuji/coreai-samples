import ChronosFeature
import Darwin
import Foundation
import Testing

@testable import ChronosDemoRuntime

struct ChronosJournalBoundaryTests {
    @Test(
        "J2 Controlled directories and files reject links nonregular entries and oversized envelopes",
        arguments: ["root", "record", "prediction", "checks", "check", "fifo", "oversized"])
    func boundaries(kind: String) async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appending(path: "journal")
        let profile = try ChronosAPIFixture.syntheticProfile(areas: NYCAPIContract.zoneIDs)
        let files = ChronosJournalFiles(directory: directory, verification: profile)
        let record = try await ChronosJournalFixture.record()
        try await files.create(record, permit: ChronosPublicationPermit())
        let actuals = try await ChronosJournalFixture.actuals(origin: record.input.snapshot.origin, profile: profile)
        let check = try ChronosActualCheck(
            id: UUID(), record: record, checkedAt: ChronosAPIFixture.now, actuals: actuals)
        try await files.append(check, permit: ChronosPublicationPermit())
        let folder = directory.appending(path: record.id.uuidString)
        let path: URL
        switch kind {
        case "root": path = directory
        case "record": path = folder
        case "checks": path = folder.appending(path: "checks")
        case "check": path = folder.appending(path: "checks/\(check.id.uuidString).json")
        default: path = folder.appending(path: "prediction.json")
        }
        let outside = root.appending(path: "outside")
        try FileManager.default.moveItem(at: path, to: outside)
        if kind == "fifo" {
            try #require(mkfifo(path.path, 0o600) == 0)
        } else if kind == "oversized" {
            try Data().write(to: path)
            let file = try FileHandle(forWritingTo: path)
            try file.truncate(atOffset: UInt64(ChronosJournalFiles.fileLimit + 1))
            try file.close()
        } else {
            try FileManager.default.createSymbolicLink(at: path, withDestinationURL: outside)
        }
        await #expect(throws: (any Error).self) { try await files.detail(record.id) }
        #expect(FileManager.default.fileExists(atPath: outside.path))
    }

    @Test("J2 Ancestor aliases outside the controlled root remain usable")
    func parentAlias() async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let real = root.appending(path: "real")
        try FileManager.default.createDirectory(at: real, withIntermediateDirectories: true)
        let alias = root.appending(path: "alias")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: real)
        let files = ChronosJournalFiles(directory: alias.appending(path: "journal"))
        let record = try await ChronosJournalFixture.record()
        try await files.create(record, permit: ChronosPublicationPermit())
        #expect(try await files.record(record.id) == record)
    }

    @Test("J1 API capture preserves nil masks and source bytes and rejects rehashed response corruption")
    func apiEvidence() async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let dataset = ChronosAPIFixture.dataset(2024)
        let fake = try ChronosAPITransportFake(
            catalog: ChronosAPIFixture.catalog([ChronosAPIFixture.item(dataset)]),
            metadata: [dataset.id: ChronosAPIFixture.metadata(dataset)])
        let profile = try ChronosAPIFixture.syntheticProfile(areas: NYCAPIContract.zoneIDs)
        let api = NYCAPIClient(
            transport: { try await fake.receive($0) }, now: { ChronosAPIFixture.now }, verification: profile)
        let history = try await api.window(
            start: ChronosAPIFixture.date("2024-11-01T04:00:00Z"), count: 1008, areaIDs: NYCAPIContract.zoneIDs)
        let snapshot = try ForecastInputSnapshot(history: history, actuals: nil, step: 0)
        let batch = try await ChronosJournalFixture.batch(snapshot: snapshot)
        let record = try ChronosForecastRecord(
            id: UUID(), savedAt: ChronosAPIFixture.now, batch: batch, device: ChronosJournalFixture.device)
        let files = ChronosJournalFiles(directory: root.appending(path: "journal"), verification: profile)
        try await files.create(record, permit: ChronosPublicationPermit())
        let copy = try await files.record(record.id)
        #expect(copy.input.snapshot.api == snapshot)
        #expect(copy.input.snapshot.series.allSatisfy { $0.values.filter { $0 == nil }.count == 4 })
        #expect(copy.input.snapshot.api?.history.responses.map(\.data) == history.responses.map(\.data))
        let path = root.appending(path: "journal/\(record.id.uuidString)/prediction.json")
        try ChronosJournalFixture.rewrite(path) { value in
            var evidence = try #require(value["input"] as? [String: Any])
            var capture = try #require(evidence["snapshot"] as? [String: Any])
            var input = try #require(capture["api"] as? [String: Any])
            var history = try #require(input["history"] as? [String: Any])
            var responses = try #require(history["responses"] as? [[String: Any]])
            let falseBytes = Data("[{\"zone\":\"230\"}]".utf8)
            responses[0]["data"] = falseBytes.base64EncodedString()
            responses[0]["sha256"] = NYCAPIContract.hash(falseBytes)
            history["responses"] = responses
            input["history"] = history
            capture["api"] = input
            let bytes = try JSONSerialization.data(
                withJSONObject: capture, options: [.sortedKeys, .withoutEscapingSlashes])
            evidence["snapshot"] = capture
            evidence["bytes"] = bytes.base64EncodedString()
            evidence["sha256"] = NYCAPIContract.hash(bytes)
            value["input"] = evidence
        }
        await #expect(throws: (any Error).self) { try await files.record(record.id) }
    }

    @Test(
        "J2 Check write failures and revocation after writing never publish partial actuals",
        arguments: ["revoked", "disk", "partial"])
    func cancelledCheck(kind: String) async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let profile = try ChronosAPIFixture.syntheticProfile(areas: NYCAPIContract.zoneIDs)
        let directory = root.appending(path: "journal")
        let files = ChronosJournalFiles(directory: directory, verification: profile)
        let record = try await ChronosJournalFixture.record()
        try await files.create(record, permit: ChronosPublicationPermit())
        let blocked = ChronosJournalFiles(
            directory: directory, verification: profile,
            write: { data, file in
                try file.write(contentsOf: kind == "partial" ? Data(data.prefix(16)) : data)
                if kind == "disk" { throw POSIXError(.ENOSPC) }
            }, beforePublish: { if kind == "revoked" { $0.revoke() } })
        let check = try ChronosActualCheck(
            id: UUID(), record: record, checkedAt: ChronosAPIFixture.now,
            actuals: await ChronosJournalFixture.actuals(origin: record.input.snapshot.origin, profile: profile))
        await #expect(throws: (any Error).self) {
            try await blocked.append(check, permit: ChronosPublicationPermit())
        }
        let detail = try await files.detail(record.id)
        #expect(detail.checks.isEmpty && detail.record == record)
    }
}
