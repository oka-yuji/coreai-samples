import ChronosFeature
import Darwin
import Dispatch
import Foundation
import Synchronization
import Testing

@testable import ChronosDemoRuntime

@Suite(.timeLimit(.minutes(1)))
struct ChronosJournalFilesTests {
    @Test(
        "J2 Active staging stays locked across owners and task cancellation is checked after the write",
        arguments: [false, true])
    func stagedOwners(cancel: Bool) async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appending(path: "journal")
        let started = AsyncStream<Void>.makeStream()
        let release = DispatchSemaphore(value: 0)
        defer {
            release.signal()
            started.continuation.finish()
        }
        let first = ChronosJournalFiles(
            directory: directory,
            beforePublish: { _ in
                started.continuation.yield(())
                release.wait()
            })
        let second = ChronosJournalFiles(directory: directory)
        let record = try await ChronosJournalFixture.record()
        let other = try await ChronosJournalFixture.record()
        let writing = Task { try await first.create(record, permit: ChronosPublicationPermit()) }
        var events = started.stream.makeAsyncIterator()
        _ = await events.next()
        let descriptor = open(directory.appending(path: ".lock").path, O_RDWR | O_NOFOLLOW)
        try #require(descriptor >= 0)
        defer { close(descriptor) }
        let probe = flock(descriptor, LOCK_EX | LOCK_NB)
        #expect(probe == -1 && errno == EWOULDBLOCK)
        if probe == 0 { flock(descriptor, LOCK_UN) }
        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        #expect(names.filter { $0.hasPrefix(".pending-") }.count == 1)
        async let competing: Void = second.create(other, permit: ChronosPublicationPermit())
        if cancel { writing.cancel() }
        release.signal()
        if cancel {
            await #expect(throws: CancellationError.self) { try await writing.value }
        } else {
            try await writing.value
        }
        try await competing
        let records = try await second.list()
        #expect(records.count == (cancel ? 1 : 2))
        #expect(try await second.record(other.id) == other)
        if !cancel { #expect(try await second.record(record.id) == record) }
    }

    @Test("J1 Native records retain every raw output, actual identity and original input through separate corrections")
    func immutable() async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let profile = try ChronosAPIFixture.syntheticProfile(areas: NYCAPIContract.zoneIDs)
        let files = ChronosJournalFiles(directory: root.appending(path: "journal"), verification: profile)
        let record = try await ChronosJournalFixture.record(step: 1)
        try await files.create(record, permit: ChronosPublicationPermit())
        let path = root.appending(path: "journal/\(record.id.uuidString)/prediction.json")
        let before = try Data(contentsOf: path)
        #expect(record.areas.reduce(0) { $0 + $1.raw.quantiles.flatMap { $0 }.count } == 6720)
        #expect(record.areas[0].raw.quantiles[0][0] == -3)
        #expect(record.identity == ChronosJournalFixture.identity)
        #expect(record.input.snapshot.series.reduce(0) { $0 + $1.values.count } == 20_160)
        #expect(try await files.record(record.id) == record)
        let origin = record.input.snapshot.origin
        let first = try ChronosActualCheck(
            id: UUID(), record: record, checkedAt: ChronosAPIFixture.now,
            actuals: await ChronosJournalFixture.actuals(origin: origin, profile: profile))
        try await files.append(first, permit: ChronosPublicationPermit())
        let checkPath = path.deletingLastPathComponent().appending(path: "checks/\(first.id.uuidString).json")
        let firstBytes = try Data(contentsOf: checkPath)
        let correction = try ChronosActualCheck(
            id: UUID(), record: record, checkedAt: ChronosAPIFixture.now.addingTimeInterval(1),
            actuals: await ChronosJournalFixture.actuals(
                origin: origin, profile: profile, revision: "2026-10-02T16:57:47.000Z"))
        try await files.append(correction, permit: ChronosPublicationPermit())
        let reopened = ChronosJournalFiles(directory: root.appending(path: "journal"), verification: profile)
        let detail = try await reopened.detail(record.id)
        #expect(detail.checks == [correction, first])
        #expect(correction.status == .unverified && correction.metrics.comparedCount == 0)
        #expect(first.metrics.comparedCount == 80 && first.metrics.weightedAbsolutePercentageError == nil)
        #expect(try Data(contentsOf: path) == before)
        #expect(try Data(contentsOf: checkPath) == firstBytes)
        #expect(detail.record.input.bytes == record.input.bytes)
        try ChronosJournalFixture.rewrite(checkPath) { value in
            var metrics = try #require(value["metrics"] as? [String: Any])
            metrics["comparedCount"] = 79
            value["metrics"] = metrics
        }
        await #expect(throws: ChronosDemoFailure.invalidData) { try await files.detail(record.id) }
        #expect(try Data(contentsOf: path) == before)
    }

    @Test(
        "J1 Rehashed envelopes cannot conceal invalid schema IDs shapes identities contexts or source bytes",
        arguments: [
            "schema", "id", "shape", "model", "quantiles", "context", "source", "inputHash", "origin", "device",
        ])
    func forged(kind: String) async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let files = ChronosJournalFiles(directory: root.appending(path: "journal"))
        let record = try await ChronosJournalFixture.record()
        try await files.create(record, permit: ChronosPublicationPermit())
        let path = root.appending(path: "journal/\(record.id.uuidString)/prediction.json")
        try ChronosJournalFixture.rewrite(path) { value in
            switch kind {
            case "schema": value["schema"] = "old-private-app/1"
            case "id": value["id"] = UUID().uuidString
            case "shape", "model":
                var areas = try #require(value["areas"] as? [[String: Any]])
                if kind == "shape" {
                    areas[19]["raw"] = ["quantiles": [[1, 2, 3]]]
                } else {
                    var identity = try #require(areas[19]["identity"] as? [String: Any])
                    identity["manifest"] = ["bytes": 123, "sha256": String(repeating: "d", count: 64)]
                    areas[19]["identity"] = identity
                }
                value["areas"] = areas
            case "quantiles": value["quantileLevels"] = [0.1, 0.5, 0.9]
            case "device": value["device"] = ["hardwareModel": "personal-name"]
            default:
                var evidence = try #require(value["input"] as? [String: Any])
                var input = try #require(evidence["snapshot"] as? [String: Any])
                if kind == "origin" { input["origin"] = 0 }
                if kind == "context" {
                    var series = try #require(input["series"] as? [[String: Any]])
                    var values = try #require(series[0]["values"] as? [Any])
                    values[0] = 999
                    series[0]["values"] = values
                    input["series"] = series
                }
                if kind == "source" {
                    var sources = try #require(input["fixedFiles"] as? [String: Any])
                    sources["manifest.json"] = Data("{}".utf8).base64EncodedString()
                    input["fixedFiles"] = sources
                }
                let bytes = try JSONSerialization.data(
                    withJSONObject: input, options: [.sortedKeys, .withoutEscapingSlashes])
                evidence["snapshot"] = input
                evidence["bytes"] = bytes.base64EncodedString()
                evidence["sha256"] =
                    kind == "inputHash" ? String(repeating: "0", count: 64) : NYCAPIContract.hash(bytes)
                value["input"] = evidence
            }
        }
        await #expect(throws: (any Error).self) { try await files.record(record.id) }
        let list = try await files.list()
        #expect(list.count == 1 && list[0].id == record.id && list[0].savedAt == nil)
    }

    @Test(
        "J2 Full disk partial writes and revocation at publication leave no successful record",
        arguments: ["disk", "partial", "revoked"])
    func failedPublication(kind: String) async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let files = ChronosJournalFiles(
            directory: root.appending(path: "journal"),
            write: { data, file in
                if kind == "disk" {
                    try file.write(contentsOf: data.prefix(16))
                    throw POSIXError(.ENOSPC)
                }
                try file.write(contentsOf: kind == "partial" ? Data(data.prefix(16)) : data)
            }, beforePublish: { if kind == "revoked" { $0.revoke() } })
        let record = try await ChronosJournalFixture.record()
        await #expect(throws: (any Error).self) { try await files.create(record, permit: ChronosPublicationPermit()) }
        #expect(try await files.list().isEmpty)
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.appending(path: "journal").path) == [".lock"])
    }

    @Test("J2 Multiple file owners never overwrite UUIDs or reclaim each other's active staging")
    func owners() async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appending(path: "journal")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let abandoned = directory.appending(path: ".pending-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: abandoned, withIntermediateDirectories: true)
        try Data("partial".utf8).write(to: abandoned.appending(path: "prediction.json"))
        let profile = try ChronosAPIFixture.syntheticProfile(areas: NYCAPIContract.zoneIDs)
        let first = ChronosJournalFiles(directory: directory, verification: profile)
        let second = ChronosJournalFiles(directory: directory, verification: profile)
        let record = try await ChronosJournalFixture.record()
        let other = try await ChronosJournalFixture.record()
        async let one: Void = first.create(record, permit: ChronosPublicationPermit())
        async let two: Void = second.create(other, permit: ChronosPublicationPermit())
        _ = try await (one, two)
        #expect(try await first.list().count == 2)
        #expect(!FileManager.default.fileExists(atPath: abandoned.path))
        await #expect(throws: (any Error).self) { try await second.create(record, permit: ChronosPublicationPermit()) }
        let actuals = try await ChronosJournalFixture.actuals(origin: record.input.snapshot.origin, profile: profile)
        let id = UUID()
        let check = try ChronosActualCheck(id: id, record: record, checkedAt: ChronosAPIFixture.now, actuals: actuals)
        let contender = try ChronosActualCheck(
            id: id, record: record, checkedAt: ChronosAPIFixture.now.addingTimeInterval(2), actuals: actuals)
        let successes = await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                do {
                    try await first.append(check, permit: ChronosPublicationPermit())
                    return true
                } catch { return false }
            }
            group.addTask {
                do {
                    try await second.append(contender, permit: ChronosPublicationPermit())
                    return true
                } catch { return false }
            }
            var count = 0
            for await success in group where success { count += 1 }
            return count
        }
        #expect(successes == 1)
        let detail = try await second.detail(record.id)
        #expect(detail.record == record && detail.checks.count == 1)
        #expect(detail.checks.first == check || detail.checks.first == contender)
    }
}
