import ChronosFeature
import Foundation
import Testing

@testable import ChronosDemoRuntime

@Suite(.timeLimit(.minutes(1)))
struct ChronosAPICacheTests {
    @Test("A4 A new save removes abandoned staging files without deleting unrelated files")
    func abandonedStaging() async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let abandoned = root.appending(path: UUID().uuidString + ".tmp")
        let unrelated = root.appending(path: "retained.txt")
        try Data("incomplete cache write".utf8).write(to: abandoned)
        try Data("retained".utf8).write(to: unrelated)
        let client = ChronosAPIFixture.client(try ChronosAPITransportFake(), cache: root)
        _ = try await client.window(
            start: ChronosAPIFixture.date("2025-01-03T22:00:00Z"), count: 1, areaIDs: ["230"])
        #expect(!FileManager.default.fileExists(atPath: abandoned.path))
        #expect(try Data(contentsOf: unrelated) == Data("retained".utf8))
        #expect(try windowFiles(root).count == 1)
    }

    @Test("A4 An intact envelope cannot conceal a modified original response")
    func rawHash() async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let fake = try ChronosAPITransportFake()
        let client = ChronosAPIFixture.client(fake, cache: root)
        let date = try ChronosAPIFixture.date("2025-01-03T22:00:00Z")
        _ = try await client.window(start: date, count: 1, areaIDs: ["230"])
        let path = try #require(windowFiles(root).first)
        var envelope = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any])
        let encodedPayload = try #require(envelope["payload"] as? String)
        let payload = try #require(Data(base64Encoded: encodedPayload))
        var window = try #require(JSONSerialization.jsonObject(with: payload) as? [String: Any])
        var responses = try #require(window["responses"] as? [[String: Any]])
        responses[0]["data"] = Data("[] ".utf8).base64EncodedString()
        window["responses"] = responses
        let changed = try JSONSerialization.data(withJSONObject: window)
        envelope["payload"] = changed.base64EncodedString()
        envelope["sha256"] = NYCAPIContract.hash(changed)
        try JSONSerialization.data(withJSONObject: envelope).write(to: path)
        await fake.setStatus(429)
        await #expect(throws: ChronosDemoFailure.rateLimited) {
            try await client.window(start: date, count: 1, areaIDs: ["230"])
        }
    }

    @Test("A4 Cache replays exact bytes and corrupt evidence cannot turn a failed fetch into success")
    func corruption() async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let fake = try ChronosAPITransportFake(rows: ChronosAPIFixture.rows())
        let client = ChronosAPIFixture.client(fake, cache: root)
        let date = try ChronosAPIFixture.date("2025-01-03T22:00:00Z")
        let first = try await client.window(start: date, count: 1, areaIDs: ["230"])
        let calls = await fake.requests.count
        let cached = try await client.window(start: date, count: 1, areaIDs: ["230"])
        #expect(cached == first)
        #expect(cached.responses[0].data == (try ChronosAPIFixture.rows()))
        #expect(await fake.requests.count == calls)
        let path = try #require(windowFiles(root).first)
        try Data("broken".utf8).write(to: path)
        await fake.setStatus(503)
        await #expect(throws: ChronosDemoFailure.network) {
            try await client.window(start: date, count: 1, areaIDs: ["230"])
        }
        await fake.setStatus(200)
        #expect(try await client.window(start: date, count: 1, areaIDs: ["230"]) == first)
    }

    @Test("A4 Cache size is checked before reading and symlinks are never followed", arguments: [false, true])
    func unsafeFile(_ link: Bool) async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let fake = try ChronosAPITransportFake()
        let cache = root.appending(path: "cache")
        let client = ChronosAPIFixture.client(fake, cache: cache)
        let date = try ChronosAPIFixture.date("2025-01-03T22:00:00Z")
        _ = try await client.window(start: date, count: 1, areaIDs: ["230"])
        let file = try #require(windowFiles(cache).first)
        if link {
            try FileManager.default.removeItem(at: file)
            let sentinel = root.appending(path: "sentinel")
            try Data("protected".utf8).write(to: sentinel)
            try FileManager.default.createSymbolicLink(at: file, withDestinationURL: sentinel)
        } else {
            let handle = try FileHandle(forWritingTo: file)
            try handle.truncate(atOffset: UInt64(NYCWindowCache.fileLimit + 1))
            try handle.close()
        }
        await fake.setStatus(429)
        await #expect(throws: ChronosDemoFailure.rateLimited) {
            try await client.window(start: date, count: 1, areaIDs: ["230"])
        }
        if link { #expect(try Data(contentsOf: root.appending(path: "sentinel")) == Data("protected".utf8)) }
    }

    @Test("A4 Cache pruning never keeps more than thirty-two windows")
    func pruning() async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let client = ChronosAPIFixture.client(try ChronosAPITransportFake(), cache: root, capacity: 50)
        let date = try ChronosAPIFixture.date("2025-01-03T22:00:00Z")
        for index in 0..<34 {
            _ = try await client.window(
                start: date.addingTimeInterval(Double(index * 1800)), count: 1, areaIDs: ["230"])
        }
        #expect(try windowFiles(root).count == 32)
    }

    private func windowFiles(_ directory: URL) throws -> [URL] {
        try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }
    }
}
