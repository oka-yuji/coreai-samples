import ForecastFeature
import Foundation
import Testing

@testable import ForecastRuntime

struct ForecastRepositoryTests {
    @Test("S1 Production admission refuses unknown or oversized bytes before parsing or caching")
    func admission() async throws {
        let root = try ForecastFixture.directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let repository = SalesDataRepository(cacheDirectory: root, fetch: { ForecastFixture.csv() })
        await #expect(throws: ForecastFailure.sourceMismatch) { try await repository.load() }
        #expect(!FileManager.default.fileExists(atPath: root.appending(path: "sales.csv").path))
        #expect(throws: ForecastFailure.responseTooLarge) {
            try SalesSource.validate(Data(repeating: 0, count: SalesSource.byteLimit + 1))
        }
    }

    @Test("S1 A verified cache reopens offline, revalidates every read and retries corruption explicitly")
    func cache() async throws {
        let root = try ForecastFixture.directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let expected = try await ForecastFixture.repository(root).load()
        let offline = SalesDataRepository(
            cacheDirectory: root, fetch: { throw ForecastFailure.network }, admit: SalesDataset.parse)
        #expect(try await offline.load() == expected)
        let file = root.appending(path: "sales.csv")
        try Data("corrupt synthetic cache".utf8).write(to: file)
        await #expect(throws: ForecastFailure.invalidData) { try await offline.load() }
        #expect(try await ForecastFixture.repository(root).load(refresh: true) == expected)
        let unrelated = root.appending(path: "unrelated.txt")
        try Data("keep".utf8).write(to: unrelated)
        _ = try await ForecastFixture.repository(root).load(refresh: true)
        #expect(try Data(contentsOf: unrelated) == Data("keep".utf8))
        let failing = SalesDataRepository(cacheDirectory: root, fetch: { Data("bad".utf8) }, admit: SalesDataset.parse)
        let before = try Data(contentsOf: file)
        await #expect(throws: ForecastFailure.invalidData) { try await failing.load(refresh: true) }
        #expect(try Data(contentsOf: file) == before)
    }
}
