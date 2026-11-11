import ChronosFeature
import Foundation
import Testing

@testable import ChronosDemoRuntime

struct ChronosAPIReplayTests {
    @Test(
        "A4 Offline replay of original NYC responses matches all 20,160 independently aggregated inputs",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_CHRONOS2_API_FIXTURES"] != nil))
    func realResponseReplay() async throws {
        let root = URL(filePath: try #require(ProcessInfo.processInfo.environment["COREAI_CHRONOS2_API_FIXTURES"]))
        let replay = try ChronosAPIReplay(root: root)
        let client = NYCAPIClient(
            transport: { try await replay.receive($0) }, now: { ChronosAPIFixture.now }, retryDelay: {})
        let origin = try ChronosAPIFixture.date("2025-01-24T22:00:00Z")
        let history = try await client.window(
            start: origin.addingTimeInterval(-1008 * 1800), count: 1008, areaIDs: NYCAPIContract.zoneIDs)
        let expected = try JSONDecoder().decode(
            ChronosAPIInput.self, from: Data(contentsOf: root.appending(path: "api-fixed-history/input.json")))
        #expect(history.series == expected.series.sorted { $0.areaID < $1.areaID })
        #expect(history.series.reduce(0) { $0 + $1.values.count } == 20_160)
        #expect(
            history.responses.map(\.sha256) == [
                "18910db9bfda8d0cf17b75ef510686d66cb1541d444f44916424628af6b72110",
                "8fabf054892995109be5c9aab54b281b84861b014b011b1b741eec8984d4c897",
                "f4c390010c072478b7874898a9d20855a1f362a1ffb41e403019994b877c7f26",
            ])
        let snapshot = try ForecastInputSnapshot(history: history, actuals: nil, step: 0)
        let hash = try await client.validate(snapshot)
        let encoded = try NYCAPIContract.encode(snapshot)
        let decoded = try NYCAPIContract.decode(ForecastInputSnapshot.self, encoded)
        #expect(try await client.validate(decoded) == hash)
        #expect(snapshot.origin == origin && snapshot.actuals == nil)
        let actuals = try await client.window(start: origin, count: 4, areaIDs: NYCAPIContract.zoneIDs)
        let truth = try JSONDecoder().decode(
            ChronosAPIInput.self, from: Data(contentsOf: root.appending(path: "api-fixed-actuals/actuals.json")))
        #expect(actuals.series == truth.series.sorted { $0.areaID < $1.areaID })
        #expect(snapshot.actuals == nil && snapshot.series == history.series)
        let advanced = try ForecastInputSnapshot(history: history, actuals: actuals, step: 1)
        for row in advanced.series {
            let prior = try #require(history.series.first { $0.areaID == row.areaID })
            let observed = try #require(actuals.series.first { $0.areaID == row.areaID })
            #expect(row.values == Array(prior.values.dropFirst()) + observed.values.prefix(1))
        }
    }
}

struct ChronosAPIInput: Decodable {
    let origin: String
    let series: [CountSeries]
}

actor ChronosAPIReplay {
    private let catalog: Data
    private let metadata: Data
    private let responses: [(String, String, Data)]

    init(root: URL) throws {
        catalog = try Data(contentsOf: root.appending(path: "api-baseline/catalog.json"))
        metadata = try Data(contentsOf: root.appending(path: "api-baseline/metadata-2025.json"))
        try #require(NYCAPIContract.hash(catalog) == "a20275e84796d07e7bc72609a4b5eb061fa451464795db9071f804b661b6b8ac")
        try #require(
            NYCAPIContract.hash(metadata) == "b4f065491b602eae169c63179c818cbef40a0599177ecd551d6a2739fb5b76be")
        var values: [(String, String, Data)] = []
        for directory in ["api-fixed-history", "api-fixed-actuals"] {
            let evidence = try JSONDecoder().decode(
                [Acquisition].self, from: Data(contentsOf: root.appending(path: directory + "/acquisition.json")))
            for (index, item) in evidence.enumerated() {
                let data = try Data(contentsOf: root.appending(path: directory + "/response-\(index).json"))
                try #require(NYCAPIContract.hash(data) == item.sha256)
                values.append((item.lower, item.upper, data))
            }
        }
        responses = values
    }

    func receive(_ request: URLRequest) throws -> NYCAPIResponse {
        let url = try #require(request.url)
        if url.host == "api.us.socrata.com" { return NYCAPIResponse(data: catalog, status: 200) }
        if url.path == "/api/views/irx2-vgm4.json" { return NYCAPIResponse(data: metadata, status: 200) }
        let query = try #require(
            URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?.first { $0.name == "$where" }?.value)
        let data = try #require(responses.first { query.contains($0.0) && query.contains($0.1) }?.2)
        return NYCAPIResponse(data: data, status: 200)
    }

    private struct Acquisition: Decodable {
        let lower: String
        let upper: String
        let sha256: String
    }
}
