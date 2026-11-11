import ChronosFeature
import Foundation
import Testing

@testable import ChronosDemoRuntime

struct ChronosAPIReplayTests {
    @Test(
        "A4 Fractional acquisition times preserve snapshot hashes through JSON round trips",
        arguments: [0.00001, 0.1234567, 0.9999999])
    func fractionalTime(_ fraction: Double) async throws {
        let now = Date(timeIntervalSince1970: 1_794_441_600 + fraction)
        let fake = try ChronosAPITransportFake()
        let client = NYCAPIClient(transport: { try await fake.receive($0) }, now: { now }, retryDelay: {})
        let history = try await client.window(
            start: ChronosAPIFixture.date("2025-01-03T22:00:00Z"), count: 1008, areaIDs: ["230"])
        let snapshot = try ForecastInputSnapshot(history: history, actuals: nil, step: 0)
        let hash = try await client.validate(snapshot)
        let bytes = try NYCAPIContract.encode(snapshot)
        let decoded = try NYCAPIContract.decode(ForecastInputSnapshot.self, bytes)
        #expect(try NYCAPIContract.encode(decoded) == bytes)
        #expect(try await client.validate(decoded) == hash)
    }

    @Test(
        "A4 A captured live snapshot retains its prediction-time hash after decoding",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_CHRONOS2_API_SNAPSHOT"] != nil))
    func capturedSnapshot() throws {
        let path = URL(filePath: try #require(ProcessInfo.processInfo.environment["COREAI_CHRONOS2_API_SNAPSHOT"]))
        let report = try NYCAPIContract.decode(SnapshotReport.self, Data(contentsOf: path))
        let encoded = try NYCAPIContract.encode(report.forecast.inputSnapshot)
        #expect(NYCAPIContract.hash(encoded) == report.forecast.inputSHA256)
    }

    @Test(
        "A1 All nine pinned official annual metadata records satisfy the historical publication contract",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_CHRONOS2_API_ANNUAL_METADATA"] != nil))
    func annualMetadata() throws {
        let root = URL(
            filePath: try #require(ProcessInfo.processInfo.environment["COREAI_CHRONOS2_API_ANNUAL_METADATA"]))
        let catalog = try Data(contentsOf: root.appending(path: "api-baseline/catalog.json"))
        try #require(NYCAPIContract.hash(catalog) == "a20275e84796d07e7bc72609a4b5eb061fa451464795db9071f804b661b6b8ac")
        let coverage = try NYCCatalogValidation.read(
            ChronosAPIFixture.evidence(catalog, url: NYCAPIContract.catalogURL()), now: ChronosAPIFixture.now)
        let hashes: [Int: String] = [
            2017: "42fffa46fe789d4e72f5cbd3465945cd0cfa8ad1b25351709479b51dcc8b13fb",
            2018: "41b757c4bf7b8baa8140bbb846cff505bfa2f982186d6392179bfa51ebce2d6f",
            2019: "d2da4e3e53f842c929788128eb55a306d010312ef00488980b427a0782dfb10c",
            2020: "32f07dfeb2de4cc247d963896b1f58b13c72b777c3fe3d0bb3339545af9a8047",
            2021: "e41e226d56683a0b38d3b139842631a76f25b0b5a98a72eae9ecd55d8b0175d2",
            2022: "bd908860357a838941a066073697a16c7832b3f45ab5e6c053a66b85b0d403cd",
            2023: "67ccc406f73bf49875fef85a82f7d033973c39a755f381ced5566eccb68e4f99",
            2024: "79ce192e836631024896124b06d5ef8b0a4f2948179bd935e5d1a89909108540",
            2025: "8f43be37dcde843cc8a2ddd508f5fc9a580ccaf7c0145674e5a238fe7a9e0a24",
        ]
        try #require(Set(coverage.datasets.map(\.year)) == Set(hashes.keys))
        var rejected: [Int] = []
        for dataset in coverage.datasets {
            let bytes = try Data(contentsOf: root.appending(path: "api-annual-metadata/\(dataset.year).json"))
            try #require(NYCAPIContract.hash(bytes) == hashes[dataset.year])
            let evidence = try ChronosAPIFixture.evidence(bytes, url: NYCAPIContract.metadataURL(dataset))
            do { try NYCCatalogValidation.metadata(evidence, dataset: dataset) } catch { rejected.append(dataset.year) }
        }
        #expect(rejected.isEmpty, "Rejected annual dataset years: \(rejected)")
    }

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

    private struct SnapshotReport: Decodable {
        struct Forecast: Decodable {
            let inputSnapshot: ForecastInputSnapshot
            let inputSHA256: String
        }
        let forecast: Forecast
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
