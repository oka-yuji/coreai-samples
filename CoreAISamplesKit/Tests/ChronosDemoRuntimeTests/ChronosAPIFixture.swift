import ChronosFeature
import Foundation
import Testing

@testable import ChronosDemoRuntime

enum ChronosAPIFixture {
    static func date(_ value: String) throws -> Date {
        try #require(ISO8601DateFormatter().date(from: value))
    }

    static var now: Date { Date(timeIntervalSince1970: 1_794_441_600) }

    static func dataset(_ year: Int = 2025) -> APIDataset {
        APIDataset(year: year, id: year == 2025 ? "irx2-vgm4" : "rwwi-khbc", revision: "2026-10-01T16:57:47.000Z")
    }

    static func item(_ dataset: APIDataset = dataset()) -> [String: Any] {
        [
            "metadata": ["domain": "data.cityofnewyork.us"], "owner": ["id": "5fuc-pqz2"],
            "resource": [
                "id": dataset.id, "name": "\(dataset.year) Yellow Taxi Trip Data", "provenance": "official",
                "data_updated_at": dataset.revision, "columns_field_name": ["tpep_pickup_datetime", "pulocationid"],
            ],
        ]
    }

    static func catalog(_ items: [[String: Any]]? = nil) throws -> Data {
        try JSONSerialization.data(withJSONObject: ["results": items ?? [item()]], options: .sortedKeys)
    }

    static func metadata(_ dataset: APIDataset = dataset(), type: String = "calendar_date") throws -> Data {
        try JSONSerialization.data(
            withJSONObject: [
                "id": dataset.id, "name": "\(dataset.year) Yellow Taxi Trip Data", "owner": ["id": "5fuc-pqz2"],
                "provenance": "official", "publicationStage": "published", "assetType": "dataset",
                "publicationDate": 1_790_700_349, "rowsUpdatedAt": 1_790_873_867,
                "columns": [
                    ["fieldName": "tpep_pickup_datetime", "dataTypeName": type],
                    ["fieldName": "pulocationid", "dataTypeName": "number"],
                ],
                "metadata": [
                    "custom_fields": [
                        "Update": ["Update Frequency": "Historical data"],
                        "Dataset Information": ["Agency": "Taxi and Limousine Commission (TLC)"],
                    ]
                ],
            ], options: .sortedKeys)
    }

    static func rows(
        day: String = "2025-01-03T00:00:00.000", hour: String = "17", zone: String = "230", trips: String = "7",
        copies: Int = 1
    ) throws -> Data {
        try JSONSerialization.data(
            withJSONObject: Array(
                repeating: [
                    "day": day, "hour": hour,
                    "second_half": false, "zone": zone, "trips": trips,
                ] as [String: Any], count: copies), options: .sortedKeys)
    }

    static func evidence(_ data: Data, url: URL) -> APIEvidence {
        APIEvidence(url: url, status: 200, acquiredAt: now, data: data, sha256: NYCAPIContract.hash(data))
    }

    static func coverage() throws -> APICoverage {
        try NYCCatalogValidation.read(evidence(catalog(), url: NYCAPIContract.catalogURL()), now: now)
    }

    static func syntheticProfile(value: Float = 0, areas: [String] = ["230"]) throws -> NYCVerificationProfile {
        let ranges = try [2024, 2025].map { year in
            try APIVerifiedRange(
                dataset: dataset(year), start: #require(NYCTime.yearStart(year)),
                end: #require(NYCTime.yearStart(year + 1)), areaIDs: areas,
                referenceSHA256: String(repeating: "a", count: 64))
        }
        let series = try ranges.map { range in
            var values: [Float?] = Array(repeating: 0, count: Int(range.end.timeIntervalSince(range.start) / 1800))
            if range.dataset.year == 2025 {
                let offset = Int(try date("2025-01-03T22:00:00Z").timeIntervalSince(range.start) / 1800)
                values[offset] = value
            }
            return areas.map { CountSeries(areaID: $0, values: values) }
        }
        return try NYCVerificationProfile(
            policy: APIVerificationPolicy(version: "synthetic-zero-fixture-v1", ranges: ranges), references: series)
    }

    static func client(
        _ fixture: ChronosAPITransportFake, cache: URL? = nil, capacity: Int = 32, value: Float = 0,
        areas: [String] = ["230"]
    )
        throws -> NYCAPIClient
    {
        try NYCAPIClient(
            cacheDirectory: cache, transport: { try await fixture.receive($0) }, maxCachedWindows: capacity,
            now: { now }, retryDelay: {}, verification: syntheticProfile(value: value, areas: areas))
    }

    static func temporary() throws -> URL {
        let path = FileManager.default.temporaryDirectory.appending(path: "chronos-api-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: path, withIntermediateDirectories: true)
        return path
    }
}

actor ChronosAPITransportFake {
    let catalog: Data
    let metadata: [String: Data]
    var rows: Data
    var status = 200
    var requests: [URLRequest] = []
    var statuses: [Int] = []
    var held = false
    var heldPath: String?
    var continuation: CheckedContinuation<Void, Never>?
    let started = AsyncStream<URLRequest>.makeStream()

    init(catalog: Data? = nil, rows: Data = Data("[]".utf8), metadata: [String: Data]? = nil) throws {
        self.catalog = try catalog ?? ChronosAPIFixture.catalog()
        self.metadata = try metadata ?? ["irx2-vgm4": ChronosAPIFixture.metadata()]
        self.rows = rows
    }

    func setStatus(_ value: Int) { status = value }
    func setStatuses(_ values: [Int]) { statuses = values }
    func hold(path: String? = nil) {
        held = true
        heldPath = path
    }
    func allowNew() { held = false }
    func finish() {
        held = false
        continuation?.resume()
        continuation = nil
    }

    func receive(_ request: URLRequest) async throws -> NYCAPIResponse {
        requests.append(request)
        started.continuation.yield(request)
        if held, heldPath.map({ request.url?.path.hasPrefix($0) == true }) ?? true {
            await withCheckedContinuation { continuation = $0 }
        }
        let data: Data
        let url = try #require(request.url)
        if url.host == "api.us.socrata.com" {
            data = catalog
        } else if url.path.hasPrefix("/api/views/") {
            data = try #require(metadata[url.deletingPathExtension().lastPathComponent])
        } else {
            data = rows
        }
        return NYCAPIResponse(data: data, status: statuses.isEmpty ? status : statuses.removeFirst())
    }
}
