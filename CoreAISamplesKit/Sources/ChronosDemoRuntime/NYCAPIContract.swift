import ChronosFeature
import CryptoKit
import Foundation

enum NYCAPIContract {
    static let byteLimit = 8 * 1024 * 1024
    static let rawLimit = 32 * 1024 * 1024
    static let rowLimit = 50_000
    static let zoneIDs = [
        "48", "50", "68", "90", "100", "107", "137", "142", "143", "161", "162", "163", "164", "170", "186", "229",
        "230", "233", "234", "246",
    ]

    static func hash(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func encode<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .millisecondsSince1970
        return try encoder.encode(value)
    }

    static func decode<T: Decodable>(_ type: T.Type, _ data: Data) throws -> T {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .millisecondsSince1970
        do { return try decoder.decode(type, from: data) } catch { throw ChronosDemoFailure.invalidData }
    }

    static func validID(_ id: String) -> Bool {
        id.range(of: "^[a-z0-9]{4}-[a-z0-9]{4}$", options: .regularExpression) != nil
    }

    static func validURL(_ url: URL) -> Bool {
        guard let parts = URLComponents(url: url, resolvingAgainstBaseURL: false),
            parts.scheme == "https", parts.port == nil || parts.port == 443,
            parts.user == nil, parts.password == nil, parts.fragment == nil
        else { return false }
        if parts.host == "api.us.socrata.com" { return parts.path == "/api/catalog/v1" }
        guard parts.host == "data.cityofnewyork.us" else { return false }
        return parts.path.range(
            of: "^/(api/views|resource)/[a-z0-9]{4}-[a-z0-9]{4}\\.json$", options: .regularExpression) != nil
    }

    static func catalogURL() throws -> URL {
        try url(
            "https://api.us.socrata.com/api/catalog/v1",
            [
                .init(name: "search_context", value: "data.cityofnewyork.us"),
                .init(name: "q", value: "Yellow Taxi Trip Data"), .init(name: "limit", value: "1000"),
            ])
    }

    static func metadataURL(_ dataset: APIDataset) throws -> URL {
        guard validID(dataset.id) else { throw ChronosDemoFailure.invalidData }
        return try url("https://data.cityofnewyork.us/api/views/\(dataset.id).json", [])
    }

    static func queryURL(_ dataset: APIDataset, start: Date, end: Date, areas: [String]) throws -> URL {
        guard validID(dataset.id), validAreas(areas) else { throw ChronosDemoFailure.invalidData }
        return try url(
            "https://data.cityofnewyork.us/resource/\(dataset.id).json",
            [
                .init(
                    name: "$select",
                    value:
                        "date_trunc_ymd(tpep_pickup_datetime) as day,date_extract_hh(tpep_pickup_datetime) as hour,date_extract_mm(tpep_pickup_datetime)>=30 as second_half,pulocationid as zone,count(*) as trips"
                ),
                .init(
                    name: "$where",
                    value:
                        "tpep_pickup_datetime >= '\(NYCTime.key(start))' AND tpep_pickup_datetime < '\(NYCTime.key(end))' AND pulocationid in(\(areas.joined(separator: ",")))"
                ),
                .init(name: "$group", value: "day,hour,second_half,zone"),
                .init(name: "$order", value: "day,hour,second_half,zone"),
                .init(name: "$limit", value: String(rowLimit)),
            ])
    }

    static func validAreas(_ areas: [String]) -> Bool {
        !areas.isEmpty && Set(areas).count == areas.count && Set(areas).isSubset(of: Set(zoneIDs))
    }

    static func verify(_ evidence: APIEvidence) throws {
        guard evidence.status == 200, validURL(evidence.url), evidence.data.count <= byteLimit,
            hash(evidence.data) == evidence.sha256, evidence.acquiredAt.timeIntervalSince1970.isFinite
        else { throw ChronosDemoFailure.invalidData }
    }

    private static func url(_ base: String, _ query: [URLQueryItem]) throws -> URL {
        guard var parts = URLComponents(string: base) else { throw ChronosDemoFailure.invalidData }
        if !query.isEmpty { parts.queryItems = query }
        guard let url = parts.url, validURL(url) else { throw ChronosDemoFailure.invalidData }
        return url
    }
}

enum NYCCatalogValidation {
    static func read(_ evidence: APIEvidence, now: Date) throws -> APICoverage {
        try NYCAPIContract.verify(evidence)
        guard evidence.url.host == "api.us.socrata.com" else { throw ChronosDemoFailure.invalidData }
        let catalog = try NYCAPIContract.decode(Catalog.self, evidence.data)
        guard catalog.results.count < 1000 else { throw ChronosDemoFailure.invalidData }
        var datasets: [APIDataset] = []
        for item in catalog.results {
            guard item.metadata?.domain == "data.cityofnewyork.us", item.owner?.id == "5fuc-pqz2",
                let resource = item.resource, resource.provenance == "official"
            else { continue }
            guard let name = resource.name else { throw ChronosDemoFailure.invalidData }
            guard name.contains("Yellow Taxi Trip Data") else { continue }
            guard let year = Int(name.prefix(4)) else { throw ChronosDemoFailure.invalidData }
            guard year >= 2017, year < NYCTime.year(now) else { continue }
            guard name == "\(year) Yellow Taxi Trip Data", let id = resource.id, NYCAPIContract.validID(id),
                let revision = resource.dataUpdatedAt, let updated = timestamp(revision), updated <= now,
                let columns = resource.columnsFieldName,
                columns.contains("tpep_pickup_datetime"), columns.contains("pulocationid")
            else { throw ChronosDemoFailure.invalidData }
            datasets.append(APIDataset(year: year, id: id, revision: revision))
        }
        guard Set(datasets.map(\.year)).count == datasets.count,
            Set(datasets.map(\.id)).count == datasets.count
        else { throw ChronosDemoFailure.invalidData }
        return APICoverage(datasets: datasets, evidence: evidence)
    }

    static func metadata(_ evidence: APIEvidence, dataset: APIDataset) throws {
        try NYCAPIContract.verify(evidence)
        let metadata = try NYCAPIContract.decode(Metadata.self, evidence.data)
        guard evidence.url == (try NYCAPIContract.metadataURL(dataset)),
            metadata.id == dataset.id, metadata.name == "\(dataset.year) Yellow Taxi Trip Data",
            metadata.owner.id == "5fuc-pqz2", metadata.provenance == "official",
            metadata.publicationStage == "published", metadata.assetType == "dataset",
            metadata.metadata.customFields["Update"]?["Update Frequency"] == "Historical data",
            metadata.metadata.customFields["Dataset Information"]?["Agency"] == "Taxi and Limousine Commission (TLC)",
            let yearEnd = NYCTime.yearStart(dataset.year + 1),
            metadata.publicationDate >= yearEnd.timeIntervalSince1970,
            metadata.publicationDate <= evidence.acquiredAt.timeIntervalSince1970,
            timestamp(dataset.revision)?.timeIntervalSince1970 == metadata.rowsUpdatedAt,
            metadata.columns.filter({ $0.fieldName == "tpep_pickup_datetime" }).map(\.dataTypeName) == [
                "calendar_date"
            ],
            metadata.columns.filter({ $0.fieldName == "pulocationid" }).map(\.dataTypeName) == ["number"]
        else { throw ChronosDemoFailure.invalidData }
    }

    static func timestamp(_ text: String) -> Date? {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = formatter.date(from: text) { return date }
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.date(from: text)
    }

    private struct Catalog: Decodable {
        struct Item: Decodable {
            struct Resource: Decodable {
                let name: String?
                let id: String?
                let provenance: String?
                let columnsFieldName: [String]?
                let dataUpdatedAt: String?
                enum CodingKeys: String, CodingKey {
                    case name, id, provenance
                    case columnsFieldName = "columns_field_name"
                    case dataUpdatedAt = "data_updated_at"
                }
            }
            struct Domain: Decodable { let domain: String? }
            struct Owner: Decodable { let id: String? }
            let resource: Resource?
            let metadata: Domain?
            let owner: Owner?
        }
        let results: [Item]
    }

    private struct Metadata: Decodable {
        struct Owner: Decodable { let id: String }
        struct Column: Decodable {
            let fieldName: String
            let dataTypeName: String
        }
        struct Details: Decodable {
            let customFields: [String: [String: String]]
            enum CodingKeys: String, CodingKey { case customFields = "custom_fields" }
        }
        let id: String
        let name: String
        let owner: Owner
        let provenance: String
        let publicationStage: String
        let assetType: String
        let publicationDate: Double
        let rowsUpdatedAt: Double
        let columns: [Column]
        let metadata: Details
    }
}
