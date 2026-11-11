import Foundation

public struct APIVerifiedRange: Codable, Equatable, Sendable {
    public let dataset: APIDataset
    public let start: Date
    public let end: Date
    public let areaIDs: [String]
    public let referenceSHA256: String

    public init(dataset: APIDataset, start: Date, end: Date, areaIDs: [String], referenceSHA256: String) {
        self.dataset = dataset
        self.start = start
        self.end = end
        self.areaIDs = areaIDs.sorted()
        self.referenceSHA256 = referenceSHA256
    }
}

public struct APIVerificationPolicy: Codable, Equatable, Sendable {
    public let version: String
    public let ranges: [APIVerifiedRange]

    public init(version: String, ranges: [APIVerifiedRange]) {
        self.version = version
        self.ranges = ranges.sorted { $0.start < $1.start }
    }

    public static let january2025 = APIVerificationPolicy(
        version: "nyc-independent-counts-v1",
        ranges: [
            APIVerifiedRange(
                dataset: APIDataset(year: 2025, id: "irx2-vgm4", revision: "2026-10-01T16:57:47.000Z"),
                start: Date(timeIntervalSince1970: 1_735_707_600),
                end: Date(timeIntervalSince1970: 1_738_386_000),
                areaIDs: [
                    "48", "50", "68", "90", "100", "107", "137", "142", "143", "161", "162", "163", "164", "170",
                    "186", "229", "230", "233", "234", "246",
                ], referenceSHA256: "2da47735c51b5987dec9f6bee72c97acd39abe4870991e413dca30a1912c6530")
        ])

    public func availability(at date: Date, datasets: [APIDataset], areaIDs: [String]) -> SlotAvailability {
        guard let dataset = datasets.first(where: { $0.year == NYCTime.year(date) }) else { return .unpublished }
        guard
            ranges.contains(where: {
                $0.dataset == dataset && $0.start <= date && date < $0.end
                    && !areaIDs.isEmpty && Set(areaIDs).isSubset(of: Set($0.areaIDs))
            })
        else { return .unverified }
        return NYCTime.isAmbiguous(date) ? .ambiguousTime : .observed
    }
}
