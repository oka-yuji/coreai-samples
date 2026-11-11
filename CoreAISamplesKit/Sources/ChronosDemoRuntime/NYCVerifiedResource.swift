import ChronosFeature
import Foundation

public struct NYCVerificationProfile: Sendable {
    let policy: APIVerificationPolicy
    let references: [[CountSeries]]

    init(policy: APIVerificationPolicy, references: [[CountSeries]]) throws {
        guard !policy.version.isEmpty, !policy.ranges.isEmpty, references.count == policy.ranges.count else {
            throw ChronosDemoFailure.invalidData
        }
        for (index, range) in policy.ranges.enumerated() {
            let duration = range.end.timeIntervalSince(range.start)
            guard range.start.timeIntervalSince1970.isFinite, duration.isFinite,
                duration > 0, duration <= 366 * 86400, duration.truncatingRemainder(dividingBy: 1800) == 0,
                range.start == NYCTime.aligned(range.start), range.end == NYCTime.aligned(range.end),
                NYCTime.year(range.start) == range.dataset.year,
                NYCTime.year(range.end.addingTimeInterval(-1800)) == range.dataset.year,
                NYCAPIContract.validID(range.dataset.id), NYCCatalogValidation.timestamp(range.dataset.revision) != nil,
                NYCAPIContract.validAreas(range.areaIDs),
                range.referenceSHA256.range(of: "^[0-9a-f]{64}$", options: .regularExpression) != nil,
                index == 0 || policy.ranges[index - 1].end <= range.start,
                references[index].count == range.areaIDs.count,
                references[index].map(\.areaID).sorted() == range.areaIDs,
                references[index].allSatisfy({
                    $0.values.count == Int(duration / 1800)
                        && $0.values.allSatisfy {
                            $0.map { $0.isFinite && $0 >= 0 && $0 <= 16_777_216 && $0.rounded() == $0 } == true
                        }
                })
            else { throw ChronosDemoFailure.invalidData }
        }
        self.policy = policy
        self.references = references
    }

    func count(at date: Date, areaID: String, dataset: APIDataset) throws -> Float {
        guard
            let index = policy.ranges.firstIndex(where: {
                $0.dataset == dataset && $0.start <= date && date < $0.end && $0.areaIDs.contains(areaID)
            }), let series = references[index].first(where: { $0.areaID == areaID })
        else {
            throw ChronosDemoFailure.invalidData
        }
        let offset = date.timeIntervalSince(policy.ranges[index].start) / 1800
        guard offset.isFinite, offset.rounded() == offset, series.values.indices.contains(Int(offset)),
            let value = series.values[Int(offset)]
        else { throw ChronosDemoFailure.invalidData }
        return value
    }
}

enum NYCVerifiedResource {
    static func bundled() throws -> NYCVerificationProfile {
        guard
            let url = Bundle.module.url(forResource: "january-2025", withExtension: "json", subdirectory: "NYCVerified")
        else { throw ChronosDemoFailure.invalidData }
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        let data = try file.read(upToCount: 88_977) ?? Data()
        return try read(data)
    }

    static func read(_ data: Data) throws -> NYCVerificationProfile {
        let policy = APIVerificationPolicy.january2025
        guard let expected = policy.ranges.first, data.count == 88_976,
            NYCAPIContract.hash(data) == expected.referenceSHA256
        else { throw ChronosDemoFailure.invalidData }
        let reference = try NYCAPIContract.decode(Reference.self, data)
        let formatter = ISO8601DateFormatter()
        guard reference.format == "nyc-verified-counts/1",
            reference.year == expected.dataset.year, reference.datasetID == expected.dataset.id,
            reference.revision == expected.dataset.revision,
            formatter.date(from: reference.start) == expected.start,
            formatter.date(from: reference.end) == expected.end,
            reference.intervalSeconds == 1800, reference.timeZone == "America/New_York",
            reference.sourceParquets == [
                Source(
                    url: "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet",
                    sha256: "9af277e4c0d3f9deb30644da822981e1e7df6af58313170fd3aa8a474485488a"),
                Source(
                    url: "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-02.parquet",
                    sha256: "037cba555a73663f3a51a2c27816e40e3feb364769942bdf122b9da31e377bd3"),
            ]
        else { throw ChronosDemoFailure.invalidData }
        return try NYCVerificationProfile(policy: policy, references: [reference.series])
    }

    private struct Source: Decodable, Equatable {
        let url: String
        let sha256: String
    }

    private struct Reference: Decodable {
        let format: String
        let year: Int
        let datasetID: String
        let revision: String
        let start: String
        let end: String
        let intervalSeconds: Int
        let timeZone: String
        let series: [CountSeries]
        let sourceParquets: [Source]
    }
}
