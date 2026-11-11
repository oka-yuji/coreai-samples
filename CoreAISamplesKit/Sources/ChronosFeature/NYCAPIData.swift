import Foundation

public struct APIDataset: Codable, Equatable, Hashable, Sendable {
    public let year: Int
    public let id: String
    public let revision: String

    public init(year: Int, id: String, revision: String) {
        self.year = year
        self.id = id
        self.revision = revision
    }
}

public struct APIEvidence: Codable, Equatable, Sendable {
    public let url: URL
    public let start: Date?
    public let end: Date?
    public let status: Int
    public let acquiredAt: Date
    public let data: Data
    public let sha256: String

    public init(
        url: URL, start: Date? = nil, end: Date? = nil, status: Int, acquiredAt: Date, data: Data, sha256: String
    ) {
        self.url = url
        self.start = start
        self.end = end
        self.status = status
        self.acquiredAt = acquiredAt
        self.data = data
        self.sha256 = sha256
    }
}

public struct APICoverage: Codable, Equatable, Sendable {
    public let datasets: [APIDataset]
    public let evidence: APIEvidence

    public init(datasets: [APIDataset], evidence: APIEvidence) {
        self.datasets = datasets.sorted { $0.year < $1.year }
        self.evidence = evidence
    }

    public var selectableRanges: [ClosedRange<Date>] {
        var ranges: [ClosedRange<Date>] = []
        var first: Int?
        var previous: Int?
        for year in datasets.map(\.year) {
            if let previous, year != previous + 1, let startYear = first {
                if let range = range(startYear, previous) { ranges.append(range) }
                first = nil
            }
            if first == nil { first = year }
            previous = year
        }
        if let first, let previous, let range = range(first, previous) { ranges.append(range) }
        return ranges
    }

    public var selectableRange: ClosedRange<Date>? {
        guard let first = selectableRanges.first, let last = selectableRanges.last else { return nil }
        return first.lowerBound...last.upperBound
    }

    public func allowsHistory(endingAt origin: Date) -> Bool {
        origin.timeIntervalSince1970.isFinite && origin == NYCTime.aligned(origin)
            && selectableRanges.contains { $0.contains(origin) }
    }

    private func range(_ first: Int, _ last: Int) -> ClosedRange<Date>? {
        guard let start = NYCTime.yearStart(first), let end = NYCTime.yearStart(last + 1) else { return nil }
        return start.addingTimeInterval(1008 * 1800)...end
    }
}

public enum SlotAvailability: String, Codable, Equatable, Sendable {
    case observed
    case notFetched
    case unpublished
    case ambiguousTime
}

public struct CountSeries: Codable, Equatable, Sendable {
    public let areaID: String
    public let values: [Float?]

    public init(areaID: String, values: [Float?]) {
        self.areaID = areaID
        self.values = values
    }
}

public struct APIWindow: Codable, Equatable, Sendable {
    public static let preprocessingVersion = "nyc-utc30m-v2"
    public let start: Date
    public let count: Int
    public let series: [CountSeries]
    public let availability: [SlotAvailability]
    public let datasets: [APIDataset]
    public let fetchedAt: Date
    public let catalog: APIEvidence
    public let metadata: [APIEvidence]
    public let responses: [APIEvidence]

    public init(
        start: Date, count: Int, series: [CountSeries], availability: [SlotAvailability], datasets: [APIDataset],
        fetchedAt: Date, catalog: APIEvidence, metadata: [APIEvidence], responses: [APIEvidence]
    ) throws {
        guard start.timeIntervalSince1970.isFinite, start == NYCTime.aligned(start),
            fetchedAt.timeIntervalSince1970.isFinite,
            (1...1024).contains(count), availability.count == count, !availability.contains(.notFetched),
            !series.isEmpty, Set(series.map(\.areaID)).count == series.count,
            series.allSatisfy({
                $0.values.count == count
                    && $0.values.allSatisfy {
                        $0.map { $0.isFinite && $0 >= 0 && $0 <= 16_777_216 && $0.rounded() == $0 } ?? true
                    }
            }),
            availability.indices.allSatisfy({ index in
                series.allSatisfy { (availability[index] == .observed) == ($0.values[index] != nil) }
            })
        else { throw ChronosDemoFailure.invalidData }
        self.start = start
        self.count = count
        self.series = series.sorted { $0.areaID < $1.areaID }
        self.availability = availability
        self.datasets = datasets.sorted { $0.year < $1.year }
        self.fetchedAt = fetchedAt
        self.catalog = catalog
        self.metadata = metadata
        self.responses = responses
    }

    public var end: Date { start.addingTimeInterval(Double(count * 1800)) }
    public var availablePrefix: Int {
        availability.firstIndex(where: { $0 == .unpublished || $0 == .notFetched }) ?? count
    }
}

public struct ForecastInputSnapshot: Codable, Equatable, Sendable {
    public let start: Date
    public let origin: Date
    public let series: [CountSeries]
    public let datasets: [APIDataset]
    public let preprocessing: String
    public let history: APIWindow
    public let actuals: APIWindow?

    public init(history: APIWindow, actuals: APIWindow?, step: Int) throws {
        guard history.count == 1008, !history.availability.contains(.unpublished), (0...12).contains(step),
            step == 0 || (actuals?.start == history.end && step <= (actuals?.availablePrefix ?? 0))
        else { throw ChronosDemoFailure.unpublishedHistory }
        var inputs: [CountSeries] = []
        for row in history.series {
            let future = actuals?.series.first { $0.areaID == row.areaID }?.values ?? []
            guard step <= future.count else { throw ChronosDemoFailure.invalidData }
            let values = Array((row.values + future.prefix(step)).suffix(1008))
            guard values.contains(where: { $0?.isFinite == true }) else { throw ChronosDemoFailure.noData }
            inputs.append(CountSeries(areaID: row.areaID, values: values))
        }
        origin = history.end.addingTimeInterval(Double(step * 1800))
        start = origin.addingTimeInterval(-1008 * 1800)
        series = inputs
        let revisions = Array(Set(history.datasets + (step == 0 ? [] : actuals?.datasets ?? [])))
        guard Set(revisions.map(\.year)).count == revisions.count else { throw ChronosDemoFailure.invalidData }
        datasets = revisions.sorted { $0.year < $1.year }
        preprocessing = APIWindow.preprocessingVersion
        self.history = history
        self.actuals = step == 0 ? nil : actuals
    }
}
