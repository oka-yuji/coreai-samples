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
    public let verification: APIVerificationPolicy

    public init(datasets: [APIDataset], evidence: APIEvidence, verification: APIVerificationPolicy = .january2025) {
        self.datasets = datasets.sorted { $0.year < $1.year }
        self.evidence = evidence
        self.verification = verification
    }

    public var verifiedRanges: [Range<Date>] {
        var intervals: [Range<Date>] = []
        for range in verification.ranges where datasets.contains(range.dataset) {
            if let last = intervals.last, range.start <= last.upperBound {
                intervals[intervals.count - 1] = last.lowerBound..<max(last.upperBound, range.end)
            } else {
                intervals.append(range.start..<range.end)
            }
        }
        return intervals
    }

    public var selectableRanges: [ClosedRange<Date>] {
        verifiedRanges.compactMap {
            let first = $0.lowerBound.addingTimeInterval(1008 * 1800)
            return first <= $0.upperBound ? first...$0.upperBound : nil
        }
    }

    public var selectableRange: ClosedRange<Date>? {
        guard let first = selectableRanges.first, let last = selectableRanges.last else { return nil }
        return first.lowerBound...last.upperBound
    }

    public func allowsHistory(endingAt origin: Date) -> Bool {
        origin.timeIntervalSince1970.isFinite && origin == NYCTime.aligned(origin)
            && selectableRanges.contains { $0.contains(origin) }
    }

    public func availability(at date: Date, areaIDs: [String]) -> SlotAvailability {
        verification.availability(at: date, datasets: datasets, areaIDs: areaIDs)
    }
}

public enum SlotAvailability: String, Codable, Equatable, Sendable {
    case observed
    case notFetched
    case unpublished
    case unverified
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
    public static let preprocessingVersion = "nyc-utc30m-verified-v3"
    public let start: Date
    public let count: Int
    public let series: [CountSeries]
    public let availability: [SlotAvailability]
    public let datasets: [APIDataset]
    public let fetchedAt: Date
    public let catalog: APIEvidence
    public let metadata: [APIEvidence]
    public let responses: [APIEvidence]
    public let verification: APIVerificationPolicy

    public init(
        start: Date, count: Int, series: [CountSeries], availability: [SlotAvailability], datasets: [APIDataset],
        fetchedAt: Date, catalog: APIEvidence, metadata: [APIEvidence], responses: [APIEvidence],
        verification: APIVerificationPolicy = .january2025
    ) throws {
        guard NYCTime.isValidWindow(start: start, count: count), NYCTime.isSupported(fetchedAt),
            availability.count == count, !availability.contains(.notFetched),
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
        self.verification = verification
    }

    public var end: Date { start.addingTimeInterval(Double(count) * 1800) }
    public var availablePrefix: Int {
        availability.firstIndex(where: { $0 == .unpublished || $0 == .unverified || $0 == .notFetched }) ?? count
    }

    public func requireVerifiedHistory() throws {
        guard count == 1008, NYCTime.isValidWindow(start: start, count: count) else {
            throw ChronosDemoFailure.invalidData
        }
        let expected = (0..<count).map {
            verification.availability(
                at: start.addingTimeInterval(Double($0 * 1800)), datasets: datasets, areaIDs: series.map(\.areaID))
        }
        if availability.contains(.unpublished) || expected.contains(.unpublished) {
            throw ChronosDemoFailure.unpublishedHistory
        }
        if availability.contains(.unverified) || expected.contains(.unverified) {
            throw ChronosDemoFailure.unverifiedHistory
        }
        guard !availability.contains(.notFetched) else { throw ChronosDemoFailure.invalidData }
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
        try history.requireVerifiedHistory()
        guard (0...12).contains(step) else { throw ChronosDemoFailure.invalidData }
        if let actuals, step > 0 {
            guard actuals.verification == history.verification,
                Set(actuals.series.map(\.areaID)) == Set(history.series.map(\.areaID))
            else { throw ChronosDemoFailure.invalidData }
            let expected = (0..<step).map {
                actuals.verification.availability(
                    at: actuals.start.addingTimeInterval(Double($0 * 1800)), datasets: actuals.datasets,
                    areaIDs: actuals.series.map(\.areaID))
            }
            if actuals.availability.prefix(step).contains(.unpublished) || expected.contains(.unpublished) {
                throw ChronosDemoFailure.unpublishedHistory
            }
            if actuals.availability.prefix(step).contains(.unverified) || expected.contains(.unverified) {
                throw ChronosDemoFailure.unverifiedHistory
            }
        }
        guard (0...12).contains(step),
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
