import ChronosFeature
import Foundation

enum NYCWindowValidation {
    struct Chunk {
        let dataset: APIDataset
        let start: Date
        let end: Date
    }

    static func datasets(start: Date, count: Int, coverage: APICoverage) -> [APIDataset] {
        let years = Set((0..<count).map { NYCTime.year(start.addingTimeInterval(Double($0 * 1800))) })
        return coverage.datasets.filter { years.contains($0.year) }
    }

    static func chunks(start: Date, count: Int, datasets: [APIDataset]) throws -> [Chunk] {
        let end = start.addingTimeInterval(Double(count * 1800))
        var chunks: [Chunk] = []
        for dataset in datasets {
            guard let first = NYCTime.yearStart(dataset.year), let last = NYCTime.yearStart(dataset.year + 1)
            else { throw ChronosDemoFailure.invalidDate }
            var cursor = max(first, start)
            let stop = min(last, end)
            while cursor < stop {
                if NYCTime.isAmbiguous(cursor) {
                    cursor.addTimeInterval(1800)
                    continue
                }
                var next = min(stop, cursor.addingTimeInterval(7 * 86400))
                while NYCTime.isAmbiguous(next), next > cursor { next.addTimeInterval(-1800) }
                if next == cursor { next = cursor.addingTimeInterval(1800) }
                chunks.append(Chunk(dataset: dataset, start: cursor, end: next))
                cursor = next
            }
        }
        return chunks
    }

    static func make(
        start: Date, count: Int, areas: [String], coverage: APICoverage,
        metadata: [APIEvidence], responses: [APIEvidence], fetchedAt: Date
    ) throws -> APIWindow {
        guard NYCAPIContract.validAreas(areas), (1...1024).contains(count),
            ([coverage.evidence] + metadata + responses).reduce(0, { $0 + $1.data.count }) <= NYCAPIContract.rawLimit
        else { throw ChronosDemoFailure.invalidData }
        let datasets = datasets(start: start, count: count, coverage: coverage)
        guard metadata.count == datasets.count else { throw ChronosDemoFailure.invalidData }
        for (dataset, evidence) in zip(datasets, metadata) {
            try NYCCatalogValidation.metadata(evidence, dataset: dataset)
        }
        let chunks = try chunks(start: start, count: count, datasets: datasets)
        guard chunks.count == responses.count else { throw ChronosDemoFailure.invalidData }
        var rows: [String: Float] = [:]
        for (chunk, evidence) in zip(chunks, responses) {
            try NYCAPIContract.verify(evidence)
            guard evidence.start == chunk.start, evidence.end == chunk.end,
                evidence.url
                    == (try NYCAPIContract.queryURL(chunk.dataset, start: chunk.start, end: chunk.end, areas: areas))
            else { throw ChronosDemoFailure.invalidData }
            try append(evidence.data, chunk: chunk, areas: areas, rows: &rows)
        }
        let years = Set(datasets.map(\.year))
        let dates = (0..<count).map { start.addingTimeInterval(Double($0 * 1800)) }
        let availability: [SlotAvailability] = dates.map {
            if !years.contains(NYCTime.year($0)) { return .unpublished }
            return NYCTime.isAmbiguous($0) ? .ambiguousTime : .observed
        }
        let keys = dates.map(NYCTime.key)
        let series = areas.map { area in
            CountSeries(
                areaID: area,
                values: keys.indices.map {
                    availability[$0] == .observed ? rows[keys[$0] + ":" + area] ?? 0 : nil
                })
        }
        return try APIWindow(
            start: start, count: count, series: series, availability: availability, datasets: datasets,
            fetchedAt: fetchedAt, catalog: coverage.evidence, metadata: metadata, responses: responses)
    }

    static func validate(_ window: APIWindow) throws {
        let coverage = try NYCCatalogValidation.read(window.catalog, now: window.catalog.acquiredAt)
        let copy = try make(
            start: window.start, count: window.count, areas: window.series.map(\.areaID), coverage: coverage,
            metadata: window.metadata, responses: window.responses, fetchedAt: window.fetchedAt)
        guard copy == window else { throw ChronosDemoFailure.invalidData }
    }

    private static func append(_ data: Data, chunk: Chunk, areas: [String], rows: inout [String: Float]) throws {
        let values = try NYCAPIContract.decode([Row].self, data)
        guard values.count < NYCAPIContract.rowLimit else { throw ChronosDemoFailure.invalidData }
        let lower = NYCTime.key(chunk.start)
        let upper = NYCTime.key(chunk.end)
        var validDays: [String: Set<String>] = [:]
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = NYCTime.calendar.timeZone
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSS"
        formatter.isLenient = false
        for row in values {
            guard let hour = integer(row.hour), (0...23).contains(hour), let count = integer(row.trips),
                (0...16_777_216).contains(count), areas.contains(row.zone),
                row.day.count == 23, row.day.hasSuffix("T00:00:00.000")
            else { throw ChronosDemoFailure.invalidData }
            if validDays[row.day] == nil {
                guard let day = formatter.date(from: row.day), formatter.string(from: day) == row.day,
                    NYCTime.year(day) == chunk.dataset.year
                else { throw ChronosDemoFailure.invalidData }
                validDays[row.day] = Set(NYCTime.slots(on: day).map(NYCTime.key))
            }
            let date = String(row.day.prefix(10)) + String(format: "T%02d:%02d:00", hour, row.secondHalf ? 30 : 0)
            guard date >= lower, date < upper, validDays[row.day]?.contains(date) == true,
                rows.updateValue(Float(count), forKey: date + ":" + row.zone) == nil
            else { throw ChronosDemoFailure.invalidData }
        }
    }

    private static func integer(_ value: String) -> Int? {
        guard !value.isEmpty, value.utf8.allSatisfy({ (48...57).contains($0) }),
            let number = Int(value), String(number) == value
        else { return nil }
        return number
    }

    private struct Row: Decodable {
        let day: String
        let hour: String
        let secondHalf: Bool
        let zone: String
        let trips: String
        enum CodingKeys: String, CodingKey {
            case day, hour, zone, trips
            case secondHalf = "second_half"
        }
    }
}
