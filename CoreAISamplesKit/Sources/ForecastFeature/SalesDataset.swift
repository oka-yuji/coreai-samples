import Foundation

public enum SalesDataset {
    public static let seriesIDs = ["B1_1", "B2_1", "B3_1", "B4_1"]
    public static let firstDay = "2014-01-01"
    public static let lastDay = "2018-12-31"

    public static func parse(_ data: Data) throws -> ForecastDataset {
        let rows = try ForecastCSV.parse(data)
        let items = [42, 45, 21, 10].enumerated().flatMap { brand, count in
            (1...count).map { "B\(brand + 1)_\($0)" }
        }
        let header = ["DATE"] + items.map { "QTY_" + $0 } + items.map { "PROMO_" + $0 }
        guard rows.first == header, rows.count > 1 else { throw ForecastFailure.invalidData }
        let calendar = try ForecastKind.productSales.calendar()
        let first = try date(firstDay, calendar: calendar)
        let last = try date(lastDay, calendar: calendar)
        var timestamps: [Date] = []
        var day = first
        while day <= last {
            timestamps.append(day)
            guard let next = calendar.date(byAdding: .day, value: 1, to: day) else {
                throw ForecastFailure.invalidData
            }
            day = next
        }
        let indices = Dictionary(uniqueKeysWithValues: timestamps.enumerated().map { ($1, $0) })
        let columns = try seriesIDs.map { id in
            guard let index = header.firstIndex(of: "QTY_" + id) else { throw ForecastFailure.invalidData }
            return index
        }
        var values = seriesIDs.map { _ in [Float?](repeating: nil, count: timestamps.count) }
        var previous: Date?
        for row in rows.dropFirst() {
            guard row.count == header.count else { throw ForecastFailure.invalidData }
            let timestamp = try date(row[0], calendar: calendar)
            guard let index = indices[timestamp], previous.map({ $0 < timestamp }) ?? true else {
                throw ForecastFailure.invalidData
            }
            previous = timestamp
            let quantities = try row[1...items.count].map(quantity)
            guard row[(items.count + 1)...].allSatisfy({ $0 == "0" || $0 == "1" }) else {
                throw ForecastFailure.invalidData
            }
            for (series, column) in columns.enumerated() { values[series][index] = quantities[column - 1] }
        }
        return try ForecastDataset(
            kind: .productSales, timestamps: timestamps,
            series: zip(seriesIDs, values).map { ForecastSeries(id: $0, values: $1) })
    }

    private static func quantity(_ text: String) throws -> Float {
        guard !text.isEmpty, text.utf8.allSatisfy({ (48...57).contains($0) }),
            let value = Double(text), value.isFinite, value >= 0, Float(value).isFinite
        else { throw ForecastFailure.invalidData }
        return Float(value)
    }

    private static func date(_ text: String, calendar: Calendar) throws -> Date {
        let bytes = Array(text.utf8)
        guard bytes.count == 10, bytes[4] == 45, bytes[7] == 45,
            bytes.enumerated().allSatisfy({ [4, 7].contains($0.offset) || (48...57).contains($0.element) }),
            let year = Int(text.prefix(4)), let month = Int(text.dropFirst(5).prefix(2)),
            let day = Int(text.suffix(2)), (2014...2018).contains(year), (1...12).contains(month),
            (1...31).contains(day),
            let date = calendar.date(from: DateComponents(year: year, month: month, day: day))
        else { throw ForecastFailure.invalidData }
        let parts = calendar.dateComponents([.year, .month, .day], from: date)
        guard parts.year == year, parts.month == month, parts.day == day else { throw ForecastFailure.invalidData }
        return date
    }
}
