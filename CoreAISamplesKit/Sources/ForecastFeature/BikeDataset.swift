import Foundation

public enum BikeDataset {
    public static let seriesID = "seoul-bike"
    static let headers = [
        "Date", "Rented Bike Count", "Hour", "Temperature(°C)", "Humidity(%)", "Wind speed (m/s)",
        "Visibility (10m)", "Dew point temperature(°C)", "Solar Radiation (MJ/m2)", "Rainfall(mm)",
        "Snowfall (cm)", "Seasons", "Holiday", "Functioning Day",
    ]

    public static func parse(_ data: Data) throws -> ForecastDataset {
        let rows = try ForecastCSV.parse(data)
        guard rows.count == 8761, rows.first == headers else { throw ForecastFailure.invalidData }
        let calendar = try ForecastKind.seoulBike.calendar()
        guard let first = calendar.date(from: DateComponents(year: 2017, month: 12, day: 1)) else {
            throw ForecastFailure.invalidData
        }
        var timestamps: [Date] = []
        var counts: [Float?] = []
        var weather: [ForecastWeatherObservation] = []
        for (index, row) in rows.dropFirst().enumerated() {
            guard row.count == headers.count else { throw ForecastFailure.invalidData }
            let hour = try integer(row[2])
            guard hour < 24 else { throw ForecastFailure.invalidData }
            let timestamp = try date(row[0], hour: hour, calendar: calendar)
            guard calendar.date(byAdding: .hour, value: index, to: first) == timestamp else {
                throw ForecastFailure.invalidData
            }
            let count = try integer(row[1])
            guard ["Holiday", "No Holiday"].contains(row[12]), ["Yes", "No"].contains(row[13]),
                row[13] == "Yes" || count == 0
            else { throw ForecastFailure.invalidData }
            let values = try row[3...10].map { text -> Float in
                guard !text.isEmpty, text == text.trimmingCharacters(in: .whitespacesAndNewlines),
                    let value = Float(text), value.isFinite
                else { throw ForecastFailure.invalidData }
                return value
            }
            let observation = ForecastWeatherObservation(
                timestamp: timestamp, temperature: values[0], humidity: values[1], windSpeed: values[2],
                visibility: values[3], dewPoint: values[4], solarRadiation: values[5], rainfall: values[6],
                snowfall: values[7], season: row[11], isFunctioning: row[13] == "Yes", isHoliday: row[12] == "Holiday")
            guard observation.isValid else { throw ForecastFailure.invalidData }
            timestamps.append(timestamp)
            counts.append(Float(count))
            weather.append(observation)
        }
        return try ForecastDataset(
            kind: .seoulBike, timestamps: timestamps,
            series: [ForecastSeries(id: seriesID, values: counts)], weather: weather)
    }

    private static func integer(_ text: String) throws -> Int {
        guard !text.isEmpty, text.utf8.allSatisfy({ (48...57).contains($0) }), let value = Int(text), value >= 0 else {
            throw ForecastFailure.invalidData
        }
        return value
    }

    private static func date(_ text: String, hour: Int, calendar: Calendar) throws -> Date {
        let bytes = Array(text.utf8)
        guard bytes.count == 10, bytes[2] == 47, bytes[5] == 47,
            bytes.enumerated().allSatisfy({ [2, 5].contains($0.offset) || (48...57).contains($0.element) }),
            let year = Int(text.suffix(4)), let month = Int(text.dropFirst(3).prefix(2)),
            let day = Int(text.prefix(2)), (2017...2018).contains(year), (1...12).contains(month),
            (1...31).contains(day),
            let date = calendar.date(from: DateComponents(year: year, month: month, day: day, hour: hour))
        else { throw ForecastFailure.invalidData }
        let parts = calendar.dateComponents([.year, .month, .day, .hour], from: date)
        guard parts.year == year, parts.month == month, parts.day == day, parts.hour == hour else {
            throw ForecastFailure.invalidData
        }
        return date
    }
}
