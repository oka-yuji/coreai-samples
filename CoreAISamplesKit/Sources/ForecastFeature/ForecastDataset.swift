import Foundation

public enum ForecastFailure: Error, Equatable, Sendable {
    case invalidData
    case invalidRequest
    case sourceMismatch
    case responseTooLarge
    case network
    case cache
    case unavailable
    case predictionFailed
    case busy
}

public enum ForecastKind: String, Equatable, Sendable, Codable {
    case productSales
    case seoulBike

    public var timeZoneID: String { self == .productSales ? "Europe/Rome" : "Asia/Seoul" }
    public var contextCount: Int { self == .productSales ? 365 : 336 }
    public var seasonLength: Int { self == .productSales ? 7 : 168 }
    public var horizons: [Int] { self == .productSales ? [7, 14] : [1, 4, 8, 16] }
    public var calendarUnit: Calendar.Component { self == .productSales ? .day : .hour }

    public func calendar() throws -> Calendar {
        guard let zone = TimeZone(identifier: timeZoneID) else { throw ForecastFailure.invalidData }
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = zone
        calendar.locale = Locale(identifier: "en_US_POSIX")
        return calendar
    }
}

public struct ForecastSeries: Sendable, Equatable {
    public let id: String
    public let values: [Float?]

    public init(id: String, values: [Float?]) {
        self.id = id
        self.values = values
    }
}

public struct ForecastDataset: Sendable, Equatable {
    public let kind: ForecastKind
    public let timestamps: [Date]
    public let series: [ForecastSeries]
    private let weather: [ForecastWeatherObservation]

    public init(
        kind: ForecastKind, timestamps: [Date], series: [ForecastSeries],
        weather: [ForecastWeatherObservation] = []
    ) throws {
        let calendar = try kind.calendar()
        guard timestamps.count > kind.contextCount, timestamps.count <= 100_000,
            timestamps.allSatisfy({
                $0.timeIntervalSince1970.isFinite && abs($0.timeIntervalSince1970) < 100_000_000_000
                    && calendar.dateInterval(of: kind.calendarUnit, for: $0)?.start == $0
            }),
            zip(timestamps, timestamps.dropFirst()).allSatisfy({
                $0 < $1 && calendar.date(byAdding: kind.calendarUnit, value: 1, to: $0) == $1
            }),
            !series.isEmpty, Set(series.map(\.id)).count == series.count,
            series.allSatisfy({ item in
                !item.id.isEmpty && item.values.count == timestamps.count
                    && item.values.allSatisfy { $0.map { $0.isFinite && $0 >= 0 } ?? true }
            })
        else { throw ForecastFailure.invalidData }
        guard
            weather.isEmpty
                || (kind == .seoulBike && weather.count == timestamps.count
                    && zip(weather, timestamps).allSatisfy { $0.timestamp == $1 && $0.isValid })
        else { throw ForecastFailure.invalidData }
        self.kind = kind
        self.timestamps = timestamps
        self.series = series
        self.weather = weather
    }

    public func origins(horizon: Int) -> ClosedRange<Int>? {
        guard kind.horizons.contains(horizon), timestamps.count - horizon >= kind.contextCount else { return nil }
        return kind.contextCount...(timestamps.count - horizon)
    }

    public func request(seriesID: String, originIndex: Int, horizon: Int) throws -> ForecastRequest {
        guard origins(horizon: horizon)?.contains(originIndex) == true,
            let selected = series.first(where: { $0.id == seriesID })
        else { throw ForecastFailure.invalidRequest }
        let context = Array(selected.values[(originIndex - kind.contextCount)..<originIndex])
        let actuals = Array(selected.values[originIndex..<(originIndex + horizon)])
        let baseline = (0..<horizon).map { selected.values[originIndex - kind.seasonLength + $0 % kind.seasonLength] }
        return ForecastRequest(
            kind: kind, seriesID: seriesID, originIndex: originIndex, horizon: horizon,
            timestamps: Array(timestamps[originIndex..<(originIndex + horizon)]),
            context: context, actuals: actuals, baseline: baseline)
    }

    public func pastWeather(originIndex: Int, count: Int) throws -> [ForecastWeatherObservation] {
        guard kind == .seoulBike, (1...kind.contextCount).contains(count),
            origins(horizon: 1)?.contains(originIndex) == true, weather.count == timestamps.count
        else { throw ForecastFailure.invalidRequest }
        return Array(weather[(originIndex - count)..<originIndex])
    }
}

public struct ForecastRequest: Sendable, Equatable, Encodable {
    public let kind: ForecastKind
    public let seriesID: String
    public let originIndex: Int
    public let horizon: Int
    public let timestamps: [Date]
    public let context: [Float?]
    public let actuals: [Float?]
    public let baseline: [Float?]
}
