import Foundation

public struct ChronosForecastMetrics: Codable, Equatable, Sendable {
    public let expectedCount: Int
    public let comparedCount: Int
    public let meanAbsoluteError: Double?
    public let weightedAbsolutePercentageError: Double?

    public static func evaluate(
        _ forecasts: [ChronosAreaForecast], actuals: [String: [Float?]]
    ) throws -> ChronosForecastMetrics {
        guard !forecasts.isEmpty, Set(forecasts.map(\.areaID)).count == forecasts.count,
            Set(actuals.keys) == Set(forecasts.map(\.areaID)),
            forecasts.allSatisfy({ $0.median.count == 4 && $0.median.allSatisfy(\.isFinite) }),
            actuals.values.allSatisfy({ $0.count == 4 && $0.allSatisfy { $0.map { $0.isFinite && $0 >= 0 } ?? true } })
        else { throw ChronosDemoFailure.invalidData }
        var compared = 0
        var absoluteError = 0.0
        var actualSum = 0.0
        for forecast in forecasts {
            guard let values = actuals[forecast.areaID] else { throw ChronosDemoFailure.invalidData }
            for (prediction, observed) in zip(forecast.median, values) {
                guard let observed else { continue }
                compared += 1
                absoluteError += abs(Double(max(0, prediction)) - Double(observed))
                actualSum += Double(observed)
            }
        }
        return ChronosForecastMetrics(
            expectedCount: forecasts.count * 4, comparedCount: compared,
            meanAbsoluteError: compared > 0 ? absoluteError / Double(compared) : nil,
            weightedAbsolutePercentageError: actualSum > 0 ? absoluteError / actualSum * 100 : nil)
    }
}
