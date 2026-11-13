import Foundation

public struct ForecastWeatherObservation: Sendable, Equatable {
    public let timestamp: Date
    public let temperature: Float
    public let humidity: Float
    public let windSpeed: Float
    public let visibility: Float
    public let dewPoint: Float
    public let solarRadiation: Float
    public let rainfall: Float
    public let snowfall: Float
    public let season: String
    public let isFunctioning: Bool
    public let isHoliday: Bool

    var isValid: Bool {
        [temperature, humidity, windSpeed, visibility, dewPoint, solarRadiation, rainfall, snowfall]
            .allSatisfy(\.isFinite)
            && (0...100).contains(humidity)
            && [windSpeed, visibility, solarRadiation, rainfall, snowfall].allSatisfy { $0 >= 0 }
            && ["Winter", "Spring", "Summer", "Autumn"].contains(season)
    }
}
