import ChronosCore
import Foundation

public struct ForecastResult: Sendable, Equatable, Encodable {
    public let request: ForecastRequest
    public let raw: ChronosPrediction
    public let identity: ChronosBundleIdentity
    public let seconds: Double
    public let loaded: Bool
    public let timingBoundary: String

    public init(
        request: ForecastRequest, raw: ChronosPrediction, identity: ChronosBundleIdentity,
        seconds: Double, loaded: Bool, timingBoundary: String
    ) throws {
        guard seconds.isFinite, seconds >= 0, request.kind.horizons.contains(request.horizon),
            request.actuals.count == request.horizon, request.baseline.count == request.horizon,
            request.timestamps.count == request.horizon, request.context.count == request.kind.contextCount
        else { throw ForecastFailure.invalidRequest }
        self.request = request
        self.raw = raw
        self.identity = identity
        self.seconds = seconds
        self.loaded = loaded
        self.timingBoundary = timingBoundary
    }

    public var median: [Float] { displayedQuantile(10) }
    public var lower: [Float] { displayedQuantile(2) }
    public var upper: [Float] { displayedQuantile(18) }

    public var metrics: ForecastMetrics {
        ForecastMetrics(actuals: request.actuals, model: median.map(Optional.some), baseline: request.baseline)
    }

    private func displayedQuantile(_ index: Int) -> [Float] {
        raw.quantiles[index].prefix(request.horizon).map { max(0, $0) }
    }
}

public struct ForecastMetrics: Equatable, Sendable, Encodable {
    public let count: Int
    public let modelMAE: Double?
    public let baselineMAE: Double?
    public let modelWAPE: Double?
    public let baselineWAPE: Double?

    public init(actuals: [Float?], model: [Float?], baseline: [Float?]) {
        var count = 0
        var modelError = 0.0
        var baselineError = 0.0
        var denominator = 0.0
        if actuals.count == model.count, actuals.count == baseline.count {
            for index in actuals.indices {
                guard let actual = actuals[index], let prediction = model[index], let naive = baseline[index],
                    actual.isFinite, prediction.isFinite, naive.isFinite, actual >= 0, naive >= 0
                else { continue }
                count += 1
                modelError += abs(Double(actual) - Double(max(0, prediction)))
                baselineError += abs(Double(actual) - Double(naive))
                denominator += Double(actual)
            }
        }
        self.count = count
        modelMAE = count > 0 ? modelError / Double(count) : nil
        baselineMAE = count > 0 ? baselineError / Double(count) : nil
        modelWAPE = denominator > 0 ? modelError / denominator : nil
        baselineWAPE = denominator > 0 ? baselineError / denominator : nil
    }
}
