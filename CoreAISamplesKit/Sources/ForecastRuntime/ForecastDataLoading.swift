import ForecastFeature

public protocol ForecastDataLoading: Sendable {
    func load(refresh: Bool) async throws -> ForecastDataset
}
