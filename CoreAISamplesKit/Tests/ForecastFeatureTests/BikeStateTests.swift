import ChronosCore
import FeatureStore
import Foundation
import Testing

@testable import ForecastFeature

struct BikeStateTests {
    @Test("B2 Bike input and weekly baseline stop before the selected hour for every horizon")
    func windows() throws {
        let dataset = try bikeFixture()
        for horizon in [1, 4, 8, 16] {
            let request = try dataset.request(seriesID: BikeDataset.seriesID, originIndex: 400, horizon: horizon)
            #expect(request.context == Array(dataset.series[0].values[64..<400]))
            #expect(request.actuals == Array(dataset.series[0].values[400..<(400 + horizon)]))
            #expect(request.baseline == Array(dataset.series[0].values[232..<(232 + horizon)]))
            #expect(request.timestamps == Array(dataset.timestamps[400..<(400 + horizon)]))
        }
        for (origin, horizon) in [(335, 1), (-1, 16), (500, 1), (499, 16), (400, 24), (Int.max, 16), (400, Int.max)] {
            #expect(throws: ForecastFailure.invalidRequest) {
                try dataset.request(seriesID: BikeDataset.seriesID, originIndex: origin, horizon: horizon)
            }
        }
    }

    @Test("B2 Whole-hour origins respect first and final days and a larger horizon clamps without predicting")
    func originBounds() throws {
        let dataset = try bikeFixture()
        #expect(dataset.origins(on: dataset.timestamps[336], horizon: 16) == 336...359)
        #expect(dataset.origins(on: dataset.timestamps[499], horizon: 16) == 480...484)
        #expect(dataset.origins(on: dataset.timestamps[499], horizon: 1) == 480...499)
        #expect(dataset.origins(on: dataset.timestamps[0], horizon: 1) == nil)
        for invalid in [Double.nan, .infinity, -Double.infinity, 1e30] {
            #expect(dataset.origins(on: Date(timeIntervalSince1970: invalid), horizon: 16) == nil)
        }
        var state = ForecastFeature.State(kind: .seoulBike, liveAvailable: true)
        #expect(ForecastFeature.reduce(&state, .appear).isEmpty)
        #expect(ForecastFeature.reduce(&state, .predict).isEmpty)
        #expect(ForecastFeature.reduce(&state, .loadData) == [.run(.load(1, refresh: false), id: .operation)])
        #expect(ForecastFeature.reduce(&state, .loaded(1, dataset)).isEmpty)
        #expect(state.phase == .ready && state.kind == .seoulBike && state.horizon == 16)
        #expect(ForecastFeature.reduce(&state, .selectHorizon(1)).isEmpty)
        #expect(ForecastFeature.reduce(&state, .selectOrigin(499)).isEmpty)
        #expect(ForecastFeature.reduce(&state, .selectHorizon(16)).isEmpty)
        #expect(state.originIndex == 484 && state.result == nil && state.canPredict)
        #expect(ForecastFeature.reduce(&state, .selectHorizon(24)).isEmpty && state.horizon == 16)
        #expect(ForecastFeature.reduce(&state, .selectOrigin(499)).isEmpty && state.originIndex == 484)
        let request = try #require(state.request)
        let ticket = state.ticket + 1
        #expect(ForecastFeature.reduce(&state, .predict) == [.run(.predict(ticket, request), id: .operation)])
        let result = try ForecastResult(
            request: request,
            raw: ChronosPrediction(quantiles: Array(repeating: Array(repeating: -1, count: 16), count: 21)),
            identity: syntheticIdentity(), seconds: 1, loaded: true, timingBoundary: "Synthetic")
        _ = ForecastFeature.reduce(&state, .stop)
        _ = ForecastFeature.reduce(&state, .predicted(ticket, result))
        #expect(state.phase == .stopping && state.result == nil)
        _ = ForecastFeature.reduce(&state, .released(state.ticket))
        #expect(state.phase == .ready)
    }

    @Test("B2 A dataset from the other kind fails visibly and does not strand loading", arguments: [false, true])
    func mismatchedKind(_ bike: Bool) throws {
        var state = ForecastFeature.State(kind: bike ? .seoulBike : .productSales)
        _ = ForecastFeature.reduce(&state, .appear)
        _ = ForecastFeature.reduce(&state, .loadData)
        _ = ForecastFeature.reduce(&state, .loaded(state.ticket, try bike ? syntheticDataset() : bikeFixture()))
        #expect(state.phase == .idle && state.error == .invalidData && state.dataset == nil && !state.canPredict)
        _ = ForecastFeature.reduce(&state, .loadData)
        _ = ForecastFeature.reduce(&state, .loaded(state.ticket, try bike ? bikeFixture() : syntheticDataset()))
        #expect(state.phase == .ready && state.error == nil && state.request?.kind == state.kind)
        #expect(ForecastFeature.reduce(&state, .predict).isEmpty)
    }

    @Test("B2 Calendar admission rejects minutes, broken hours and out-of-domain dates before arithmetic")
    func admission() throws {
        let dataset = try bikeFixture()
        for invalid in [
            Double.nan, .infinity, -Double.infinity, 1e30, dataset.timestamps[0].timeIntervalSince1970 + 60,
        ] {
            var dates = dataset.timestamps
            dates[0] = Date(timeIntervalSince1970: invalid)
            #expect(throws: ForecastFailure.invalidData) {
                try ForecastDataset(kind: .seoulBike, timestamps: dates, series: dataset.series)
            }
        }
        var dates = dataset.timestamps
        dates[400] = dates[399]
        #expect(throws: ForecastFailure.invalidData) {
            try ForecastDataset(kind: .seoulBike, timestamps: dates, series: dataset.series)
        }
        #expect(throws: ForecastFailure.invalidRequest) {
            try syntheticDataset().pastWeather(originIndex: 400, count: 24)
        }
    }
}

func bikeFixture() throws -> ForecastDataset {
    let calendar = try ForecastKind.seoulBike.calendar()
    let start = try #require(calendar.date(from: DateComponents(year: 2017, month: 12, day: 1)))
    let dates = try (0..<500).map { try #require(calendar.date(byAdding: .hour, value: $0, to: start)) }
    var counts: [Float?] = (0..<500).map { Float($0) }
    counts[233] = nil
    counts[401] = 0
    return try ForecastDataset(
        kind: .seoulBike, timestamps: dates, series: [ForecastSeries(id: BikeDataset.seriesID, values: counts)])
}
