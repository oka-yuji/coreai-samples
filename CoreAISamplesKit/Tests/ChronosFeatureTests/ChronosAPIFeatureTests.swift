import ChronosCore
import ChronosFeature
import FeatureStore
import Foundation
import Testing

struct ChronosAPIFeatureTests {
    @Test("A2 Missing masks remain null and only elapsed actuals enter the next input")
    func masks() throws {
        let history = try window(count: 1008, missing: [100])
        let actuals = try window(start: history.end, count: 16, missing: [0])
        let original = try ForecastInputSnapshot(history: history, actuals: actuals, step: 0)
        let next = try ForecastInputSnapshot(history: history, actuals: actuals, step: 1)
        #expect(original.actuals == nil && original.series[0].values[100] == nil)
        #expect(next.series[0].values.last == .some(nil))
        #expect(next.series[0].values[99] == nil && next.series[0].values[100] == 1)
        #expect(next.origin == history.end.addingTimeInterval(1800))
        let encoded = try JSONEncoder().encode(original)
        #expect(String(decoding: encoded, as: UTF8.self).contains("null"))
        #expect(!String(decoding: encoded, as: UTF8.self).contains("NaN"))
    }

    @Test(
        "A3 Selection changes cancel pending work and retain the previous valid display",
        arguments: ["date", "source", "suspend", "ended"])
    func stale(_ change: String) throws {
        var state = try ready()
        let previous = state.catalog
        _ = ChronosFeature.reduce(&state, .selectSource(.api))
        _ = ChronosFeature.reduce(&state, .refreshCoverage)
        _ = ChronosFeature.reduce(&state, .coverageLoaded(state.ticket, try coverage()))
        _ = ChronosFeature.reduce(&state, .fetchHistory)
        let old = state.ticket
        let action: ChronosFeature.Action
        switch change {
        case "date": action = .selectDate(try window(count: 1008).end.addingTimeInterval(1800))
        case "source": action = .selectSource(.fixed)
        case "suspend": action = .suspend
        default: action = .ended
        }
        let commands = ChronosFeature.reduce(&state, action)
        #expect(commands == [.cancelAll, .run(.release(state.ticket), id: .release)])
        _ = ChronosFeature.reduce(&state, .historyLoaded(old, try window(count: 1008)))
        #expect(state.catalog == previous && state.phase == .stopping)
        if change == "date" {
            _ = ChronosFeature.reduce(&state, .released(state.ticket))
            _ = ChronosFeature.reduce(&state, .fetchHistory)
            let selected = try #require(state.requestedOrigin)
            _ = ChronosFeature.reduce(
                &state,
                .historyLoaded(
                    state.ticket,
                    try window(start: selected.addingTimeInterval(-1008 * 1800), count: 1008)))
            let fresh = state.catalog
            _ = ChronosFeature.reduce(&state, .historyLoaded(old, try window(count: 1008)))
            _ = ChronosFeature.reduce(&state, .failed(old, .network))
            #expect(state.catalog == fresh && state.catalog?.origin == selected && state.error == nil)
        }
    }

    @Test("A3 Failed history and coverage refresh retain the prior forecast and mode")
    func failedRefresh() throws {
        var state = try ready()
        _ = ChronosFeature.reduce(&state, .selectSource(.api))
        _ = ChronosFeature.reduce(&state, .refreshCoverage)
        let coverage = try coverage()
        _ = ChronosFeature.reduce(&state, .coverageLoaded(state.ticket, coverage))
        let previous = state.catalog
        _ = ChronosFeature.reduce(&state, .fetchHistory)
        _ = ChronosFeature.reduce(&state, .failed(state.ticket, .network))
        #expect(state.catalog == previous && state.coverage == coverage && state.mode == .live)
        #expect(state.error == .network && state.phase == .ready)
        _ = ChronosFeature.reduce(&state, .refreshCoverage)
        _ = ChronosFeature.reduce(&state, .failed(state.ticket, .rateLimited))
        #expect(state.coverage == coverage && state.catalog == previous)
    }

    @Test("A4 API input rejects previews and actuals remain an explicit post-prediction action")
    func explicitPrediction() throws {
        var state = try ready()
        _ = ChronosFeature.reduce(&state, .selectSource(.api))
        _ = ChronosFeature.reduce(&state, .refreshCoverage)
        _ = ChronosFeature.reduce(&state, .coverageLoaded(state.ticket, try coverage()))
        _ = ChronosFeature.reduce(&state, .fetchHistory)
        let commands = ChronosFeature.reduce(
            &state, .historyLoaded(state.ticket, try window(count: 1008, missing: [50])))
        #expect(commands.isEmpty && !state.canFetchActuals && !state.canAdvance)
        let catalog = try #require(state.catalog)
        #expect(catalog.request(scenarioID: state.scenarioID, step: 0, mode: .precomputed) == nil)
        #expect(ChronosFeature.reduce(&state, .fetchActuals).isEmpty)
        let request = try #require(catalog.request(scenarioID: state.scenarioID, step: 0, mode: .live))
        #expect(request.series[0].values[50].isNaN)
        #expect(request == catalog.request(scenarioID: state.scenarioID, step: 0, mode: .live))
        _ = ChronosFeature.reduce(&state, .predict)
        let raw = try ChronosPrediction(quantiles: Array(repeating: Array(repeating: -2, count: 16), count: 21))
        let batch = ChronosForecastBatch(
            scenarioID: state.scenarioID, originStep: 0, mode: .live,
            areas: [ChronosAreaForecast(areaID: "230", raw: raw)], processingSeconds: 1,
            identity: ChronosBundleIdentity(
                source: .expected, manifest: .init(bytes: 1, sha256: String(repeating: "a", count: 64)), files: [:]),
            inputSnapshot: request.snapshot, inputSHA256: String(repeating: "b", count: 64))
        #expect(ChronosFeature.reduce(&state, .predicted(state.ticket, batch)).isEmpty)
        #expect(state.canFetchActuals && state.forecast?.areas[0].raw == raw)
        _ = ChronosFeature.reduce(&state, .fetchActuals)
        _ = ChronosFeature.reduce(
            &state, .actualsLoaded(state.ticket, try window(start: catalog.origin, count: 16, missing: [0])))
        #expect(state.forecast == batch && state.canAdvance && state.selectedAvailability == .ambiguousTime)
        _ = ChronosFeature.reduce(&state, .advance)
        #expect(state.clockStep == 1 && state.canPredict)
        #expect(
            state.catalog?.request(scenarioID: state.scenarioID, step: 1, mode: .live)?.series[0].values.last?.isNaN
                == true)
        #expect(state.forecast?.inputSnapshot?.actuals == nil)
        _ = ChronosFeature.reduce(&state, .predict)
        _ = ChronosFeature.reduce(&state, .failed(state.ticket, .predictionFailed))
        #expect(state.forecast == batch && state.mode == .live)
        _ = ChronosFeature.reduce(&state, .selectDate(catalog.origin.addingTimeInterval(1800)))
        _ = ChronosFeature.reduce(&state, .fetchHistory)
        _ = ChronosFeature.reduce(&state, .failed(state.ticket, .network))
        #expect(state.forecast == batch && state.catalog?.origin == catalog.origin)
    }

    private func ready() throws -> ChronosFeature.State {
        let origin = try window(count: 1008).end
        let area = ChronosArea(
            id: "230", name: .init(ja: "230", en: "230"), district: "Manhattan", grid: .init(x: 0, y: 0))
        let scene = ChronosScenario(
            id: "fixed", name: .init(ja: "Fixed", en: "Fixed"), focusArea: "230",
            history: ["230": Array(repeating: 1, count: 1008)], truth: ["230": Array(repeating: 2, count: 16)],
            origin: origin)
        let catalog = ChronosCatalog(
            areas: [area], scenarios: [scene], origin: origin, intervalSeconds: 1800, historySteps: 1008, truthSteps: 16
        )
        var state = ChronosFeature.State()
        _ = ChronosFeature.reduce(&state, .appear)
        _ = ChronosFeature.reduce(
            &state, .loaded(state.ticket, ChronosLoadedDemo(catalog: catalog, liveAvailable: true)))
        return state
    }

    private func coverage() throws -> APICoverage {
        APICoverage(
            datasets: APIVerificationPolicy.january2025.ranges.map(\.dataset), evidence: try evidence(),
            verification: APIVerificationPolicy(
                version: "feature-fixture-v1", ranges: APIVerificationPolicy.january2025.ranges))
    }

    private func evidence() throws -> APIEvidence {
        APIEvidence(
            url: try #require(URL(string: "https://api.us.socrata.com/api/catalog/v1")), status: 200,
            acquiredAt: Date(timeIntervalSince1970: 0), data: Data(), sha256: "fixture")
    }

    private func window(start: Date = Date(timeIntervalSince1970: 1_735_941_600), count: Int, missing: Set<Int> = [])
        throws -> APIWindow
    {
        try APIWindow(
            start: start, count: count,
            series: [CountSeries(areaID: "230", values: (0..<count).map { missing.contains($0) ? nil : 1 })],
            availability: (0..<count).map { missing.contains($0) ? .ambiguousTime : .observed },
            datasets: coverage().datasets, fetchedAt: Date(timeIntervalSince1970: 0), catalog: evidence(), metadata: [],
            responses: [], verification: coverage().verification)
    }
}
