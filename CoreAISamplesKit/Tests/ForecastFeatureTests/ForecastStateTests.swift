import ChronosCore
import FeatureStore
import Testing

@testable import ForecastFeature

struct ForecastStateTests {
    @Test("S2 Opening and changing selections never download or infer and discard stale predictions")
    func explicitActions() throws {
        var state = ForecastFeature.State(liveAvailable: true)
        #expect(ForecastFeature.reduce(&state, .appear).isEmpty && state.phase == .idle)
        #expect(ForecastFeature.reduce(&state, .predict).isEmpty)
        let commands = ForecastFeature.reduce(&state, .loadData)
        #expect(commands == [.run(.load(1, refresh: false), id: .operation)])
        let data = try syntheticDataset()
        #expect(ForecastFeature.reduce(&state, .loaded(1, data)).isEmpty)
        #expect(state.phase == .ready && state.result == nil && state.canPredict)
        #expect(ForecastFeature.reduce(&state, .selectOrigin(400)).isEmpty)
        let request = try #require(state.request)
        #expect(ForecastFeature.reduce(&state, .predict) == [.run(.predict(3, request), id: .operation)])
        let result = try ForecastResult(
            request: request,
            raw: ChronosPrediction(quantiles: Array(repeating: Array(repeating: 4, count: 16), count: 21)),
            identity: syntheticIdentity(), seconds: 2, loaded: true, timingBoundary: "Synthetic")
        _ = ForecastFeature.reduce(&state, .predicted(3, result))
        #expect(state.result == result && state.phase == .ready)
        #expect(ForecastFeature.reduce(&state, .selectSeries("B2_1")).isEmpty && state.result == nil)
        _ = ForecastFeature.reduce(&state, .predicted(3, result))
        #expect(state.result == nil)
        _ = ForecastFeature.reduce(&state, .selectHorizon(7))
        _ = ForecastFeature.reduce(&state, .selectOrigin(493))
        _ = ForecastFeature.reduce(&state, .selectHorizon(14))
        #expect(state.originIndex == 486 && state.horizon == 14)
        _ = ForecastFeature.reduce(&state, .selectHorizon(28))
        #expect(state.horizon == 14)
    }

    @Test("S2 Failure retries explicitly, stop joins before retry and old tickets cannot revive a closed session")
    func lifecycle() throws {
        var state = ForecastFeature.State(liveAvailable: true)
        _ = ForecastFeature.reduce(&state, .appear)
        _ = ForecastFeature.reduce(&state, .loadData)
        _ = ForecastFeature.reduce(&state, .failed(1, .network))
        #expect(state.error == .network && state.phase == .idle)
        #expect(ForecastFeature.reduce(&state, .loadData) == [.run(.load(2, refresh: true), id: .operation)])
        #expect(ForecastFeature.reduce(&state, .stop) == [.cancelAll, .run(.release(3), id: .release)])
        _ = ForecastFeature.reduce(&state, .loaded(2, try syntheticDataset()))
        #expect(state.dataset == nil && state.phase == .stopping && state.error == nil)
        #expect(ForecastFeature.reduce(&state, .loadData).isEmpty)
        _ = ForecastFeature.reduce(&state, .released(2))
        #expect(state.phase == .stopping)
        _ = ForecastFeature.reduce(&state, .released(3))
        #expect(state.phase == .idle)
        _ = ForecastFeature.reduce(&state, .loadData)
        _ = ForecastFeature.reduce(&state, .loaded(4, try syntheticDataset()))
        _ = ForecastFeature.reduce(&state, .predict)
        _ = ForecastFeature.reduce(&state, .suspend)
        _ = ForecastFeature.reduce(&state, .appear)
        #expect(state.phase == .stopping && state.active)
        _ = ForecastFeature.reduce(&state, .released(state.ticket))
        #expect(state.phase == .ready && state.result == nil)
        #expect(ForecastFeature.reduce(&state, .ended).first == .cancelAll)
        let ended = state
        _ = ForecastFeature.reduce(&state, .appear)
        _ = ForecastFeature.reduce(&state, .loaded(state.ticket, try syntheticDataset()))
        #expect(state == ended)
    }

    @Test("S2 Preview can inspect data but never requests a native or substitute prediction")
    func preview() throws {
        var state = ForecastFeature.State()
        _ = ForecastFeature.reduce(&state, .appear)
        _ = ForecastFeature.reduce(&state, .loadData)
        _ = ForecastFeature.reduce(&state, .loaded(1, try syntheticDataset()))
        #expect(!state.canPredict && ForecastFeature.reduce(&state, .predict).isEmpty)
    }

    @Test(
        "S2 Reloading data after a prediction failure preserves the verified offline cache",
        arguments: [ForecastFailure.predictionFailed, .unavailable, .invalidRequest, .busy])
    func predictionFailureKeepsCache(_ failure: ForecastFailure) throws {
        var state = ForecastFeature.State(liveAvailable: true)
        _ = ForecastFeature.reduce(&state, .appear)
        _ = ForecastFeature.reduce(&state, .loadData)
        _ = ForecastFeature.reduce(&state, .loaded(1, try syntheticDataset()))
        _ = ForecastFeature.reduce(&state, .predict)
        _ = ForecastFeature.reduce(&state, .failed(state.ticket, failure))
        let next = state.ticket + 1
        #expect(ForecastFeature.reduce(&state, .loadData) == [.run(.load(next, refresh: false), id: .operation)])
    }
}
