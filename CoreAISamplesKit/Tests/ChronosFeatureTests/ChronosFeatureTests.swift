import ChronosCore
import ChronosFeature
import FeatureStore
import Foundation
import Testing

struct ChronosFeatureTests {
    @Test("D1 Area and interval selection use the same fixed forecast origin without running effects")
    func selection() throws {
        var state = ready()
        let request = try start(&state)
        let batch = try batch(request)
        _ = ChronosFeature.reduce(&state, .predicted(state.ticket, batch))
        let areaCommands = ChronosFeature.reduce(&state, .selectArea("b"))
        let leadCommands = ChronosFeature.reduce(&state, .selectLead(2))
        #expect(areaCommands.isEmpty && leadCommands.isEmpty)
        #expect(state.displayStep == 2 && state.selectedForecast?.areaID == "b")
        #expect(state.displayedValue(areaID: "b") == 102)
        #expect(state.displayEnd?.timeIntervalSince(state.displayStart ?? .distantPast) == 1800)
        #expect(state.selectedTruth == nil)
        _ = ChronosFeature.reduce(&state, .selectScenario("second"))
        #expect(state.forecast == nil && state.clockStep == 0 && state.areaID == "b" && state.lead == 0)
    }

    @Test("D2 Advancing reveals one elapsed bin and a new request appends only elapsed truth")
    func advances() throws {
        var state = ready()
        let first = try start(&state)
        _ = ChronosFeature.reduce(&state, .predicted(state.ticket, try batch(first)))
        let original = state.forecast
        _ = ChronosFeature.reduce(&state, .advance)
        #expect(state.forecast == original && state.forecast?.originStep == 0)
        #expect(state.clockStep == 1 && state.displayStep == 0 && state.selectedTruth == 100)
        _ = ChronosFeature.reduce(&state, .selectLead(1))
        #expect(state.selectedTruth == nil && state.displayStep == 1)
        let next = try start(&state)
        #expect(next.originStep == 1 && next.series.first?.values == [2, 3, 100])
        _ = ChronosFeature.reduce(&state, .predicted(state.ticket, try batch(next)))
        #expect(state.displayStep == 1 && state.selectedTruth == nil)
        _ = ChronosFeature.reduce(&state, .reset)
        #expect(state.forecast == nil && state.clockStep == 0)
    }

    @Test("D2 The last available four-step forecast starts at step twelve")
    func endBoundary() throws {
        var state = ready()
        for step in 0...12 {
            let request = try start(&state)
            #expect(request.originStep == step)
            _ = ChronosFeature.reduce(&state, .predicted(state.ticket, try batch(request)))
            _ = ChronosFeature.reduce(&state, .advance)
        }
        #expect(!state.canPredict)
        for _ in 0..<3 { _ = ChronosFeature.reduce(&state, .advance) }
        #expect(state.clockStep == 16 && !state.canAdvance && state.selectedTruth == 115)
        #expect(state.forecast?.originStep == 12 && state.displayStep == 15)
    }

    @Test("D3 Cancellation joins stop before allowing another request and ignores stale results")
    func cancelAndOldTickets() throws {
        var state = ready()
        let request = try start(&state)
        let old = state.ticket
        let commands = ChronosFeature.reduce(&state, .cancelPrediction)
        #expect(commands == [.cancelAll, .run(.release(state.ticket), id: .release)])
        #expect(!state.canPredict && state.phase == .stopping)
        _ = ChronosFeature.reduce(&state, .predicted(old, try batch(request)))
        _ = ChronosFeature.reduce(&state, .failed(old, .predictionFailed))
        _ = ChronosFeature.reduce(&state, .released(old))
        #expect(state.phase == .stopping && state.forecast == nil && state.error == nil)
        _ = ChronosFeature.reduce(&state, .released(state.ticket))
        #expect(state.canPredict)
        let stopped = state.ticket
        _ = try start(&state)
        _ = ChronosFeature.reduce(&state, .released(stopped))
        #expect(state.phase == .predicting)
    }

    @Test("D3 An ended owner never accepts appearance or results again")
    func ended() throws {
        var state = ready()
        let request = try start(&state)
        let ticket = state.ticket
        let commands = ChronosFeature.reduce(&state, .ended)
        #expect(commands == [.cancelAll, .run(.release(state.ticket), id: .release)])
        let ended = state
        for action in [
            ChronosFeature.Action.appear, .predicted(ticket, try batch(request)), .released(state.ticket), .reset,
            .selectArea("b"),
        ] {
            let result = ChronosFeature.reduce(&state, action)
            #expect(result.isEmpty && state == ended)
        }
    }

    @Test("D4 Missing models use explicit preview and native failure never changes mode")
    func modes() throws {
        var preview = ready()
        #expect(preview.mode == .precomputed && !preview.liveAvailable)
        _ = ChronosFeature.reduce(&preview, .selectMode(.live))
        #expect(preview.mode == .precomputed)
        var live = ready(live: true)
        let request = try start(&live)
        _ = ChronosFeature.reduce(&live, .failed(live.ticket, .predictionFailed))
        #expect(live.mode == .live && live.error == .predictionFailed && live.forecast == nil)
        _ = ChronosFeature.reduce(&live, .selectMode(.precomputed))
        #expect(live.mode == .precomputed && live.forecast == nil && live.error == nil)
        #expect(request.mode == .live)
    }

    @Test("D1 Results with wrong origins, modes or area inventories are rejected")
    func wrongResults() throws {
        for variant in 0..<4 {
            var state = ready()
            let request = try start(&state)
            let correct = try batch(request)
            let invalid = ChronosForecastBatch(
                scenarioID: variant == 0 ? "wrong" : request.scenarioID,
                originStep: variant == 1 ? 1 : 0, mode: variant == 2 ? .live : .precomputed,
                areas: variant == 3 ? [correct.areas[0], correct.areas[0]] : correct.areas, processingSeconds: nil)
            _ = ChronosFeature.reduce(&state, .predicted(state.ticket, invalid))
            #expect(state.forecast == nil && state.error == .predictionFailed)
        }
    }

    @Test("D3 A preview can resume only after its pending stop acknowledgement")
    func previewReappears() {
        var state = ready()
        _ = ChronosFeature.reduce(&state, .suspend)
        _ = ChronosFeature.reduce(&state, .appear)
        #expect(!state.canPredict && state.phase == .stopping)
        _ = ChronosFeature.reduce(&state, .released(state.ticket))
        #expect(state.canPredict)
    }

    @Test("D2 History preserves missing values and rejects invalid or empty retained windows")
    func history() throws {
        let missing = try #require(
            ChronosHistoryWindow.make(history: [1, .nan], truth: [5, .infinity], revealed: 1, maxLength: 3))
        #expect(missing[0] == 1 && missing[1].isNaN && missing[2] == 5)
        #expect(ChronosHistoryWindow.make(history: [1, .nan], truth: [], revealed: 0, maxLength: 1) == nil)
        #expect(ChronosHistoryWindow.make(history: [], truth: [1], revealed: 1, maxLength: 3) == nil)
        #expect(ChronosHistoryWindow.make(history: [1], truth: [], revealed: 0, maxLength: 0) == nil)
        #expect(ChronosHistoryWindow.make(history: [1], truth: [.infinity], revealed: 1, maxLength: 2) == nil)
    }

    @Test("Raw native quantiles stay unchanged when the display clips negative counts")
    func rawOutput() throws {
        let raw = try ChronosPrediction(quantiles: Array(repeating: Array(repeating: -2, count: 16), count: 21))
        let area = ChronosAreaForecast(areaID: "a", raw: raw)
        #expect(area.raw == raw && area.raw?.quantiles[20][15] == -2)
        #expect(area.lower == [0, 0, 0, 0] && area.median == [0, 0, 0, 0] && area.upper == [0, 0, 0, 0])
    }

    private func ready(live: Bool = false) -> ChronosFeature.State {
        let areas = ["a", "b"].map {
            ChronosArea(id: $0, name: .init(ja: $0, en: $0), district: "Manhattan", grid: .init(x: 0, y: 0))
        }
        let scenes = ["first", "second"].map {
            ChronosScenario(
                id: $0, name: .init(ja: $0, en: $0), focusArea: $0 == "first" ? "a" : "b",
                history: ["a": [1, 2, 3], "b": [4, 5, 6]],
                truth: ["a": (100..<116).map(Float.init), "b": (200..<216).map(Float.init)],
                origin: Date(timeIntervalSince1970: 0))
        }
        let catalog = ChronosCatalog(
            areas: areas, scenarios: scenes, origin: Date(timeIntervalSince1970: 0), intervalSeconds: 1800,
            historySteps: 3, truthSteps: 16)
        var state = ChronosFeature.State()
        _ = ChronosFeature.reduce(&state, .appear)
        _ = ChronosFeature.reduce(
            &state, .loaded(state.ticket, ChronosLoadedDemo(catalog: catalog, liveAvailable: live)))
        return state
    }

    private func start(_ state: inout ChronosFeature.State) throws -> ChronosForecastRequest {
        let commands = ChronosFeature.reduce(&state, .predict)
        guard case .run(.predict(_, let request), id: .prediction) = try #require(commands.first) else {
            throw ChronosDemoFailure.invalidData
        }
        return request
    }

    private func batch(_ request: ChronosForecastRequest) throws -> ChronosForecastBatch {
        let areas = try request.series.enumerated().map { index, series in
            try ChronosAreaForecast(
                areaID: series.areaID, lower: [1, 2, 3, 4], median: (0..<4).map { Float(index * 100 + $0) },
                upper: [300, 301, 302, 303])
        }
        return ChronosForecastBatch(
            scenarioID: request.scenarioID, originStep: request.originStep, mode: request.mode, areas: areas,
            processingSeconds: nil)
    }
}
