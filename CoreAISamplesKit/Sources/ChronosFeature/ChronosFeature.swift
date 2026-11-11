import FeatureStore
import Foundation

public enum ChronosFeature: Feature {
    public enum Phase: Equatable, Sendable {
        case idle
        case loading
        case ready
        case predicting
        case stopping
    }

    public struct State: Equatable, Sendable {
        public var catalog: ChronosCatalog?
        public var liveAvailable = false
        public var scenarioID = "friday-evening"
        public var areaID = "230"
        public var clockStep = 0
        public var lead = 0
        public var mode: ChronosForecastMode = .precomputed
        public var forecast: ChronosForecastBatch?
        public var phase: Phase = .idle
        public var error: ChronosDemoFailure?
        public var ticket = 0
        public var active = false
        public var ended = false

        public init() {}

        public var canPredict: Bool {
            guard active, phase == .ready, let catalog else { return false }
            return clockStep <= catalog.truthSteps - catalog.horizon && (mode == .precomputed || liveAvailable)
        }

        public var canAdvance: Bool {
            guard active, phase == .ready, let catalog, let forecast else { return false }
            return clockStep < min(catalog.truthSteps, forecast.originStep + catalog.horizon)
        }

        public var selectedArea: ChronosArea? { catalog?.areas.first { $0.id == areaID } }
        public var selectedScenario: ChronosScenario? { catalog?.scenario(scenarioID) }
        public var selectedForecast: ChronosAreaForecast? { forecast?.area(areaID) }
        public var displayStep: Int { forecast.map { $0.originStep + lead } ?? (clockStep - 1) }
        public var displayStart: Date? { catalog?.time(at: displayStep, scenarioID: scenarioID) }
        public var displayEnd: Date? { catalog?.time(at: displayStep + 1, scenarioID: scenarioID) }

        public func displayedValue(areaID: String) -> Float? {
            if let prediction = forecast?.area(areaID), prediction.median.indices.contains(lead) {
                return prediction.median[lead]
            }
            return catalog?.history(scenarioID: scenarioID, areaID: areaID, step: clockStep)?.last
        }

        public var selectedTruth: Float? {
            guard let forecast else { return nil }
            return catalog?.revealedValue(
                scenarioID: scenarioID, areaID: areaID,
                targetStep: forecast.originStep + lead, clockStep: clockStep)
        }
    }

    public enum Action: Sendable {
        case appear
        case suspend
        case ended
        case loaded(Int, ChronosLoadedDemo)
        case failed(Int, ChronosDemoFailure)
        case cancelled(Int)
        case released(Int)
        case selectScenario(String)
        case selectArea(String)
        case selectLead(Int)
        case selectMode(ChronosForecastMode)
        case predict
        case cancelPrediction
        case predicted(Int, ChronosForecastBatch)
        case advance
        case reset
        case dismissError
    }

    public enum Effect: Equatable, Sendable {
        case load(Int)
        case predict(Int, ChronosForecastRequest)
        case release(Int)
    }

    public enum EffectID: Hashable, Sendable {
        case load
        case prediction
        case release
    }

    public static func reduce(_ state: inout State, _ action: Action) -> [Command<Effect, EffectID>] {
        guard !state.ended else { return [] }
        switch action {
        case .appear:
            state.active = true
            if state.phase == .stopping { return [] }
            guard state.phase == .idle else { return [] }
            state.ticket += 1
            state.phase = .loading
            return [.run(.load(state.ticket), id: .load)]
        case .ended:
            let commands = reduce(&state, .suspend)
            state.ended = true
            return commands
        case .suspend:
            guard state.active else { return [] }
            state.active = false
            state.ticket += 1
            state.phase = .stopping
            return [.cancelAll, .run(.release(state.ticket), id: .release)]
        case .released(let ticket):
            guard ticket == state.ticket, state.phase == .stopping else { return [] }
            state.phase = state.catalog == nil ? .idle : .ready
            if state.active, state.catalog == nil { return reduce(&state, .appear) }
        case .loaded(let ticket, let demo):
            guard ticket == state.ticket, state.active, state.phase == .loading else { return [] }
            state.catalog = demo.catalog
            state.liveAvailable = demo.liveAvailable
            state.mode = demo.liveAvailable ? .live : .precomputed
            if let scenario = demo.catalog.scenarios.first {
                state.scenarioID = scenario.id
                state.areaID = scenario.focusArea
            }
            state.phase = .ready
            state.error = nil
        case .selectScenario(let id):
            guard state.active, state.phase == .ready, let scenario = state.catalog?.scenario(id) else { return [] }
            state.scenarioID = id
            state.areaID = scenario.focusArea
            state.clockStep = 0
            state.forecast = nil
            state.lead = 0
            state.error = nil
            state.ticket += 1
        case .selectArea(let id):
            guard state.active, state.catalog?.areas.contains(where: { $0.id == id }) == true else { return [] }
            state.areaID = id
        case .selectLead(let lead):
            guard state.active, (0..<4).contains(lead) else { return [] }
            state.lead = lead
        case .selectMode(let mode):
            guard state.active, state.phase == .ready, mode == .precomputed || state.liveAvailable else { return [] }
            state.mode = mode
            state.forecast = nil
            state.error = nil
            state.ticket += 1
        case .cancelPrediction:
            guard state.active, state.phase == .predicting else { return [] }
            state.ticket += 1
            state.phase = .stopping
            return [.cancelAll, .run(.release(state.ticket), id: .release)]
        case .predict:
            guard state.canPredict,
                let request = state.catalog?.request(
                    scenarioID: state.scenarioID, step: state.clockStep, mode: state.mode)
            else { return [] }
            state.ticket += 1
            state.phase = .predicting
            state.error = nil
            return [.run(.predict(state.ticket, request), id: .prediction)]
        case .predicted(let ticket, let batch):
            guard ticket == state.ticket, state.active, state.phase == .predicting else { return [] }
            let expected = Set(state.catalog?.areas.map(\.id) ?? [])
            let validSource: Bool
            switch batch.mode {
            case .live:
                validSource = batch.identity != nil && batch.areas.allSatisfy { $0.raw != nil }
            case .precomputed:
                validSource = batch.processingSeconds == nil && batch.areas.allSatisfy { $0.raw == nil }
            }
            guard batch.scenarioID == state.scenarioID, batch.originStep == state.clockStep, batch.mode == state.mode,
                Set(batch.areas.map(\.areaID)) == expected, batch.areas.count == expected.count, validSource
            else {
                state.phase = .ready
                state.error = .predictionFailed
                return []
            }
            state.forecast = batch
            state.lead = 0
            state.phase = .ready
        case .failed(let ticket, let failure):
            guard ticket == state.ticket, state.active, state.phase == .loading || state.phase == .predicting else {
                return []
            }
            state.phase = state.catalog == nil ? .idle : .ready
            state.error = failure
        case .cancelled(let ticket):
            guard ticket == state.ticket, state.active else { return [] }
            if state.phase == .predicting { state.phase = .ready }
            if state.phase == .loading { state.phase = .idle }
        case .advance:
            guard state.canAdvance else { return [] }
            state.clockStep += 1
            if let forecast = state.forecast { state.lead = min(3, state.clockStep - forecast.originStep - 1) }
        case .reset:
            guard state.active, state.phase == .ready else { return [] }
            state.clockStep = 0
            state.lead = 0
            state.forecast = nil
            state.error = nil
            state.ticket += 1
        case .dismissError:
            state.error = nil
        }
        return []
    }
}
