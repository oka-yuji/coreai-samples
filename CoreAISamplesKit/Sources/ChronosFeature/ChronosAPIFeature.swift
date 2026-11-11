import FeatureStore
import Foundation

extension ChronosFeature {
    static func reduceAPI(_ state: inout State, _ action: Action) -> [Command<Effect, EffectID>] {
        switch action {
        case .selectSource(let source):
            guard state.active, source != state.source else { return [] }
            state.source = source
            let commands = invalidateAcquisition(&state)
            if source == .fixed, let catalog = state.fixedCatalog, let scene = catalog.scenarios.first {
                state.catalog = catalog
                state.scenarioID = scene.id
                state.clockStep = 0
                state.lead = 0
                state.forecast = nil
            }
            return commands
        case .selectDate(let date):
            guard state.active, state.source == .api, date.timeIntervalSince1970.isFinite,
                date == NYCTime.aligned(date), state.coverage?.allowsHistory(endingAt: date) == true,
                date != state.requestedOrigin
            else { return [] }
            state.requestedOrigin = date
            return invalidateAcquisition(&state)
        case .refreshCoverage:
            guard state.active, state.phase == .ready, state.source == .api else { return [] }
            state.ticket += 1
            state.error = nil
            state.phase = .fetchingCoverage
            return [.run(.coverage(state.ticket), id: .acquisition)]
        case .coverageLoaded(let ticket, let coverage):
            guard accepts(state, ticket, .fetchingCoverage) else { return [] }
            state.coverage = coverage
            if state.requestedOrigin == nil {
                state.requestedOrigin =
                    state.catalog.map(\.origin).flatMap {
                        coverage.allowsHistory(endingAt: $0) ? $0 : nil
                    } ?? coverage.selectableRange?.upperBound
            }
            state.phase = .ready
        case .fetchHistory:
            guard state.canFetchHistory, let date = state.requestedOrigin, let catalog = state.fixedCatalog else {
                return []
            }
            state.ticket += 1
            state.phase = .fetchingHistory
            state.error = nil
            return [.run(.history(state.ticket, date, catalog.areas.map(\.id)), id: .acquisition)]
        case .historyLoaded(let ticket, let window):
            guard accepts(state, ticket, .fetchingHistory) else { return [] }
            guard window.end == state.requestedOrigin,
                let catalog = try? state.fixedCatalog?.replacingWithAPI(window), let scene = catalog.scenarios.first
            else { return reduce(&state, .failed(ticket, .invalidData)) }
            state.catalog = catalog
            state.scenarioID = scene.id
            state.clockStep = 0
            state.lead = 0
            state.forecast = nil
            state.phase = .ready
        case .fetchActuals:
            guard state.canFetchActuals, let catalog = state.catalog else { return [] }
            state.ticket += 1
            state.phase = .fetchingActuals
            state.error = nil
            return [.run(.actuals(state.ticket, catalog.origin, catalog.areas.map(\.id)), id: .acquisition)]
        case .actualsLoaded(let ticket, let window):
            guard accepts(state, ticket, .fetchingActuals) else { return [] }
            guard let history = state.selectedScenario?.apiHistory,
                let catalog = try? state.catalog?.replacingWithAPI(history, actuals: window)
            else { return reduce(&state, .failed(ticket, .invalidData)) }
            state.catalog = catalog
            state.phase = .ready
        default: break
        }
        return []
    }

    private static func accepts(_ state: State, _ ticket: Int, _ phase: Phase) -> Bool {
        state.active && state.source == .api && state.ticket == ticket && state.phase == phase
    }

    private static func invalidateAcquisition(_ state: inout State) -> [Command<Effect, EffectID>] {
        state.error = nil
        guard state.phase != .stopping else { return [] }
        state.ticket += 1
        guard state.isWorking else { return [] }
        state.phase = .stopping
        return [.cancelAll, .run(.release(state.ticket), id: .release)]
    }
}
