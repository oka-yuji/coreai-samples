import FeatureStore
import Foundation

public enum ForecastFeature: Feature {
    public enum Phase: Sendable, Equatable { case idle, loading, ready, predicting, stopping }

    public struct State: Sendable, Equatable {
        public let kind: ForecastKind
        public var dataset: ForecastDataset?
        public var seriesID = "B1_1"
        public var originIndex = 365
        public var horizon = 14
        public var result: ForecastResult?
        public var phase = Phase.idle
        public var error: ForecastFailure?
        public var ticket = 0
        public var active = false
        public var ended = false
        public let liveAvailable: Bool

        public init(kind: ForecastKind = .productSales, liveAvailable: Bool = false) {
            self.kind = kind
            self.liveAvailable = liveAvailable
            seriesID = kind == .productSales ? "B1_1" : BikeDataset.seriesID
            originIndex = kind.contextCount
            horizon = kind == .productSales ? 14 : 16
        }
        public var isWorking: Bool { phase == .loading || phase == .predicting }
        public var canSelect: Bool { active && phase == .ready && !ended }
        public var request: ForecastRequest? {
            guard dataset?.kind == kind else { return nil }
            return try? dataset?.request(seriesID: seriesID, originIndex: originIndex, horizon: horizon)
        }
        public var canPredict: Bool {
            canSelect && liveAvailable && request?.context.contains(where: { $0 != nil }) == true
        }
    }

    public enum Action: Sendable {
        case appear
        case loadData
        case loaded(Int, ForecastDataset)
        case selectSeries(String)
        case selectOrigin(Int)
        case selectHorizon(Int)
        case predict
        case predicted(Int, ForecastResult)
        case failed(Int, ForecastFailure)
        case cancelled(Int)
        case stop
        case suspend
        case ended
        case released(Int)
    }

    public enum Effect: Equatable, Sendable {
        case load(Int, refresh: Bool)
        case predict(Int, ForecastRequest)
        case release(Int)
    }

    public enum EffectID: Hashable, Sendable { case operation, release }

    public static func reduce(_ state: inout State, _ action: Action) -> [Command<Effect, EffectID>] {
        guard !state.ended else { return [] }
        switch action {
        case .appear:
            state.active = true
        case .loadData:
            guard state.active, state.phase == .idle || state.phase == .ready else { return [] }
            let refresh =
                switch state.error {
                case .invalidData, .sourceMismatch, .responseTooLarge, .network, .cache: true
                case .invalidRequest, .unavailable, .predictionFailed, .busy, nil: false
                }
            state.ticket += 1
            state.phase = .loading
            state.error = nil
            state.result = nil
            return [.run(.load(state.ticket, refresh: refresh), id: .operation)]
        case .loaded(let ticket, let dataset):
            guard state.active, ticket == state.ticket, state.phase == .loading else { return [] }
            guard dataset.kind == state.kind, let first = dataset.series.first,
                let origins = dataset.origins(horizon: state.horizon)
            else {
                state.phase = state.dataset == nil ? .idle : .ready
                state.error = .invalidData
                return []
            }
            state.dataset = dataset
            state.seriesID = first.id
            state.originIndex = origins.upperBound
            state.phase = .ready
            state.error = nil
        case .selectSeries(let id):
            guard state.canSelect, state.dataset?.series.contains(where: { $0.id == id }) == true else { return [] }
            state.seriesID = id
            clear(&state)
        case .selectOrigin(let index):
            guard state.canSelect, state.dataset?.origins(horizon: state.horizon)?.contains(index) == true else {
                return []
            }
            state.originIndex = index
            clear(&state)
        case .selectHorizon(let horizon):
            guard state.canSelect, let origins = state.dataset?.origins(horizon: horizon) else { return [] }
            state.horizon = horizon
            state.originIndex = min(state.originIndex, origins.upperBound)
            clear(&state)
        case .predict:
            guard state.canPredict, let request = state.request else { return [] }
            clear(&state)
            state.phase = .predicting
            return [.run(.predict(state.ticket, request), id: .operation)]
        case .predicted(let ticket, let result):
            guard state.active, ticket == state.ticket, state.phase == .predicting else { return [] }
            state.phase = .ready
            guard result.request == state.request else {
                state.error = .predictionFailed
                return []
            }
            state.result = result
        case .failed(let ticket, let error):
            guard state.active, ticket == state.ticket, state.isWorking else { return [] }
            state.phase = state.dataset == nil ? .idle : .ready
            state.error = error
        case .cancelled(let ticket):
            guard state.active, ticket == state.ticket, state.isWorking else { return [] }
            state.phase = state.dataset == nil ? .idle : .ready
        case .stop:
            guard state.active, state.isWorking else { return [] }
            return stopping(&state)
        case .suspend:
            guard state.active else { return [] }
            state.active = false
            return stopping(&state)
        case .ended:
            state.active = false
            state.ended = true
            return stopping(&state)
        case .released(let ticket):
            guard state.ticket == ticket, state.phase == .stopping else { return [] }
            state.phase = state.dataset == nil ? .idle : .ready
        }
        return []
    }

    private static func clear(_ state: inout State) {
        state.ticket += 1
        state.result = nil
        state.error = nil
    }

    private static func stopping(_ state: inout State) -> [Command<Effect, EffectID>] {
        clear(&state)
        state.phase = .stopping
        return [.cancelAll, .run(.release(state.ticket), id: .release)]
    }
}
