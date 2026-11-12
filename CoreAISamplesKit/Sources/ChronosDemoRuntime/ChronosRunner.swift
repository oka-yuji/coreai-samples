import ChronosFeature
import FeatureStore
import Foundation

@MainActor
public final class ChronosRunner {
    private let service: any ChronosService
    private let permits: @MainActor () -> Bool
    private let journal: Store<ChronosJournalFeature>?
    private var busy = false
    private var closing = false
    private var operations: [CheckedContinuation<Void, Never>] = []
    private var releases: [CheckedContinuation<Void, Never>] = []

    public init(
        service: any ChronosService, permits: @escaping @MainActor () -> Bool = { true },
        journal: Store<ChronosJournalFeature>? = nil
    ) {
        self.service = service
        self.permits = permits
        self.journal = journal
    }

    public func run(_ effect: ChronosFeature.Effect, send: (ChronosFeature.Action) -> Void) async {
        if case .release(let ticket) = effect {
            await release()
            if !Task.isCancelled { send(.released(ticket)) }
            return
        }
        let ticket: Int
        switch effect {
        case .load(let value), .predict(let value, _), .release(let value), .coverage(let value),
            .history(let value, _, _), .actuals(let value, _, _):
            ticket = value
        }
        guard !busy, !closing else {
            send(.failed(ticket, .busy))
            return
        }
        guard permits(), !Task.isCancelled else {
            send(.cancelled(ticket))
            return
        }
        busy = true
        defer {
            busy = false
            let waiting = operations
            operations.removeAll()
            for waiter in waiting { waiter.resume() }
        }
        do {
            let action: ChronosFeature.Action
            switch effect {
            case .load: action = .loaded(ticket, try await service.load())
            case .predict(_, let request): action = .predicted(ticket, try await service.predict(request))
            case .coverage: action = .coverageLoaded(ticket, try await service.coverage())
            case .history(_, let origin, let areas):
                action = .historyLoaded(ticket, try await service.history(endingAt: origin, areaIDs: areas))
            case .actuals(_, let origin, let areas):
                action = .actualsLoaded(ticket, try await service.actuals(startingAt: origin, areaIDs: areas))
            case .release: return
            }
            try Task.checkCancellation()
            guard permits(), !closing else { throw CancellationError() }
            send(action)
            if case .predicted(_, let batch) = action { journal?.send(.capture(batch)) }
        } catch {
            guard !Task.isCancelled, !closing, permits() else { return }
            if error is CancellationError {
                send(.cancelled(ticket))
                return
            }
            send(.failed(ticket, (error as? ChronosDemoFailure) ?? .predictionFailed))
        }
    }

    public func release() async {
        journal?.send(.suspend)
        if closing {
            await withCheckedContinuation { releases.append($0) }
            return
        }
        closing = true
        if busy { await withCheckedContinuation { operations.append($0) } }
        await withTaskCancellationShield { await service.release() }
        closing = false
        let waiting = releases
        releases.removeAll()
        for waiter in waiting { waiter.resume() }
    }
}
