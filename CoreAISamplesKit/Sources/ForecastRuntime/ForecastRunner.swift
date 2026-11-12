import ForecastFeature
import Foundation

@MainActor
public final class ForecastRunner {
    private let service: any ForecastService
    private let permits: @MainActor () -> Bool
    private var busy = false
    private var closing = false
    private var operations: [CheckedContinuation<Void, Never>] = []
    private var releases: [CheckedContinuation<Void, Never>] = []

    public init(service: any ForecastService, permits: @escaping @MainActor () -> Bool = { true }) {
        self.service = service
        self.permits = permits
    }

    public func run(_ effect: ForecastFeature.Effect, send: (ForecastFeature.Action) -> Void) async {
        if case .release(let ticket) = effect {
            await release()
            if !Task.isCancelled { send(.released(ticket)) }
            return
        }
        let ticket: Int
        switch effect {
        case .load(let value, _), .predict(let value, _), .release(let value): ticket = value
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
            let action: ForecastFeature.Action
            switch effect {
            case .load(_, let refresh): action = .loaded(ticket, try await service.load(refresh: refresh))
            case .predict(_, let request): action = .predicted(ticket, try await service.predict(request))
            case .release: return
            }
            try Task.checkCancellation()
            guard permits(), !closing else { throw CancellationError() }
            send(action)
        } catch {
            guard !Task.isCancelled, !closing, permits() else { return }
            if error is CancellationError {
                send(.cancelled(ticket))
            } else {
                send(.failed(ticket, (error as? ForecastFailure) ?? .predictionFailed))
            }
        }
    }

    public func release() async {
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
