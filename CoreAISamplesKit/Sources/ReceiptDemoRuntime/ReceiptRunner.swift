import Foundation
import ReceiptFeature

@MainActor
public final class ReceiptRunner {
    private let service: any ReceiptService
    private let permits: @MainActor () -> Bool
    private var active: ReceiptTicket?
    private var completed: ReceiptFinishedOperation?
    private var waiters: [CheckedContinuation<Void, Never>] = []

    public init(service: any ReceiptService, permits: @escaping @MainActor () -> Bool = { true }) {
        self.service = service
        self.permits = permits
    }

    public func run(_ effect: ReceiptEffect, send: (ReceiptAction) -> Void) async {
        let ticket = effect.ticket
        switch effect {
        case .release:
            await release()
            if !Task.isCancelled { send(.released(ticket)) }
            return
        case .stop(_, let interrupted):
            await service.stop()
            await join()
            let result = completed?.ticket == interrupted ? completed : nil
            if !Task.isCancelled { send(.stopped(ticket, result)) }
            return
        default: break
        }
        guard permits(), !Task.isCancelled else { return }
        guard active == nil else {
            send(.failed(ticket, "Wait for the current receipt operation to finish."))
            return
        }
        active = ticket
        completed = nil
        defer { finish(ticket) }
        do {
            let result: ReceiptCompletion
            switch effect {
            case .samples: result = .samples(try await service.samples())
            case .prepare(_, let bundle):
                try await service.prepare(bundle: bundle)
                result = .prepared
            case .read(_, let url): result = .read(try await service.read(id: ticket.item, url: url))
            case .classify(_, let record): result = .classified(try await service.classify(record))
            case .remove:
                try await service.remove(id: ticket.item)
                result = .removed
            case .export(_, let ledger, let directory):
                result = .exported(try await service.export(ledger, to: directory))
            case .stop, .release: return
            }
            let operation = ReceiptFinishedOperation(ticket: ticket, completion: result)
            completed = operation
            guard !Task.isCancelled, permits() else { return }
            send(.finished(operation))
        } catch {
            guard !Task.isCancelled, permits() else { return }
            if error is CancellationError {
                send(.stop)
            } else {
                send(.failed(ticket, Self.message(error)))
            }
        }
    }

    public func release() async {
        await service.release()
        await join()
        completed = nil
    }

    private func join() async {
        if active != nil { await withCheckedContinuation { waiters.append($0) } }
    }

    private func finish(_ ticket: ReceiptTicket) {
        guard active == ticket else { return }
        active = nil
        let pending = waiters
        waiters.removeAll()
        for waiter in pending { waiter.resume() }
    }

    private static func message(_ error: any Error) -> String {
        if let text = (error as? any LocalizedError)?.errorDescription { return text }
        if error is CocoaError { return error.localizedDescription }
        return "The receipt operation failed. Review the input and try again."
    }
}
