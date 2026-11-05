import Foundation
import ReceiptCore

public enum ReceiptCompletion: Equatable, Sendable {
    case samples([ReceiptInput])
    case prepared
    case read(ReceiptRead)
    case classified(ReceiptDecision)
    case removed
    case exported(ReceiptSaved)
}

public struct ReceiptFinishedOperation: Equatable, Sendable {
    public let ticket: ReceiptTicket
    public let completion: ReceiptCompletion

    public init(ticket: ReceiptTicket, completion: ReceiptCompletion) {
        self.ticket = ticket
        self.completion = completion
    }
}

public enum ReceiptAction: Equatable, Sendable {
    case appeared, ended
    case add([ReceiptInput])
    case samples, process
    case select(Int)
    case edit(Int, ReceiptField, String)
    case category(Int, ReceiptCategory)
    case classify(Int)
    case confirm(Int)
    case confirmTogether
    case remove(Int)
    case finished(ReceiptFinishedOperation)
    case failed(ReceiptTicket, String)
    case stop
    case stopped(ReceiptTicket, ReceiptFinishedOperation?)
    case released(ReceiptTicket)
    case requestExport
    case exportSelected(ReceiptTicket, URL?)
    case uiError(String)
}

public enum ReceiptEffect: Equatable, Sendable {
    case samples(ReceiptTicket)
    case prepare(ReceiptTicket, URL)
    case read(ReceiptTicket, URL)
    case classify(ReceiptTicket, ReceiptRecord)
    case remove(ReceiptTicket)
    case export(ReceiptTicket, ReceiptLedger, URL)
    case stop(ReceiptTicket, interrupted: ReceiptTicket?)
    case release(ReceiptTicket)

    public var ticket: ReceiptTicket {
        switch self {
        case .samples(let value), .prepare(let value, _), .read(let value, _), .classify(let value, _),
            .remove(let value), .export(let value, _, _), .stop(let value, _), .release(let value):
            value
        }
    }
}

public enum ReceiptEffectID: Hashable, Sendable { case operation, stop }
