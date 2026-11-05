import Foundation
import ReceiptCore

public struct ReceiptInput: Equatable, Sendable {
    public let url: URL
    public let purpose: String
    public let title: String?

    public init(url: URL, purpose: String = "", title: String? = nil) {
        self.url = url
        self.purpose = purpose
        self.title = title
    }
}

public struct ReceiptItem: Equatable, Identifiable, Sendable {
    public let id: Int
    public let input: ReceiptInput
    public var revision = 0
    public var record: ReceiptRecord?
    public var previews: [Data] = []
    public var ocrSeconds: Double?
    public var decisionSeconds: Double?
    public var tokenCount: Int?
    public var failure: String?
    public var needsClassification = true

    public var title: String {
        record?.fields.merchant.isEmpty == false
            ? record?.fields.merchant ?? "" : input.title ?? input.url.lastPathComponent
    }
    public var isConfirmed: Bool { record?.isConfirmed == true }
    public var needsReview: Bool { record != nil && !isConfirmed }
}

public struct ReceiptTicket: Equatable, Hashable, Sendable {
    public let session: Int
    public let request: Int
    public let item: Int
    public let revision: Int

    public init(session: Int, request: Int, item: Int = 0, revision: Int = 0) {
        self.session = session
        self.request = request
        self.item = item
        self.revision = revision
    }
}

public struct ReceiptRead: Equatable, Sendable {
    public let record: ReceiptRecord
    public let previews: [Data]
    public let seconds: Double

    public init(record: ReceiptRecord, previews: [Data], seconds: Double) {
        self.record = record
        self.previews = previews
        self.seconds = seconds
    }
}

public struct ReceiptDecision: Equatable, Sendable {
    public let classification: ReceiptClassification
    public let tokenCount: Int
    public let seconds: Double

    public init(classification: ReceiptClassification, tokenCount: Int, seconds: Double) {
        self.classification = classification
        self.tokenCount = tokenCount
        self.seconds = seconds
    }
}

public struct ReceiptSaved: Equatable, Sendable {
    public let directory: URL
    public let count: Int
    public let totalYen: Int64

    public init(directory: URL, count: Int, totalYen: Int64) {
        self.directory = directory
        self.count = count
        self.totalYen = totalYen
    }
}

public struct ReceiptExportSelection: Equatable, Identifiable, Sendable {
    public let id: ReceiptTicket
    public let ledger: ReceiptLedger
}

public struct ReceiptState: Equatable, Sendable {
    public enum Phase: Equatable, Sendable {
        case idle, samples, preparing, reading, classifying, removing, exporting, stopping, closing, closed
    }

    public let bundle: URL
    public var presented = false
    public var ended = false
    public var phase: Phase = .idle
    public var items: [ReceiptItem] = []
    public var selection: Int?
    public var active: ReceiptTicket?
    public var message: String?
    public var saved: ReceiptSaved?
    public var exportSelection: ReceiptExportSelection?
    public var modelPrepared = false
    public var batchTotal = 0
    public var batchDone = 0
    var session = 1
    var request = 0
    var nextItem = 0
    var queue: [Int] = []
    var exportingLedger: ReceiptLedger?
    var interrupted: ReceiptInterrupted?

    public init(bundle: URL, session: Int = 1) {
        self.bundle = bundle
        self.session = session
    }

    public var busy: Bool { phase != .idle && phase != .closed }
    public var editable: Bool { presented && !ended && !busy && exportSelection == nil }
    public var selected: ReceiptItem? { items.first { $0.id == selection } }
    public var confirmedCount: Int { items.filter(\.isConfirmed).count }
    public var readCount: Int { items.filter { $0.record != nil }.count }
    public var ledger: ReceiptLedger? { try? ReceiptLedger(records: items.compactMap(\.record)) }
    public var canExport: Bool { editable && ledger != nil }
    public var canProcess: Bool { editable && items.contains { $0.record == nil || $0.needsClassification } }
    public var canConfirmTogether: Bool {
        editable && items.contains { !$0.isConfirmed && $0.record?.canConfirmTogether == true }
    }
    public var ledgerProblem: String? {
        guard confirmedCount > 0 else { return nil }
        do {
            _ = try ReceiptLedger(records: items.compactMap(\.record))
            return nil
        } catch ReceiptValidationError.amountOverflow {
            return "The confirmed total is too large. Review the receipt amounts."
        } catch { return "The confirmed ledger could not be prepared. Review the receipts." }
    }

}

struct ReceiptInterrupted: Equatable, Sendable {
    let ticket: ReceiptTicket
    let phase: ReceiptState.Phase
}
