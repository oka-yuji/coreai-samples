import Foundation
import ReceiptCore

public enum ReceiptRuntimeError: String, Error, LocalizedError, Sendable {
    case busy
    case released
    case noModel
    case invalidSource
    case tooLarge
    case tooManyReceipts
    case unsupportedFile
    case unsupportedPDF
    case sourceChanged
    case duplicateReceipt
    case sourceMismatch
    case incompleteText
    case invalidDestination
    case destinationExists
    case cleanupFailed

    public var errorDescription: String? {
        switch self {
        case .busy: "Wait for the current receipt operation to finish."
        case .released: "Reopen Receipt Desk to start a new session."
        case .noModel: "Prepare JEV before classifying receipts."
        case .invalidSource: "Choose an original, regular receipt file."
        case .tooLarge: "Each receipt must be at most 20 MiB."
        case .tooManyReceipts: "A Receipt Desk session supports up to 20 receipts."
        case .unsupportedFile: "Choose a JPEG, PNG, HEIC, TIFF or PDF receipt."
        case .unsupportedPDF: "Choose an unlocked PDF with one to five pages."
        case .sourceChanged: "The receipt changed while it was being imported. Try importing it again."
        case .duplicateReceipt: "This exact receipt is already in the session."
        case .sourceMismatch: "The retained original does not match this receipt. Import it again."
        case .incompleteText: "Some receipt text could not be read. Review the original and enter its fields."
        case .invalidDestination: "Choose a writable folder for the receipt export."
        case .destinationExists: "The export folder already exists. Choose a new export."
        case .cleanupFailed:
            "The export failed and its temporary folder could not be removed. Check the selected folder."
        }
    }
}

public struct ReceiptRecognizedDocument: Sendable {
    public let lines: [ReceiptTextLine]
    public let previews: [Data]
    public let fileExtension: String
    public let unreadablePages: [Int]

    public init(lines: [ReceiptTextLine], previews: [Data], fileExtension: String, unreadablePages: [Int] = []) {
        self.lines = lines
        self.previews = previews
        self.fileExtension = fileExtension
        self.unreadablePages = unreadablePages
    }
}

public struct ReceiptReadResult: Equatable, Sendable {
    public let record: ReceiptRecord
    public let previews: [Data]
    public let seconds: Double
}

public struct ReceiptDecisionInput: Equatable, Sendable {
    public let state: String
    public let question: String
    public let options: [String]
}

public struct ReceiptDecisionValues: Sendable {
    public let probabilities: [Float]
    public let tokenCount: Int

    public init(probabilities: [Float], tokenCount: Int) {
        self.probabilities = probabilities
        self.tokenCount = tokenCount
    }
}

public protocol ReceiptDecisionExecuting: Sendable {
    func decide(_ input: ReceiptDecisionInput) async throws -> ReceiptDecisionValues
}

public struct ReceiptDecisionResult: Equatable, Sendable {
    public let classification: ReceiptClassification
    public let tokenCount: Int
    public let seconds: Double
}

public struct ReceiptExportResult: Equatable, Sendable {
    public let directory: URL
    public let count: Int
    public let totalYen: Int64
}
