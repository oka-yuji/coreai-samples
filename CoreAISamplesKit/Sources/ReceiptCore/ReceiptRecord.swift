import Foundation

public enum ReceiptField: String, CaseIterable, Sendable, Codable {
    case merchant
    case date
    case amount
    case purpose
}

public enum ReceiptIssue: String, CaseIterable, Sendable, Codable {
    case missingMerchant
    case missingDate
    case ambiguousDate
    case missingAmount
    case ambiguousAmount
    case unreadableAmount
    case unsupportedCurrency
    case lowRecognitionConfidence
}

public struct ReceiptFields: Equatable, Sendable, Codable {
    public var merchant: String
    public var date: String
    public var amount: String
    public var purpose: String
    public var category: ReceiptCategory

    public init(
        merchant: String = "", date: String = "", amount: String = "", purpose: String = "",
        category: ReceiptCategory = .review
    ) {
        self.merchant = merchant
        self.date = date
        self.amount = amount
        self.purpose = purpose
        self.category = category
    }

    public var parsedDate: ReceiptDate? { try? ReceiptDate(iso8601: date) }
    public var parsedAmount: ReceiptAmount? { try? ReceiptAmount(text: amount) }
    public var isComplete: Bool {
        !merchant.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && merchant.utf8.count <= 1024
            && parsedDate != nil && parsedAmount != nil && category != .review
    }
}

public struct ReceiptExtraction: Equatable, Sendable {
    public var fields: ReceiptFields
    public var issues: [ReceiptIssue]

    public init(fields: ReceiptFields, issues: [ReceiptIssue] = []) {
        self.fields = fields
        self.issues = issues
    }
}

public struct ReceiptRecord: Identifiable, Equatable, Sendable {
    public let id: Int
    public let sourceName: String
    public let sourceExtension: String
    public let sourceSHA256: String
    public let text: String
    public private(set) var fields: ReceiptFields
    public private(set) var issues: [ReceiptIssue]
    public private(set) var classification: ReceiptClassification?
    public private(set) var categoryWasEdited = false
    public private(set) var isConfirmed = false

    public init(
        id: Int, sourceName: String, sourceExtension: String, sourceSHA256: String, text: String,
        extraction: ReceiptExtraction
    ) {
        self.id = id
        self.sourceName = sourceName
        self.sourceExtension = sourceExtension
        self.sourceSHA256 = sourceSHA256
        self.text = text
        fields = extraction.fields
        issues = extraction.issues
    }

    public var canConfirm: Bool { fields.isComplete && !issues.contains(.unsupportedCurrency) }
    public var canConfirmTogether: Bool { canConfirm && issues.isEmpty }

    public mutating func edit(_ field: ReceiptField, value: String) {
        isConfirmed = false
        switch field {
        case .merchant:
            fields.merchant = value
            issues.removeAll { $0 == .missingMerchant }
        case .date:
            fields.date = value
            issues.removeAll { $0 == .missingDate || $0 == .ambiguousDate }
        case .amount:
            fields.amount = value
            issues.removeAll { $0 == .missingAmount || $0 == .ambiguousAmount || $0 == .unreadableAmount }
        case .purpose: fields.purpose = value
        }
        if field == .merchant || field == .purpose {
            classification = nil
            fields.category = .review
            categoryWasEdited = false
        }
    }

    public mutating func setCategory(_ category: ReceiptCategory) {
        fields.category = category
        categoryWasEdited = true
        classification = nil
        isConfirmed = false
    }

    public mutating func apply(_ classification: ReceiptClassification) {
        self.classification = classification
        fields.category = classification.category
        categoryWasEdited = false
        isConfirmed = false
    }

    public mutating func confirm() throws {
        guard canConfirm else { throw ReceiptValidationError.invalidFields }
        isConfirmed = true
    }

    public mutating func unconfirm() { isConfirmed = false }
}
