import Foundation

public enum ReceiptValidationError: String, Error, LocalizedError, Sendable {
    case invalidAmount
    case invalidDate
    case invalidFields
    case unsupportedCurrency
    case invalidClassification
    case amountOverflow
    case noConfirmedReceipts
    case duplicateIdentifier
    case invalidSource
    case invalidRules

    public var errorDescription: String? {
        switch self {
        case .invalidAmount: "Enter a nonnegative whole-yen amount within the supported range."
        case .invalidDate: "Enter a valid date in YYYY-MM-DD format."
        case .invalidFields: "Review the merchant, date, amount and category before confirming."
        case .unsupportedCurrency: "Only JPY receipts are supported."
        case .invalidClassification: "The model returned an invalid category distribution."
        case .amountOverflow: "The total exceeds the supported amount range."
        case .noConfirmedReceipts: "Confirm at least one complete receipt before exporting."
        case .duplicateIdentifier: "Receipt identifiers must be unique."
        case .invalidSource: "The receipt source name or file type is invalid."
        case .invalidRules: "The receipt extraction rules are invalid."
        }
    }
}
