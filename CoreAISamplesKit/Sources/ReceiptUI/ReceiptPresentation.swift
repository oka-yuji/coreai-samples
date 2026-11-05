#if os(macOS)
    import ReceiptCore
    import ReceiptFeature
    import SwiftUI

    func receiptText(_ key: String) -> Text { Text(LocalizedStringKey(key), bundle: .module) }

    extension ReceiptCategory {
        var symbol: String {
            switch self {
            case .travel: "tram"
            case .lodging: "bed.double"
            case .supplies: "pencil.and.ruler"
            case .meals: "cup.and.saucer"
            case .communications: "antenna.radiowaves.left.and.right"
            case .books: "books.vertical"
            case .other: "tray"
            case .review: "questionmark.folder"
            }
        }
    }

    extension ReceiptIssue {
        var message: String {
            switch self {
            case .missingMerchant: "Enter the merchant shown on the receipt."
            case .missingDate: "Enter the receipt date."
            case .ambiguousDate: "More than one date was found. Check the original."
            case .missingAmount: "Enter the total shown on the receipt."
            case .ambiguousAmount: "Different totals were found. Check the original."
            case .unreadableAmount: "A total could not be read reliably. Check the original."
            case .unsupportedCurrency: "This demo supports JPY receipts only. Foreign currency cannot be confirmed."
            case .lowRecognitionConfidence: "OCR may contain mistakes. Compare the fields with the original."
            case .unreadablePage: "A page could not be read. Check every page before confirming."
            }
        }
    }

    extension ReceiptState.Phase {
        var label: String {
            switch self {
            case .idle: "Ready for review"
            case .samples: "Opening sample receipts…"
            case .preparing: "Preparing JEV on this Mac…"
            case .reading: "Reading receipt text…"
            case .classifying: "JEV is suggesting a category…"
            case .removing: "Removing receipt from this session…"
            case .exporting: "Saving the reviewed ledger and receipt copies…"
            case .stopping: "Finishing the current operation…"
            case .closing, .closed: "Closing Receipt Desk…"
            }
        }
    }
#endif
