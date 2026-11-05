import Foundation
import ReceiptCore
import Testing

struct ReceiptLedgerTests {
    @Test("Only confirmed complete receipts contribute to the exact total and relative export plan")
    func confirmedOnly() throws {
        let a = try record(id: 1, amount: "1000", category: .supplies)
        let b = try record(id: 2, amount: "250", category: .meals)
        var pending = try record(id: 3, amount: "9999", category: .books)
        pending.unconfirm()
        let ledger = try ReceiptLedger(records: [b, pending, a])
        #expect(ledger.rows.map(\.id) == [1, 2])
        #expect(ledger.totalYen == 1250)
        #expect(ledger.categoryTotals == [.supplies: 1000, .meals: 250])
        #expect(ledger.rows[0].relativePath == "2026/11/supplies/2026-11-06-1.png")
        #expect(ledger.rows[0].sourceSHA256 == String(repeating: "a", count: 64))
        #expect(ledger.rows[1].relativePath == "2026/11/meals/2026-11-06-2.png")
    }

    @Test("Editing a confirmed amount immediately removes it from the exportable total")
    func editInvalidatesTotal() throws {
        var a = try record(id: 1, amount: "1000")
        a.edit(.amount, value: "2000")
        #expect(throws: ReceiptValidationError.noConfirmedReceipts) { try ReceiptLedger(records: [a]) }
        try a.confirm()
        #expect(try ReceiptLedger(records: [a]).totalYen == 2000)
    }

    @Test("Overflow and duplicate identifiers fail without producing a partial ledger")
    func rejectsInvalidLedger() throws {
        let a = try record(id: 1, amount: String(Int64.max))
        let b = try record(id: 2, amount: "1")
        #expect(throws: ReceiptValidationError.amountOverflow) { try ReceiptLedger(records: [a, b]) }
        #expect(throws: ReceiptValidationError.duplicateIdentifier) { try ReceiptLedger(records: [a, a]) }
    }

    @Test("Untrusted text cannot create formula cells or alter the CSV column structure")
    func csvEscaping() throws {
        let merchant = "  =HYPERLINK(\"example\")"
        let purpose = "Line one, quoted \"value\"\nLine two"
        let a = try record(id: 1, amount: "100", merchant: merchant, purpose: purpose)
        let csv = String(decoding: try ReceiptLedger(records: [a]).csv, as: UTF8.self)
        #expect(csv.contains("\"'  =HYPERLINK(\"\"example\"\")\""))
        #expect(csv.contains("\"Line one, quoted \"\"value\"\"\nLine two\""))
        #expect(csv.contains(",100,\"2026/11/supplies/2026-11-06-1.png\""))
    }

    @Test("Invalid source components cannot escape the generated receipt folders")
    func rejectsPaths() throws {
        for (name, ext) in [("../receipt.png", "png"), ("receipt.png", "../png"), ("receipt", "exe")] {
            var a = ReceiptRecord(
                id: 1, sourceName: name, sourceExtension: ext, sourceSHA256: String(repeating: "a", count: 64),
                text: "",
                extraction: ReceiptExtraction(
                    fields: ReceiptFields(merchant: "Shop", date: "2026-11-06", amount: "100", category: .supplies)))
            try a.confirm()
            #expect(throws: ReceiptValidationError.invalidSource) { try ReceiptLedger(records: [a]) }
        }
    }

    @Test("A ledger preserves its source fingerprint and rejects malformed fingerprints")
    func fingerprints() throws {
        for digest in ["", String(repeating: "a", count: 63), String(repeating: "G", count: 64)] {
            var value = ReceiptRecord(
                id: 1, sourceName: "receipt.png", sourceExtension: "png", sourceSHA256: digest, text: "",
                extraction: ReceiptExtraction(
                    fields: ReceiptFields(merchant: "Shop", date: "2026-11-06", amount: "100", category: .supplies)))
            try value.confirm()
            #expect(throws: ReceiptValidationError.invalidSource) { try ReceiptLedger(records: [value]) }
        }
    }

    private func record(
        id: Int, amount: String, category: ReceiptCategory = .supplies, merchant: String = "Shop", purpose: String = ""
    ) throws -> ReceiptRecord {
        var value = ReceiptRecord(
            id: id, sourceName: "receipt.png", sourceExtension: "png", sourceSHA256: String(repeating: "a", count: 64),
            text: "Receipt",
            extraction: ReceiptExtraction(
                fields: ReceiptFields(
                    merchant: merchant, date: "2026-11-06", amount: amount, purpose: purpose, category: category)))
        try value.confirm()
        return value
    }
}
