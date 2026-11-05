import Foundation
import ReceiptCore
import Testing

struct ReceiptAmountTests {
    @Test("Whole yen amounts preserve every digit without floating point rounding")
    func wholeYen() throws {
        #expect(try ReceiptAmount(text: "1,234").yen == 1234)
        #expect(try ReceiptAmount(text: "9,007,199,254,740,993").yen == 9_007_199_254_740_993)
        #expect(try ReceiptAmount(text: "0").yen == 0)
    }

    @Test(
        "Invalid, fractional, negative and overflowing amounts are rejected",
        arguments: [
            "", "-1", "+1", "1.00", "1,23", "00", "1 000", "1e3", "9223372036854775808", "$100",
        ])
    func rejectsInvalidAmounts(_ text: String) {
        #expect(throws: ReceiptValidationError.invalidAmount) { try ReceiptAmount(text: text) }
    }

    @Test("Decoding cannot bypass the whole-yen invariant")
    func decoding() throws {
        #expect(try JSONDecoder().decode(ReceiptAmount.self, from: Data(#""1234""#.utf8)).yen == 1234)
        #expect(throws: ReceiptValidationError.invalidAmount) {
            try JSONDecoder().decode(ReceiptAmount.self, from: Data(#""-1""#.utf8))
        }
        let exact = try ReceiptAmount(text: "9007199254740993")
        let encoded = try JSONEncoder().encode(exact)
        #expect(String(decoding: encoded, as: UTF8.self) == #""9007199254740993""#)
        #expect(try JSONDecoder().decode(ReceiptAmount.self, from: encoded) == exact)
    }

    @Test("Numeric JSON cannot silently round a fractional amount into whole yen")
    func rejectsNumericJSON() {
        for json in ["1.0000000000000001", "1e3", "-0", "9007199254740993.0"] {
            #expect(throws: (any Error).self) {
                try JSONDecoder().decode(ReceiptAmount.self, from: Data(json.utf8))
            }
        }
    }
}

struct ReceiptDateTests {
    @Test("Dates use the Gregorian leap-year rules and preserve the declared calendar day")
    func calendarDay() throws {
        #expect(try ReceiptDate(iso8601: "2024-02-29").iso8601 == "2024-02-29")
        #expect(try ReceiptDate(year: 2000, month: 2, day: 29).monthText == "02")
        for text in ["2026-02-29", "1900-02-29", "2026-04-31", "2026-00-01", "2026-1-01", "0000-01-01"] {
            #expect(throws: ReceiptValidationError.invalidDate) { try ReceiptDate(iso8601: text) }
        }
    }
}

struct ReceiptClassificationTests {
    @Test("Only the fixed ordered categories receive valid probability distributions")
    func categories() throws {
        #expect(
            ReceiptCategory.allCases.map(\.rawValue) == [
                "travel", "lodging", "supplies", "meals", "communications", "books", "other", "review",
            ])
        let result = try ReceiptClassification(probabilities: [0.01, 0.01, 0.92, 0.02, 0.01, 0.01, 0.01, 0.01])
        #expect(result.category == .supplies)
        #expect(result.probability == 0.92)
        #expect(ReceiptCategory.allCases.allSatisfy { !$0.decisionOption.contains("\n") })
    }

    @Test("Low probabilities and an explicit review choice remain reviewable")
    func uncertain() throws {
        let tied = try ReceiptClassification(probabilities: [0.5, 0.5, 0, 0, 0, 0, 0, 0])
        #expect(tied.category == .review)
        #expect(tied.topCategory == .travel)
        #expect(try ReceiptClassification(probabilities: [0, 0, 0, 0, 0, 0, 0, 1]).category == .review)
        for values: [Float] in [[1], [1, 1, 0, 0, 0, 0, 0, 0], [.nan, 0, 0, 0, 0, 0, 0, 1]] {
            #expect(throws: ReceiptValidationError.invalidClassification) {
                try ReceiptClassification(probabilities: values)
            }
        }
    }
}

struct ReceiptRecordTests {
    @Test("Editing invalidates confirmation and semantic edits invalidate the old classification")
    func edit() throws {
        var record = makeRecord()
        try record.confirm()
        #expect(record.isConfirmed)
        record.edit(.amount, value: "1200")
        #expect(!record.isConfirmed)
        #expect(record.fields.category == .supplies)
        try record.confirm()
        record.edit(.purpose, value: "Client meeting")
        #expect(!record.isConfirmed)
        #expect(record.fields.category == .review)
        #expect(!record.canConfirm)
        #expect(throws: ReceiptValidationError.invalidFields) { try record.confirm() }
    }

    @Test("Manual categories preserve monetary fields and unsupported currency remains blocked")
    func review() throws {
        var record = makeRecord(issues: [.unsupportedCurrency])
        record.setCategory(.travel)
        #expect(record.fields.amount == "1000")
        #expect(record.fields.date == "2026-11-06")
        #expect(!record.canConfirm)
        #expect(record.categoryWasEdited)
        record = makeRecord(issues: [.lowRecognitionConfidence])
        #expect(record.canConfirm)
        #expect(!record.canConfirmTogether)
        try record.confirm()
        #expect(record.isConfirmed)
    }

    func makeRecord(issues: [ReceiptIssue] = []) -> ReceiptRecord {
        ReceiptRecord(
            id: 1, sourceName: "receipt.png", sourceExtension: "png", sourceSHA256: String(repeating: "a", count: 64),
            text: "Shop\nTotal 1000",
            extraction: ReceiptExtraction(
                fields: ReceiptFields(merchant: "Shop", date: "2026-11-06", amount: "1000", category: .supplies),
                issues: issues))
    }
}
