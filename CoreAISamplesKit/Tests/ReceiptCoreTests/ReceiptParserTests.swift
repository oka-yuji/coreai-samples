import Foundation
import ReceiptCore
import Testing

struct ReceiptParserTests {
    @Test("Explicit receipt fields are extracted without guessing missing or ambiguous values")
    func examples() throws {
        let fixture = try fixture()
        let parser = try ReceiptTextParser(rules: fixture.rules)
        for item in fixture.cases {
            let value = parser.extract([ReceiptTextLine(item.text)])
            #expect(value.fields.merchant == item.merchant)
            #expect(value.fields.date == item.date)
            #expect(value.fields.amount == item.amount)
            #expect(Set(value.issues.map(\.rawValue)) == Set(item.issues))
        }
    }

    @Test("Low recognition confidence remains visible even when the text parses")
    func confidence() throws {
        let parser = try ReceiptTextParser(rules: fixture().rules)
        let result = parser.extract([ReceiptTextLine("Store\n2026-11-06\nTotal 1000", confidence: 0.3)])
        #expect(result.fields.amount == "1000")
        #expect(result.issues == [.lowRecognitionConfidence])
    }

    @Test("Invalid extraction patterns fail before reading a receipt")
    func invalidRules() throws {
        let rules = try fixture().rules
        let broken = ReceiptExtractionRules(
            totalLabels: rules.totalLabels, currencyMarkers: rules.currencyMarkers,
            merchantExclusions: [], merchantExcludedPrefixes: [], datePattern: "[",
            foreignCurrencyPattern: rules.foreignCurrencyPattern)
        #expect(throws: ReceiptValidationError.invalidRules) { try ReceiptTextParser(rules: broken) }
    }

    private func fixture() throws -> Fixture {
        let url = try #require(
            Bundle.module.url(forResource: "parser", withExtension: "json", subdirectory: "Fixtures"))
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    private struct Fixture: Decodable {
        let rules: ReceiptExtractionRules
        let cases: [Example]
    }

    private struct Example: Decodable {
        let text: String
        let merchant: String
        let date: String
        let amount: String
        let issues: [String]
    }
}
