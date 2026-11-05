import Foundation
import ReceiptCore
import Testing

@testable import ReceiptRuntime

struct ReceiptRuntimeTests {
    @Test("The bundled extraction profile preserves a Japanese receipt total")
    func profile() throws {
        let parser = try ReceiptTextParser(rules: ReceiptPipeline.extractionRules())
        let result = parser.extract([ReceiptTextLine("Shop\n2026-11-06\nTotal JPY 1000")])
        #expect(result.fields.amount == "1000")
        #expect(result.fields.date == "2026-11-06")
        #expect(result.issues.isEmpty)
    }
}
