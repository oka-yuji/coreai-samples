import ChronosCore
import Testing

struct ChronosCoreTests {
    @Test("Context lengths retain raw values and pad only on the left", arguments: [1, 15, 16, 17, 32, 1008, 8192])
    func contextLengths(count: Int) throws {
        let values = (0..<count).map { Float($0 - 5) }
        let context = try ChronosContext(values)
        #expect(context.rawCount == count)
        #expect(context.paddedValues.count == ((count + 15) / 16) * 16)
        #expect(Array(context.paddedValues.suffix(count)) == values)
        let paddingIsMissing = context.paddedValues.dropLast(count).allSatisfy { $0.isNaN }
        #expect(paddingIsMissing)
    }

    @Test("Missing values and finite extremes retain their original bit patterns")
    func missingValues() throws {
        let values: [Float] = [
            .nan, -.greatestFiniteMagnitude, -0.0, .leastNonzeroMagnitude, .greatestFiniteMagnitude, .nan,
        ]
        let context = try ChronosContext(values)
        #expect(context.paddedValues.suffix(values.count).map(\.bitPattern) == values.map(\.bitPattern))
    }

    @Test("Empty oversized wholly missing and infinite contexts are rejected")
    func invalidContexts() {
        for values: [Float] in [
            [], [.nan], [.nan, .nan], [1, .infinity], [-.infinity, 1], Array(repeating: 1, count: 8193),
        ] {
            #expect(throws: ChronosError.invalidContext) { try ChronosContext(values) }
        }
    }

    @Test("Every raw quantile value keeps its position sign and magnitude")
    func predictions() throws {
        let rows = (0..<21).map { row in (0..<16).map { Float(1000 - row * 16 - $0) * -3 } }
        #expect(try ChronosPrediction(quantiles: rows).quantiles == rows)
        #expect(throws: ChronosError.invalidPrediction) { try ChronosPrediction(quantiles: Array(rows.dropLast())) }
        var short = rows
        short[20].removeLast()
        #expect(throws: ChronosError.invalidPrediction) { try ChronosPrediction(quantiles: short) }
        for invalid: Float in [.nan, .infinity, -.infinity] {
            var changed = rows
            changed[20][15] = invalid
            #expect(throws: ChronosError.invalidPrediction) { try ChronosPrediction(quantiles: changed) }
        }
    }
}
