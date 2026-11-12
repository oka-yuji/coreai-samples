import ChronosCore
import Foundation
import Testing

struct ChronosPredictionDecodingTests {
    @Test("Decoding uses the same complete finite 21 by 16 prediction contract", arguments: [0, 3, 20, 22])
    func shape(rows: Int) throws {
        let data = try JSONSerialization.data(withJSONObject: [
            "quantiles": Array(repeating: Array(repeating: 0, count: 16), count: rows)
        ])
        #expect(throws: ChronosError.invalidPrediction) { try JSONDecoder().decode(ChronosPrediction.self, from: data) }
    }

    @Test("Decoding rejects a short last quantile and nonfinite values")
    func values() throws {
        var values: [[Any]] = Array(repeating: Array(repeating: 0, count: 16), count: 21)
        values[20].removeLast()
        let short = try JSONSerialization.data(withJSONObject: ["quantiles": values])
        #expect(throws: ChronosError.invalidPrediction) {
            try JSONDecoder().decode(ChronosPrediction.self, from: short)
        }
        values[20].append("NaN")
        let data = try JSONSerialization.data(withJSONObject: ["quantiles": values])
        let decoder = JSONDecoder()
        decoder.nonConformingFloatDecodingStrategy = .convertFromString(
            positiveInfinity: "Inf", negativeInfinity: "-Inf", nan: "NaN")
        #expect(throws: ChronosError.invalidPrediction) { try decoder.decode(ChronosPrediction.self, from: data) }
    }
}
