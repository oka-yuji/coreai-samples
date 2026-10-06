import Foundation
import GLiNERCore
import Testing

private func scalarWords(_ text: String) -> [WordSpan] {
    text.unicodeScalars.enumerated().map { WordSpan(text: String($1), scalarRange: $0..<$0 + 1) }
}

private func filled(rows: Int, columns: Int, with value: Float, _ cells: [Int: [Float]] = [:]) -> Matrix {
    Matrix(
        rows: rows, columns: columns,
        values: (0..<rows).flatMap { row in
            (0..<columns).map { column in cells[row].map { column < $0.count ? $0[column] : value } ?? value }
        })
}

struct DecodingTests {
    @Test("The pool ignores labels and boundaries outside the input and reserves each label's best pairs in order")
    func poolUsesOnlyLabelsAndBoundariesInUse() {
        let pool = CandidatePool(
            startLogits: filled(rows: 16, columns: 4, with: 100, [0: [0.5, 0, 0.25]]),
            endLogits: filled(rows: 16, columns: 4, with: 100, [0: [0, 0.25, 0.5]]),
            poolStart: filled(rows: 4, columns: 2, with: 9, [0: [0, 0], 1: [0, 0], 2: [0, 0]]),
            poolEnd: filled(rows: 4, columns: 2, with: 9, [0: [0, 0], 1: [0, 0], 2: [0, 0]]),
            wordCount: 2, labelCount: 1)
        let slots = pool.mask.indices.filter { pool.mask[$0] != 0 }
        #expect(slots == [0, 1, 2])
        #expect(slots.map { pool.starts[$0]..<pool.ends[$0] } == [0..<2, 0..<1, 1..<2])
        #expect(pool.starts.count == CandidatePool.size && pool.priors.allSatisfy { $0 == 0 })
    }

    @Test("Spans over the appended period are cut or dropped, and only a null probability over 0.5 abstains")
    func decodesWithPeriodCutAndAbstention() throws {
        let input = try PreparedInput(
            text: "ab", schema: .entities([EntityLabel(name: "x"), EntityLabel(name: "y"), EntityLabel(name: "z")]),
            splitWords: scalarWords, tokenize: { _ in [7] })
        try #require(input.appendedPeriod && input.words.count == 3)
        let zeros = filled(rows: 16, columns: 4, with: 0)
        let pool = CandidatePool(
            startLogits: zeros, endLogits: zeros, poolStart: filled(rows: 4, columns: 2, with: 0),
            poolEnd: filled(rows: 4, columns: 2, with: 0), wordCount: 3, labelCount: 3)
        // Label 0: "b." -> "b". Label 1: "a" at exactly 0.5 is kept, "." alone is dropped. Label 2 abstains.
        let logits: [Range<Int32>: [Float]] = [1..<3: [3, -10, 9], 0..<1: [-0.001, 0, 9], 2..<3: [-10, 4, 9]]
        let pairLogits = Matrix(
            rows: CandidatePool.size, columns: 16,
            values: pool.mask.indices.flatMap { slot in
                let row = pool.mask[slot] == 0 ? [] : logits[pool.starts[slot]..<pool.ends[slot]] ?? []
                return (0..<16).map { label in pool.mask[slot] == 0 ? 100 : label < row.count ? row[label] : -10 }
            })
        let result = ExtractionResult(
            pairLogits: pairLogits, nullLogits: [-5, 0, 0.001] + Array(repeating: -5, count: 13), pool: pool,
            input: input, text: "ab")
        let expected = ExtractionResult(
            spans: [
                ExtractedSpan(labelIndex: 0, scalarRange: 1..<2, text: "b", probability: 1 / (1 + exp(-3))),
                ExtractedSpan(labelIndex: 1, scalarRange: 0..<1, text: "a", probability: 0.5),
            ],
            abstained: [2], tokenCapacity: 256)
        #expect(result == expected)
    }
}
