import Foundation
import Testing

@testable import OCRCore

@Suite("OCR vision input contracts")
struct OCRVisionInputTests {
    private static func fixture(_ name: String) throws -> VisionFixture.Case {
        try #require(try OCRFixtures.load("vision", as: VisionFixture.self).cases.first { $0.name == name })
    }

    private static func originalOrder(_ input: OCRVisionInput) -> [Float] {
        var values = [Float]()
        values.reserveCapacity(input.tokenRows.count * 4 * OCRVisionInput.rowWidth)
        for row in input.tokenRows {
            let start = row * 4 * OCRVisionInput.rowWidth
            values.append(contentsOf: input.pixelValues[start..<(start + 4 * OCRVisionInput.rowWidth)])
        }
        return values
    }

    private static func paddedRow(unpadded index: Int, lengths: [Int]) -> Int {
        var remaining = index
        for (window, length) in lengths.enumerated() {
            if remaining < length { return window * OCRVisionInput.rowsPerWindow + remaining }
            remaining -= length
        }
        return -1
    }

    @Test(
        "Match patches, window order, inverse order, key validity, and rotary values for 448, 1036, and landscape inputs",
        arguments: ["square448", "square1036", "landscape300x200"])
    func windowContract(name: String) throws {
        let expected = try Self.fixture(name)
        let rotary = try #require(expected.rotary)
        let input = try OCRVisionInput(try OCRFixtures.pattern(width: expected.width, height: expected.height))
        #expect([1, input.gridHeight, input.gridWidth] == expected.grid)
        let original = Self.originalOrder(input)
        #expect(OCRFixtures.digest(original) == expected.pixelValuesSHA256)
        for row in try #require(expected.pixelRows) {
            let start = row.index * OCRVisionInput.rowWidth
            #expect(Array(original[start..<(start + OCRVisionInput.rowWidth)]) == OCRFixtures.floats(row.values))
        }

        #expect(input.windowCount == expected.windowLengths.count)
        #expect(input.rowCount == expected.windowLengths.count * 64)
        var order = [Int](repeating: -1, count: input.rowCount / 4)
        for (unit, row) in input.tokenRows.enumerated() { order[row] = unit }
        #expect(order.filter { $0 >= 0 } == expected.windowIndex)
        for (window, length) in expected.windowLengths.enumerated() {
            let flags = input.keyValid[(window * 64)..<(window * 64 + 64)]
            #expect(Array(flags) == [Float](repeating: 1, count: length) + [Float](repeating: 0, count: 64 - length))
        }
        let features = (0..<(input.rowCount / 4)).flatMap { [Float($0), -Float($0)] }
        let restored = try input.tokenFeatures(from: features)
        let restore = try #require(expected.restore)
        #expect(restored.count == restore.count * 2)
        for (unit, position) in restore.enumerated() {
            let row = Self.paddedRow(unpadded: position * 4, lengths: expected.windowLengths) / 4
            #expect(restored[unit * 2] == Float(row) && restored[unit * 2 + 1] == -Float(row))
        }

        let cosine = OCRFixtures.floats(rotary.cosineTableBits)
        let sine = OCRFixtures.floats(rotary.sineTableBits)
        var difference: Float = 0
        for position in 0..<rotary.firstAxisPositions.count {
            let row = Self.paddedRow(unpadded: position, lengths: expected.windowLengths)
            for index in 0..<OCRVisionInput.rotaryWidth {
                let axis = index % 40 < 20 ? rotary.firstAxisPositions[position] : rotary.secondAxisPositions[position]
                let table = axis * 20 + index % 20
                difference = max(difference, abs(input.cos[row * 80 + index] - cosine[table]))
                difference = max(difference, abs(input.sin[row * 80 + index] - sine[table]))
            }
        }
        #expect(difference <= 1e-6)
        for sample in rotary.rows {
            let row = Self.paddedRow(unpadded: sample.index, lengths: expected.windowLengths)
            let slice = (row * 80)..<(row * 80 + 80)
            #expect(OCRFixtures.maximumDifference(input.cos[slice], OCRFixtures.floats(sample.cosineBits)) <= 1e-6)
            #expect(OCRFixtures.maximumDifference(input.sin[slice], OCRFixtures.floats(sample.sineBits)) <= 1e-6)
        }
        #expect(OCRVisionInput.inverseFrequencies.map(\.bitPattern) == rotary.inverseFrequencyBits)
    }

    @Test("Fill padded rows with zero pixels, invalid keys, and an identity rotation")
    func paddedRows() throws {
        let expected = try Self.fixture("square1036")
        let input = try OCRVisionInput(try OCRFixtures.pattern(width: expected.width, height: expected.height))
        let padded = input.keyValid.indices.filter { input.keyValid[$0] == 0 }
        #expect(padded.count == input.rowCount - 74 * 74)
        for row in padded {
            #expect(input.pixelValues[(row * 1_176)..<(row * 1_176 + 1_176)].allSatisfy { $0 == 0 })
            #expect(input.cos[(row * 80)..<(row * 80 + 80)].allSatisfy { $0 == 1 })
            #expect(input.sin[(row * 80)..<(row * 80 + 80)].allSatisfy { $0 == 0 })
        }
    }

    @Test("Resize a page-sized image and lay out its patches like the processor")
    func pageInput() throws {
        let expected = try Self.fixture("page1240x1754")
        let input = try OCRVisionInput(try OCRFixtures.pattern(width: expected.width, height: expected.height))
        #expect([1, input.gridHeight, input.gridWidth] == expected.grid)
        #expect(input.windowCount == expected.windowLengths.count)
        #expect(OCRFixtures.digest(Self.originalOrder(input)) == expected.pixelValuesSHA256)
    }

    @Test("Reject vision features whose row count does not match the padded layout")
    func invalidFeatures() throws {
        let input = try OCRVisionInput(try OCRFixtures.pattern(width: 56, height: 56))
        #expect(throws: OCRError.invalidShape) { try input.tokenFeatures(from: []) }
        #expect(throws: OCRError.invalidShape) {
            try input.tokenFeatures(from: [Float](repeating: 0, count: input.rowCount / 4 + 1))
        }
    }
}
