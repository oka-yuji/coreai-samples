import GLiNERCore
import Testing

private func scalarWords(_ text: String) -> [WordSpan] {
    text.unicodeScalars.enumerated().compactMap { offset, scalar in
        scalar.properties.isWhitespace ? nil : WordSpan(text: String(scalar), scalarRange: offset..<offset + 1)
    }
}

private func oneTokenEach(_ piece: String) -> [Int32] { [7] }

struct PreparedInputTests {
    @Test(
        "The trailing period rule holds, text over 256 tokens moves to 512, and text over 512 tokens is an error",
        arguments: [
            ("abc", true), ("abc.", false), ("abc!", false), ("abc?", false), ("", true),
            ("abc\u{3002}", true), ("abc ", true), ("abc.\u{301}", true),
        ])
    func appendsPeriodUnlessTextEndsWithStop(text: String, appended: Bool) throws {
        let input = try PreparedInput(
            text: text, schema: .classification(name: "n", labels: ["a"]), splitWords: scalarWords,
            tokenize: oneTokenEach)
        #expect(input.appendedPeriod == appended)
        #expect(input.words.map(\.text) == scalarWords(appended ? text + "." : text).map(\.text))
    }

    @Test(
        "The trailing period rule holds, text over 256 tokens moves to 512, and text over 512 tokens is an error",
        arguments: [(246, 256), (247, 512), (502, 512)])
    func picksTheSmallestCapacityThatFits(letters: Int, capacity: Int) throws {
        let input = try PreparedInput(
            text: String(repeating: "x", count: letters), schema: .classification(name: "n", labels: ["a"]),
            splitWords: scalarWords, tokenize: oneTokenEach)
        #expect(input.attentionMask.filter { $0 == 1 }.count == letters + 10)
        #expect(input.tokenCapacity == capacity)
        #expect(input.inputIDs.count == capacity)
        #expect(input.wordIndex.count == capacity)
        #expect(input.wordMask.count == capacity)
    }

    @Test("The trailing period rule holds, text over 256 tokens moves to 512, and text over 512 tokens is an error")
    func rejectsTextLongerThan512Tokens() {
        #expect(throws: GLiNERPreprocessingError.inputTooLong(tokens: 513, limit: 512)) {
            try PreparedInput(
                text: String(repeating: "x", count: 503), schema: .classification(name: "n", labels: ["a"]),
                splitWords: scalarWords, tokenize: oneTokenEach)
        }
    }

    @Test("More than 16 labels is an error")
    func rejectsMoreThan16Labels() {
        let labels = (0..<17).map { "l\($0)" }
        #expect(throws: GLiNERPreprocessingError.tooManyLabels(count: 17, limit: 16)) {
            try PreparedInput(
                text: "x", schema: .classification(name: "n", labels: labels), splitWords: scalarWords,
                tokenize: oneTokenEach)
        }
    }

    @Test("Entities route [E] to query and classification routes [L] to cls, with each description after its label")
    func laysOutSchemaMarkersAndWords() throws {
        let vocabulary: [String: [Int32]] = [
            "(": [1], "[P]": [2], "entities [DESCRIPTION] a: d": [3, 4], "[E]": [5], "a": [6], "b": [7, 8],
            ")": [9], "[SEP_TEXT]": [10], "x": [11], "y": [12, 13], ".": [14], "n": [15], "[L]": [16],
        ]
        let tokenize = { (piece: String) in vocabulary[piece] ?? [0] }
        let entities = try PreparedInput(
            text: "x y", schema: .entities([EntityLabel(name: "a", description: "d"), EntityLabel(name: "b")]),
            splitWords: scalarWords, tokenize: tokenize)
        let ids: [Int32] = [1, 2, 3, 4, 1, 5, 6, 5, 7, 8, 9, 9, 10, 11, 12, 13, 14]
        #expect(entities.inputIDs == ids + Array(repeating: 0, count: 256 - ids.count))
        #expect(
            entities.attentionMask == Array(repeating: 1, count: ids.count)
                + Array(repeating: 0, count: 256 - ids.count))
        #expect(entities.wordIndex == [13, 14, 16] + Array(repeating: 0, count: 253))
        #expect(entities.wordMask == [1, 1, 1] + Array(repeating: 0, count: 253))
        #expect(entities.queryIndex == [5, 7] + Array(repeating: 0, count: 14))
        #expect(entities.queryMask == [1, 1] + Array(repeating: 0, count: 14))
        #expect(entities.classIndex == Array(repeating: 0, count: 16))
        #expect(entities.classMask == Array(repeating: 0, count: 16))
        #expect(entities.words.map(\.scalarRange) == [0..<1, 2..<3, 3..<4])
        #expect(entities.labelCount == 2)

        let classes = try PreparedInput(
            text: "x y", schema: .classification(name: "n", labels: ["a", "b"]), splitWords: scalarWords,
            tokenize: tokenize)
        #expect(Array(classes.inputIDs.prefix(15)) == [1, 2, 15, 1, 16, 6, 16, 7, 8, 9, 9, 10, 11, 12, 13])
        #expect(classes.classIndex == [4, 6] + Array(repeating: 0, count: 14))
        #expect(classes.classMask == [1, 1] + Array(repeating: 0, count: 14))
        #expect(classes.queryIndex == Array(repeating: 0, count: 16))
        #expect(classes.queryMask == Array(repeating: 0, count: 16))
    }

    @Test(
        "Scalar and UTF-16 offsets convert back and forth on text with characters outside the BMP",
        arguments: ["a\u{1F600}b\u{20BB7}c", "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}", "e\u{301}\u{1D11E}."])
    func scalarAndUTF16RangesRoundTrip(text: String) {
        let count = text.unicodeScalars.count
        for lower in 0...count {
            for upper in lower...count {
                let utf16 = text.utf16Range(fromScalarRange: lower..<upper)
                #expect(utf16.flatMap(text.scalarRange(fromUTF16Range:)) == lower..<upper)
            }
        }
        #expect(text.utf16Range(fromScalarRange: 0..<(count + 1)) == nil)
        #expect(text.utf16Range(fromScalarRange: -1..<1) == nil)
    }

    @Test("Scalar and UTF-16 offsets convert back and forth on text with characters outside the BMP")
    func utf16RangesCountSurrogatePairs() {
        let text = "a\u{1F600}b"
        #expect(text.utf16Range(fromScalarRange: 1..<2) == 1..<3)
        #expect(text.utf16Range(fromScalarRange: 2..<3) == 3..<4)
        #expect(text.scalarRange(fromUTF16Range: 3..<4) == 2..<3)
        #expect(text.scalarRange(fromUTF16Range: 2..<3) == nil)
    }
}
