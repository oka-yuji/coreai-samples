import Foundation
import Testing

@testable import OCRCore

@Suite("OCR prompt tokens and positions")
struct OCRPromptTests {
    private static let instructions = [
        "f1Text448": OCRPrompt.text, "layout1036": OCRPrompt.layout, "table300x200": OCRPrompt.table,
    ]

    private static func prompt(_ item: PromptFixture.Case, _ fixture: PromptFixture) throws -> OCRPrompt {
        let encodings = Dictionary(uniqueKeysWithValues: fixture.segments.map { ($0.text, $0.ids) })
        let instruction = try #require(instructions[item.name])
        return try OCRPrompt(instruction: instruction, gridHeight: item.gridHeight, gridWidth: item.gridWidth) { text in
            try #require(encodings[text], "missing segment \(text.debugDescription)")
        }
    }

    @Test(
        "Match the chat template tokens, image slot, three-axis positions, and rope delta of the reference prompts",
        arguments: ["f1Text448", "layout1036", "table300x200"])
    func promptContract(name: String) throws {
        let fixture = try OCRFixtures.load("prompts", as: PromptFixture.self)
        let item = try #require(fixture.cases.first { $0.name == name })
        #expect(Self.instructions[name] == item.prompt)
        let prompt = try Self.prompt(item, fixture)
        #expect(prompt.ids == item.ids)
        #expect(prompt.imageRange == item.imageStart..<(item.imageStart + item.imageCount))
        #expect(prompt.ids[prompt.imageRange].allSatisfy { $0 == OCRPrompt.imageToken })
        #expect([prompt.positions.temporal, prompt.positions.height, prompt.positions.width] == item.positions)
        #expect(prompt.ropeDelta == item.ropeDelta)
    }

    @Test("Merge the mRoPE sections within 1e-6 of the reference for every F1 prompt row and for decode positions")
    func textRotary() throws {
        let fixture = try OCRFixtures.load("prompts", as: PromptFixture.self)
        let rotary = fixture.textRotary
        let item = try #require(fixture.cases.first { $0.name == "f1Text448" })
        let prompt = try Self.prompt(item, fixture)
        let merged = OCRTextRotary.cosSin(prompt.positions)
        let cosine = OCRFixtures.floats(rotary.cosineTableBits)
        let sine = OCRFixtures.floats(rotary.sineTableBits)
        var difference: Float = 0
        for row in 0..<item.ids.count {
            for index in 0..<OCRTextRotary.width {
                let table = item.positions[rotary.sectionAxes[index]][row] * 64 + rotary.frequencyIndex[index]
                difference = max(difference, abs(merged.cos[row * 128 + index] - cosine[table]))
                difference = max(difference, abs(merged.sin[row * 128 + index] - sine[table]))
            }
        }
        #expect(difference <= 1e-6)
        for sample in rotary.rows {
            let slice = (sample.index * 128)..<(sample.index * 128 + 128)
            #expect(OCRFixtures.maximumDifference(merged.cos[slice], OCRFixtures.floats(sample.cosineBits)) <= 1e-6)
            #expect(OCRFixtures.maximumDifference(merged.sin[slice], OCRFixtures.floats(sample.sineBits)) <= 1e-6)
        }
        for sample in rotary.decode {
            let step = OCRTextRotary.cosSin(
                OCRRopePositions(writePosition: sample.writePosition, count: 1, delta: prompt.ropeDelta))
            #expect(OCRFixtures.maximumDifference(step.cos, OCRFixtures.floats(sample.cosineBits)) <= 1e-6)
            #expect(OCRFixtures.maximumDifference(step.sin, OCRFixtures.floats(sample.sineBits)) <= 1e-6)
        }
        let ulps = zip(OCRTextRotary.inverseFrequencies.map(\.bitPattern), rotary.inverseFrequencyBits).map {
            abs(Int($0) - Int($1))
        }
        #expect(ulps.allSatisfy { $0 <= 1 })
    }

    @Test("Continue decode positions from the write position plus the rope delta on all three axes")
    func decodePositions() {
        let positions = OCRRopePositions(writePosition: 286, count: 3, delta: -240)
        #expect(
            positions.temporal == [46, 47, 48] && positions.height == [46, 47, 48] && positions.width == [46, 47, 48])
    }

    @Test("Reject image grids that cannot be merged into two by two tokens")
    func invalidGrid() {
        #expect(throws: OCRError.invalidShape) {
            try OCRPrompt(instruction: OCRPrompt.text, gridHeight: 3, gridWidth: 4) { _ in [] }
        }
        #expect(throws: OCRError.invalidShape) {
            try OCRPrompt(instruction: OCRPrompt.text, gridHeight: 0, gridWidth: 4) { _ in [] }
        }
    }
}
