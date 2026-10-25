import Foundation
import OCRCore
import Testing

private struct GenerationFixture: Decodable {
    struct Logits: Decodable {
        let baseBits: UInt32
        let overrides: [[Int]]

        var values: [Float] {
            var values = [Float](repeating: Float(bitPattern: baseBits), count: 151_936)
            for pair in overrides { values[pair[0]] = Float(bitPattern: UInt32(pair[1])) }
            return values
        }
    }

    struct Penalty: Decodable {
        let name: String
        let logits: Logits
        let seen: [Int32]
        let expected: [[Int]]
        let argmax: Int32
    }

    struct Sequence: Decodable {
        let name: String
        let prompt: [Int32]
        let maxNewTokens: Int
        let steps: [Logits]
        let tokens: [Int32]
        let finish: String
    }

    let penalty: [Penalty]
    let sequences: [Sequence]
}

@Suite("OCR greedy decoding with repetition penalty")
struct OCRGenerationTests {
    @Test("Apply the repetition penalty bit for bit and pick the same argmax as the reference")
    func penalty() throws {
        let fixture = try OCRFixtures.load("generation", as: GenerationFixture.self)
        for item in fixture.penalty {
            let logits = item.logits.values
            let penalized = try OCRGeneration.penalized(logits, seen: Set(item.seen))
            for pair in item.expected {
                #expect(penalized[pair[0]].bitPattern == UInt32(pair[1]), "\(item.name) token \(pair[0])")
            }
            var generation = OCRGeneration(prompt: item.seen)
            #expect(try generation.select(logits) == item.argmax, "\(item.name)")
        }
    }

    @Test("Stop at either stop token or the token budget exactly where the reference loop stops")
    func sequences() throws {
        let fixture = try OCRFixtures.load("generation", as: GenerationFixture.self)
        for item in fixture.sequences {
            var generation = OCRGeneration(prompt: item.prompt, maxNewTokens: item.maxNewTokens)
            for step in item.steps {
                #expect(generation.finish == nil, "\(item.name)")
                _ = try generation.select(step.values)
            }
            #expect(generation.tokens == item.tokens, "\(item.name)")
            #expect(generation.finish?.rawValue == item.finish, "\(item.name)")
            #expect(throws: OCRError.generationFinished) { try generation.select(item.steps[0].values) }
        }
    }

    @Test("Reject non-finite logits and seen tokens outside the vocabulary")
    func invalidLogits() {
        var generation = OCRGeneration(prompt: [1])
        #expect(throws: OCRError.nonFiniteLogits) { try generation.select([0, .nan, 1]) }
        #expect(throws: OCRError.nonFiniteLogits) { try generation.select([0, .infinity, 1]) }
        #expect(throws: OCRError.invalidShape) { try OCRGeneration.penalized([0, 1], seen: [2]) }
        #expect(throws: OCRError.invalidShape) { try generation.select([]) }
        #expect(generation.tokens.isEmpty && generation.finish == nil)
    }
}
