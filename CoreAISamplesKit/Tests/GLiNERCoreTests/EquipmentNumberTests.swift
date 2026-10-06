import Foundation
import Testing

@testable import GLiNERCore

/// `Resources/equipment_number_examples.json`: the words and the ten examples of rule N1 as the Python reference
/// (`g9_remedies.py`) has them, and examples of the folding.
struct EquipmentNumberExamples: Decodable {
    enum Decision: String, Decodable {
        case keep, drop
    }

    struct Example: Decodable, CustomTestStringConvertible {
        /// A note whose only span is the first `candidate` in it, of the field `field`.
        let text: String
        let field: String
        let candidate: String
        let expect: Decision
        let why: String

        var testDescription: String { why }
    }

    let words: EquipmentNumberWords
    let examples: [Example]
    let folding: [Example]

    static func load() throws -> EquipmentNumberExamples {
        try resource(EquipmentNumberExamples.self, "equipment_number_examples.json")
    }
}

/// Whether `filter` keeps the candidate of `example`.
func keeps(_ example: EquipmentNumberExamples.Example, _ filter: CandidateFilter) throws -> Bool {
    let form = try formU1()
    let label = try #require(form.fields.firstIndex { $0.key == example.field })
    let range = try #require(scalarRanges(of: example.candidate, in: example.text).first)
    let mapping = FormMapper(schema: form, correctionWords: [], candidateFilter: filter).map(
        text: example.text, extraction: makeExtraction([makeSpan(label, range, in: example.text)])
    ) { _, _ in ClassificationResult(probabilities: []) }
    let field = mapping.fields[label]
    try #require(field.candidates.count + field.droppedCandidates.count == 1)
    return !field.candidates.isEmpty
}

struct EquipmentNumberTests {
    @Test("Rule N1 reads the equipment markers and numeral words of the Python reference")
    func readsTheReferenceWords() throws {
        let examples = try EquipmentNumberExamples.load()
        #expect(EquipmentNumberWords.bundled == examples.words)
        #expect(examples.examples.count == 10)
    }

    @Test(
        "Rule N1 drops a number candidate whose every digit run is attached to an equipment marker and keeps one with a plain run",
        arguments: try EquipmentNumberExamples.load().examples)
    func dropsEquipmentNumbers(example: EquipmentNumberExamples.Example) throws {
        #expect(try keeps(example, .clauseFieldNameAndEquipmentNumber) == (example.expect == .keep))
    }

    @Test(
        "Rule N1 compares full-width letters, digits and punctuation as half-width lower case without moving positions",
        arguments: try EquipmentNumberExamples.load().folding)
    func comparesFoldedText(example: EquipmentNumberExamples.Example) throws {
        #expect(try keeps(example, .clauseFieldNameAndEquipmentNumber) == (example.expect == .keep))
    }
}
