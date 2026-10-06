import Foundation
import Testing

@testable import GLiNERCore

/// `Resources/equipment_number_examples.json`: the words and the ten examples of rule N1 recorded from the lab's
/// `data/g9_variants.json` (`n1.before`, `n1.after`, `selftest.n1`) and `data/g6b_numerals.json` (`tokens`), the data
/// of the Python reference `g9_remedies.py`; then examples of the folding, and of where runs and markers end.
struct EquipmentNumberExamples: Decodable {
    enum Decision: String, Decodable {
        case keep, drop
    }

    struct Example: Decodable, Equatable, CustomTestStringConvertible {
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
    let boundaries: [Example]

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

/// The lab's `models/coreai/gliner`, as `FormMappingParityTests` finds it.
private let labDirectory = URL(
    filePath: ProcessInfo.processInfo.environment["GLINER_FIXTURES_DIR"]
        ?? NSHomeDirectory() + "/Desktop/SelfApps/CoreLLM/models/coreai/gliner")
private let labVariantsPath = "data/g9_variants.json"
private let labNumeralsPath = "data/g6b_numerals.json"
private let hasLabFiles = [labVariantsPath, labNumeralsPath].allSatisfy {
    FileManager.default.fileExists(atPath: labDirectory.appending(path: $0).path(percentEncoded: false))
}

/// `g9_variants.json`: the markers of rule N1 and the examples of its self-test.
private struct LabVariants: Decodable {
    struct Markers: Decodable {
        let before: [String]
        let after: [String]
    }

    struct SelfTest: Decodable {
        let n1: [EquipmentNumberExamples.Example]
    }

    let n1: Markers
    let selftest: SelfTest
}

/// `g6b_numerals.json`: the numeral words.
private struct LabNumerals: Decodable {
    let tokens: [String]
}

struct EquipmentNumberTests {
    @Test("Rule N1 reads the equipment markers and numeral words recorded from the Python reference")
    func readsTheRecordedWords() throws {
        let examples = try EquipmentNumberExamples.load()
        #expect(EquipmentNumberWords.bundled == examples.words)
        // A test with no arguments passes without running.
        let counts: [Int] = [examples.examples.count, examples.folding.count, examples.boundaries.count]
        #expect(counts == [10, 6, 3])
    }

    @Test(
        "Rule N1's markers, numeral words and ten examples are the ones in the lab's g9_variants.json and g6b_numerals.json",
        .enabled(if: hasLabFiles, "GLINER_FIXTURES_DIR must be the lab's models/coreai/gliner, with its data folder"))
    func matchesTheLabFiles() throws {
        let decoder = JSONDecoder()
        let variants = try decoder.decode(
            LabVariants.self, from: Data(contentsOf: labDirectory.appending(path: labVariantsPath)))
        let numerals = try decoder.decode(
            LabNumerals.self, from: Data(contentsOf: labDirectory.appending(path: labNumeralsPath)))
        let lab = EquipmentNumberWords(before: variants.n1.before, after: variants.n1.after, numerals: numerals.tokens)
        #expect(EquipmentNumberWords.bundled == lab)
        #expect(try EquipmentNumberExamples.load().examples == variants.selftest.n1)
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

    @Test(
        "Rule N1 reads the longest numeral word, ends a run at a decimal point, and takes a marker only right next to the run",
        arguments: try EquipmentNumberExamples.load().boundaries)
    func findsRunAndMarkerBoundaries(example: EquipmentNumberExamples.Example) throws {
        #expect(try keeps(example, .clauseFieldNameAndEquipmentNumber) == (example.expect == .keep))
    }
}
