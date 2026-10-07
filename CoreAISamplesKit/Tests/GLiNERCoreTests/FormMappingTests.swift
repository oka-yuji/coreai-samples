import Foundation
import GLiNERCore
import Testing

struct MissingResource: Error {
    let path: String
}

/// A JSON file of this target's `Resources` folder.
func resource<Value: Decodable>(_ type: Value.Type, _ path: String) throws -> Value {
    guard let folder = Bundle.module.url(forResource: "Resources", withExtension: nil) else {
        throw MissingResource(path: path)
    }
    return try JSONDecoder().decode(type, from: Data(contentsOf: folder.appending(path: path)))
}

/// Form U1, with the wording GL6 chose for the clause classification.
func formU1() throws -> FormSchema { try resource(FormSchema.self, "forms/inspection_u1.json") }

func correctionWords() throws -> [String] { try resource([String].self, "correction_words.json") }

func makeSpan(_ label: Int, _ range: Range<Int>, in text: String, probability: Float = 0.9) -> ExtractedSpan {
    let scalars = Array(text.unicodeScalars)[range]
    return ExtractedSpan(
        labelIndex: label, scalarRange: range, text: String(String.UnicodeScalarView(scalars)), probability: probability
    )
}

func makeExtraction(_ spans: [ExtractedSpan]) -> ExtractionResult {
    ExtractionResult(spans: spans, abstained: [], tokenCapacity: PreparedInput.tokenCapacities[0])
}

func scalarRanges(of piece: String, in text: String) -> [Range<Int>] {
    Array(text.unicodeScalars).ranges(of: Array(piece.unicodeScalars))
}

private func decimal(_ text: String) throws -> Decimal { try #require(Decimal(string: text)) }

/// `Resources/mapping_examples.json`: notes in Japanese for the tests below.
struct MappingExamples: Decodable {
    struct NumberExample: Decodable, CustomTestStringConvertible {
        let text: String
        let value: String?
        let unit: String?
        let normalizedUnit: String?

        var testDescription: String { text }
    }

    struct Sentence: Decodable {
        let text: String
        let piece: String
    }

    struct Cues: Decodable {
        let text: String
        let words: [String]
    }

    struct FilterExample: Decodable {
        /// A span `piece` of field `key`, in the clause `clause`, and whether R1 keeps it.
        struct Candidate: Decodable {
            let key: String
            let clause: String
            let piece: String
            let kept: Bool
        }

        let text: String
        let candidates: [Candidate]
    }

    let numbers: [NumberExample]
    /// Two pressures with a correction word between them; `piece` is the word.
    let twoValues: Sentence
    /// A pressure in kana; `piece` is the number.
    let kanaNumber: Sentence
    /// Two clauses that end in the same word; `piece` is the word.
    let twoClauses: Sentence
    let cues: Cues
    let filter: FilterExample
}

struct FormMappingTests {
    let examples: MappingExamples
    let form: FormSchema
    /// Keeps every candidate; the filter has its own tests.
    let mapper: FormMapper

    init() throws {
        examples = try resource(MappingExamples.self, "mapping_examples.json")
        form = try formU1()
        mapper = FormMapper(schema: form, correctionWords: try correctionWords(), candidateFilter: .none)
    }

    /// The spans of `examples.filter`, and whether R1 keeps each, by `outcome(of:)`'s description.
    func filterSpans() throws -> (spans: [ExtractedSpan], kept: [String: Bool]) {
        let text = examples.filter.text
        var kept: [String: Bool] = [:]
        let spans = try examples.filter.candidates.map { candidate in
            let clause = try #require(scalarRanges(of: candidate.clause, in: text).first)
            let piece = try #require(scalarRanges(of: candidate.piece, in: candidate.clause).first)
            let range = (clause.lowerBound + piece.lowerBound)..<(clause.lowerBound + piece.upperBound)
            kept["\(candidate.key) \(candidate.clause) \(range)"] = candidate.kept
            return makeSpan(try #require(form.fields.firstIndex { $0.key == candidate.key }), range, in: text)
        }
        return (spans, kept)
    }

    /// Whether each candidate of `mapping` of `examples.filter` is kept, by field key, clause text and span range.
    func outcome(of mapping: FormMapping) -> [String: Bool] {
        let scalars = Array(examples.filter.text.unicodeScalars)
        var outcome: [String: Bool] = [:]
        for field in mapping.fields {
            for (candidates, kept) in [(field.candidates, true), (field.droppedCandidates, false)] {
                for candidate in candidates {
                    let clause = String(String.UnicodeScalarView(scalars[candidate.clause]))
                    outcome["\(field.field.key) \(clause) \(candidate.span.scalarRange)"] = kept
                }
            }
        }
        return outcome
    }

    @Test(
        "A field without spans is not detected, both values stay as candidates in a sentence with two, and a sentence with a correction word gets a cue"
    )
    func keepsBothValuesAndMarksTheCorrection() throws {
        let text = examples.twoValues.text
        let old = try #require(scalarRanges(of: "0.45", in: text).first)
        let new = try #require(scalarRanges(of: "0.54MPa", in: text).first)
        let cue = try #require(scalarRanges(of: examples.twoValues.piece, in: text).first)
        // The later value scores higher, as a correction would; the candidates still come in text order.
        let spans = [makeSpan(0, new, in: text, probability: 0.95), makeSpan(0, old, in: text, probability: 0.8)]
        let mapping = mapper.map(text: text, extraction: makeExtraction(spans)) { _, _ in
            Issue.record("No list field has a span to classify")
            return ClassificationResult(probabilities: [])
        }
        let numbers = [
            ParsedNumber(value: try decimal("0.45"), unit: nil, normalizedUnit: nil),
            ParsedNumber(value: try decimal("0.54"), unit: "MPa", normalizedUnit: "MPa"),
        ]
        #expect(mapping.fields.map(\.state) == [.candidate] + Array(repeating: .notDetected, count: 5))
        #expect(mapping.fields[0].candidates.map(\.span.scalarRange) == [old, new])
        #expect(mapping.fields[0].candidates.map(\.number) == numbers)
        #expect(mapping.corrections == [CorrectionCue(range: cue, word: examples.twoValues.piece)])
    }

    @Test("Correction words are found left to right without overlap, the longest where several start at one place")
    func findsTheLongestCorrectionWords() throws {
        let cues = CorrectionCue.find(in: examples.cues.text, words: try correctionWords())
        let scalars = Array(examples.cues.text.unicodeScalars)
        #expect(cues.map(\.word) == examples.cues.words)
        #expect(cues.map { String(String.UnicodeScalarView(scalars[$0.range])) } == examples.cues.words)
    }

    @Test(
        "Number parsing turns 0.45MPa, 72 with the kanji for degree, full-width 0.5MPa and 72 with the degree Celsius sign into a number and a unit, and a number in kana stays a candidate as written",
        arguments: try resource(MappingExamples.self, "mapping_examples.json").numbers)
    func parsesTheNumberAndUnit(example: MappingExamples.NumberExample) throws {
        let expected = try example.value.map {
            ParsedNumber(value: try decimal($0), unit: example.unit, normalizedUnit: example.normalizedUnit)
        }
        #expect(ParsedNumber(parsing: example.text) == expected)
    }

    @Test(
        "Number parsing turns 0.45MPa, 72 with the kanji for degree, full-width 0.5MPa and 72 with the degree Celsius sign into a number and a unit, and a number in kana stays a candidate as written"
    )
    func keepsANumberInKanaAsACandidate() throws {
        let text = examples.kanaNumber.text
        let range = try #require(scalarRanges(of: examples.kanaNumber.piece, in: text).first)
        let mapping = mapper.map(text: text, extraction: makeExtraction([makeSpan(0, range, in: text)])) { _, _ in
            Issue.record("No list field has a span to classify")
            return ClassificationResult(probabilities: [])
        }
        #expect(mapping.fields[0].state == .candidate)
        #expect(mapping.fields[0].candidates.map(\.span.text) == [examples.kanaNumber.piece])
        #expect(mapping.fields[0].candidates.map(\.number) == [nil])
        #expect(mapping.corrections.isEmpty)
    }

    @Test(
        "Each span of a list field is a candidate whose suggestions are the two best options of its clause, by option name"
    )
    func suggestsTheTwoBestOptionsOfEachClause() throws {
        let text = examples.twoClauses.text
        let scalars = Array(text.unicodeScalars)
        let comma = try #require(scalars.firstIndex(of: "\u{3001}"))
        let clauses = [0..<comma, (comma + 1)..<scalars.count]
        let clauseTexts = clauses.map { String(String.UnicodeScalarView(scalars[$0])) }
        let words = scalarRanges(of: examples.twoClauses.piece, in: text)
        try #require(words.count == 2)
        // Oil leak and abnormal sound both point at both words, as in sentence 12 of the PyTorch run.
        let listFields = [2, 3]
        let spans = listFields.flatMap { label in words.map { makeSpan(label, $0, in: text) } }
        var calls: [(text: String, schema: GLiNERSchema)] = []
        let mapping = mapper.map(text: text, extraction: makeExtraction(spans)) { clause, schema in
            calls.append((clause, schema))
            guard case .classification(_, let labels) = schema else { return ClassificationResult(probabilities: []) }
            let rising = labels.indices.map { Float($0 + 1) }
            return ClassificationResult(probabilities: clause == clauseTexts[0] ? rising : rising.reversed())
        }
        var schemas: [GLiNERSchema] = []
        for index in listFields {
            let field = form.fields[index]
            guard case .list(let options, let modelTask) = field.kind else {
                Issue.record("\(field.key) is not a list field")
                continue
            }
            let schema = GLiNERSchema.classification(name: modelTask ?? field.name, labels: options.map(\.modelLabel))
            schemas += [schema, schema]
            let names = options.map(\.name)
            let best = Float(names.count)
            #expect(mapping.fields[index].candidates.map(\.clause) == clauses)
            #expect(
                mapping.fields[index].candidates.map(\.suggestions) == [
                    [
                        OptionSuggestion(option: names[names.count - 1], probability: best),
                        OptionSuggestion(option: names[names.count - 2], probability: best - 1),
                    ],
                    [
                        OptionSuggestion(option: names[0], probability: best),
                        OptionSuggestion(option: names[1], probability: best - 1),
                    ],
                ])
        }
        #expect(calls.map(\.text) == clauseTexts + clauseTexts)
        #expect(calls.map(\.schema) == schemas)
    }

    @Test(
        "The classification requests are the clause and schema pairs map asks for, once each in the order first asked, and map gives the same result from their looked-up answers"
    )
    func listsEachClassificationOnce() throws {
        let text = examples.twoClauses.text
        let scalars = Array(text.unicodeScalars)
        let comma = try #require(scalars.firstIndex(of: "\u{3001}"))
        let clauseTexts = [0..<comma, (comma + 1)..<scalars.count].map { String(String.UnicodeScalarView(scalars[$0])) }
        let words = scalarRanges(of: examples.twoClauses.piece, in: text)
        try #require(words.count == 2)
        // Oil leak has two spans in the first clause and one in the second; abnormal sound one in the second; the
        // pressure span (a number field) is never classified.
        let spans = [
            makeSpan(0, words[0], in: text), makeSpan(2, 0..<comma, in: text), makeSpan(2, words[0], in: text),
            makeSpan(2, words[1], in: text), makeSpan(3, words[1], in: text),
        ]
        let oilLeak = form.classificationSchema(for: form.fields[2])
        let sound = form.classificationSchema(for: form.fields[3])
        let requests = mapper.classificationRequests(text: text, extraction: makeExtraction(spans))
        #expect(
            requests == [
                ClassificationRequest(clause: clauseTexts[0], schema: oilLeak),
                ClassificationRequest(clause: clauseTexts[1], schema: oilLeak),
                ClassificationRequest(clause: clauseTexts[1], schema: sound),
            ])

        func answer(_ clause: String, _ schema: GLiNERSchema) -> ClassificationResult {
            guard case .classification(_, let labels) = schema else { return ClassificationResult(probabilities: []) }
            let rising = labels.indices.map { Float($0 + 1) }
            return ClassificationResult(probabilities: clause == clauseTexts[0] ? rising : rising.reversed())
        }
        let answers = Dictionary(uniqueKeysWithValues: requests.map { ($0, answer($0.clause, $0.schema)) })
        let looked = mapper.map(text: text, extraction: makeExtraction(spans)) { clause, schema in
            answers[ClassificationRequest(clause: clause, schema: schema)] ?? ClassificationResult(probabilities: [])
        }
        #expect(looked == mapper.map(text: text, extraction: makeExtraction(spans), classify: answer))
    }

    @Test(
        "A candidate in a clause with its own field name is kept, one in a clause with only another field's name is dropped, and one in a clause without field names is kept (hand-made example)"
    )
    func dropsCandidatesFromAnotherFieldsClause() throws {
        let (spans, kept) = try filterSpans()
        // The example also has a number candidate in a clause with a list field's name, and a list candidate in a
        // clause with only a number field's name: R1 reads list fields and their names only.
        let mapping = FormMapper(schema: form, correctionWords: []).map(
            text: examples.filter.text, extraction: makeExtraction(spans)
        ) { _, _ in ClassificationResult(probabilities: []) }
        #expect(outcome(of: mapping) == kept)
        #expect(
            mapping.fields.filter { $0.state == .notDetected }.map(\.field.key) == [
                "discharge_pressure", "abnormal_sound",
            ])
    }

    @Test("The clause of a dropped candidate is not classified, and the filter .none keeps every candidate")
    func classifiesOnlyTheKeptCandidates() throws {
        let (spans, kept) = try filterSpans()
        let text = examples.filter.text
        let extraction = makeExtraction(spans)
        let expected = try examples.filter.candidates.compactMap { candidate -> ClassificationRequest? in
            let field = try #require(form.fields.first { $0.key == candidate.key })
            guard candidate.kept, case .list = field.kind else { return nil }
            return ClassificationRequest(clause: candidate.clause, schema: form.classificationSchema(for: field))
        }
        let filtered = FormMapper(schema: form, correctionWords: [])
        #expect(Set(filtered.classificationRequests(text: text, extraction: extraction)) == Set(expected))

        let unfiltered = FormMapper(schema: form, correctionWords: [], candidateFilter: .none)
        let mapping = unfiltered.map(text: text, extraction: extraction) { _, _ in
            ClassificationResult(probabilities: [])
        }
        #expect(outcome(of: mapping) == kept.mapValues { _ in true })
    }

    @Test("Rule R1+N1 leaves list fields to rule R1 and keeps every number candidate of .none and .clauseFieldName")
    func appliesRuleN1ToNumberFieldsOnly() throws {
        func mapper(_ filter: CandidateFilter) -> FormMapper {
            FormMapper(schema: form, correctionWords: [], candidateFilter: filter)
        }
        func noClassification(_: String, _: GLiNERSchema) -> ClassificationResult {
            ClassificationResult(probabilities: [])
        }
        // The R1 example has no equipment number: R1+N1 keeps, drops and classifies what R1 does.
        let (spans, kept) = try filterSpans()
        let text = examples.filter.text
        let extraction = makeExtraction(spans)
        let both = mapper(.clauseFieldNameAndEquipmentNumber)
        #expect(outcome(of: both.map(text: text, extraction: extraction, classify: noClassification)) == kept)
        #expect(
            both.classificationRequests(text: text, extraction: extraction)
                == mapper(.clauseFieldName).classificationRequests(text: text, extraction: extraction))

        // An equipment number, P-3, as a pressure and as an oil leak: only the pressure candidate is dropped.
        let note = "P-3 0.5MPa x"
        let equipment = makeExtraction([
            makeSpan(0, 0..<3, in: note), makeSpan(0, 4..<10, in: note), makeSpan(2, 0..<3, in: note),
        ])
        let mapped = both.map(text: note, extraction: equipment, classify: noClassification)
        #expect(mapped.fields[0].candidates.map(\.span.text) == ["0.5MPa"])
        #expect(mapped.fields[0].droppedCandidates.map(\.span.text) == ["P-3"])
        #expect(mapped.fields[2].candidates.map(\.span.text) == ["P-3"])

        let n1 = try EquipmentNumberExamples.load()
        for filter in [CandidateFilter.none, .clauseFieldName] {
            let all = mapper(filter).map(text: note, extraction: equipment, classify: noClassification)
            #expect(all.fields[0].candidates.map(\.span.text) == ["P-3", "0.5MPa"], "\(filter)")
            #expect(all.fields.flatMap(\.droppedCandidates).isEmpty, "\(filter)")
            for example in n1.examples + n1.folding + n1.boundaries {
                #expect(try keeps(example, filter), "\(filter): \(example.why)")
            }
        }
    }
}
