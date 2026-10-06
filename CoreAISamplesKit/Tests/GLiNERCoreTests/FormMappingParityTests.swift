import Foundation
import GLiNERCore
import Testing

private let fixturesDirectory = URL(
    filePath: ProcessInfo.processInfo.environment["GLINER_FIXTURES_DIR"]
        ?? NSHomeDirectory() + "/Desktop/SelfApps/CoreLLM/models/coreai/gliner")
private let preregistrationPath = "data/g1_preregistration.json"
private let wordingVariantsPath = "data/g6_variants.json"
/// The records of the lab checkout that `GLINER_FIXTURES_DIR` (its `models/coreai/gliner`) belongs to.
private let recordsPath = "../../../docs/results/data/"
private let baselinePath = recordsPath + "2026-10-06-coreai-gliner25-pytorch-baseline/"
private let outputsPath = baselinePath + "e3_outputs_N_i.json"
private let itemsPath = baselinePath + "e3_items_N_i.tsv"
private let summaryPath = baselinePath + "e3_summary.json"
private let wordingSummaryPath = recordsPath + "2026-10-06-coreai-gliner25-option-wording/g6_summary.json"
private let filterPath = recordsPath + "2026-10-06-coreai-gliner25-candidate-filter/"
private let droppedPath = filterPath + "g6b_dropped.tsv"
private let judgmentsPath = filterPath + "g6b_judgments.tsv"
private let hasFixtures = [
    preregistrationPath, wordingVariantsPath, outputsPath, itemsPath, summaryPath, wordingSummaryPath, droppedPath,
    judgmentsPath,
].allSatisfy {
    FileManager.default.fileExists(atPath: fixturesDirectory.appending(path: $0).path(percentEncoded: false))
}

private func load<Value: Decodable>(_ type: Value.Type, _ path: String) throws -> Value {
    try JSONDecoder().decode(type, from: Data(contentsOf: fixturesDirectory.appending(path: path)))
}

/// `g1_preregistration.json`: form U1 and the entity labels of method (i).
private struct Preregistration: Decodable {
    struct Field: Decodable {
        let key: String
        let name: String
        let kind: String
        let unit: String?
        let options: [String]?
    }

    struct Entity: Decodable {
        let label: String?
        let description: String?

        /// `number_entities` also holds a "note", which is a string.
        init(from decoder: any Decoder) throws {
            let container = try? decoder.container(keyedBy: CodingKeys.self)
            label = try container?.decode(String.self, forKey: .label)
            description = try container?.decode(String.self, forKey: .description)
        }

        private enum CodingKeys: String, CodingKey {
            case label, description
        }
    }

    struct MethodI: Decodable {
        let valueEntities: [String: Entity]

        private enum CodingKeys: String, CodingKey {
            case valueEntities = "value_entities"
        }
    }

    let form: [Field]
    let numberEntities: [String: Entity]
    let methodI: MethodI

    private enum CodingKeys: String, CodingKey {
        case form
        case numberEntities = "number_entities"
        case methodI = "method_i"
    }
}

/// `g6_variants.json` and the choice in `g6_summary.json`: the wording of the clause classification.
private struct WordingVariants: Decodable {
    struct Wording: Decodable {
        /// `{name}` stands for the field's name.
        let task: String
        let labels: [String]
    }

    let statusItems: [String]
    let statusVariants: [String: Wording]
    let oilItem: String
    let oilVariants: [String: Wording]

    private enum CodingKeys: String, CodingKey {
        case statusItems = "status_items"
        case statusVariants = "status_variants"
        case oilItem = "oil_item"
        case oilVariants = "oil_variants"
    }
}

private struct WordingSummary: Decodable {
    struct Chosen: Decodable {
        let status: String
        let oil: String
    }

    let chosen: Chosen
}

/// `e3_outputs_N_i.json`: the spans of method (i) per sentence, and the classifications PyTorch ran.
private struct BaselineOutputs: Decodable {
    struct Sentence: Decodable {
        let number: Int
        let text: String
        let spans: [Span]
        /// The scoring of each field from its best span; a list field's has the clause and its probabilities.
        let out: [String: Item?]
        /// The clauses of the answer spans and their probabilities.
        let given: [String: Saved]

        private enum CodingKeys: String, CodingKey {
            case number = "case"
            case text, spans, out, given
        }
    }

    /// `[key, start, end, text, probability]`, offsets in code points.
    struct Span: Decodable {
        let key: String
        let range: Range<Int>
        let text: String
        let probability: Float

        init(from decoder: any Decoder) throws {
            var container = try decoder.unkeyedContainer()
            key = try container.decode(String.self)
            let start = try container.decode(Int.self)
            range = start..<(try container.decode(Int.self))
            text = try container.decode(String.self)
            probability = try container.decode(Float.self)
        }
    }

    struct Item: Decodable {
        let clause: String?
        let labels: [String]?
        let probs: [Float]?
    }

    struct Saved: Decodable {
        let clause: String
        let probs: [Float]
    }

    let outputs: [Sentence]
}

private struct BaselineSummary: Decodable {
    struct CorrectionRule: Decodable {
        let flaggedCases: [Int]

        private enum CodingKeys: String, CodingKey {
            case flaggedCases = "flagged_cases"
        }
    }

    let correctionRule: CorrectionRule

    private enum CodingKeys: String, CodingKey {
        case correctionRule = "correction_rule"
    }
}

/// A row of `e3_items_N_i.tsv`: one field of one sentence, scored from the field's best span (4 decimals).
private struct BaselineRow {
    static let header = "case\titem\texpected\tvalue\tmatch\tspan_text\tconfidence\tclause\tmax\tmargin\tprobs"

    let sentence: Int
    let key: String
    let value: String
    let spanText: String
    let confidence: Float?
    let clause: String
    let best: Float?
    let margin: Float?

    init?(_ line: Substring) {
        let columns = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
        guard columns.count == 11, let sentence = Int(columns[0]) else { return nil }
        self.sentence = sentence
        key = columns[1]
        value = columns[3]
        spanText = columns[5]
        confidence = Float(columns[6])
        clause = columns[7]
        best = Float(columns[8])
        margin = Float(columns[9])
    }
}

private func baselineRows() throws -> [BaselineRow] {
    let lines = try String(contentsOf: fixturesDirectory.appending(path: itemsPath), encoding: .utf8)
        .split(separator: "\n")
    try #require(lines.first == Substring(BaselineRow.header))
    return try lines.dropFirst().map { try #require(BaselineRow($0)) }
}

/// The rows of a GL6b table for rule R1 on dev30, each as `columns` joined by spaces.
private func gl6bRows(_ path: String, _ columns: [String]) throws -> [String] {
    let lines = try String(contentsOf: fixturesDirectory.appending(path: path), encoding: .utf8)
        .split(separator: "\n")
    let header = try #require(lines.first).split(separator: "\t").map(String.init)
    let rows = try lines.dropFirst().map { line in
        let values = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
        try #require(values.count == header.count)
        return Dictionary(uniqueKeysWithValues: zip(header, values))
    }
    return rows.filter { $0["rule"] == "R1" && $0["set"] == "dev30" }.map { row in
        columns.map { row[$0, default: "?"] }.joined(separator: " ")
    }
}

private struct SavedCall: Hashable {
    let task: String
    let labels: [String]
    let text: String
}

/// The classifications PyTorch ran with the GL1 wording (the field's name as the task, the options as labels): the
/// clause of each list field's best span, and the clauses of the answer spans.
private func savedClassifications(
    _ sentences: [BaselineOutputs.Sentence], _ preregistration: Preregistration
) -> [SavedCall: [Float]] {
    let fields = Dictionary(uniqueKeysWithValues: preregistration.form.map { ($0.key, $0) })
    var saved: [SavedCall: [Float]] = [:]
    for sentence in sentences {
        for (key, item) in sentence.out {
            guard let field = fields[key], let clause = item?.clause, let labels = item?.labels,
                let probabilities = item?.probs
            else { continue }
            saved[SavedCall(task: field.name, labels: labels, text: clause)] = probabilities
        }
        for (key, given) in sentence.given {
            guard let field = fields[key], let options = field.options else { continue }
            saved[SavedCall(task: field.name, labels: options, text: given.clause)] = given.probs
        }
    }
    return saved
}

/// Form U1 with the wording GL1 used to classify the clauses: the field's name as the task, the option names as labels.
private func gl1Wording(_ form: FormSchema) -> FormSchema {
    FormSchema(
        fields: form.fields.map { field in
            guard case .list(let options, _) = field.kind else { return field }
            var plain = field
            plain.kind = .list(options: options.map { FormOption(name: $0.name) })
            return plain
        })
}

/// What differs between the mapping of a field and the PyTorch row of that field; empty when they agree. The row
/// scores the best span, so the suggestions are compared on the candidate with the highest probability.
private func differences(
    _ mapped: FieldMapping, _ row: BaselineRow, spans: [BaselineOutputs.Span], text: String
) -> [String] {
    let expected = spans.filter { $0.key == row.key }
        .sorted { ($0.range.lowerBound, $0.range.upperBound) < ($1.range.lowerBound, $1.range.upperBound) }
    var checks = [
        (
            "candidate spans",
            mapped.candidates.map { "\($0.span.scalarRange) \($0.span.text) \($0.span.probability)" }
                == expected.map { "\($0.range) \($0.text) \($0.probability)" }
        ),
        ("state", mapped.state == (row.spanText.isEmpty ? .notDetected : .candidate)),
    ]
    if let best = mapped.candidates.max(by: { $0.span.probability < $1.span.probability }) {
        let first = best.suggestions.first
        let margin = first.map { $0.probability - (best.suggestions.dropFirst().first?.probability ?? 0) }
        let clause = String(String.UnicodeScalarView(Array(text.unicodeScalars)[best.clause]))
        checks += [
            ("span text", best.span.text == row.spanText),
            ("confidence", isClose(best.span.probability, row.confidence)),
        ]
        if case .number = mapped.field.kind {
            checks.append(("number", best.number?.value == Decimal(string: row.value)))
        } else {
            checks += [
                ("clause", clause == row.clause), ("option", first?.option == row.value),
                ("max", isClose(first?.probability, row.best)), ("margin", isClose(margin, row.margin)),
            ]
        }
    }
    return checks.filter { !$0.1 }.map(\.0)
}

/// Within the 4 decimals of the record; true when the record has no value.
private func isClose(_ value: Float?, _ recorded: Float?) -> Bool {
    guard let recorded else { return true }
    return abs((value ?? .nan) - recorded) <= 1e-4
}

@Suite(
    .enabled(if: hasFixtures, "GLINER_FIXTURES_DIR must be the lab's models/coreai/gliner, with its PyTorch records"))
struct FormMappingParityTests {
    @Test("Labels and descriptions built from form U1 (6 fields) are the same strings as PyTorch method (i)")
    func labelsMatchMethodI() throws {
        let preregistration = try load(Preregistration.self, preregistrationPath)
        let form = try formU1()
        let expected = try preregistration.form.map { field in
            let entity = preregistration.numberEntities[field.key] ?? preregistration.methodI.valueEntities[field.key]
            return EntityLabel(name: try #require(entity?.label), description: try #require(entity?.description))
        }
        try #require(expected.count == 6)
        #expect(form.extractionSchema() == .entities(expected))
        #expect(form.fields.map(\.key) == preregistration.form.map(\.key))
        #expect(form.fields.map(\.name) == preregistration.form.map(\.name))
        let plain = gl1Wording(form)
        for (field, reference) in zip(plain.fields, preregistration.form) {
            switch field.kind {
            case .number(let unit):
                #expect(reference.kind == "number" && unit == reference.unit, "\(field.key)")
            case .list(let options, _):
                #expect(reference.kind == "list" && options.map(\.name) == reference.options, "\(field.key)")
                #expect(
                    plain.classificationSchema(for: field)
                        == .classification(name: reference.name, labels: reference.options ?? []), "\(field.key)")
            }
        }
    }

    @Test(
        "Form U1 classifies the clauses with the wording GL6 chose: the status wording for abnormal sound and vibration, the oil wording for oil leak, the field name and options for next action"
    )
    func classificationUsesTheGL6Wording() throws {
        let variants = try load(WordingVariants.self, wordingVariantsPath)
        let chosen = try load(WordingSummary.self, wordingSummaryPath).chosen
        let status = try #require(variants.statusVariants[chosen.status])
        let oil = try #require(variants.oilVariants[chosen.oil])
        let form = try formU1()
        var listFields = 0
        for field in form.fields {
            guard case .list(let options, _) = field.kind else { continue }
            listFields += 1
            let wording = variants.statusItems.contains(field.key) ? status : field.key == variants.oilItem ? oil : nil
            let expected = GLiNERSchema.classification(
                name: wording?.task.replacing("{name}", with: field.name) ?? field.name,
                labels: wording?.labels ?? options.map(\.name))
            #expect(form.classificationSchema(for: field) == expected, "\(field.key)")
        }
        #expect(listFields == 4)
    }

    @Test(
        "With the candidate filter .none, mapping the saved extraction and classification outputs (30 sentences) gives the candidates and suggestions of PyTorch e3_items_N_i.tsv for every field in all 30 sentences"
    )
    func mappingMatchesPyTorch() throws {
        let preregistration = try load(Preregistration.self, preregistrationPath)
        let sentences = try load(BaselineOutputs.self, outputsPath).outputs
        let rows = try baselineRows()
        let flagged = try load(BaselineSummary.self, summaryPath).correctionRule.flaggedCases
        try #require(sentences.count == 30 && rows.count == 180)
        let form = gl1Wording(try formU1())
        let mapper = FormMapper(schema: form, correctionWords: try correctionWords(), candidateFilter: .none)
        let labelIndex = Dictionary(uniqueKeysWithValues: form.fields.enumerated().map { ($1.key, $0) })
        let saved = savedClassifications(sentences, preregistration)
        var mismatches: [String] = []
        for sentence in sentences {
            let spans = try sentence.spans.map { span in
                ExtractedSpan(
                    labelIndex: try #require(labelIndex[span.key]), scalarRange: span.range, text: span.text,
                    probability: span.probability)
            }
            let mapping = mapper.map(text: sentence.text, extraction: makeExtraction(spans)) { text, schema in
                guard case .classification(let task, let labels) = schema else {
                    return ClassificationResult(probabilities: [])
                }
                // PyTorch classified only the clause of the best span (and of the answer spans); other spans get
                // zeros, and their suggestions are not compared.
                let call = SavedCall(task: task, labels: labels, text: text)
                return ClassificationResult(probabilities: saved[call] ?? Array(repeating: 0, count: labels.count))
            }
            let sentenceRows = rows.filter { $0.sentence == sentence.number }
            try #require(sentenceRows.count == form.fields.count)
            for row in sentenceRows {
                let mapped = try #require(mapping.fields.first { $0.field.key == row.key })
                let found = differences(mapped, row, spans: sentence.spans, text: sentence.text)
                if !found.isEmpty { mismatches.append("#\(sentence.number) \(row.key): \(found)") }
            }
            if mapping.corrections.isEmpty == flagged.contains(sentence.number) {
                mismatches.append("#\(sentence.number): correction cues \(mapping.corrections)")
            }
        }
        #expect(mismatches.isEmpty, "\(mismatches.joined(separator: "\n"))")
    }

    @Test("Applying R1 to the saved dev30 outputs drops the same candidates as the GL6b list (its dev30 rows)")
    func filterDropsTheGL6bCandidates() throws {
        let sentences = try load(BaselineOutputs.self, outputsPath).outputs
        try #require(sentences.count == 30)
        let form = try formU1()
        let mapper = FormMapper(schema: form, correctionWords: [])
        let labelIndex = Dictionary(uniqueKeysWithValues: form.fields.enumerated().map { ($1.key, $0) })
        var dropped: [String] = []
        var kept: [String] = []
        for sentence in sentences {
            let spans = try sentence.spans.map { span in
                ExtractedSpan(
                    labelIndex: try #require(labelIndex[span.key]), scalarRange: span.range, text: span.text,
                    probability: span.probability)
            }
            let mapping = mapper.map(text: sentence.text, extraction: makeExtraction(spans)) { _, _ in
                ClassificationResult(probabilities: [])
            }
            let scalars = Array(sentence.text.unicodeScalars)
            for field in mapping.fields {
                dropped += field.droppedCandidates.map { candidate in
                    let range = candidate.span.scalarRange
                    return [
                        "\(sentence.number)", field.field.key, candidate.span.text, "\(range.lowerBound)",
                        "\(range.upperBound)", String(String.UnicodeScalarView(scalars[candidate.clause])),
                    ].joined(separator: " ")
                }
                // GL6b's judgments give what each field keeps: the candidate count and the most probable text.
                let best = field.candidates.max { $0.span.probability < $1.span.probability }
                kept.append("\(sentence.number) \(field.field.key) \(field.candidates.count) \(best?.span.text ?? "")")
            }
        }
        let expectedDropped = try gl6bRows(droppedPath, ["id", "field", "text", "start", "end", "clause"])
        let expectedKept = try gl6bRows(judgmentsPath, ["id", "field", "candidates", "primary_text"])
        try #require(expectedDropped.count == 12 && expectedKept.count == 180)
        #expect(dropped.sorted() == expectedDropped.sorted())
        #expect(kept.sorted() == expectedKept.sorted())
    }
}
