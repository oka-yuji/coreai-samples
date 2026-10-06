/// The candidates for each field of a form, from one note. Nothing is chosen: people confirm the values.
public struct FormMapping: Equatable, Sendable {
    /// One per field, in form order.
    public var fields: [FieldMapping]
    /// The correction words in the note, found by rule.
    public var corrections: [CorrectionCue]

    public init(fields: [FieldMapping], corrections: [CorrectionCue]) {
        self.fields = fields
        self.corrections = corrections
    }
}

public struct FieldMapping: Equatable, Sendable {
    public enum State: Equatable, Sendable {
        /// One or more spans for people to confirm.
        case candidate
        /// No span. The model may have missed a mention, so this does not mean the note leaves the field out.
        case notDetected
    }

    public var field: FormField
    /// The spans of the field's label that `FormMapper.candidateFilter` keeps, in text order.
    public var candidates: [FieldCandidate]
    /// The spans the filter drops, in text order. People do not see them; the export keeps them.
    public var droppedCandidates: [FieldCandidate]

    public var state: State { candidates.isEmpty ? .notDetected : .candidate }

    public init(field: FormField, candidates: [FieldCandidate], droppedCandidates: [FieldCandidate] = []) {
        self.field = field
        self.candidates = candidates
        self.droppedCandidates = droppedCandidates
    }
}

public struct FieldCandidate: Equatable, Sendable {
    public var span: ExtractedSpan
    /// The clause that holds the span: the text between delimiters, in Unicode scalars of the note.
    public var clause: Range<Int>
    /// The number and unit in the span text of a number field. Nil when the text has no digits, and in a list field.
    public var number: ParsedNumber?
    /// The two best options for the clause of a list field, highest first; empty in a number field and in a dropped
    /// candidate, whose clause is not classified. The probabilities are not calibrated.
    public var suggestions: [OptionSuggestion]

    public init(span: ExtractedSpan, clause: Range<Int>, number: ParsedNumber?, suggestions: [OptionSuggestion]) {
        self.span = span
        self.clause = clause
        self.number = number
        self.suggestions = suggestions
    }
}

public struct OptionSuggestion: Equatable, Sendable {
    /// The `name` of the option.
    public var option: String
    public var probability: Float

    public init(option: String, probability: Float) {
        self.option = option
        self.probability = probability
    }
}

/// One clause classification of `FormMapper`: the clause text and the classification schema of a list field.
public struct ClassificationRequest: Hashable, Sendable {
    public var clause: String
    public var schema: GLiNERSchema

    public init(clause: String, schema: GLiNERSchema) {
        self.clause = clause
        self.schema = schema
    }
}

/// The candidates `FormMapper` drops before people see them.
public enum CandidateFilter: Sendable {
    /// Keep every candidate.
    case none
    /// Rule R1: drop a candidate of a list field when its clause holds the name of another list field and not the
    /// field's own name. Names match as substrings, in Unicode scalars. Number fields keep every candidate.
    case clauseFieldName
    /// Rule R1 for list fields, and rule N1 for number fields: drop a candidate that overlaps one or more numeral runs
    /// of the note when an equipment marker touches every one of them, as in `No.3` or `P-3`. A candidate with an
    /// unmarked run, or with no run, is kept. Markers and numeral words are in `Resources/equipment_numbers.json`.
    case clauseFieldNameAndEquipmentNumber
}

/// Method (i): the spans of the value labels, then each span's clause classified over the options of its field.
public struct FormMapper: Sendable {
    public var schema: FormSchema
    /// The words `CorrectionCue.find(in:words:)` looks for.
    public var correctionWords: [String]
    /// Rule R1 unless set.
    public var candidateFilter: CandidateFilter

    public init(schema: FormSchema, correctionWords: [String], candidateFilter: CandidateFilter = .clauseFieldName) {
        self.schema = schema
        self.correctionWords = correctionWords
        self.candidateFilter = candidateFilter
    }

    /// - Parameters:
    ///   - text: The note.
    ///   - extraction: The spans of `schema.extractionSchema()` in `text`.
    ///   - classify: The probabilities of the labels of a classification schema for a text, in label order.
    public func map(
        text: String, extraction: ExtractionResult, classify: (String, GLiNERSchema) -> ClassificationResult
    ) -> FormMapping {
        let scalars = Array(text.unicodeScalars)
        let numeralRuns =
            candidateFilter == .clauseFieldNameAndEquipmentNumber
            ? EquipmentNumberWords.bundled.numeralRuns(in: scalars) : []
        let fields = schema.fields.enumerated().map { index, field in
            let spans = extraction.spans.filter { $0.labelIndex == index }.sorted {
                ($0.scalarRange.lowerBound, $0.scalarRange.upperBound)
                    < ($1.scalarRange.lowerBound, $1.scalarRange.upperBound)
            }
            let candidates = spans.map { span in
                let clause = Self.clause(around: span.scalarRange, in: scalars)
                guard case .list(let options, _) = field.kind else {
                    return FieldCandidate(
                        span: span, clause: clause, number: ParsedNumber(parsing: span.text), suggestions: [])
                }
                // Rule R1 reads names only, so the clause of a dropped candidate is not classified.
                guard !drops(field, clause: scalars[clause]) else {
                    return FieldCandidate(span: span, clause: clause, number: nil, suggestions: [])
                }
                let result = classify(
                    String(String.UnicodeScalarView(scalars[clause])), schema.classificationSchema(for: field))
                let suggestions = zip(options, result.probabilities).enumerated()
                    .sorted { (-$0.element.1, $0.offset) < (-$1.element.1, $1.offset) }
                    .prefix(2)
                    .map { OptionSuggestion(option: $0.element.0.name, probability: $0.element.1) }
                return FieldCandidate(span: span, clause: clause, number: nil, suggestions: suggestions)
            }
            func dropped(_ candidate: FieldCandidate) -> Bool {
                drops(field, clause: scalars[candidate.clause])
                    || drops(field, span: candidate.span.scalarRange, numeralRuns: numeralRuns)
            }
            return FieldMapping(
                field: field, candidates: candidates.filter { !dropped($0) },
                droppedCandidates: candidates.filter(dropped))
        }
        return FormMapping(fields: fields, corrections: CorrectionCue.find(in: text, words: correctionWords))
    }

    /// Whether `candidateFilter` drops a candidate of `field` whose clause is `clause`: rule R1, list fields only.
    private func drops(_ field: FormField, clause: ArraySlice<Unicode.Scalar>) -> Bool {
        guard candidateFilter != .none, case .list = field.kind else { return false }
        func holds(_ name: String) -> Bool { clause.contains(Array(name.unicodeScalars)) }
        return !holds(field.name)
            && schema.fields.contains { other in
                guard case .list = other.kind else { return false }
                return holds(other.name)
            }
    }

    /// Whether `candidateFilter` drops a candidate of `field` over `span`: rule N1, number fields only.
    private func drops(
        _ field: FormField, span: Range<Int>, numeralRuns: [(range: Range<Int>, marked: Bool)]
    ) -> Bool {
        guard candidateFilter == .clauseFieldNameAndEquipmentNumber, case .number = field.kind else { return false }
        let own = numeralRuns.filter { $0.range.overlaps(span) }
        return !own.isEmpty && own.allSatisfy { $0.marked }
    }

    /// The classifications `map(text:extraction:classify:)` asks for, in the order it first asks for each, without
    /// repeats. Run them (asynchronously, if need be), then pass `map` a `classify` that looks the results up.
    public func classificationRequests(text: String, extraction: ExtractionResult) -> [ClassificationRequest] {
        var requests: [ClassificationRequest] = []
        _ = map(text: text, extraction: extraction) { clause, schema in
            let request = ClassificationRequest(clause: clause, schema: schema)
            if !requests.contains(request) { requests.append(request) }
            return ClassificationResult(probabilities: [])
        }
        return requests
    }

    /// Japanese and ASCII punctuation and the line break. The ASCII period is a decimal point in these notes, not a
    /// delimiter.
    private static let delimiters: Set<Unicode.Scalar> = [
        "\u{3001}", "\u{3002}", "\u{FF0C}", "\u{FF0E}", ",", "!", "?", "\u{FF01}", "\u{FF1F}", "\n", "\u{FF1B}", ";",
    ]

    /// The text between the delimiters around `range`, without delimiters at its ends; the clauses a span crosses are
    /// joined. When nothing is left, `range` itself.
    private static func clause(around range: Range<Int>, in scalars: [Unicode.Scalar]) -> Range<Int> {
        let span = range.clamped(to: 0..<scalars.count)
        var lower = span.lowerBound
        var upper = span.upperBound
        while lower > 0, !delimiters.contains(scalars[lower - 1]) { lower -= 1 }
        while upper < scalars.count, !delimiters.contains(scalars[upper]) { upper += 1 }
        while lower < upper, delimiters.contains(scalars[lower]) { lower += 1 }
        while upper > lower, delimiters.contains(scalars[upper - 1]) { upper -= 1 }
        return lower < upper ? lower..<upper : span
    }
}
