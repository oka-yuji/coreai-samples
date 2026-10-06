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
    /// Every span of the field's label, in text order.
    public var candidates: [FieldCandidate]

    public var state: State { candidates.isEmpty ? .notDetected : .candidate }

    public init(field: FormField, candidates: [FieldCandidate]) {
        self.field = field
        self.candidates = candidates
    }
}

public struct FieldCandidate: Equatable, Sendable {
    public var span: ExtractedSpan
    /// The clause that holds the span: the text between delimiters, in Unicode scalars of the note.
    public var clause: Range<Int>
    /// The number and unit in the span text of a number field. Nil when the text has no digits, and in a list field.
    public var number: ParsedNumber?
    /// The two best options for the clause of a list field, highest first; empty in a number field. The
    /// probabilities are not calibrated.
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

/// Method (i): the spans of the value labels, then each span's clause classified over the options of its field.
public struct FormMapper: Sendable {
    public var schema: FormSchema
    /// The words `CorrectionCue.find(in:words:)` looks for.
    public var correctionWords: [String]

    public init(schema: FormSchema, correctionWords: [String]) {
        self.schema = schema
        self.correctionWords = correctionWords
    }

    /// - Parameters:
    ///   - text: The note.
    ///   - extraction: The spans of `schema.extractionSchema()` in `text`.
    ///   - classify: The probabilities of the labels of a classification schema for a text, in label order.
    public func map(
        text: String, extraction: ExtractionResult, classify: (String, GLiNERSchema) -> ClassificationResult
    ) -> FormMapping {
        let scalars = Array(text.unicodeScalars)
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
                let result = classify(
                    String(String.UnicodeScalarView(scalars[clause])), schema.classificationSchema(for: field))
                let suggestions = zip(options, result.probabilities).enumerated()
                    .sorted { (-$0.element.1, $0.offset) < (-$1.element.1, $1.offset) }
                    .prefix(2)
                    .map { OptionSuggestion(option: $0.element.0.name, probability: $0.element.1) }
                return FieldCandidate(span: span, clause: clause, number: nil, suggestions: suggestions)
            }
            return FieldMapping(field: field, candidates: candidates)
        }
        return FormMapping(fields: fields, corrections: CorrectionCue.find(in: text, words: correctionWords))
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
