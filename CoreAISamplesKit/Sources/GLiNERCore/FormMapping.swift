public struct FormMapping: Equatable, Sendable {

    public var fields: [FieldMapping]

    public var corrections: [CorrectionCue]

    public init(fields: [FieldMapping], corrections: [CorrectionCue]) {
        self.fields = fields
        self.corrections = corrections
    }
}

public struct FieldMapping: Equatable, Sendable {
    public enum State: Equatable, Sendable {

        case candidate

        case notDetected
    }

    public var field: FormField

    public var candidates: [FieldCandidate]

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

    public var clause: Range<Int>

    public var number: ParsedNumber?

    public var suggestions: [OptionSuggestion]

    public init(span: ExtractedSpan, clause: Range<Int>, number: ParsedNumber?, suggestions: [OptionSuggestion]) {
        self.span = span
        self.clause = clause
        self.number = number
        self.suggestions = suggestions
    }
}

public struct OptionSuggestion: Equatable, Sendable {

    public var option: String
    public var probability: Float

    public init(option: String, probability: Float) {
        self.option = option
        self.probability = probability
    }
}

public struct ClassificationRequest: Hashable, Sendable {
    public var clause: String
    public var schema: GLiNERSchema

    public init(clause: String, schema: GLiNERSchema) {
        self.clause = clause
        self.schema = schema
    }
}

public enum CandidateFilter: Sendable {

    case none

    case clauseFieldName

    case clauseFieldNameAndEquipmentNumber
}

public struct FormMapper: Sendable {
    public var schema: FormSchema

    public var correctionWords: [String]

    public var candidateFilter: CandidateFilter

    public init(
        schema: FormSchema, correctionWords: [String],
        candidateFilter: CandidateFilter = .clauseFieldNameAndEquipmentNumber
    ) {
        self.schema = schema
        self.correctionWords = correctionWords
        self.candidateFilter = candidateFilter
    }

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

    private func drops(_ field: FormField, clause: ArraySlice<Unicode.Scalar>) -> Bool {
        guard candidateFilter != .none, case .list = field.kind else { return false }
        func holds(_ name: String) -> Bool { clause.contains(Array(name.unicodeScalars)) }
        return !holds(field.name)
            && schema.fields.contains { other in
                guard case .list = other.kind else { return false }
                return holds(other.name)
            }
    }

    private func drops(
        _ field: FormField, span: Range<Int>, numeralRuns: [(range: Range<Int>, marked: Bool)]
    ) -> Bool {
        guard candidateFilter == .clauseFieldNameAndEquipmentNumber, case .number = field.kind else { return false }
        let own = numeralRuns.filter { $0.range.overlaps(span) }
        return !own.isEmpty && own.allSatisfy { $0.marked }
    }

    public func classificationRequests(text: String, extraction: ExtractionResult) -> [ClassificationRequest] {
        var requests: [ClassificationRequest] = []
        _ = map(text: text, extraction: extraction) { clause, schema in
            let request = ClassificationRequest(clause: clause, schema: schema)
            if !requests.contains(request) { requests.append(request) }
            return ClassificationResult(probabilities: [])
        }
        return requests
    }

    private static let delimiters: Set<Unicode.Scalar> = [
        "\u{3001}", "\u{3002}", "\u{FF0C}", "\u{FF0E}", ",", "!", "?", "\u{FF01}", "\u{FF1F}", "\n", "\u{FF1B}", ";",
    ]

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
