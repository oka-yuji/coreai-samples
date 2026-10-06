import Foundation
import GLiNERCore

/// The JSON people export: the input, the form, the model's output, the rule's marks and people's choices, each in
/// its own section. Positions are given in Unicode scalars (the model's) and in UTF-16 (`NSString`'s).
public struct ExportDocument: Codable, Equatable, Sendable {
    public struct Offsets: Codable, Equatable, Sendable {
        public var start: Int
        public var end: Int
    }

    public struct Position: Codable, Equatable, Sendable {
        public var scalars: Offsets
        public var utf16: Offsets

        init(_ range: Range<Int>, in text: String) {
            let utf16 = text.utf16Range(fromScalarRange: range) ?? range
            scalars = Offsets(start: range.lowerBound, end: range.upperBound)
            self.utf16 = Offsets(start: utf16.lowerBound, end: utf16.upperBound)
        }
    }

    public struct Input: Codable, Equatable, Sendable {
        public var text: String
        public var version: Int
    }

    public struct Suggestion: Codable, Equatable, Sendable {
        public var option: String
        public var probability: Float
    }

    public struct Candidate: Codable, Equatable, Sendable {
        /// The span as written in the note.
        public var text: String
        public var position: Position
        public var probability: Float
        /// The clause a list field's options were scored on.
        public var clause: Position
        /// The number in the span of a number field, and the unit written after it.
        public var number: String?
        public var unit: String?
        /// The two best options of a list field, highest first.
        public var suggestions: [Suggestion]
    }

    public struct FieldOutput: Codable, Equatable, Sendable {
        public var key: String
        public var name: String
        /// `candidate` or `notDetected`.
        public var state: String
        public var candidates: [Candidate]
    }

    public struct ModelOutput: Codable, Equatable, Sendable {
        /// The version of the text and the form the output is for.
        public var version: Int
        /// The probabilities are model scores; they are not calibrated.
        public var probabilitiesCalibrated: Bool
        public var fields: [FieldOutput]
        public var timing: ExtractTiming
    }

    public struct Cue: Codable, Equatable, Sendable {
        public var word: String
        public var position: Position
    }

    public struct RuleMarks: Codable, Equatable, Sendable {
        /// Correction words found by string match. They mark the place only.
        public var correctionCues: [Cue]
    }

    public struct Confirmation: Codable, Equatable, Sendable {
        public var key: String
        public var name: String
        public var choice: ExtractChoice
    }

    public var input: Input
    public var form: FormSchema
    /// Nil before the current text and form have a result.
    public var model: ModelOutput?
    public var rules: RuleMarks?
    /// People's choices, in form order.
    public var confirmations: [Confirmation]

    public init(_ state: ExtractState) {
        let text = state.text
        input = Input(text: text, version: state.version)
        form = state.form
        model = state.result.map { result in
            ModelOutput(
                version: result.version, probabilitiesCalibrated: false,
                fields: result.mapping.fields.map { field in
                    FieldOutput(
                        key: field.field.key, name: field.field.name,
                        state: field.state == .candidate ? "candidate" : "notDetected",
                        candidates: field.candidates.map { Self.candidate($0, in: text) })
                }, timing: result.timing)
        }
        rules = state.result.map { result in
            RuleMarks(
                correctionCues: result.mapping.corrections.map {
                    Cue(word: $0.word, position: Position($0.range, in: text))
                })
        }
        confirmations = state.form.fields.compactMap { field in
            state.confirmations[field.key].map { Confirmation(key: field.key, name: field.name, choice: $0) }
        }
    }

    private static func candidate(_ candidate: FieldCandidate, in text: String) -> Candidate {
        Candidate(
            text: candidate.span.text, position: Position(candidate.span.scalarRange, in: text),
            probability: candidate.span.probability, clause: Position(candidate.clause, in: text),
            number: candidate.number.map { "\($0.value)" }, unit: candidate.number?.unit,
            suggestions: candidate.suggestions.map { Suggestion(option: $0.option, probability: $0.probability) })
    }
}
