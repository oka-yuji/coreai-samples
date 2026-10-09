import Foundation
import GLiNERCore

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

        public var text: String
        public var position: Position
        public var probability: Float

        public var clause: Position

        public var number: String?
        public var unit: String?

        public var suggestions: [Suggestion]
    }

    public struct FieldOutput: Codable, Equatable, Sendable {
        public var key: String
        public var name: String

        public var state: String
        public var candidates: [Candidate]

        public var droppedCandidates: [Candidate]
    }

    public struct ModelOutput: Codable, Equatable, Sendable {

        public var version: Int

        public var probabilitiesCalibrated: Bool

        public var candidateFilter: String

        public var assetRevision: String?

        public var appVersion: String?
        public var fields: [FieldOutput]
        public var timing: ExtractTiming
    }

    public struct Cue: Codable, Equatable, Sendable {
        public var word: String
        public var position: Position
    }

    public struct RuleMarks: Codable, Equatable, Sendable {

        public var correctionCues: [Cue]
    }

    public struct Confirmation: Codable, Equatable, Sendable {
        public var key: String
        public var name: String
        public var choice: ExtractChoice
    }

    public var input: Input

    public var mode: String

    public var form: FormSchema

    public var model: ModelOutput?
    public var rules: RuleMarks?

    public var confirmations: [Confirmation]

    public init(_ state: ExtractState) {
        let text = state.text
        input = Input(text: text, version: state.version)
        mode = state.mode.rawValue
        form = state.schema
        model = state.result.map { result in
            ModelOutput(
                version: result.version, probabilitiesCalibrated: false, candidateFilter: state.appliedFilter.name,
                assetRevision: result.assetRevision, appVersion: state.appVersion,
                fields: result.mapping.fields.map { field in
                    FieldOutput(
                        key: field.field.key, name: field.field.name,
                        state: field.state == .candidate ? "candidate" : "notDetected",
                        candidates: field.candidates.map { Self.candidate($0, in: text) },
                        droppedCandidates: field.droppedCandidates.map { Self.candidate($0, in: text) })
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

extension CandidateFilter {

    public var name: String {
        switch self {
        case .none: "none"
        case .clauseFieldName: "clauseFieldName"
        case .clauseFieldNameAndEquipmentNumber: "clauseFieldNameAndEquipmentNumber"
        }
    }
}
