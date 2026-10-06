import ExtractFeature
import Foundation
import GLiNERCore
import Testing

struct MissingResource: Error {
    let path: String
}

func resourceData(_ path: String) throws -> Data {
    guard let folder = Bundle.module.url(forResource: "Resources", withExtension: nil) else {
        throw MissingResource(path: path)
    }
    return try Data(contentsOf: folder.appending(path: path))
}

/// `Resources/evaluation_examples.json`: five hand-made gold sentences (and one excluded) with fixed outputs.
struct EvaluationExamples: Decodable {
    struct Suggestion: Decodable {
        let option: String
        let probability: Float
    }

    struct Candidate: Decodable {
        let start: Int
        let end: Int
        let probability: Float
        let suggestions: [Suggestion]
    }

    struct Cue: Decodable {
        let start: Int
        let end: Int
        let word: String
    }

    struct Output: Decodable {
        let candidates: [String: [Candidate]]
        let corrections: [Cue]
    }

    let outputs: [String: Output]

    /// The fixed output of `sentence` as a mapping of `form`, as `FormMapper` would build it.
    func mapping(of sentence: GoldSentence, form: FormSchema) -> FormMapping {
        let scalars = Array(sentence.text.unicodeScalars)
        let output = outputs[sentence.id]
        let fields = form.fields.enumerated().map { index, field in
            let candidates = (output?.candidates[field.key] ?? []).map { candidate in
                let range = candidate.start..<candidate.end
                let text = String(String.UnicodeScalarView(scalars[range]))
                let isNumber = if case .number = field.kind { true } else { false }
                return FieldCandidate(
                    span: ExtractedSpan(
                        labelIndex: index, scalarRange: range, text: text, probability: candidate.probability),
                    clause: range, number: isNumber ? ParsedNumber(parsing: text) : nil,
                    suggestions: candidate.suggestions.map {
                        OptionSuggestion(option: $0.option, probability: $0.probability)
                    })
            }
            return FieldMapping(field: field, candidates: candidates)
        }
        let cues = (output?.corrections ?? []).map { CorrectionCue(range: $0.start..<$0.end, word: $0.word) }
        return FormMapping(fields: fields, corrections: cues)
    }
}

struct ExtractEvaluationTests {
    let report: EvaluationReport

    init() throws {
        let data = try resourceData("evaluation_examples.json")
        let form = try JSONDecoder().decode(FormSchema.self, from: resourceData("forms/inspection_u1.json"))
        let examples = try JSONDecoder().decode(EvaluationExamples.self, from: data)
        let wrapper = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let gold = try GoldSentence.decode(JSONSerialization.data(withJSONObject: wrapper?["gold"] ?? []))
        report = ExtractEvaluation.evaluate(gold.map { ($0, examples.mapping(of: $0, form: form)) })
    }

    @Test(
        "The scoring of five hand-made sentences gives the mention precision and recall, the fabrication and miss rates, the option and number scores and the span match by the rules"
    )
    func scoresTheFiveSentences() {
        let overall = report.overall
        #expect(overall.sentences == 5)
        #expect(overall.fields == 30)
        #expect(
            [overall.mention.found, overall.mention.fabricated, overall.mention.missed, overall.mention.misplaced]
                == [11, 1, 1, 1] as [Int])
        #expect(overall.mention.absent == 16)
        #expect(overall.mention.precision == 11.0 / 13)
        #expect(overall.mention.recall == 11.0 / 13)
        #expect(overall.fabrication == Ratio(1, of: 17))
        #expect(overall.mainFabrication == Ratio(1, of: 5))
        #expect(overall.miss == Ratio(1, of: 13))
        #expect(overall.option == Ratio(3, of: 6))
        #expect(overall.optionWithCandidate == Ratio(3, of: 5))
        #expect(overall.optionTopTwo == Ratio(4, of: 6))
        #expect(overall.optionNull == 1)
        #expect(overall.number == Ratio(3, of: 6))
        #expect(overall.numberInCandidates == Ratio(4, of: 6))
        #expect(overall.spans.matched == Ratio(8, of: 14))
        #expect(overall.spans.found == Ratio(8, of: 12))
        #expect(overall.spans.exactMatched == Ratio(7, of: 14))
        #expect(overall.spans.exactFound == Ratio(7, of: 12))
        #expect(overall.agreement == Ratio(22, of: 30))
        #expect(overall.allFields == Ratio(1, of: 5))
        let cues = overall.correctionCues
        #expect([cues?.tagged, cues?.taggedWithCue, cues?.withCue, cues?.withCueTagged] == [1, 1, 1, 1] as [Int?])
        #expect(report.unscoredFields.isEmpty)
    }

    @Test("The scores are also given for each field and for each feature tag, and an excluded sentence counts nowhere")
    func scoresEachFieldAndFeature() throws {
        let pressure = try #require(report.fields["discharge_pressure"])
        #expect([pressure.mention.found, pressure.mention.misplaced, pressure.mention.absent] == [2, 1, 2] as [Int])
        #expect(pressure.number == Ratio(1, of: 3))
        #expect(pressure.numberInCandidates == Ratio(2, of: 3))
        #expect(pressure.spans.matched == Ratio(2, of: 4))
        #expect(pressure.spans.found == Ratio(2, of: 3))
        #expect(pressure.agreement == Ratio(3, of: 5))
        #expect(pressure.allFields == nil)

        let negation = try #require(report.features["negation"])
        #expect(negation.sentences == 2)
        #expect([negation.mention.found, negation.mention.misplaced, negation.mention.absent] == [5, 1, 6] as [Int])
        #expect(negation.allFields == Ratio(1, of: 2))
        #expect(
            report.features.keys.sorted() == [
                "correction", "explicit_empty", "negation", "no_matching_option", "number_format", "paraphrase",
            ])

        #expect(report.sentences.map(\.id) == ["E1", "E2", "E3", "E4", "E5"])
        let fourth = report.sentences[3].fields.map(\.mention)
        #expect(fourth == [.misplaced, .found, .found, .absent, .absent, .absent])
    }

    @Test("The timing percentiles are nearest-rank over the warm runs, stage by stage")
    func summarizesTheTiming() {
        #expect(ExtractEvaluation.percentile((1...20).map(Double.init), 0.5) == 10)
        #expect(ExtractEvaluation.percentile((1...20).map(Double.init), 0.95) == 19)
        #expect(ExtractEvaluation.percentile([], 0.5) == nil)
        let runs = (1...20).map { value in
            let value = Double(value)
            return ExtractTiming(
                preprocess: value, encode: 2 * value, pair: 0, classify: 0, classifications: 4, map: 0, total: 3 * value
            )
        }
        let summary = ExtractEvaluation.timing(cold: runs[0], warm: runs)
        #expect(summary.runs == 20)
        #expect(summary.cold == runs[0])
        #expect(summary.p95?.encode == 38)
        #expect(summary.p95?.total == 57)
        #expect(summary.p50?.classifications == 4)
    }
}
