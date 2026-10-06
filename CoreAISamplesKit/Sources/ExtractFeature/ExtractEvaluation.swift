import Foundation
import GLiNERCore

/// One sentence of a gold file. Positions count Unicode scalars and the end is exclusive.
public struct GoldSentence: Decodable, Equatable, Sendable {
    public struct Span: Decodable, Equatable, Sendable {
        public var start: Int
        public var end: Int
        public var text: String

        public var range: Range<Int> { start..<end }
    }

    public struct Field: Decodable, Equatable, Sendable {
        public var mentioned: Bool
        public var explicitEmpty: Bool?
        public var statementSpan: Span?
        public var valueSpan: Span?
        /// A number field's value, normalized ("0.5").
        public var value: String?
        public var unit: String?
        /// A list field's option; nil when no option fits.
        public var option: String?
    }

    public var id: String
    public var text: String
    /// Excluded sentences are left out of every count.
    public var excluded: Bool?
    /// The field each sentence names in advance as not mentioned, counted on its own.
    public var mainUnmentionedField: String?
    public var features: [String]?
    public var fields: [String: Field]

    /// The sentences of a gold file (a JSON array with snake-case keys).
    public static func decode(_ data: Data) throws -> [GoldSentence] {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode([GoldSentence].self, from: data)
    }
}

/// Scores mappings against gold sentences.
///
/// - Mention: a field is found when one of its candidates overlaps the gold statement span, fabricated when it is not
///   mentioned but has candidates, missed when it is mentioned without candidates, and misplaced when it is mentioned
///   and no candidate overlaps its statement span (counted as both a false positive and a false negative).
/// - Span match: a candidate overlaps the field's gold value span, lies in the same sentence (ended by an
///   ideographic full stop or a line break) and overlaps no other field's value span (other than an identical one).
/// - The policy takes the candidate with the highest probability (the first on a tie): its top suggestion for a
///   list field, its number and unit (the form's unit when none is written) for a number field. Option and number
///   scores count mentioned fields that are not explicitly empty; a null gold option is never matched. A field agrees
///   when the policy's value equals the gold, or when the gold is empty (not mentioned, or explicitly empty) and the
///   field has no candidates.
public enum ExtractEvaluation {
    public static func evaluate(_ sentences: [(gold: GoldSentence, mapping: FormMapping)]) -> EvaluationReport {
        let judged = sentences.filter { $0.gold.excluded != true }.map { judge($0.gold, $0.mapping) }
        let keys = sentences.first?.mapping.fields.map(\.field.key) ?? []
        let goldKeys = Set(sentences.flatMap { $0.gold.fields.keys })
        let features = Set(judged.flatMap(\.features)).sorted()
        return EvaluationReport(
            overall: summarize(judged),
            fields: Dictionary(keys.map { ($0, summarize(judged, field: $0)) }) { first, _ in first },
            features: Dictionary(
                uniqueKeysWithValues: features.map { feature in
                    (feature, summarize(judged.filter { $0.features.contains(feature) }))
                }),
            unscoredFields: goldKeys.symmetricDifference(keys).sorted(),
            sentences: judged)
    }

    /// The nearest-rank percentile (0 < `fraction` <= 1) of `values`; nil when there are none.
    public static func percentile(_ values: [Double], _ fraction: Double) -> Double? {
        guard !values.isEmpty else { return nil }
        let rank = Int((fraction * Double(values.count)).rounded(.up))
        return values.sorted()[min(max(rank - 1, 0), values.count - 1)]
    }

    /// The percentiles of each stage of `warm` on its own, and the first (cold) run as measured.
    public static func timing(cold: ExtractTiming?, warm: [ExtractTiming]) -> TimingSummary {
        func stages(_ fraction: Double) -> ExtractTiming? {
            func value(_ stage: (ExtractTiming) -> Double) -> Double { percentile(warm.map(stage), fraction) ?? 0 }
            return warm.isEmpty
                ? nil
                : ExtractTiming(
                    preprocess: value(\.preprocess), encode: value(\.encode), pair: value(\.pair),
                    classify: value(\.classify), classifications: Int(value { Double($0.classifications) }),
                    map: value(\.map), total: value(\.total))
        }
        return TimingSummary(runs: warm.count, cold: cold, p50: stages(0.5), p95: stages(0.95))
    }

    static func judge(_ gold: GoldSentence, _ mapping: FormMapping) -> SentenceJudgment {
        let scalars = Array(gold.text.unicodeScalars)
        let valueSpans = gold.fields.compactMapValues { field in
            field.mentioned && field.explicitEmpty != true ? field.valueSpan?.range : nil
        }
        let fields = mapping.fields.compactMap { field in
            gold.fields[field.field.key].map {
                judge(
                    field, gold: $0, isMain: gold.mainUnmentionedField == field.field.key, valueSpans: valueSpans,
                    scalars: scalars)
            }
        }
        return SentenceJudgment(
            id: gold.id, text: gold.text, features: gold.features ?? [], allFieldsAgree: fields.allSatisfy(\.agrees),
            correctionCues: mapping.corrections.map(\.word),
            correctionTagged: gold.features?.contains("correction") ?? false, fields: fields)
    }

    private static func judge(
        _ mapping: FieldMapping, gold: GoldSentence.Field, isMain: Bool, valueSpans: [String: Range<Int>],
        scalars: [Unicode.Scalar]
    ) -> FieldJudgment {
        let key = mapping.field.key
        let candidates = mapping.candidates
        let isEmpty = !gold.mentioned || gold.explicitEmpty == true
        let goldValue = valueSpans[key]
        let others = valueSpans.filter { $0.key != key && $0.value != goldValue }.values
        func matches(_ range: Range<Int>) -> Bool {
            guard let goldValue, range.overlaps(goldValue) else { return false }
            let sentence = Sentence.range(around: goldValue, in: scalars)
            return sentence.lowerBound <= range.lowerBound && range.upperBound <= sentence.upperBound
                && !others.contains { $0.overlaps(range) }
        }

        let mention: FieldJudgment.Mention
        if !gold.mentioned {
            mention = candidates.isEmpty ? .absent : .fabricated
        } else if candidates.isEmpty {
            mention = .missed
        } else if let statement = gold.statementSpan?.range ?? goldValue,
            !candidates.contains(where: { $0.span.scalarRange.overlaps(statement) })
        {
            mention = .misplaced
        } else {
            mention = .found
        }

        var judgment = FieldJudgment(
            key: key, mentioned: gold.mentioned, explicitEmpty: gold.explicitEmpty == true, mainUnmentioned: isMain,
            mention: mention,
            candidates: candidates.map { candidate in
                let range = candidate.span.scalarRange
                return FieldJudgment.Candidate(
                    start: range.lowerBound, end: range.upperBound, text: candidate.span.text,
                    probability: candidate.span.probability, number: candidate.number.map { "\($0.value)" },
                    unit: candidate.number?.normalizedUnit,
                    suggestions: candidate.suggestions.map { "\($0.option) \($0.probability)" },
                    matchesGoldSpan: matches(range), exactGoldSpan: range == goldValue)
            },
            goldSpanMatched: goldValue.map { _ in candidates.contains { matches($0.span.scalarRange) } },
            goldSpanExact: goldValue.map { value in candidates.contains { $0.span.scalarRange == value } },
            agrees: isEmpty && candidates.isEmpty)
        guard !isEmpty else { return judgment }

        let best = candidates.max { $0.span.probability < $1.span.probability }
        switch mapping.field.kind {
        case .list:
            let suggestion = best?.suggestions.first?.option
            judgment.suggestion = suggestion
            judgment.goldOption = .some(gold.option)
            judgment.optionCorrect = gold.option != nil && suggestion == gold.option
            judgment.optionInTopTwo =
                gold.option.map { option in
                    best?.suggestions.prefix(2).contains { $0.option == option } ?? false
                } ?? false
            judgment.agrees = judgment.optionCorrect == true
        case .number(let unit):
            guard let value = gold.value.flatMap({ Decimal(string: $0) }) else { return judgment }
            let goldUnit = gold.unit ?? unit
            func agrees(_ candidate: FieldCandidate) -> Bool {
                guard let number = candidate.number else { return false }
                return number.value == value && (number.normalizedUnit ?? unit) == goldUnit
            }
            judgment.goldNumber = "\(value) \(goldUnit)"
            judgment.numberCorrect = best.map(agrees) ?? false
            judgment.numberInCandidates = candidates.contains(where: agrees)
            judgment.agrees = judgment.numberCorrect == true
        }
        return judgment
    }

    static func summarize(_ sentences: [SentenceJudgment], field key: String? = nil) -> Summary {
        let fields = sentences.flatMap { $0.fields.filter { key == nil || $0.key == key } }
        func count(_ include: (FieldJudgment) -> Bool) -> Int { fields.count(where: include) }
        func ratio(_ hit: (FieldJudgment) -> Bool?, among include: (FieldJudgment) -> Bool) -> Ratio {
            let pool = fields.filter(include)
            return Ratio(pool.count { hit($0) == true }, of: pool.count)
        }
        let mention = MentionSummary(
            found: count { $0.mention == .found }, fabricated: count { $0.mention == .fabricated },
            missed: count { $0.mention == .missed }, misplaced: count { $0.mention == .misplaced },
            absent: count { $0.mention == .absent })
        let candidates = fields.flatMap(\.candidates)
        let spans = SpanSummary(
            matched: Ratio(candidates.count(where: \.matchesGoldSpan), of: candidates.count),
            found: ratio({ $0.goldSpanMatched }, among: { $0.goldSpanMatched != nil }),
            exactMatched: Ratio(candidates.count(where: \.exactGoldSpan), of: candidates.count),
            exactFound: ratio({ $0.goldSpanExact }, among: { $0.goldSpanExact != nil }))
        let tagged = sentences.filter(\.correctionTagged)
        let cued = sentences.filter { !$0.correctionCues.isEmpty }
        return Summary(
            sentences: sentences.count, fields: fields.count, mention: mention,
            fabrication: ratio({ $0.mention == .fabricated }, among: { !$0.mentioned }),
            mainFabrication: ratio({ $0.mention == .fabricated }, among: \.mainUnmentioned),
            miss: ratio({ $0.mention == .missed }, among: \.mentioned),
            option: ratio({ $0.optionCorrect }, among: { $0.optionCorrect != nil }),
            optionWithCandidate: ratio(
                { $0.optionCorrect }, among: { $0.optionCorrect != nil && !$0.candidates.isEmpty }),
            optionTopTwo: ratio({ $0.optionInTopTwo }, among: { $0.optionInTopTwo != nil }),
            optionNull: count { $0.optionCorrect != nil && $0.goldOption == .some(nil) },
            number: ratio({ $0.numberCorrect }, among: { $0.numberCorrect != nil }),
            numberInCandidates: ratio({ $0.numberInCandidates }, among: { $0.numberInCandidates != nil }),
            spans: spans, agreement: ratio({ $0.agrees }, among: { _ in true }),
            allFields: key == nil ? Ratio(sentences.count(where: \.allFieldsAgree), of: sentences.count) : nil,
            correctionCues: key == nil
                ? CueSummary(
                    tagged: tagged.count, taggedWithCue: tagged.count { !$0.correctionCues.isEmpty },
                    withCue: cued.count,
                    withCueTagged: cued.count(where: \.correctionTagged))
                : nil)
    }
}

public struct EvaluationReport: Encodable, Equatable, Sendable {
    public var overall: Summary
    /// By field key.
    public var fields: [String: Summary]
    /// By the gold's `features` tags: the sentences that carry the tag.
    public var features: [String: Summary]
    /// Keys in the form or in the gold but not in both; they are not scored.
    public var unscoredFields: [String]
    public var sentences: [SentenceJudgment]
}

public struct Ratio: Encodable, Equatable, Sendable {
    public var count: Int
    public var total: Int
    public var rate: Double?

    public init(_ count: Int, of total: Int) {
        self.count = count
        self.total = total
        rate = total == 0 ? nil : Double(count) / Double(total)
    }
}

public struct MentionSummary: Encodable, Equatable, Sendable {
    public var found: Int
    public var fabricated: Int
    public var missed: Int
    public var misplaced: Int
    public var absent: Int
    public var precision: Double?
    public var recall: Double?
    public var f1: Double?

    init(found: Int, fabricated: Int, missed: Int, misplaced: Int, absent: Int) {
        self.found = found
        self.fabricated = fabricated
        self.missed = missed
        self.misplaced = misplaced
        self.absent = absent
        precision = Ratio(found, of: found + fabricated + misplaced).rate
        recall = Ratio(found, of: found + missed + misplaced).rate
        f1 = harmonicMean(precision, recall)
    }
}

public struct SpanSummary: Encodable, Equatable, Sendable {
    /// Candidates that match a gold value span (precision), of all candidates.
    public var matched: Ratio
    /// Gold value spans that a candidate matches (recall), of all gold value spans.
    public var found: Ratio
    public var f1: Double?
    /// The same with identical ranges only (for reference).
    public var exactMatched: Ratio
    public var exactFound: Ratio

    init(matched: Ratio, found: Ratio, exactMatched: Ratio, exactFound: Ratio) {
        self.matched = matched
        self.found = found
        self.exactMatched = exactMatched
        self.exactFound = exactFound
        f1 = harmonicMean(matched.rate, found.rate)
    }
}

public struct CueSummary: Encodable, Equatable, Sendable {
    /// Sentences the gold tags `correction`, and those of them with a cue.
    public var tagged: Int
    public var taggedWithCue: Int
    /// Sentences with a cue, and those of them the gold tags `correction`.
    public var withCue: Int
    public var withCueTagged: Int
}

public struct Summary: Encodable, Equatable, Sendable {
    public var sentences: Int
    public var fields: Int
    public var mention: MentionSummary
    /// Fabricated, of the fields not mentioned.
    public var fabrication: Ratio
    /// Fabricated, of the fields named as not mentioned in advance.
    public var mainFabrication: Ratio
    /// Missed, of the mentioned fields.
    public var miss: Ratio
    /// List fields whose policy suggestion is the gold option.
    public var option: Ratio
    public var optionWithCandidate: Ratio
    public var optionTopTwo: Ratio
    /// List fields in `option` whose gold option is null (never matched).
    public var optionNull: Int
    /// Number fields whose policy number and unit are the gold's.
    public var number: Ratio
    /// Number fields with any candidate of the gold number and unit.
    public var numberInCandidates: Ratio
    public var spans: SpanSummary
    /// Fields that agree with the gold under the policy.
    public var agreement: Ratio
    /// Sentences whose fields all agree; nil in a summary of one field.
    public var allFields: Ratio?
    /// Nil in a summary of one field.
    public var correctionCues: CueSummary?
}

public struct TimingSummary: Encodable, Equatable, Sendable {
    public var runs: Int
    public var cold: ExtractTiming?
    /// Each stage's own percentile, in milliseconds; the stages do not add up to `total`.
    public var p50: ExtractTiming?
    public var p95: ExtractTiming?
}

public struct SentenceJudgment: Encodable, Equatable, Sendable {
    public var id: String
    public var text: String
    public var features: [String]
    public var allFieldsAgree: Bool
    /// The correction words the rule found.
    public var correctionCues: [String]
    public var correctionTagged: Bool
    public var fields: [FieldJudgment]
}

public struct FieldJudgment: Encodable, Equatable, Sendable {
    public enum Mention: String, Encodable, Sendable {
        case found
        case fabricated
        case missed
        case misplaced
        /// Not mentioned and no candidates.
        case absent
    }

    public struct Candidate: Encodable, Equatable, Sendable {
        public var start: Int
        public var end: Int
        public var text: String
        public var probability: Float
        public var number: String?
        public var unit: String?
        /// The two best options and their probabilities, highest first.
        public var suggestions: [String]
        public var matchesGoldSpan: Bool
        public var exactGoldSpan: Bool
    }

    public var key: String
    public var mentioned: Bool
    public var explicitEmpty: Bool
    public var mainUnmentioned: Bool
    public var mention: Mention
    public var candidates: [Candidate]
    /// Nil without a gold value span.
    public var goldSpanMatched: Bool?
    public var goldSpanExact: Bool?
    /// The gold option of a scored list field; `.some(nil)` when no option fits.
    public var goldOption: String??
    public var suggestion: String?
    /// Nil outside the option score.
    public var optionCorrect: Bool?
    public var optionInTopTwo: Bool?
    /// The gold number and unit of a scored number field.
    public var goldNumber: String?
    /// Nil outside the number score.
    public var numberCorrect: Bool?
    public var numberInCandidates: Bool?
    public var agrees: Bool
}

private func harmonicMean(_ lhs: Double?, _ rhs: Double?) -> Double? {
    guard let lhs, let rhs, lhs + rhs > 0 else { return nil }
    return 2 * lhs * rhs / (lhs + rhs)
}
