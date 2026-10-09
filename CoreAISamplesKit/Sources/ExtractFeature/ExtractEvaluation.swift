import Foundation
import GLiNERCore

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

        public var value: String?
        public var unit: String?

        public var option: String?
    }

    public var id: String
    public var text: String

    public var excluded: Bool?

    public var mainUnmentionedField: String?
    public var features: [String]?
    public var fields: [String: Field]

    public static func decode(_ data: Data) throws -> [GoldSentence] {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode([GoldSentence].self, from: data)
    }
}

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

    public static func percentile(_ values: [Double], _ fraction: Double) -> Double? {
        guard !values.isEmpty else { return nil }
        let rank = Int((fraction * Double(values.count)).rounded(.up))
        return values.sorted()[min(max(rank - 1, 0), values.count - 1)]
    }

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
            let sentence = Sentence.range(around: goldValue, in: scalars, ends: Sentence.fullStop)
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
            judgment.goldNumberWithoutDigits = gold.valueSpan.map { ParsedNumber(parsing: $0.text) == nil }
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
            optionNull: ratio(
                { $0.suggestion == nil }, among: { $0.optionCorrect != nil && $0.goldOption == .some(nil) }),
            number: ratio({ $0.numberCorrect }, among: { $0.numberCorrect != nil }),
            numberInCandidates: ratio({ $0.numberInCandidates }, among: { $0.numberInCandidates != nil }),
            numberWithoutDigits: count { $0.numberCorrect != nil && $0.goldNumberWithoutDigits == true },
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

    public var fields: [String: Summary]

    public var features: [String: Summary]

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

    public var matched: Ratio

    public var found: Ratio
    public var f1: Double?

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

    public var tagged: Int
    public var taggedWithCue: Int

    public var withCue: Int
    public var withCueTagged: Int
}

public struct Summary: Encodable, Equatable, Sendable {
    public var sentences: Int
    public var fields: Int
    public var mention: MentionSummary

    public var fabrication: Ratio

    public var mainFabrication: Ratio

    public var miss: Ratio

    public var option: Ratio
    public var optionWithCandidate: Ratio
    public var optionTopTwo: Ratio

    public var optionNull: Ratio

    public var number: Ratio

    public var numberInCandidates: Ratio

    public var numberWithoutDigits: Int
    public var spans: SpanSummary

    public var agreement: Ratio

    public var allFields: Ratio?

    public var correctionCues: CueSummary?
}

public struct TimingSummary: Encodable, Equatable, Sendable {
    public var runs: Int
    public var cold: ExtractTiming?

    public var p50: ExtractTiming?
    public var p95: ExtractTiming?
}

public struct SentenceJudgment: Encodable, Equatable, Sendable {
    public var id: String
    public var text: String
    public var features: [String]
    public var allFieldsAgree: Bool

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

        case absent
    }

    public struct Candidate: Encodable, Equatable, Sendable {
        public var start: Int
        public var end: Int
        public var text: String
        public var probability: Float
        public var number: String?
        public var unit: String?

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

    public var goldSpanMatched: Bool?
    public var goldSpanExact: Bool?

    public var goldOption: String??
    public var suggestion: String?

    public var optionCorrect: Bool?
    public var optionInTopTwo: Bool?

    public var goldNumber: String?

    public var goldNumberWithoutDigits: Bool?

    public var numberCorrect: Bool?
    public var numberInCandidates: Bool?
    public var agrees: Bool
}

private func harmonicMean(_ lhs: Double?, _ rhs: Double?) -> Double? {
    guard let lhs, let rhs, lhs + rhs > 0 else { return nil }
    return 2 * lhs * rhs / (lhs + rhs)
}
