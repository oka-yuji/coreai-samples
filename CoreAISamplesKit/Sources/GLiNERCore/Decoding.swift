import Foundation

public struct ExtractedSpan: Equatable, Sendable {

    public var labelIndex: Int

    public var scalarRange: Range<Int>
    public var text: String
    public var probability: Float

    public init(labelIndex: Int, scalarRange: Range<Int>, text: String, probability: Float) {
        self.labelIndex = labelIndex
        self.scalarRange = scalarRange
        self.text = text
        self.probability = probability
    }
}

public struct ExtractionResult: Equatable, Sendable {

    public var spans: [ExtractedSpan]

    public var abstained: [Int]

    public var tokenCapacity: Int

    public init(spans: [ExtractedSpan], abstained: [Int], tokenCapacity: Int) {
        self.spans = spans
        self.abstained = abstained
        self.tokenCapacity = tokenCapacity
    }
}

extension ExtractionResult {
    public static let threshold: Float = 0.5
    public static let abstentionThreshold: Float = 0.5

    public init(pairLogits: Matrix, nullLogits: [Float], pool: CandidatePool, input: PreparedInput, text: String) {
        let scalars = Array(text.unicodeScalars)
        var spans: [ExtractedSpan] = []
        var abstained: [Int] = []
        for label in 0..<input.labelCount {
            guard sigmoid(nullLogits[label]) <= Self.abstentionThreshold else {
                abstained.append(label)
                continue
            }
            let candidates = pool.mask.indices.filter {
                pool.mask[$0] != 0 && sigmoid(pairLogits[$0, label]) >= Self.threshold
            }
            let scored = candidates.enumerated().map { index, slot in
                ScoredSpan(
                    score: Double(sigmoid(pairLogits[slot, label])), start: Int(pool.starts[slot]),
                    end: Int(pool.ends[slot]), index: index)
            }
            for span in resolveFlat(scored) where 0 <= span.start && span.start < span.end {
                guard span.end <= input.words.count else { continue }
                let lower = min(input.words[span.start].scalarRange.lowerBound, scalars.count)
                let upper = min(input.words[span.end - 1].scalarRange.upperBound, scalars.count)
                guard scalars[lower..<upper].contains(where: { !$0.isPythonWhitespace }) else { continue }
                spans.append(
                    ExtractedSpan(
                        labelIndex: label, scalarRange: lower..<upper,
                        text: String(String.UnicodeScalarView(scalars[lower..<upper])), probability: Float(span.score)))
            }
        }
        self.init(spans: spans, abstained: abstained, tokenCapacity: input.tokenCapacity)
    }
}

public struct ClassificationResult: Equatable, Sendable {

    public var probabilities: [Float]

    public init(probabilities: [Float]) {
        self.probabilities = probabilities
    }
}

extension ClassificationResult {

    public init(logits: [Float]) {
        let largest = logits.max() ?? 0
        let exponentials = logits.map { exp($0 - largest) }
        let total = exponentials.reduce(0, +)
        self.init(probabilities: exponentials.map { $0 / total })
    }
}

private func sigmoid(_ logit: Float) -> Float { 1 / (1 + exp(-logit)) }

private struct ScoredSpan {

    let score: Double
    let start: Int
    let end: Int
    let index: Int
}

private func ranksBefore(_ lhs: ScoredSpan, _ rhs: ScoredSpan) -> Bool {
    (-lhs.score, lhs.start, lhs.end, lhs.index) < (-rhs.score, rhs.start, rhs.end, rhs.index)
}

private func resolveFlat(_ items: [ScoredSpan]) -> [ScoredSpan] {
    var seen = Set<Range<Int>>()
    let distinct = items.sorted(by: ranksBefore).filter { seen.insert($0.start..<$0.end).inserted }
    let byEnd = distinct.sorted {
        ($0.end, $0.start, -$0.score, $0.index) < ($1.end, $1.start, -$1.score, $1.index)
    }
    func ranked(_ picks: [Int]) -> [ScoredSpan] { picks.map { byEnd[$0] }.sorted(by: ranksBefore) }
    func prefers(_ lhs: (total: Double, picks: [Int]), over rhs: (total: Double, picks: [Int])) -> Bool {
        if lhs.total != rhs.total { return lhs.total > rhs.total }
        if lhs.picks.count != rhs.picks.count { return lhs.picks.count > rhs.picks.count }
        return ranked(lhs.picks).lexicographicallyPrecedes(ranked(rhs.picks), by: ranksBefore)
    }
    var best: [(total: Double, picks: [Int])] = [(0, [])]
    for (position, span) in byEnd.enumerated() {
        let previous = best[byEnd[..<position].prefix { $0.end <= span.start }.count]
        let with = (total: previous.total + span.score, picks: previous.picks + [position])
        best.append(prefers(with, over: best[position]) ? with : best[position])
    }
    return ranked(best[byEnd.count].picks)
}
