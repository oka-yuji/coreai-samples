/// A row-major matrix of `Float`: a model output without its batch axis.
public struct Matrix: Equatable, Sendable {
    public let rows: Int
    public let columns: Int
    public let values: [Float]

    public init(rows: Int, columns: Int, values: [Float]) {
        precondition(values.count == rows * columns, "a \(rows) x \(columns) matrix needs \(rows * columns) values")
        self.rows = rows
        self.columns = columns
        self.values = values
    }

    public subscript(row: Int, column: Int) -> Float { values[row * columns + column] }
}

/// The candidate spans `pair_t<T>` scores: gliner2's `DocumentCandidatePool` at inference.
///
/// Boundaries up to the word count take their best start and end logits over the labels. The best 32 starts and
/// 32 ends make 32 x 32 pairs (end after start), each with `compat = pool_start[s] . pool_end[e] / sqrt(d)`.
/// Every label reserves its best 8 pairs by `start + end + compat`, scored 5000 + 8 ... 1; every pair also enters
/// with `compat + best start + best end`. A key `s * (T + 1) + e` keeps its best score, and the best 192 fill the
/// slots. Sorts break ties by position, as NumPy's stable argsort does.
public struct CandidatePool: Equatable, Sendable {
    public static let size = 192
    public static let boundaryLimit = 32
    public static let pairsPerLabel = 8
    public static let maskLogit: Float = -10_000
    public static let reservedScore: Float = 5_000

    /// Words `starts[slot]..<ends[slot]` of each slot; 0 in an empty slot.
    public let starts: [Int32]
    public let ends: [Int32]
    /// 1 for a candidate, 0 for an empty slot.
    public let mask: [Int32]
    /// The candidate's `compat`; 0 in an empty slot.
    public let priors: [Float]

    /// - Parameters:
    ///   - startLogits: `start_logits`, labels x boundaries (T + 1).
    ///   - endLogits: `end_logits`, labels x boundaries.
    ///   - poolStart: `pool_start`, boundaries x d.
    ///   - poolEnd: `pool_end`, boundaries x d.
    ///   - wordCount: Words of the model text; boundaries after the last word are ignored.
    ///   - labelCount: Labels in use; the rows after them are ignored.
    public init(
        startLogits: Matrix, endLogits: Matrix, poolStart: Matrix, poolEnd: Matrix, wordCount: Int, labelCount: Int
    ) {
        let boundaryCount = startLogits.columns
        let floor = Self.maskLogit
        let scale = Float(Double(poolStart.columns).squareRoot())
        func compat(_ start: Int, _ end: Int) -> Float {
            var sum: Float = 0
            for column in 0..<poolStart.columns { sum += poolStart[start, column] * poolEnd[end, column] }
            return sum / scale
        }

        let boundaries = labelCount > 0 ? 0..<min(wordCount + 1, boundaryCount) : 0..<0
        var bestStart = [Float](repeating: floor, count: boundaryCount)
        var bestEnd = bestStart
        for boundary in boundaries {
            for label in 0..<labelCount {
                bestStart[boundary] = max(bestStart[boundary], startLogits[label, boundary])
                bestEnd[boundary] = max(bestEnd[boundary], endLogits[label, boundary])
            }
        }
        let topCount = min(Self.boundaryLimit, boundaryCount)
        func top(_ scores: [Float]) -> [Int?] {
            let order = stableDescending((0..<boundaryCount).map { boundaries.contains($0) ? scores[$0] : floor })
            return order.prefix(topCount).map { boundaries.contains($0) ? $0 : nil }
        }
        let topEnds = top(bestEnd)
        let pairs = top(bestStart).flatMap { start in
            topEnds.map { end in
                guard let start, let end, end > start else { return Pair(start: 0, end: 0, compat: 0, isValid: false) }
                return Pair(start: start, end: end, compat: compat(start, end), isValid: true)
            }
        }

        var entries: [Entry] = []
        let reserved = min(Self.pairsPerLabel, pairs.count)
        for label in 0..<labelCount {
            let scores = pairs.map { pair in
                pair.isValid ? startLogits[label, pair.start] + endLogits[label, pair.end] + pair.compat : floor
            }
            for (rank, index) in stableDescending(scores).prefix(reserved).enumerated() {
                let pair = pairs[index]
                let score = Self.reservedScore + Float(reserved - rank)
                entries.append(Entry(key: pair.start * boundaryCount + pair.end, score: score, isValid: pair.isValid))
            }
        }
        for pair in pairs {
            let score = pair.compat + bestStart[pair.start] + bestEnd[pair.end]
            entries.append(Entry(key: pair.start * boundaryCount + pair.end, score: score, isValid: pair.isValid))
        }

        let slots = Self.slots(entries, boundaryCount: boundaryCount)
        starts = slots.map { Int32($0?.start ?? 0) }
        ends = slots.map { Int32($0?.end ?? 0) }
        mask = slots.map { $0 == nil ? 0 : 1 }
        priors = slots.map { $0.map { compat($0.start, $0.end) } ?? 0 }
    }

    /// gliner2's `_deduplicate_pool`: order by score, then by key; keep the first entry of each key; fill the slots
    /// by score. `nil` is an empty slot.
    private static func slots(_ entries: [Entry], boundaryCount: Int) -> [(start: Int, end: Int)?] {
        let keys = entries.map { $0.isValid ? $0.key : boundaryCount * boundaryCount }
        let scores = entries.map { $0.isValid ? $0.score : maskLogit }
        let byScore = stableDescending(scores)
        let byKey = byScore.indices.sorted { (keys[byScore[$0]], $0) < (keys[byScore[$1]], $1) }.map { byScore[$0] }
        let kept = byKey.indices.map { position in
            entries[byKey[position]].isValid && (position == 0 || keys[byKey[position]] != keys[byKey[position - 1]])
        }
        let chosen = stableDescending(byKey.indices.map { kept[$0] ? scores[byKey[$0]] : maskLogit })
        return (0..<size).map { slot in
            guard slot < chosen.count, kept[chosen[slot]] else { return nil }
            let (start, end) = keys[byKey[chosen[slot]]].quotientAndRemainder(dividingBy: boundaryCount)
            return (start, end)
        }
    }
}

private struct Pair {
    let start: Int
    let end: Int
    let compat: Float
    let isValid: Bool
}

private struct Entry {
    let key: Int
    let score: Float
    let isValid: Bool
}

/// Indices sorted by value, largest first; equal values keep index order (NumPy's stable argsort of -values).
private func stableDescending(_ values: [Float]) -> [Int] {
    values.indices.sorted { values[$0] > values[$1] || (values[$0] == values[$1] && $0 < $1) }
}
