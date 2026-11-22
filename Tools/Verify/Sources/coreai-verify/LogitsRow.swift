import Foundation

enum LogitsRow: Equatable {
    case nonFinite(Int)
    case finite(TopTwo)

    init(_ logits: [Float16]) {
        let nonFinite = logits.lazy.filter { !$0.isFinite }.count
        guard nonFinite == 0 else {
            self = .nonFinite(nonFinite)
            return
        }
        var first = 0
        var second = -1
        for id in logits.indices.dropFirst() {
            if logits[id] > logits[first] {
                second = first
                first = id
            } else if second < 0 || logits[id] > logits[second] {
                second = id
            }
        }
        let maximum = Double(logits[first])
        let total = logits.reduce(0) { $0 + exp(Double($1) - maximum) }
        func probability(_ id: Int) -> Double { exp(Double(logits[id]) - maximum) / total }
        self = .finite(
            TopTwo(
                first: Int32(first), firstProbability: probability(first), second: Int32(second),
                secondProbability: probability(second)))
    }
}
