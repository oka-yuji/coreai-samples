enum LogitComparison: Equatable, Sendable {
    case finite(identical: Bool, maxAbsDiff: Float)
    case nonFinite(own: Int, official: Int)

    init(own: [Float16], official: [Float16]) {
        let ownCount = own.lazy.filter { !$0.isFinite }.count
        let officialCount = official.lazy.filter { !$0.isFinite }.count
        guard ownCount == 0, officialCount == 0 else {
            self = .nonFinite(own: ownCount, official: officialCount)
            return
        }
        var identical = own.count == official.count
        var difference: Float = 0
        for (a, b) in zip(own, official) where a.bitPattern != b.bitPattern {
            identical = false
            difference = max(difference, abs(Float(a) - Float(b)))
        }
        self = .finite(identical: identical, maxAbsDiff: difference)
    }
}
