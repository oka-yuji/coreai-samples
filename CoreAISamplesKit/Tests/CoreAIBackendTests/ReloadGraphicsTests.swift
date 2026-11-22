import Testing

@testable import CoreAIBackend

@Suite struct ReloadGraphicsTests {
    typealias Policy = LocalLanguageModel.ReloadPolicy

    static let qwen08 = (
        modelID: "Qwen/Qwen3.5-0.8B", mainHash: "41b46bd34a69c7a25e017122323e72c338ff1fb1a78f17888e5e04bb0c77a2ba"
    )
    static let off = Policy(readInterval: 0, footprintLimitBytes: 0)

    private func reloads(
        _ policy: Policy, state: ReloadState = ReloadState(), isNew: Bool = false, footprint: UInt64 = 0,
        growth: UInt64?, limit: UInt64
    ) -> Bool {
        state.shouldReload(
            policy: policy, isNewConversation: isNew, footprint: footprint, graphicsGrowth: growth, graphicsLimit: limit
        )
    }

    @Test("When the graphics growth is at or above its limit, a reload is selected with the other two conditions off")
    func growthAtLimitReloads() {
        #expect(reloads(Self.off, growth: 100, limit: 100))
        #expect(reloads(Self.off, isNew: true, footprint: .max, growth: 101, limit: 100))
    }

    @Test("When the growth is below its limit, unsampled or disabled, only the existing two conditions decide")
    func belowLimitKeepsExistingConditions() {
        var counted = ReloadState()
        for _ in 0..<10 { counted.completedRead() }
        let footprint = Policy(readInterval: 0, footprintLimitBytes: 100)
        let interval = Policy(readInterval: 10, footprintLimitBytes: 0)
        #expect(reloads(Self.off, growth: 100, limit: 100))
        #expect(!reloads(Self.off, growth: 99, limit: 100))
        #expect(!reloads(Self.off, growth: nil, limit: 100))
        #expect(!reloads(Self.off, growth: .max, limit: 0))
        #expect(reloads(footprint, footprint: 100, growth: 99, limit: 100))
        #expect(!reloads(footprint, footprint: 99, growth: nil, limit: 100))
        #expect(reloads(interval, state: counted, isNew: true, growth: nil, limit: 100))
        #expect(!reloads(interval, state: counted, isNew: false, growth: 99, limit: 100))
        #expect(counted.shouldReload(policy: footprint, isNewConversation: false, footprint: 100))
        #expect(!counted.shouldReload(policy: footprint, isNewConversation: false, footprint: 99))
    }

    @Test("Before a read finishes after a load, the growth is unknown and no baseline is taken")
    func noBaselineBeforeFinishedRead() {
        var state = ReloadState()
        var growth = (0..<3).map { _ in state.graphicsGrowth(sample: 100) }
        state.finishedRead()
        growth += [500, 700].map { state.graphicsGrowth(sample: $0) }
        #expect(growth == [nil, nil, nil, 0, 200])
    }

    @Test("A missing sample gives no growth and neither takes nor moves the baseline")
    func missingSampleKeepsBaseline() {
        var state = ReloadState()
        state.finishedRead()
        let samples: [UInt64?] = [nil, 500, nil, 900]
        let growth = samples.map { state.graphicsGrowth(sample: $0) }
        #expect(growth == [nil, 0, nil, 400])
    }

    @Test("A sample at or below the baseline gives zero growth and leaves the baseline in place")
    func lowerSampleGivesZero() {
        var state = ReloadState()
        state.finishedRead()
        let growth = [500, 500, 300, 700].map { state.graphicsGrowth(sample: $0) }
        #expect(growth == [0, 0, 0, 200])
    }

    @Test("After a load the growth is unknown until a read finishes, and the next sample becomes the new baseline")
    func loadRetakesBaseline() {
        var state = ReloadState()
        state.finishedRead()
        var growth = [500, 900].map { state.graphicsGrowth(sample: $0) }
        state.completedRead()
        state.didLoad()
        growth.append(state.graphicsGrowth(sample: 900))
        state.finishedRead()
        growth += [800, 1_000].map { state.graphicsGrowth(sample: $0) }
        #expect(growth == [0, 400, nil, 0, 200])
        #expect(state.completedReads == 0)
    }

    @Test("An explicit limit wins even at zero; otherwise only a table asset matching both model ID and hash has one")
    func limitResolution() {
        let (id, hash) = Self.qwen08
        let assets: [(String?, String?)] = [
            (id, hash), (id, "00"), ("Qwen/Qwen3-0.6B", hash), ("Qwen/Qwen3-0.6B", "00"), (id, nil), (nil, hash),
            (nil, nil),
        ]
        let byTable = Policy()
        let resolved = assets.map { byTable.graphicsGrowthLimit(modelID: $0.0, assetHash: $0.1) }
        #expect(resolved == [536_870_912, 0, 0, 0, 0, 0, 0])
        for explicit: UInt64 in [0, 1_000] {
            let policy = Policy(graphicsGrowthLimitBytes: explicit)
            let limits = assets.map { policy.graphicsGrowthLimit(modelID: $0.0, assetHash: $0.1) }
            #expect(limits == Array(repeating: explicit, count: assets.count))
        }
        #expect(
            Policy.verifiedGraphicsGrowthLimits.map { "\($0.modelID) \($0.mainHash) \($0.bytes)" }
                == ["\(id) \(hash) 536870912"])
    }

    @Test("The default policy has no explicit graphics growth limit and equality compares that field")
    func defaultAndEquality() {
        #expect(Policy().graphicsGrowthLimitBytes == nil)
        #expect(
            Policy(graphicsGrowthLimitBytes: 7)
                == Policy(readInterval: 10, footprintLimitBytes: 18_000_000_000, graphicsGrowthLimitBytes: 7))
        #expect(Policy(graphicsGrowthLimitBytes: 7) != Policy())
        #expect(Policy(graphicsGrowthLimitBytes: 0) != Policy())
        #expect(Policy(graphicsGrowthLimitBytes: 7) != Policy(graphicsGrowthLimitBytes: 8))
    }
}
