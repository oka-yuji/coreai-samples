import Testing
import VoxCPMCore

@Suite("VoxCPM2 patch limit and stop rule")
struct VoxCPMStopRuleTests {
    @Test(
        "The patch limit for T text tokens is min(6T + 10, capacity - T - 1)",
        arguments: [(1024, 28, 178), (1024, 144, 874), (1024, 145, 878), (1024, 1022, 1), (64, 7, 52), (64, 8, 55)])
    func patchLimit(capacity: Int, textTokenCount: Int, limit: Int) throws {
        #expect(try VoxCPMStopRule.patchLimit(textTokenCount: textTokenCount, capacity: capacity) == limit)
    }

    @Test(
        "A text that leaves no room for a single patch is rejected",
        arguments: [(1024, 1023), (1024, 4096), (64, 63)])
    func textWithoutRoom(capacity: Int, textTokenCount: Int) {
        #expect(throws: VoxCPMError.self) {
            try VoxCPMStopRule.patchLimit(textTokenCount: textTokenCount, capacity: capacity)
        }
    }

    @Test("With the fixture stop logits generation stops at patch 33, the first stop decision after patch 2")
    func fixtureStop() throws {
        let logits = try VoxCPMReference.load().stopLogits.map(VoxCPMReference.floats)
        try #require(logits.count == 34)
        let steps = try logits.indices.map {
            try VoxCPMStopRule.step(patchIndex: $0, limit: 178, stopLogits: logits[$0])
        }
        #expect(steps == [VoxCPMPatchStep](repeating: .next, count: 33) + [.stopped])
    }

    @Test("Reaching the patch limit is reported apart from a stop, and a stop at the last allowed patch is a stop")
    func limitReached() throws {
        let logits = try VoxCPMReference.load().stopLogits.map(VoxCPMReference.floats)
        try #require(logits.count == 34)
        let steps = try (0..<10).map { try VoxCPMStopRule.step(patchIndex: $0, limit: 10, stopLogits: logits[$0]) }
        #expect(steps == [VoxCPMPatchStep](repeating: .next, count: 9) + [.reachedLimit])
        #expect(try VoxCPMStopRule.step(patchIndex: 33, limit: 34, stopLogits: logits[33]) == .stopped)
    }

    @Test("Stop decisions at patches 0 through 2 are ignored, and equal logits are not a stop decision")
    func earlyStopsIgnored() throws {
        let steps = try (0..<4).map { try VoxCPMStopRule.step(patchIndex: $0, limit: 10, stopLogits: [0, 1]) }
        #expect(steps == [.next, .next, .next, .stopped])
        #expect(try VoxCPMStopRule.step(patchIndex: 3, limit: 10, stopLogits: [0.5, 0.5]) == .next)
    }

    @Test("Stop logits other than two values and patch indices outside the limit are rejected")
    func invalidSteps() {
        #expect(throws: VoxCPMError.self) { try VoxCPMStopRule.step(patchIndex: 3, limit: 10, stopLogits: [1]) }
        #expect(throws: VoxCPMError.self) { try VoxCPMStopRule.step(patchIndex: 3, limit: 10, stopLogits: [0, 1, 2]) }
        #expect(throws: VoxCPMError.self) { try VoxCPMStopRule.step(patchIndex: 10, limit: 10, stopLogits: [1, 0]) }
        #expect(throws: VoxCPMError.self) { try VoxCPMStopRule.step(patchIndex: -1, limit: 10, stopLogits: [1, 0]) }
    }
}
