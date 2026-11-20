import Testing
import VoxCPMCore

@Suite("VoxCPM2 patch limit and stop rule")
struct VoxCPMStopRuleTests {
    @Test(
        "The patch limit for T text tokens is min(6T + 10, capacity - T - 1)",
        arguments: [
            (1024, 28, 178), (1024, 144, 874), (1024, 145, 878), (1024, 1022, 1), (64, 7, 52), (64, 8, 55),
            (Int.max, Int.max / 2, Int.max - Int.max / 2 - 1),
        ])
    func patchLimit(capacity: Int, textTokenCount: Int, limit: Int) throws {
        #expect(try VoxCPMStopRule.patchLimit(textTokenCount: textTokenCount, capacity: capacity) == limit)
    }

    @Test(
        "A text that leaves no room for a single patch is rejected",
        arguments: [(1024, 1023), (1024, 4096), (64, 63), (1024, Int.max), (Int.min, 0)])
    func textWithoutRoom(capacity: Int, textTokenCount: Int) {
        #expect(throws: VoxCPMError.self) {
            try VoxCPMStopRule.patchLimit(textTokenCount: textTokenCount, capacity: capacity)
        }
    }

    @Test(
        "The patch limit of a prompt with T target tokens and length L is min(6T + 10, maximum patches, capacity - L)",
        arguments: [
            (64, 4_096, 22), (17, 4_096, 10), (8, 4_096, 1), (64, 5, 5), (Int.max, 4_096, 22), (Int.max, Int.max, 22),
        ]
    )
    func promptLimit(capacity: Int, maximumPatches: Int, limit: Int) throws {
        let prompt = try VoxCPMReferencePrompt(
            textAndStartTokenIDs: [11, 22, 101], referencePatchCount: 2, prefillLength: 7)
        #expect(
            try VoxCPMStopRule.patchLimit(prompt: prompt, capacity: capacity, maximumPatches: maximumPatches) == limit)
    }

    @Test("A prompt that leaves no cache position for a patch is rejected", arguments: [7, 6, 0, -1, Int.min])
    func promptWithoutRoom(capacity: Int) throws {
        let prompt = try VoxCPMReferencePrompt(
            textAndStartTokenIDs: [11, 22, 101], referencePatchCount: 2, prefillLength: 7)
        #expect(throws: VoxCPMError.textTooLong) {
            try VoxCPMStopRule.patchLimit(prompt: prompt, capacity: capacity, maximumPatches: 4_096)
        }
    }

    @Test("A maximum patch count below one is rejected", arguments: [0, -1, Int.min])
    func nonpositiveMaximum(maximumPatches: Int) throws {
        let prompt = try VoxCPMReferencePrompt(
            textAndStartTokenIDs: [11, 22, 101], referencePatchCount: 2, prefillLength: 7)
        #expect(throws: VoxCPMError.invalidSize) {
            try VoxCPMStopRule.patchLimit(prompt: prompt, capacity: 64, maximumPatches: maximumPatches)
        }
    }

    @Test(
        "Full 145-position prompts with five-second, ten-second or no reference audio keep the text budget",
        arguments: [(32, 110, 670), (63, 79, 484), (0, 144, 874)])
    func fullPromptLimit(patches: Int, tokens: Int, limit: Int) throws {
        let prompt = try VoxCPMReferencePrompt(
            textAndStartTokenIDs: [Int32](repeating: 11, count: tokens) + [101], referencePatchCount: patches,
            prefillLength: 145)
        #expect(prompt.inputIDs.count == 145 && prompt.textTokenCount == tokens)
        #expect(try VoxCPMStopRule.patchLimit(prompt: prompt, capacity: 1_024, maximumPatches: 4_096) == limit)
    }

    @Test("Without reference audio the prompt limit equals the text token limit", arguments: [1, 2, 28, 144, 145])
    func plainPromptLimit(tokens: Int) throws {
        let prompt = try VoxCPMReferencePrompt(
            textAndStartTokenIDs: [Int32](repeating: 11, count: tokens) + [101], referencePatchCount: 0,
            prefillLength: 8_192)
        let expected = try VoxCPMStopRule.patchLimit(textTokenCount: tokens, capacity: 1_024)
        #expect(try VoxCPMStopRule.patchLimit(prompt: prompt, capacity: 1_024, maximumPatches: 4_096) == expected)
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
