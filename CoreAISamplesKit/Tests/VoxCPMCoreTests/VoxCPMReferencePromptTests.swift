import Testing

@testable import VoxCPMCore

struct VoxCPMReferencePromptTests {
    @Test("VPR-01: No reference preserves the text and audio start token")
    func preservesTextOnlyInput() throws {
        let prompt = try VoxCPMReferencePrompt(
            textAndStartTokenIDs: [11, 22, 101], referencePatchCount: 0, prefillLength: 3)
        #expect(prompt.inputIDs == [11, 22, 101])
        #expect(prompt.audioPositions == 0..<0)
        #expect(prompt.textTokenCount == 2)
    }

    @Test("VPR-02: Reference positions precede the text in the upstream order")
    func arrangesReferencePrefix() throws {
        let prompt = try VoxCPMReferencePrompt(
            textAndStartTokenIDs: [11, 22, 101], referencePatchCount: 2, prefillLength: 7)
        #expect(prompt.inputIDs == [103, 0, 0, 104, 11, 22, 101])
        #expect(prompt.audioPositions == 1..<3)
        #expect(prompt.textTokenCount == 2)
    }

    @Test("VPR-02: Token values do not determine reference audio positions")
    func identifiesAudioByPosition() throws {
        let prompt = try VoxCPMReferencePrompt(
            textAndStartTokenIDs: [0, 103, 104, 101], referencePatchCount: 1, prefillLength: 7)
        #expect(prompt.inputIDs == [103, 0, 104, 0, 103, 104, 101])
        #expect(prompt.audioPositions == 1..<2)
        #expect(prompt.textTokenCount == 3)
    }

    @Test("VPR-03: Malformed text is rejected", arguments: [[Int32](), [101], [3, 100], [-1, 101]])
    func rejectsMalformedText(inputIDs: [Int32]) {
        #expect(throws: VoxCPMError.invalidSize) {
            try VoxCPMReferencePrompt(
                textAndStartTokenIDs: inputIDs, referencePatchCount: 0, prefillLength: 145)
        }
    }

    @Test("VPR-03: Negative reference counts are rejected", arguments: [-1, Int.min])
    func rejectsNegativeReferenceCount(referencePatches: Int) {
        #expect(throws: VoxCPMError.invalidSize) {
            try VoxCPMReferencePrompt(
                textAndStartTokenIDs: [7, 101], referencePatchCount: referencePatches, prefillLength: 145)
        }
    }

    @Test("VPR-03: Nonpositive prefill lengths are rejected", arguments: [0, -1, Int.min])
    func rejectsNonpositivePrefillLength(prefillLength: Int) {
        #expect(throws: VoxCPMError.invalidSize) {
            try VoxCPMReferencePrompt(
                textAndStartTokenIDs: [7, 101], referencePatchCount: 0, prefillLength: prefillLength)
        }
    }

    @Test("VPR-03: Exact fit succeeds and one position less is rejected", arguments: [0, 1, 2, 31, 32, 63])
    func checksPrefillBoundary(referencePatches: Int) throws {
        let length = referencePatches == 0 ? 2 : referencePatches + 4
        let prompt = try VoxCPMReferencePrompt(
            textAndStartTokenIDs: [7, 101], referencePatchCount: referencePatches, prefillLength: length)
        #expect(prompt.inputIDs.count == length)
        #expect(prompt.textTokenCount == 1)
        #expect(throws: VoxCPMError.textTooLong) {
            try VoxCPMReferencePrompt(
                textAndStartTokenIDs: [7, 101], referencePatchCount: referencePatches, prefillLength: length - 1)
        }
    }

    @Test("VPR-03: Huge patch counts are rejected without overflow", arguments: [Int.max, Int.max - 1, Int.max - 3])
    func rejectsHugePatchCount(referencePatches: Int) {
        #expect(throws: VoxCPMError.textTooLong) {
            try VoxCPMReferencePrompt(
                textAndStartTokenIDs: [7, 101], referencePatchCount: referencePatches, prefillLength: Int.max)
        }
    }

    @Test("VPR-03: A large capacity does not allocate padding")
    func returnsOnlyValidPositions() throws {
        let prompt = try VoxCPMReferencePrompt(
            textAndStartTokenIDs: [7, 101], referencePatchCount: 0, prefillLength: Int.max)
        #expect(prompt.inputIDs == [7, 101])
        #expect(prompt.audioPositions.isEmpty)
    }

    @Test("VPR-03: Reference input at the upstream capacity is accepted")
    func acceptsUpstreamCapacity() throws {
        let prompt = try VoxCPMReferencePrompt(
            textAndStartTokenIDs: [7, 101], referencePatchCount: 8_188, prefillLength: Int.max)
        #expect(prompt.inputIDs.count == 8_192)
        #expect(prompt.audioPositions == 1..<8_189)
        #expect(prompt.textTokenCount == 1)
    }

    @Test(
        "VPR-03: Reference input exceeding the upstream capacity is rejected",
        arguments: [8_189, Int.max - 4, Int.max / 8])
    func rejectsReferenceBeyondUpstreamCapacity(referencePatches: Int) {
        #expect(throws: VoxCPMError.textTooLong) {
            try VoxCPMReferencePrompt(
                textAndStartTokenIDs: [7, 101], referencePatchCount: referencePatches, prefillLength: Int.max)
        }
    }

    @Test("VPR-03: Text input obeys the upstream capacity without tensor padding")
    func checksTextAtUpstreamCapacity() throws {
        let input = [Int32](repeating: 7, count: 8_191) + [101]
        let prompt = try VoxCPMReferencePrompt(
            textAndStartTokenIDs: input, referencePatchCount: 0, prefillLength: Int.max)
        #expect(prompt.inputIDs == input)
        #expect(prompt.audioPositions.isEmpty)
        #expect(throws: VoxCPMError.textTooLong) {
            try VoxCPMReferencePrompt(
                textAndStartTokenIDs: input + [101], referencePatchCount: 0, prefillLength: Int.max)
        }
    }

}
