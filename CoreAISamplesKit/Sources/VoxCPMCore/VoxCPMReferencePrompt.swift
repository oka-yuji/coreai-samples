public struct VoxCPMReferencePrompt: Equatable, Sendable {
    public let inputIDs: [Int32]
    public let audioPositions: Range<Int>
    public let textTokenCount: Int

    public init(textAndStartTokenIDs inputIDs: [Int32], referencePatchCount patches: Int, prefillLength length: Int)
        throws
    {
        guard inputIDs.count >= 2 else { throw VoxCPMError.invalidSize }
        guard inputIDs.last == VoxCPMPrompt.audioStartTokenID else { throw VoxCPMError.invalidSize }
        guard inputIDs.allSatisfy({ $0 >= 0 }) else { throw VoxCPMError.invalidSize }
        guard patches >= 0 else { throw VoxCPMError.invalidSize }
        guard length > 0 else { throw VoxCPMError.invalidSize }

        let textTokenCount = inputIDs.count - 1
        var validLength = inputIDs.count
        if patches > 0 {
            let patched = validLength.addingReportingOverflow(patches)
            guard !patched.overflow else { throw VoxCPMError.textTooLong }
            validLength = patched.partialValue
            let prefixed = validLength.addingReportingOverflow(2)
            guard !prefixed.overflow else { throw VoxCPMError.textTooLong }
            validLength = prefixed.partialValue
        }
        guard validLength <= length else { throw VoxCPMError.textTooLong }

        if patches == 0 {
            self.inputIDs = inputIDs
            self.audioPositions = 0..<0
        } else {
            self.inputIDs = [103] + [Int32](repeating: 0, count: patches) + [104] + inputIDs
            self.audioPositions = 1..<(patches + 1)
        }
        self.textTokenCount = textTokenCount
    }
}
