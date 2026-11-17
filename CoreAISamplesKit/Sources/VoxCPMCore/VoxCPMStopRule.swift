public enum VoxCPMStopRule {
    public static let patchesPerTextToken = 6
    public static let extraPatches = 10
    public static let minimumLength = 2

    public static func patchLimit(textTokenCount: Int, capacity: Int) throws -> Int {
        guard textTokenCount >= 0 else { throw VoxCPMError.invalidTextTokenCount }
        let room = capacity.subtractingReportingOverflow(textTokenCount)
        let byCapacity = room.partialValue.subtractingReportingOverflow(1)
        guard !room.overflow, !byCapacity.overflow, byCapacity.partialValue >= 1 else {
            throw VoxCPMError.textTooLong
        }
        let scaled = patchesPerTextToken.multipliedReportingOverflow(by: textTokenCount)
        let byText = scaled.partialValue.addingReportingOverflow(extraPatches)
        guard !scaled.overflow, !byText.overflow else { return byCapacity.partialValue }
        return min(byText.partialValue, byCapacity.partialValue)
    }

    public static func step(patchIndex: Int, limit: Int, stopLogits: [Float]) throws -> VoxCPMPatchStep {
        guard stopLogits.count == 2 else { throw VoxCPMError.invalidSize }
        guard patchIndex >= 0, patchIndex < limit else { throw VoxCPMError.invalidPatchIndex }
        if patchIndex > minimumLength, stopLogits[1] > stopLogits[0] {
            return .stopped
        }
        return patchIndex == limit - 1 ? .reachedLimit : .next
    }
}
