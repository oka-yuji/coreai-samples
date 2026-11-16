public enum VoxCPMStopRule {
    private static let patchesPerTextToken = 6
    private static let extraPatches = 10
    private static let minimumLength = 2

    public static func patchLimit(textTokenCount: Int, capacity: Int) throws -> Int {
        guard textTokenCount >= 0 else { throw VoxCPMError.invalidTextTokenCount }
        let limit = min(patchesPerTextToken * textTokenCount + extraPatches, capacity - textTokenCount - 1)
        guard limit >= 1 else { throw VoxCPMError.textTooLong }
        return limit
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
