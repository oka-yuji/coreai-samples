#if os(macOS) || os(iOS)
    enum VoxCPMGenerationError: Error, Equatable, Sendable {
        case invalidTokenIDs
        case textTooLong(Int)
        case reachedPatchLimit(Int)
        case nonFiniteValue(String)
        case invalidInput(String)
        case invalidOutput(String)
    }
#endif
