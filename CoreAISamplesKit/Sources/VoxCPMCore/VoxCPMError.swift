public enum VoxCPMError: Error, Equatable, Sendable {
    case unknownToken
    case invalidTextTokenCount
    case textTooLong
    case invalidSize
    case invalidPatchIndex
    case solverFinished
}
