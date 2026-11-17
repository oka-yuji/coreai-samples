public enum VoxCPMError: Error, Equatable, Sendable {
    case emptyText
    case unknownToken
    case invalidTextTokenCount
    case textTooLong
    case invalidSize
    case invalidPatchIndex
    case solverFinished
}
