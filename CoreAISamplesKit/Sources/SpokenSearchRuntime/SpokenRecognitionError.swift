public enum SpokenRecognitionError: Error, Equatable, Sendable {
    case unavailable
    case languageUnavailable
    case unreadableFile
    case recognitionFailed
    case busy
}
