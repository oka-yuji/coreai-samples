public enum SpokenNotice: Equatable, Sendable {
    case unavailable
    case languageUnavailable
    case unreadableFile
    case recognitionFailed
    case noSpeech
    case playbackFailed
}
