#if os(macOS)
    enum MacSpeechRecordingEvent: Sendable {
        case permissionAnswered(Bool)
        case deviceFinished(Bool)
        case finishRequested
        case cancelled
    }
#endif
