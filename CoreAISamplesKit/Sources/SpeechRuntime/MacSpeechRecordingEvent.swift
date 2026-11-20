#if os(macOS)
    enum MacSpeechRecordingEvent: Sendable {
        case permissionAnswered(Bool)
        case deviceFinished(Bool)
        case finishRequested
        case deadlineFired(ContinuousClock.Instant)
        case cancelled
    }
#endif
