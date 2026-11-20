#if os(macOS)
    import Foundation

    protocol MacSpeechRecordingDevice: AnyObject {
        func prepare() -> Bool
        func start(duration: TimeInterval) -> Bool
        func stop()
    }
#endif
