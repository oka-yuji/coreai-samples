#if os(macOS)
    import Foundation

    struct MacSpeechRecordingBackend: Sendable {
        let requestPermission: @Sendable (@escaping @Sendable (Bool) -> Void) -> Void
        let reserveFile: @Sendable () throws -> URL
        let makeDevice:
            @Sendable (URL, [String: Any], @escaping @Sendable (Bool) -> Void) throws -> any MacSpeechRecordingDevice
    }
#endif
