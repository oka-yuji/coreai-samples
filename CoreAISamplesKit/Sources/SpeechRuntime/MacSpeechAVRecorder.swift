#if os(macOS)
    import AVFAudio
    import Foundation

    final class MacSpeechAVRecorder: MacSpeechRecordingDevice {
        private let recorder: AVAudioRecorder
        private let receiver: MacSpeechRecordingDelegate

        init(url: URL, settings: [String: Any], finished: @escaping @Sendable (Bool) -> Void) throws {
            recorder = try AVAudioRecorder(url: url, settings: settings)
            receiver = MacSpeechRecordingDelegate(finished: finished)
            recorder.delegate = receiver
        }

        func prepare() -> Bool {
            recorder.prepareToRecord()
        }

        func start(duration: TimeInterval) -> Bool {
            recorder.record(forDuration: duration)
        }

        func stop() {
            recorder.delegate = nil
            recorder.stop()
        }
    }
#endif
