#if os(macOS)
    import AVFAudio
    import Foundation

    final class MacSpeechRecordingDelegate: NSObject, AVAudioRecorderDelegate {
        private let finished: @Sendable (Bool) -> Void

        init(finished: @escaping @Sendable (Bool) -> Void) {
            self.finished = finished
        }

        func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
            finished(flag)
        }

        func audioRecorderEncodeErrorDidOccur(_ recorder: AVAudioRecorder, error: (any Error)?) {
            finished(false)
        }
    }
#endif
