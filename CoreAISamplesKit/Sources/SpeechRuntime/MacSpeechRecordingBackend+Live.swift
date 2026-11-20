#if os(macOS)
    import AVFoundation
    import Foundation

    extension MacSpeechRecordingBackend {
        static let live = MacSpeechRecordingBackend(
            requestPermission: { answer in
                AVCaptureDevice.requestAccess(for: .audio, completionHandler: answer)
            },
            reserveFile: {
                let url = URL.temporaryDirectory.appending(path: "irodori-reference-\(UUID().uuidString).wav")
                try Data().write(to: url, options: .withoutOverwriting)
                return url
            },
            makeDevice: { url, settings, finished in
                try MacSpeechAVRecorder(url: url, settings: settings, finished: finished)
            })
    }
#endif
