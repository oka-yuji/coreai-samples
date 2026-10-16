#if os(iOS)
    import AVFoundation
    import Foundation
    import Speech
    import TruckInspectionRuntime

    @MainActor
    final class LiveInspectionSpeech: TruckInspectionSpeechService {
        private let audio = InspectionAudioCapture()
        private let speaker = InspectionSpeaker()

        func prepare() async throws -> String {
            try Task.checkCancellation()
            let authorization = await withCheckedContinuation { continuation in
                SFSpeechRecognizer.requestAuthorization { @Sendable status in continuation.resume(returning: status) }
            }
            try Task.checkCancellation()
            guard authorization == .authorized else { throw InspectionSpeechError.permissionDenied }
            let microphone = await AVAudioApplication.requestRecordPermission()
            try Task.checkCancellation()
            guard microphone else { throw InspectionSpeechError.permissionDenied }
            _ = try InspectionRecognition.makeRecognizer()
            guard let voice = InspectionSpeaker.voice else { throw InspectionSpeechError.noJapaneseVoice }
            let route = try await audio.configure()
            return "permissions: microphone=granted speech=authorized on-device=ja-JP voice=\(voice.name) | \(route)"
        }

        func speak(_ text: String) async throws { try await speaker.speak(text) }

        func startListening(hints: [String]) async throws -> AsyncStream<InspectionSpeechEvent> {
            try await audio.start(hints: hints)
        }

        func endAudio() async { await audio.endAudio() }
        func finishListening() async { await audio.finish() }

        func stop() async -> String? {
            speaker.stop()
            return await audio.stopSession()
        }
    }
#endif
